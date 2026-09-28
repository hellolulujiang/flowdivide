"""flowdivide.py -- the FlowDivide workflow from a flow-direction grid and the HydroBASINS Level-03
vectors to the partition, its views, the figures and the attributes, on one machine, step by step.

    python flowdivide.py run south-america --out-root /data/flowdivide
    python flowdivide.py run north-america --out-root /data/flowdivide --drop acc,aca,l3
    python flowdivide.py run merit-global  --out-root /data/flowdivide --steps fd1,fd2 --drop aca,l3
    python flowdivide.py run south-america --out-root /data/flowdivide --steps fd3 --attributes shv,ord --capacity 2^31
    python flowdivide.py run mygrid --dir /data/dir.tif --convention merit --group-vectors /data/basins.shp --out-root /data/flowdivide
    python flowdivide.py list

Three grids are built in (the three of the manuscript): HydroSHEDS v2 at 1 arc-second over South
America and over North America, and MERIT Hydro at 3 arc-seconds over the whole globe (periodic in
longitude).  Any other flow-direction grid runs with --dir and --convention, at any resolution and in any
coding: the codings of MERIT Hydro, ESRI, HydroSHEDS, TauDEM, GRASS and the degrees are known by name, and
any other is given as --convention custom --direction-codes E=1,SE=2,... with --sink-code, --nodata-code
and --mouth-code.  The basin groups come from --group-vectors, or are made here along a Hilbert curve.

The inputs are the flow directions and the Level-03 vectors, nothing else.  The chain, in the order of
the paper:

    FD1  fd1.0 recode -> fd1.4 the Level-03 vectors burned -> fd1.1 accumulation -> fd1.2 outlets
         -> fd1.3 delineation -> fd2.1 the basin views (the automatic grouping reads one)
         -> fd1.4 the basin groups: by Level-03 unit, and automatically along a Hilbert curve
         -> fd1.5 the Pfafstetter cut and the regions, once per capacity
    FD2  the Level-02 tables; the group and region views at every factor, as rasters and coloured polygons
    figures  Figures 3, 4 and 5 of the manuscript (South America and North America; not for MERIT)
    FD3  the attributes on one capacity: shv ldn hck lup lfp ord, or those asked for

The products are laid out as follows:

    <out-root>/<dataset>/global/fineresolution/{dir,upg,upa,str,l3,bsn}/   the continental rasters
    <out-root>/<dataset>/global/table/                                     outlets, basins, groups
    <out-root>/<dataset>/global/coarseresolution/{bsn,grp}/, global/vector/  the views (GeoParquet, GeoPackage or both: --vector)
    <out-root>/<dataset>/partitions/<capacity>/global/{table,preview,fineresolution,coarseresolution,vector}/
    <out-root>/<dataset>/figures/, _logs/

Every step leaves a marker in _logs/ with its arguments, its wall time, the time of its own checks and
its peak memory, so a run that stopped is taken up where it stopped and a step is never repeated with
the same arguments; _logs/timings_<dataset>.csv gathers the times.  Every raster written to disk is a
variable with a keep option (--drop): a dropped variable is deleted as soon as the last step of the run
that reads it is done.  The variables and the steps that read them:

    dir     the recoded flow directions       read by everything; never dropped
    acc     upstream pixel count (upg)        fd1.2, fd1.5 (the cut)
    aca     upstream area (upa)               fd1.2, fd1.5 (the cut, for the areas of the pieces), fd3 hck
    str     channel mask (>= 1 km2, 10 on MERIT)  fd1.5 (the cut), fd3 shv hck ord
    l3      the Level-03 codes burned         fd1.4
    bsn     the basin of every pixel          fd1.4, fd1.5, fd2, figures
    rgn     the region of every pixel         fd2, figures
    pieces  the piece rasters of the cut basins   fd1.5 (the regions), fd2 (the piece views), figures
    shv ldn hck lup lfp ord                   products of fd3 (lfp reads the member table of ldn, not its raster)
"""
import argparse
import glob
import hashlib
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd1_partition as fd1
import fd_tables
import fd2_views as fd2
import fd3_attributes as fd3
from fd1_partition import FlowDivideError, log

VERSION = "1.0.0"          # the package (pyproject.toml and fd2_views.FLOWDIVIDE_VERSION say the same)
RULES_VERSION = 1          # the rules that make the results; a step's marker is tied to this, not to the
                           # package version, so that a change of wording or timing reruns nothing
REGIONS_RULES_VERSION = 2  # the rules of fd1.5.regions_final that are not its arguments (the order of the
                           # joins of merge=level2, partition_config.txt)
FD2_VECTOR_RULES_VERSION = 4  # the rules of the polygons of stage 2: the boundary lines carry their table row as the
                           # GeoPackage fid; the polygons from 4-connected cells as GDAL gives
                           # them, OGC-valid, the boundary lines from the 8-connected ones (fd2_views.VECTOR_RULE_TAG);
                           # a GeoPackage view carries the default style on color_id (fd2_views.embed_color_id_style);
                           # named in the arguments of the steps that write polygons, so that polygons made under other
                           # rules are made again.  The automatic (Hilbert) grouping reads the basin views and so
                           # inherits the signature of fd2_basin_views: where it runs, it is made again as well, and a
                           # partition cut on the Hilbert groups with it, stage 3 included (the results do not change)
CHANNEL_RULES_VERSION = 2  # the test of a channel pixel in fd1.1: the threshold in the raster's unit and in float32
                           # against the float32 area, as in FD3 (a double threshold decides otherwise at
                           # 1.00000001 km2).  In fd1.1's arguments, so that a mask of another test is made again,
                           # and asked of fd1.1's marker when fd3 runs alone
FD3_RULES_VERSION = 5      # the rules of stage 3 alone: the distance and the upstream flow length cover every basin
                           # however small, their tables still list the basins of min_basin_area_km2 and more; the
                           # basin tables are printed column by column, with the longitudes and
                           # latitudes transformed on a projected grid; the lfp outputs carry .done markers;
                           # the distance raster's nodata is -1, and two donors of equal area are
                           # separated by their place in the window; a break in the channel
                           # mask is answered for by following the flow, not by the first break and the members'
                           # rectangles, so a product made under other rules may have passed a break it should
                           # have stopped at
# The inputs of the three built-in grids come from environment variables, with no default path: a run of a
# built-in grid whose inputs are not set stops and names the variables (unset_inputs).
HYBAS_ROOT_DEFAULT = os.environ.get("FLOWDIVIDE_HYBAS", "")
# HydroSHEDS v2's own mosaics, <continent>_DIR_1s_v2r0.tif with the _ACC_ and _ACA_ mosaics beside it.
# FLOWDIVIDE_HYDROSHEDS names the directory.
HYDROSHEDS_RAW_DEFAULT = os.environ.get("FLOWDIVIDE_HYDROSHEDS", "")
MERIT_RAW_DEFAULT = os.environ.get("FLOWDIVIDE_MERIT", "")
# MERIT Hydro's own upstream count and upstream area, which the MERIT 90 m products are built on: the same
# two files the published products were made from.  The published BasinFull v1.0 and v1.1 took their basin areas
# and their basin numbers from these rasters, and a re-run must carry the same numbers, so this line reads
# them instead of computing its own area.
MERIT_ACC_DEFAULT = os.environ.get("FLOWDIVIDE_MERIT_ACC", "")
MERIT_ACA_DEFAULT = os.environ.get("FLOWDIVIDE_MERIT_ACA", "")
# The outlet table of MERIT BasinFull v1.0 / v1.1, whose basin ids a re-run carries over pixel by pixel.
# Why carried over and not worked out again: no rule computed from today's rasters
# reproduces those ids (ordering one band of rows by MERIT's own Float32 upstream area agrees with the
# published numbers on 85.9% of neighbouring pairs, and the pairs that disagree are one or two Float32
# steps apart), because the areas the published run sorted on differ from the distributed raster by about
# 1e-5 and cannot be reconstructed.  A re-run gives every basin the id v1.0 and v1.1 published.
MERIT_BASIN_IDS_DEFAULT = os.environ.get("FLOWDIVIDE_MERIT_BASIN_IDS", "")
ALL_ATTRIBUTES = ["shv", "ldn", "hck", "lup", "lfp", "ord"]
DROPPABLE = ["acc", "aca", "str", "l3", "bsn", "rgn", "pieces"] + ALL_ATTRIBUTES
RASTER_FILE_NAME = {"acc": "upg", "aca": "upa"}       # the variables of the paper and their file names
PREVIEW_STEP = 30                                       # the piece preview of Figure 4: one cell per 30 x 30 pixels
GROUPINGS = ["l3", "hilbert"]
DISK_RESERVE_GB = 20.0
# what a step adds to the disk, in bytes per pixel of the grid, from the South America products at 1
# arc-second (the DIR 6.3 GB, the upstream count 22.7 GB and area about as much, the members under
# 1 GB, lup 44 GB, ldn 18 GB); a guard, not an accounting
DISK_BYTES_PER_PIXEL = {"fd1.0": 0.12, "fd1.1": 0.85, "fd1.2": 0.01, "fd1.3": 0.02, "fd1.4_level3": 0.01, "fd1.4_groups": 0.01, "fd1.5_cut": 0.02, "fd1.5_regions": 0.03,
                        "fd2": 0.03, "figures": 0.0, "fd3_shv": 0.02, "fd3_ldn": 0.35, "fd3_hck": 0.02, "fd3_lup": 0.80, "fd3_lfp": 0.01, "fd3_ord": 0.02}
METRES_PER_DEGREE = 40075017.0 / 360.0                 # along the equator (WGS84)


def view_name_of_factor(grid, factor):
    """the name of a view whose cell is `factor` pixels wide, for a grid this package has no fixed names for:
    the size of the cell, without a space and without a fraction, as a file name can carry it -- 100m, 1km,
    30s (arc-seconds on a geographic grid).  The name is only a name; the factor is what the view is made
    with, so any grid and any resolution gets a name a reader understands."""
    width = abs(grid.pixel_width) * factor
    if grid.geographic:
        seconds = width * 3600.0
        if abs(seconds - round(seconds)) < 1e-6:
            return "%ds" % int(round(seconds))
        per_degree = 1.0 / width
        if abs(per_degree - round(per_degree)) < 1e-6 * per_degree:
            return "%dth" % int(round(per_degree))
        return "%gdeg" % width
    if width >= 1000.0:
        return "%gkm" % (width / 1000.0)
    return "%gm" % width


def views_by_cell_size(grid, factors=(10, 50, 250)):
    """the views of a grid this package has no fixed names for: three coarse grids, each a whole number of
    pixels to a cell and a divisor of the block, so that a block holds whole cells; named by their cell size"""
    chosen = []
    for wanted in factors:
        factor = max((f for f in range(1, max(wanted, 1) + 1) if grid.block_pixels % f == 0), default=1)
        name = view_name_of_factor(grid, factor)
        if factor not in [f for _, f in chosen] and name not in [n for n, _ in chosen]:
            chosen.append((name, factor))
    return chosen or [(view_name_of_factor(grid, 1), 1)]


def resolution_text(pixel_width, geographic, factor=1):
    """the size of a pixel as a reader says it: the 1 and 3 arc-second grids by their usual names, 30 m
    and 90 m (not 1/3600 degree); any other geographic cell as the
    fraction of a degree with the metres along the equator, 1/400° (280 m); a projected grid in metres.
    The factor is that of a view, whose cell is factor pixels wide"""
    width = abs(pixel_width) * factor
    if not geographic:
        return "%g m" % width
    seconds = width * 3600.0
    if abs(seconds - round(seconds)) < 1e-6 and int(round(seconds)) in (1, 3):
        return "%d m" % int(round(seconds * METRES_PER_DEGREE / 3600.0 / 10.0) * 10)
    per_degree = 1.0 / width
    fraction = "1/%d\u00b0" % int(round(per_degree)) if abs(per_degree - round(per_degree)) < 1e-6 * per_degree else "%g\u00b0" % width
    metres = width * METRES_PER_DEGREE
    if metres >= 10000:
        about = "%d km" % int(round(metres / 1000.0))
    elif metres >= 1000:
        about = "%.1f km" % (metres / 1000.0)
    elif metres >= 20:
        about = "%d m" % int(round(metres / 10.0) * 10)
    elif metres >= 1:
        about = "%d m" % int(round(metres))
    else:
        about = "%.1f m" % metres
    return "%s (%s)" % (fraction, about)


# =============================================================================
#  [1] The datasets
# =============================================================================

class Dataset:
    """everything the chain needs to know about one grid"""

    def __init__(self, name, raw_dir, convention, group_layers=(), continent=None, level1=None, periodic=False, block_pixels=None, capacities=(), merge_policy="none",
                 land_cut_above=None, land_merge_below=0, island_reach_blocks=6, island_max_blocks=36, cut_cap_pixels=2 ** 31 - 1, views=(), hilbert=True,
                 figures=False, figure_basin_id=1, figure_basin_name="", figure_view=None, figure_ticks=None, figure3_size=(7.48, 4.3), figure3_stacked=False,
                 figure5_size=(7.48, 4.3), vectorise_basins=True, acc_path=None, aca_path=None, aca_unit="m2",
                 direction_codes=None, sink_code=None, nodata_code=None, mouth_code=None, level3_layers=None,
                 earth_model=fd1.EARTH_MODEL_WGS84_ZONE, basin_order="area-then-count", basin_id_table=None,
                 aca_single_pixel_tolerance=1.0e-3, channel_threshold_km2=1.0):
        self.name = name
        self.continent = continent or name                # the word in every file name
        self.level1 = level1                            # the level-1 number of the island and seam groups (6 South America, 7 North America, 0 the whole grid); None: from the codes
        self.acc_path = acc_path                        # a provider's upstream count and area given instead of fd1.1's own (None: computed)
        self.aca_path = aca_path
        self.aca_unit = aca_unit if aca_path else "m2"  # fd1.1's own area is in square metres; a given one in its unit (km2 for HydroSHEDS and MERIT)
        self.raw_dir = raw_dir                          # one file, or a directory of tiles
        self.convention = convention                    # a name of fd1.DIRECTION_CONVENTIONS, or "custom" with direction_codes
        self.direction_codes = direction_codes          # "E=1,SE=2,..." of a coding this package does not know by name
        self.sink_code = sink_code                      # the producer's value of an inland sink, no data and a river mouth,
        self.nodata_code = nodata_code                  # where they differ from the named coding's (None: the coding's own)
        self.mouth_code = mouth_code
        # the vectors that give the basin groups: HydroBASINS Level-03 for the three grids of the paper, any
        # polygons of whole basins for a grid of one's own (level3_layers, another name for it, is accepted too)
        self.group_layers = list(group_layers if group_layers else (level3_layers or []))
        self.periodic = periodic
        # which pixel area the products of this grid are published with: the exact area on the WGS84
        # ellipsoid, or MERIT Hydro's own on the MERIT 90 m grid, so that the upstream area is MERIT Hydro's
        # pixel for pixel.  It goes into every step's arguments, so changing it makes
        # the steps that computed an area run again.
        if earth_model not in fd1.EARTH_MODELS:
            raise ValueError("unknown earth model '%s': it must be one of %s" % (earth_model, ", ".join(fd1.EARTH_MODELS)))
        self.earth_model = earth_model
        # how the basins are numbered (fd1.BASIN_ORDER_RULES): the MERIT 90 m products keep the rule their
        # v1.0 and v1.1 were numbered with, so that a re-run carries the published basin ids
        if basin_order not in fd1.BASIN_ORDER_RULES:
            raise ValueError("unknown basin order '%s': it must be one of %s" % (basin_order, ", ".join(fd1.BASIN_ORDER_RULES)))
        if basin_order == "from-a-published-table" and basin_id_table is None:
            raise ValueError("the basin order 'from-a-published-table' needs basin_id_table")
        self.basin_order = basin_order
        self.basin_id_table = basin_id_table            # the table whose ids are carried over, with that rule
        # fd1.2: how far the upstream area of a one-pixel basin may be off the pixel area of its row
        # (0.1 %, and 0.5 % on MERIT, whose provider has an area formula of its own)
        self.aca_single_pixel_tolerance = aca_single_pixel_tolerance
        # the upstream area a channel pixel has at least on this grid, when --channel-threshold-km2 does not
        # say otherwise: 1 km2 on the HydroSHEDS 30 m grids, 10 km2 on MERIT 90 m
        self.channel_threshold_km2 = channel_threshold_km2
        self.block_pixels = block_pixels                # None: one degree on a geographic grid
        self.capacities = list(capacities)              # [(name, blocks, cut levels), ...], the largest first
        self.merge_policy = merge_policy
        self.land_cut_above = land_cut_above            # None: the capacity itself (no cut for the land)
        self.land_merge_below = land_merge_below
        self.island_reach_blocks = island_reach_blocks
        self.island_max_blocks = island_max_blocks
        self.cut_cap_pixels = cut_cap_pixels            # the needs_cut column of the outlet table
        self.views = list(views)                        # [(name, factor), ...]
        self.hilbert = hilbert                          # the automatic grouping as well (Figure 3c)
        self.figures = figures
        self.figure_basin_id = figure_basin_id
        self.figure_basin_name = figure_basin_name
        self.figure_view = figure_view                  # lon min, lon max, lat min, lat max
        self.figure_ticks = figure_ticks                # (longitude ticks, latitude ticks)
        self.figure3_size = figure3_size
        self.figure3_stacked = figure3_stacked
        self.figure5_size = figure5_size
        self.vectorise_basins = vectorise_basins

    @property
    def capacity_names(self):
        return [name for name, _, _ in self.capacities]

    @property
    def aca_to_km2(self):
        return 1.0 if self.aca_unit == "km2" else 1e-6

    @property
    def has_given_groups(self):
        """are the basin groups given as vectors (the l3 grouping), or made here along a Hilbert curve?"""
        return len(self.group_layers) > 0

    # other names for the two, still accepted
    level3_layers = property(lambda self: self.group_layers)
    has_level3 = property(lambda self: self.has_given_groups)


def earth_in_the_arguments(dataset):
    """" earth=<model>" when this grid is not on the default earth model, an empty string when it is: a
    step's arguments decide whether its marker still holds, and the grids published under the default must
    keep theirs"""
    return "" if dataset.earth_model == fd1.EARTH_MODEL_WGS84_ZONE else " earth=%s" % dataset.earth_model


def order_in_the_arguments(dataset):
    """the same for the basin numbering rule, with the table's identity when the ids are carried over"""
    if dataset.basin_order == "area-then-count":
        return ""
    if dataset.basin_order == "from-a-published-table":
        return " order=%s ids=%s" % (dataset.basin_order, file_identity(dataset.basin_id_table))
    return " order=%s" % dataset.basin_order


def hybas(root, code):
    return os.path.join(root, "hybas_%s_lev03_v1c.shp" % code)


def builtin_datasets(hybas_root=HYBAS_ROOT_DEFAULT, hydrosheds_raw=HYDROSHEDS_RAW_DEFAULT, merit_raw=MERIT_RAW_DEFAULT):
    """the three grids of the paper, with the settings of the paper.  A capacity is named by a
    power of two and realised as a round number of whole blocks below it: at 1 arc-second 2^31 pixels is
    165.7 blocks of one degree and the partition uses 160 (2^30: 80, 2^29: 40); on the 3 arc-second grid
    the tiers are 1,400, 700 and 350 blocks (round numbers under 2^31, 2^30 and 2^29).  The Pfafstetter
    cut goes three levels deep at the two larger capacities and four at the smallest.  The 30 m grids
    keep the land balance of the paper (a group of whole basins is also cut when it holds more than
    7.5e8 pixels of land, and a part under 1.5e8 is merged back); the 90 m grid looks at the window only
    and takes the Level-02 units as its regions (merge=level2); its basins are not vectorised and the
    automatic grouping is not made."""
    return {
        "south-america": Dataset(
            "south-america", os.path.join(hydrosheds_raw, "south-america_DIR_1s_v2r0.tif"), "hydrosheds", [(hybas(hybas_root, "sa"), None)], level1=6,
            # the provider's own upstream count and area: what a product uses is
            # the number that was downloaded, on both lines, so that the two are one standard.  fd1.1 then
            # makes the channel mask only.
            acc_path=os.path.join(hydrosheds_raw, "south-america_ACC_1s_v2r0.tif"),
            aca_path=os.path.join(hydrosheds_raw, "south-america_ACA_1s_v2r0.tif"), aca_unit="km2",
            capacities=[("2^31", 160, 3), ("2^30", 80, 3), ("2^29", 40, 4)], merge_policy="none", land_cut_above=750000000, land_merge_below=150000000,
            island_reach_blocks=6, island_max_blocks=36, cut_cap_pixels=1000000000, views=[("400th", 9), ("200th", 18), ("120th", 30), ("20th", 180)], hilbert=True,
            figures=True, figure_basin_id=1, figure_basin_name="Amazon", figure_view=(-93.0, -32.0, -57.0, 15.0), figure_ticks=([-90, -75, -60, -45], [-45, -30, -15, 0]),
            figure3_size=(7.48, 4.24), figure3_stacked=False, figure5_size=(7.48, 4.3), vectorise_basins=True),
        "north-america": Dataset(
            "north-america", os.path.join(hydrosheds_raw, "north-america_DIR_1s_v2r0.tif"), "hydrosheds",
            [(hybas(hybas_root, "na"), None), (hybas(hybas_root, "ar"), "PFAF_ID != 353")], level1=7,
            acc_path=os.path.join(hydrosheds_raw, "north-america_ACC_1s_v2r0.tif"),
            aca_path=os.path.join(hydrosheds_raw, "north-america_ACA_1s_v2r0.tif"), aca_unit="km2",
            capacities=[("2^31", 160, 3), ("2^30", 80, 3), ("2^29", 40, 4)], merge_policy="none", land_cut_above=750000000, land_merge_below=150000000,
            island_reach_blocks=6, island_max_blocks=36, cut_cap_pixels=1000000000, views=[("400th", 9), ("200th", 18), ("120th", 30), ("20th", 180)], hilbert=True,
            figures=True, figure_basin_id=1, figure_basin_name="Mississippi", figure_view=(-170.0, -52.0, 7.0, 84.0), figure_ticks=([-160, -140, -120, -100, -80, -60], [15, 30, 45, 60, 75]),
            figure3_size=(7.48, 13.8), figure3_stacked=True, figure5_size=(7.48, 5.0), vectorise_basins=True),
        "merit-global": Dataset(
            "merit-global", merit_raw, "merit", [(hybas(hybas_root, code), None) for code in ("af", "ar", "as", "au", "eu", "gr", "na", "sa", "si")], continent="global", level1=0,
            periodic=True, capacities=[("1400deg2", 1400, 3), ("700deg2", 700, 3), ("350deg2", 350, 4)], aca_single_pixel_tolerance=5.0e-3, merge_policy="level2", land_cut_above=None, land_merge_below=0,
            # the upstream count and the upstream area are MERIT Hydro's own rasters, read as they are, from
            # fd1.1 on.  So this line's basin areas are the published ones to the bit, and fd1.1 only makes
            # the channel mask.  MERIT's pixel area is CaMa-Flood's rgetara, 2.3e-3 smaller than the exact
            # figure on the WGS84 ellipsoid.  The areas of the pieces and the members are differences of
            # that same upstream area (the outlet's less the children's), so they are on MERIT's footing
            # too and add up to the basin exactly; the exact ellipsoid is what this code integrates pixel
            # areas with -- the region windows and the land share.
            acc_path=MERIT_ACC_DEFAULT, aca_path=MERIT_ACA_DEFAULT, aca_unit="km2",
            # The basin ids are the published ones, carried over pixel by pixel: a re-run gives every basin
            # the id v1.0 and v1.1 published.  No sort of today's areas gives
            # them back (see fd1.BASIN_ORDER_RULES), so the table is the only source of the ids.
            basin_order="from-a-published-table", basin_id_table=MERIT_BASIN_IDS_DEFAULT,
            island_reach_blocks=18, island_max_blocks=1296, cut_cap_pixels=2 ** 31 - 1, views=[("400th", 3), ("200th", 6), ("120th", 10), ("20th", 60)], hilbert=False,
            figures=False, vectorise_basins=False,
            # a river pixel of this grid has at least 10 km2 upstream, the fd3 default here
            channel_threshold_km2=10.0),
    }


def unset_inputs(name, steps, hybas_root, acc_given=False, attributes=()):
    """the inputs a run of the built-in grid `name` needs whose environment variable (or option) is not set;
    there is no default path.  The flow directions are read by every run, the upstream area by FD1 and by FD3
    when it computes the Hack order (hck), the other inputs by FD1 only"""
    unset = []
    if name in ("south-america", "north-america"):
        if not HYDROSHEDS_RAW_DEFAULT:
            unset.append("FLOWDIVIDE_HYDROSHEDS (the directory of HydroSHEDS v2's DIR, ACC and ACA mosaics)")
    else:
        if not MERIT_RAW_DEFAULT:
            unset.append("FLOWDIVIDE_MERIT (MERIT Hydro's flow directions, one file)")
        if "fd1" in steps and not MERIT_ACC_DEFAULT and not acc_given:
            unset.append("FLOWDIVIDE_MERIT_ACC (MERIT Hydro's upstream pixel count; or --acc)")
        if ("fd1" in steps or ("fd3" in steps and "hck" in attributes)) and not MERIT_ACA_DEFAULT:
            unset.append("FLOWDIVIDE_MERIT_ACA (MERIT Hydro's upstream area, read by FD1 and by FD3's hck)")
        if "fd1" in steps and not MERIT_BASIN_IDS_DEFAULT:
            unset.append("FLOWDIVIDE_MERIT_BASIN_IDS (the outlet table whose basin ids are carried over)")
    if "fd1" in steps and not hybas_root:
        unset.append("FLOWDIVIDE_HYBAS (the directory of the HydroBASINS Level-03 shapefiles; or --hybas-root)")
    return unset


def file_identity(path):
    """what identifies an input file's content well enough for the markers: its size and its modification
    time in nanoseconds (a shapefile: those of its .shp and .dbf)"""
    paths = [path]
    if path.lower().endswith(".shp"):
        paths += [path[:-4] + extension for extension in (".dbf", ".shx", ".prj")]
    return ";".join("%d:%d" % (os.path.getsize(p), os.stat(p).st_mtime_ns) if os.path.exists(p) else "absent" for p in paths)


def capacity_blocks_of(name, block_pixels):
    """a capacity written as 2^k, or as a whole number of blocks (160, 160deg2, 1400blocks): the number of whole blocks:
    the top tier, 2^31, is (2^31 - 1) / block^2 rounded down to two significant
    digits (160 blocks of 3600 pixels, 1400 of 1200), and every other tier that halved or doubled"""
    if name.startswith("2^"):
        power = int(name[2:])
        if not 20 <= power <= 40:
            raise FlowDivideError("the capacity must be 2^k with k from 20 to 40, got %s" % name)
        top = ((1 << 31) - 1) // (block_pixels * block_pixels)
        unit = 1
        while top // unit >= 100:
            unit *= 10
        top = (top // unit) * unit
        blocks = top >> (31 - power) if power <= 31 else top << (power - 31)
        if blocks < 1:
            raise FlowDivideError("%s is less than one block of %d x %d pixels" % (name, block_pixels, block_pixels))
        return blocks
    digits = name.replace("deg2", "").replace("blocks", "")
    if not digits.isdigit() or int(digits) <= 0:
        raise FlowDivideError("the capacity must be 2^k or a whole number of blocks such as 160deg2, got %s" % name)
    return int(digits)


# =============================================================================
#  [2] The layout of the output directory: the directories and file names
# =============================================================================

class Layout:
    """where every product of one dataset lives under the output root"""

    def __init__(self, root, dataset, grid=None):
        self.root = os.path.join(root, dataset.name)
        self.dataset = dataset
        self.name = dataset.continent
        self.grid = grid
        self.tag = None                                 # the resolution tag of the file names (1s, 3s, 10m), known with the grid
        self.external = {}                              # a variable given from outside (acc, aca): its path, never written or dropped here
        for directory in ("global/fineresolution/dir", "global/table", "global/coarseresolution/bsn", "global/coarseresolution/grp", "global/vector", "_logs", "figures"):
            fd1.ensure_directory(os.path.join(self.root, directory))

    def set_grid(self, grid):
        self.grid = grid
        if grid.geographic:
            seconds = abs(grid.pixel_width) * 3600.0
            self.tag = "%ds" % int(round(seconds)) if abs(seconds - round(seconds)) < 1e-6 else "%gs" % seconds
        else:
            self.tag = "%gm" % abs(grid.pixel_width)

    # ---- the continental products ----
    def raster(self, variable):
        """the file of a variable of stage 1: dir_<c>_<tag>_merit.tif, upg_, upa_, str_, l3_, bsn_; or the file
        given from outside for acc and aca"""
        if variable in self.external:
            return self.external[variable]
        if variable == "dir":
            return os.path.join(self.root, "global", "fineresolution", "dir", "dir_%s_%s_merit.tif" % (self.name, self.tag))
        file_name = RASTER_FILE_NAME.get(variable, variable)
        fd1.ensure_directory(os.path.join(self.root, "global", "fineresolution", file_name))
        return os.path.join(self.root, "global", "fineresolution", file_name, "%s_%s_%s.tif" % (file_name, self.name, self.tag))

    def dir_path_of(self, grid_of_native):
        """the recoded flow directions are named before the grid is known: from the native file"""
        self.set_grid(grid_of_native)
        return self.raster("dir")

    def table(self, name):
        return os.path.join(self.root, "global", "table", "%s_%s.csv" % (name, self.name))

    def basin_table(self):
        """the one basin table of the run"""
        return fd_tables.basin_table_path(self.root, self.name)

    def group_table(self, grouping):
        return self.table("group_%s_fine" % grouping)

    def basin_group_table(self, grouping):
        return self.table("basin_group_%s_fine" % grouping)

    def basin_view(self, view_name):
        return os.path.join(self.root, "global", "coarseresolution", "bsn", "bsn_%s_%s.tif" % (self.name, view_name))

    def group_view(self, grouping, view_name):
        return os.path.join(self.root, "global", "coarseresolution", "grp", "grp_%s_%s_%s.tif" % (grouping, self.name, view_name))

    def vector(self, stem, view_name):
        """bsn_, bnd_, grp_<g>_ polygons of one view: the path without its extension (.parquet, .gpkg: --vector)"""
        return os.path.join(self.root, "global", "vector", "%s_%s_%s" % (stem, self.name, view_name))

    @staticmethod
    def vector_files(stem_path, formats):
        """the files a vector stem is written as, one per format"""
        return [stem_path + (".parquet" if name == "geoparquet" else ".gpkg") for name in formats]

    # ---- the partitions ----
    def capacity_directory_name(self, blocks, grouping="l3"):
        """<N>deg2 on a geographic grid with blocks of one degree, else <N>blocks; a block other than the
        grid's own (one degree, or 5000 pixels on a projected grid) is part of the name,
        since two block sizes can give the same number of blocks but not the same windows"""
        grid = self.grid
        block_is_one_degree = grid is not None and grid.geographic and abs(grid.block_pixels * abs(grid.pixel_width) - 1.0) < 1e-6
        text = "%ddeg2" % blocks if block_is_one_degree else "%dblocks" % blocks
        if grid is not None and not block_is_one_degree and not (not grid.geographic and grid.block_pixels == 5000):
            text += "_block%d" % grid.block_pixels
        return text if grouping == "l3" else "hilbert_%s" % text

    def partition(self, blocks, grouping="l3"):
        directory = os.path.join(self.root, "partitions", self.capacity_directory_name(blocks, grouping))
        for sub in ("global/table", "global/preview", "global/fineresolution", "global/coarseresolution/rgn", "global/vector"):
            fd1.ensure_directory(os.path.join(directory, sub))
        return directory

    def partition_table(self, blocks, grouping, name):
        return os.path.join(self.partition(blocks, grouping), "global", "table", "%s_%s.csv" % (name, self.name))

    def partition_raster(self, blocks, grouping, variable):
        fd1.ensure_directory(os.path.join(self.partition(blocks, grouping), "global", "fineresolution", variable))
        return os.path.join(self.partition(blocks, grouping), "global", "fineresolution", variable, "%s_%s_%s.tif" % (variable, self.name, self.tag))

    def region_view(self, blocks, grouping, view_name, kind="tif"):
        """the region view raster, or (kind "vector") the path of its polygons without the extension"""
        if kind == "tif":
            return os.path.join(self.partition(blocks, grouping), "global", "coarseresolution", "rgn", "rgn_%s_%s.tif" % (self.name, view_name))
        return os.path.join(self.partition(blocks, grouping), "global", "vector", "rgn_%s_%s" % (self.name, view_name))

    def cut_listing(self, blocks, grouping):
        return os.path.join(self.partition(blocks, grouping), "global", "table", "cut_basins_%s.txt" % self.name)

    def piece_table(self, blocks, grouping, basin_id):
        return os.path.join(self.partition(blocks, grouping), "global", "table", "pfafstetter_pieces_%s_basin%d.csv" % (self.name, basin_id))

    def mainstem_table(self, blocks, grouping, basin_id):
        return os.path.join(self.partition(blocks, grouping), "global", "table", "pfafstetter_mainstem_%s_basin%d.csv" % (self.name, basin_id))

    def piece_codes(self, blocks, grouping, basin_id):
        return os.path.join(self.partition(blocks, grouping), "global", "table", "pfafstetter_codes_%s_basin%d.csv" % (self.name, basin_id))

    def piece_raster(self, blocks, grouping, basin_id, view_name=None):
        return os.path.join(self.partition(blocks, grouping), "global", "preview", "pfafstetter_pieces_%s_basin%d_%s.tif" % (self.name, basin_id, view_name or self.tag))

    def attribute(self, code, blocks, grouping="l3"):
        return self.partition_raster(blocks, grouping, code)

    def attribute_table(self, code, blocks, grouping="l3"):
        return self.partition_table(blocks, grouping, "%s_basin" % fd3.ATTRIBUTES[code]["table"])

    def attribute_member_table(self, code, blocks, grouping="l3"):
        return self.partition_table(blocks, grouping, "%s_member" % fd3.ATTRIBUTES[code]["table"])

    def piece_preview(self, blocks, grouping, basin_id):
        """the preview of the pieces of a cut basin, one cell per 30 x 30 pixels of its rectangle, the pixel at the
        corner of each cell"""
        return os.path.join(self.partition(blocks, grouping), "global", "preview", "pfafstetter_pieces_%s_basin%d_%s.tif" % (self.name, basin_id, self.preview_tag()))

    def preview_tag(self):
        grid = self.grid
        if grid.geographic:
            seconds = abs(grid.pixel_width) * 3600.0 * PREVIEW_STEP
            return "%ds" % int(round(seconds)) if abs(seconds - round(seconds)) < 1e-6 else "%gs" % seconds
        return "%gm" % (abs(grid.pixel_width) * PREVIEW_STEP)

    def cut_basins(self, blocks, grouping):
        """the basins a finished cut step listed"""
        listing = self.cut_listing(blocks, grouping)
        if not os.path.exists(listing):
            return []
        with open(listing) as handle:
            return [int(line) for line in handle.read().split()]

    def cuts(self, blocks, grouping):
        """{basin id: (piece table, piece raster)} of a finished cut step"""
        return {basin_id: (self.piece_table(blocks, grouping, basin_id), self.piece_raster(blocks, grouping, basin_id)) for basin_id in self.cut_basins(blocks, grouping)}

    def files_of_variable(self, variable, partitions):
        """every file on disk that belongs to a variable (for the keep option); partitions: [(blocks, grouping)]"""
        if variable in self.external:
            return []                                   # a file given from outside is never deleted
        if variable in ("acc", "aca", "str", "l3", "bsn", "dir"):
            return [self.raster(variable)]
        if variable == "rgn":
            return [self.partition_raster(b, g, variable) for b, g in partitions]
        if variable == "pieces":
            files = []
            for b, g in partitions:
                files += [raster for _, raster in self.cuts(b, g).values()]
            return files
        if variable in ALL_ATTRIBUTES:
            return [self.attribute(variable, b, g) for b, g in partitions]
        return []


# =============================================================================
#  [3] The chain: one step after another, with markers, timing and the keep option
# =============================================================================

def run_in_child_process(function, args):
    """one step run in a forked child process, so that its memory goes back to the system when it ends
    and its own peak memory is measured; the result, the peak and the time of the step's own checks
    come back through a pipe, an error is raised again in the parent"""
    import multiprocessing
    import traceback
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child():
        try:
            fd1.reset_timing()
            result = function(*args)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
            sender.send(("ok", result, peak, fd1.CHECK_SECONDS[0], dict(fd1.IO_SECONDS)))
        except BaseException as error:
            sender.send(("error", "%s: %s\n%s" % (type(error).__name__, error, traceback.format_exc()), 0, 0.0, {"input": 0.0, "output": 0.0}))
        finally:
            sender.close()

    process = context.Process(target=child)
    process.start()
    sender.close()
    try:
        status, payload, peak, check_seconds, io_seconds = receiver.recv()
    except EOFError:
        process.join()
        raise FlowDivideError("the step's process ended without a result (exit code %s); killed for memory?" % process.exitcode)
    process.join()
    if status != "ok":
        raise FlowDivideError("the step failed in its process:\n%s" % payload)
    return payload, peak, check_seconds, io_seconds


class Chain:
    """runs the steps of one dataset in order, keeps a marker per step, drops the variables that are
    not kept once nothing left in the run reads them"""

    def __init__(self, dataset, root, steps, attributes, capacity_name, drop, tile, min_basin_area_km2, channel_threshold_km2, open_figures, grouping="l3",
                 levels=None, vector_formats=("geoparquet",), colours=fd2.COLOURS_AT_LEAST, separate_processes=True):
        self.dataset = dataset
        self.separate_processes = separate_processes
        self.vector_formats = tuple(vector_formats)     # geoparquet, gpkg, or both (--vector)
        self.colours = colours                          # at least this many colour numbers (--colours)
        self.steps = steps
        self.attributes = attributes
        self.drop = set(drop)
        self.tile = tile
        self.min_basin_area_km2 = min_basin_area_km2
        self.channel_threshold_km2 = channel_threshold_km2
        self.open_figures = open_figures
        self.grouping = grouping                        # the grouping the partition is built on: l3 or hilbert
        self.levels = levels                            # None: the depth of each capacity from the dataset; "auto" or 2..4 for every capacity
        self.layout = Layout(root, dataset)
        self.summary = os.path.join(self.layout.root, "_logs", "chain_summary.txt")
        self.summary_only = False                       # --summary-only: the page and the plan, nothing run, nothing logged
        self.only = None                                # --only: the labels of the steps allowed to execute (None: every planned step)
        self.held_back = []                              # the planned steps --only held back, reported once
        self.timing = True                              # --timing on|off: the split of every step's time recorded, or only its total
        self.done = set()
        self.grid = None
        self.capacity_name = capacity_name
        self.blocks_of = {name: blocks for name, blocks, _ in dataset.capacities}
        self.levels_of = {name: levels for name, _, levels in dataset.capacities}
        if capacity_name not in self.blocks_of:
            raise FlowDivideError("the capacity %s is not one of the dataset's: %s" % (capacity_name, ", ".join(self.blocks_of)))
        if grouping == "hilbert" and not dataset.hilbert:
            raise FlowDivideError("the automatic grouping is not made for this dataset; the partition can only follow the Level-03 groups")
        if grouping == "l3" and not dataset.has_given_groups:
            raise FlowDivideError("no group vectors were given (--group-vectors): the partition can only follow the automatic groups (--groups hilbert)")
        for variable, path in (("acc", dataset.acc_path), ("aca", dataset.aca_path)):
            if path is not None:
                if not os.path.exists(path):
                    # the count is read by FD1 only, the area by FD1 and by FD3 when it computes the Hack order;
                    # any other run works from the products already on disk, and must not be stopped because the
                    # drive holding the provider's mosaics is not mounted
                    reads_it = "fd1" in self.steps or (variable == "aca" and "fd3" in self.steps
                                                       and "hck" in self.attributes)
                    if reads_it:
                        raise FlowDivideError("the %s raster given, %s, is not there" % (variable, path))
                    log("flowdivide", "the %s raster %s is not there; no step of this run reads it" % (variable, path))
                    continue
                self.layout.external[variable] = path
        if (dataset.acc_path is None) != (dataset.aca_path is None):
            raise FlowDivideError("the upstream count and the upstream area are given together or not at all")

    # ---- markers and the plan ----
    def marker(self, label):
        return os.path.join(self.layout.root, "_logs", label.replace("/", "_") + ".ok")

    def marker_signature(self, label):
        """the signature a finished step recorded in its marker, or None"""
        path = self.marker(label)
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            for line in handle.read().splitlines():
                if line.startswith("signature: "):
                    return line[len("signature: "):]
        return None

    def marker_arguments(self, label):
        """the arguments a finished step recorded in its marker ("args: ..."), or None"""
        path = self.marker(label)
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            for line in handle.read().splitlines():
                if line.startswith("args: "):
                    return line[len("args: "):]
        return None

    def marker_channel_threshold(self, label):
        """the channel threshold a finished fd1.1 recorded in its arguments ("threshold=..."), or None"""
        path = self.marker(label)
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            for line in handle.read().splitlines():
                if line.startswith("args: "):
                    found = re.search(r"threshold=([^ ]+)", line)
                    return float(found.group(1)) if found else None
        return None

    def signature(self, step, steps_by_label, cache):
        """what identifies a run of the step: its own arguments and the signatures of the steps it depends
        on, so that a change of a parameter upstream (the channel threshold, the merge policy) changes the
        signature of everything after it.  A dependency that is not in this run's list contributes the
        signature its marker recorded, or "missing" when it never ran."""
        if step.label in cache:
            return cache[step.label]
        parts = ["flowdivide rules %d" % RULES_VERSION, step.arguments]
        # an input from outside the run's own tree (the provider's mosaics, the Level-03 vectors) enters with its
        # size and its time to the nanosecond, so that a mosaic replaced in place makes the steps that read it run again
        # (without it, an ACA replaced in place leaves the Hack order of the old one standing)
        parts += self.external_identities(step)
        for label in step.depends_on:
            if label in steps_by_label:
                parts.append(self.signature(steps_by_label[label], steps_by_label, cache))
            else:
                # a step not in this run is taken from its marker only while the outside inputs it recorded are as
                # they were; an ACA replaced in place since fd1.1 leaves the channel mask of the old one, and fd3
                # alone would read it under a new signature
                changed = self.marker_external_that_changed(label)
                if changed:
                    raise FlowDivideError("%s was made from %s, which has changed since; run that step again (add its stage "
                                          "to --steps)" % (label, changed))
                parts.append(self.marker_signature(label) or "missing")
        cache[step.label] = hashlib.sha1("|".join(parts).encode("utf8")).hexdigest()[:16]
        return cache[step.label]

    def external_identities(self, step):
        """an input from outside the run's own tree (the provider's mosaics, the Level-03 vectors), as "input
        <path> <bytes> <time in ns>", so that a mosaic replaced in place makes the steps that read it run again
        (without it, an ACA replaced in place leaves the Hack order of the old one standing).  The provider's
        mosaics the run was given (acc, aca) enter every step, since what a step reads was made from them upstream even
        when the step does not read them itself"""
        root = os.path.abspath(self.layout.root) + os.sep
        external_paths = [path for path in getattr(step, "inputs", None) or [] if path]
        external_paths += [path for path in getattr(self.layout, "external", {}).values() if path]
        identities = []
        for path in sorted(set(external_paths)):
            if not os.path.abspath(path).startswith(root) and os.path.exists(path):
                status = os.stat(path)
                identities.append("input %s %d %d" % (os.path.abspath(path), status.st_size, status.st_mtime_ns))
        return identities

    def marker_external_that_changed(self, label):
        """the first outside input a finished step's marker recorded ("external: input <path> <bytes>
        <ns>") that is no longer of that size and time, or None; a marker without such lines says nothing"""
        path = self.marker(label)
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            for line in handle.read().splitlines():
                if not line.startswith("external: input "):
                    continue
                recorded_path, recorded_bytes, recorded_ns = line[len("external: input "):].rsplit(" ", 2)
                if not os.path.exists(recorded_path):
                    return recorded_path
                status = os.stat(recorded_path)
                if status.st_size != int(recorded_bytes) or status.st_mtime_ns != int(recorded_ns):
                    return recorded_path
        return None

    def is_done(self, step):
        return self.marker_signature(step.label) == step.signature

    @staticmethod
    def product_paths(step):
        """the paths of a step's products: its dict maps a variable to one path or to a list of paths"""
        paths = []
        for value in (getattr(step, "products", None) or {}).values():
            paths.extend(value if isinstance(value, (list, tuple)) else [value])
        return [path for path in paths if path]

    def an_output_lost_its_marker(self, step):
        """an output that is empty, or that carried a .done marker when the step finished and carries none now
        (a table left half written or its marker taken away by a failed rerun), is not taken as done"""
        for path in list(step.outputs) + self.product_paths(step):
            if os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) == 0:
                return True
        path = self.marker(step.label)
        if not os.path.exists(path):
            return False
        with open(path) as handle:
            for line in handle.read().splitlines():
                if line.startswith("done_marked: "):
                    return any(marked and not os.path.exists(marked + ".done")
                               for marked in line[len("done_marked: "):].split("|"))
        # a marker without that line does not say which outputs had .done markers; the step is made again once
        # so that its marker says it
        return True

    def note(self, text):
        with open(self.summary, "a") as handle:
            handle.write(text + "\n")
        print(text, flush=True)

    class Step:
        """one step of the chain as it is planned: its label, the arguments that identify a run of it,
        the function, the files it must leave (outputs), the variables it makes (products: name ->
        file or files), the files it needs (inputs), the steps it depends on, and the kind of step for
        the disk estimate"""

        def __init__(self, label, arguments, function, args, outputs=(), products=None, inputs=(), depends_on=(), kind=None,
                     refresh=None):
            self.label = label
            self.arguments = arguments
            self.function = function
            self.args = args
            self.outputs = list(outputs)
            self.products = products or {}
            self.inputs = list(inputs)
            self.depends_on = list(depends_on)          # the labels of the steps whose results this one reads
            self.kind = kind or label
            self.signature = None
            self.will_execute = False
            self.refresh = refresh                      # () -> (outputs, products), after the step ran

    def plan(self, steps):
        """Which steps will actually execute: a step runs when its marker is missing or carries another
        signature, when one of its outputs is missing, or when a product of it is missing that a step
        executing later needs, or that was not dropped.  Settled from the end backwards until nothing
        changes, so that a step made necessary by a later one is caught."""
        executing = {}
        # the steps planned earlier in this run (the recode) count as part of this plan: their signatures and
        # whether they executed carry over
        steps_by_label = dict(getattr(self, "planned", {}))
        steps_by_label.update({step.label: step for step in steps})
        for label, step in getattr(self, "planned", {}).items():
            executing[label] = step.will_execute
        cache = {}
        for step in steps:
            step.signature = self.signature(step, steps_by_label, cache)
            executing[step.label] = (not self.is_done(step)) or any(not os.path.exists(path) for path in step.outputs) \
                or self.an_output_lost_its_marker(step)
            if not executing[step.label] and step.label == "figures":
                with open(step.outputs[0]) as handle:
                    executing[step.label] = any(not os.path.exists(path) for path in handle.read().split("\n") if path)
        changed = True
        while changed:
            changed = False
            # a step that executes changes its results for the steps after it: they execute too
            for step in steps:
                if not executing[step.label] and any(executing.get(label, False) for label in step.depends_on):
                    executing[step.label] = True
                    changed = True
            for step in steps:
                if executing[step.label]:
                    continue
                for variable, paths in step.products.items():
                    for path in ([paths] if isinstance(paths, str) else paths):
                        if os.path.exists(path):
                            continue
                        needed = variable not in self.drop or any(executing.get(label, False) for label in self.consumers(variable))
                        if needed:
                            executing[step.label] = True
                            changed = True
                            break
                    if executing[step.label]:
                        break
        for step in steps:
            step.will_execute = executing[step.label]
        # --only: run just the steps named, whatever else the plan holds.  What it is for: a step whose
        # marker or product was lost invalidates every step after it, so a change in one late step would
        # drag the whole chain along, where only the products that change need to be made again.  A step
        # still refuses to run when a file it reads is not on disk, so this cannot make a product out of
        # nothing; what it can do is leave the tree with old and new parts in it, so the log says plainly
        # which planned steps were held back.
        if self.only is not None:
            held_back_here = []
            for step in steps:
                if step.will_execute and step.label not in self.only:
                    step.will_execute = False
                    held_back_here.append(step.label)
            if held_back_here:
                self.held_back.extend(held_back_here)
                self.note("--only holds back %d planned steps: %s" % (len(held_back_here), ", ".join(held_back_here)))
            # A step held back leaves its own products as they are, which is the point; what must not
            # happen is a selected step running on products that are out of date and then recording a
            # marker made with the signature of the step that did not run -- the next run would then
            # find everything in order.  So every step a selected step depends on,
            # directly or through others, must carry the signature this run works out for it.
            for step in steps:
                if not step.will_execute:
                    continue
                # the step it rests on directly is enough: a marker's signature is made of that step's
                # arguments and the signatures of the steps IT rests on, so a marker that still matches
                # says the whole chain above it is as it was.  A product that is gone rather than out of
                # date is caught by execute(), which refuses a step whose inputs are not on disk.
                for label in step.depends_on:
                    if label in self.only:
                        continue
                    other = steps_by_label.get(label)
                    if other is None:
                        continue
                    recorded = self.marker_signature(label)
                    if recorded != other.signature:
                        raise FlowDivideError(
                            "--only would run %s while %s, which it rests on, is out of date (%s); run %s as well, "
                            "or leave --only out" % (step.label, label,
                                                     "it has never run" if recorded is None else "its arguments or its own inputs have changed",
                                                     label))
        planned = dict(getattr(self, "planned", {}))
        planned.update({step.label: step for step in steps})
        self.planned = planned
        return steps

    def free_disk_gb(self):
        return shutil.disk_usage(self.layout.root).free / 1e9

    def check_disk(self, step):
        """the step must have room for what it adds, and the reserve, before it starts"""
        pixels = max(self.grid.nrow * self.grid.ncol if self.grid is not None else 0, getattr(self, "native_pixels", 0))
        needed = DISK_BYTES_PER_PIXEL.get(step.kind, 0.02) * pixels / 1e9
        free = self.free_disk_gb()
        if free < needed + DISK_RESERVE_GB:
            raise FlowDivideError("%s needs about %.0f GB of disk and the reserve of %.0f GB; %.0f GB are free under %s" % (step.label, needed, DISK_RESERVE_GB, free, self.layout.root))

    def execute(self, step):
        """one planned step: skipped when it need not execute, else its inputs and the disk checked, run,
        timed (its own checks apart), its peak memory noted, its marker written on success"""
        if not step.will_execute:
            self.note("%s done already" % step.label)
            self.done.add(step.label)
            return None
        for path in step.inputs:
            if not os.path.exists(path):
                raise FlowDivideError("%s needs %s, which is not on disk; it was dropped or its step has not run" % (step.label, path))
        self.check_disk(step)
        if os.path.exists(self.marker(step.label)):
            os.remove(self.marker(step.label))               # a step that fails must not keep the marker of an earlier success
        started = time.perf_counter()
        self.note("%s start %s :: %s" % (step.label, time.strftime("%m-%d %H:%M:%S"), step.arguments))
        if self.separate_processes:
            result, peak, check_seconds, io_seconds = run_in_child_process(step.function, step.args)
            how = "peak memory of the step"
        else:
            fd1.reset_timing()
            result = step.function(*step.args)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
            check_seconds = fd1.CHECK_SECONDS[0]
            io_seconds = dict(fd1.IO_SECONDS)
            how = "peak memory of the process so far"
        seconds = time.perf_counter() - started
        # the split of the step's time: input (reading), computing, output (writing), and its own checks apart
        compute_seconds = max(0.0, seconds - check_seconds - io_seconds["input"] - io_seconds["output"])
        if self.timing:
            self.note("%s done  %s  %.1f s = input %.1f + compute %.1f + output %.1f (+ checks %.1f)  %s %.1f GB" % (
                step.label, time.strftime("%m-%d %H:%M:%S"), seconds, io_seconds["input"], compute_seconds, io_seconds["output"], check_seconds, how, peak / 1e9))
        else:
            self.note("%s done  %s  %.1f s  %s %.1f GB" % (step.label, time.strftime("%m-%d %H:%M:%S"), seconds, how, peak / 1e9))
        # a step whose files are known only once it has run (the cut: which basins it cut) names them now, so that
        # the marker below lists their .done markers; planned before the first cut, the list would be empty and a
        # piece table that later lost its .done would pass as done
        if step.refresh is not None:
            step.outputs, step.products = step.refresh()
        with open(self.marker(step.label), "w") as handle:
            handle.write("%s version %s\nargs: %s\nsignature: %s\nseconds: %.1f\npeak_bytes: %d\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), VERSION, step.arguments, step.signature, seconds, peak))
            for identity in self.external_identities(step):
                handle.write("external: %s\n" % identity)
            # the outputs that carried their own .done marker when the step finished, asked for again on a resume
            done_marked = [path for path in list(step.outputs) + self.product_paths(step)
                           if os.path.exists(path + ".done")]
            handle.write("done_marked: %s\n" % "|".join(done_marked))
            if self.timing:
                handle.write("input_seconds: %.1f\ncompute_seconds: %.1f\noutput_seconds: %.1f\ncheck_seconds: %.1f\n" % (
                    io_seconds["input"], compute_seconds, io_seconds["output"], check_seconds))
        self.done.add(step.label)
        if self.timing:
            self.write_timings()
        return result

    def write_timings(self):
        """_logs/timings_<dataset>.csv from every marker (--timing on): the wall time of each step split
        into input (reading), compute and output (writing), the time of its own checks, the time without
        the checks (work_seconds, what the paper reports), the peak memory.  A marker written with
        --timing off has the total and the peak only"""
        rows = []
        for path in sorted(glob.glob(os.path.join(self.layout.root, "_logs", "*.ok"))):
            record = {"step": os.path.basename(path)[:-3], "finished": "", "seconds": np.nan, "input_seconds": np.nan, "compute_seconds": np.nan,
                      "output_seconds": np.nan, "check_seconds": np.nan, "peak_gb": np.nan}
            with open(path) as handle:
                for line in handle.read().splitlines():
                    for key in ("seconds", "input_seconds", "compute_seconds", "output_seconds", "check_seconds"):
                        if line.startswith(key + ": "):
                            record[key] = float(line.split()[1])
                    if line.startswith("peak_bytes: "):
                        record["peak_gb"] = int(line.split()[1]) / 1e9
                    elif " version " in line:
                        record["finished"] = line.split(" version ")[0].replace(" ", "T")
            record["work_seconds"] = record["seconds"] - (0.0 if np.isnan(record["check_seconds"]) else record["check_seconds"])
            rows.append(record)
        table = pd.DataFrame(rows, columns=["step", "finished", "seconds", "input_seconds", "compute_seconds", "output_seconds", "check_seconds", "work_seconds", "peak_gb"])
        fd1.write_table(table, os.path.join(self.layout.root, "_logs", "timings_%s.csv" % self.dataset.name), float_format="%.1f")

    def write_run_summary(self, steps):
        """one page, printed when the run starts and saved as _logs/run_summary_<dataset>.txt: what the run
        reads (the flow directions; the Level-03 vectors, or none; the upstream count and area, or none:
        computed), the options in effect, and every product with its resolution, its path, and whether it
        is kept or dropped (and after which step).  Every input and product carries its resolution as a
        reader says it, 1/3600 degree (30 m) or 1/1200 degree (90 m)"""
        dataset = self.dataset
        layout = self.layout
        grid = self.grid
        fine = resolution_text(grid.pixel_width, grid.geographic)
        views = ", ".join(resolution_text(grid.pixel_width, grid.geographic, factor) for _, factor in dataset.views)
        first_view = resolution_text(grid.pixel_width, grid.geographic, dataset.views[0][1])
        if grid.geographic:
            block_degrees = grid.block_pixels * abs(grid.pixel_width)
            block = "blocks of %d pixels (%s)" % (grid.block_pixels, "one degree" if abs(block_degrees - 1.0) < 1e-6 else resolution_text(grid.pixel_width, True, grid.block_pixels).split(" (")[0])
        else:
            block = "blocks of %d pixels (%g m)" % (grid.block_pixels, grid.block_pixels * abs(grid.pixel_width))
        lines = []
        add = lines.append
        add("FlowDivide %s   run %s   %s" % (VERSION, dataset.name, time.strftime("%Y-%m-%d %H:%M")))
        add("  command              python %s" % " ".join([os.path.basename(sys.argv[0])] + [a for a in sys.argv[1:] if a != "--summary-only"]))
        add("")
        add("inputs")
        add("  flow directions      %s" % dataset.raw_dir)
        if grid.geographic:
            seconds = abs(grid.pixel_width) * 3600.0
            unit = "%g arc-second%s (%s)" % (seconds, "" if abs(seconds - 1.0) < 1e-6 else "s", fine)
        else:
            unit = "%s pixels" % fine
        add("                       %s convention, %s, %d rows x %d columns%s" % (dataset.convention, unit, grid.nrow, grid.ncol, ", periodic in longitude" if grid.periodic else ""))
        add("                       %s" % block)
        # the basin groups the partition follows: predefined units given as polygons (HydroBASINS Level-03),
        # or none, and the basins are grouped automatically
        if dataset.has_given_groups:
            for k, (path, query) in enumerate(dataset.group_layers):
                add("  %-20s %s%s" % ("basin groups" if k == 0 else "", ("predefined units: " if k == 0 else "" ) + path, "   (%s)" % query if query else ""))
        else:
            add("  basin groups         none: the basins are grouped automatically along a Hilbert curve")
        add("  upstream count       %s" % ("%s, on the same grid, %s" % (dataset.acc_path, fine) if dataset.acc_path else "none: computed by fd1.1 at %s" % fine))
        add("  upstream area        %s" % ("%s (%s), on the same grid, %s" % (dataset.aca_path, dataset.aca_unit, fine) if dataset.aca_path else "none: computed by fd1.1 at %s, square metres" % fine))
        add("")
        add("options")
        add("  steps                %s" % ", ".join(self.steps))
        for k, (name, blocks, _) in enumerate(dataset.capacities):
            levels = self.cut_levels_of(name)
            add("  %-20s %s = %d blocks%s = %d pixels of %s, the cut %s" % (
                "capacities" if k == 0 else "", name, blocks, " (%d square degrees)" % blocks if grid.geographic and "one degree" in block else "",
                blocks * grid.block_pixels * grid.block_pixels, fine, "%s levels deep" % levels if levels else "as deep as the capacity needs (auto)"))
        add("  basin groups         %s%s" % ("the predefined units (Level-03)" if self.grouping == "l3" else "the automatic groups (along a Hilbert curve)",
                                              "; the automatic groups made as well, for Figure 3c" if self.grouping == "l3" and dataset.hilbert else ""))
        add("  merge policy         %s" % dataset.merge_policy)
        add("  channel threshold    %g km2;  attributes for basins of at least %g km2" % (self.channel_threshold_km2, self.min_basin_area_km2))
        add("  work tile            %d pixels of %s" % (self.tile, fine))
        add("  views                %s" % ";  ".join("%s = %d pixels a cell, basins of %.4f km2 and more" % (
            resolution_text(grid.pixel_width, grid.geographic, factor), factor, self.min_basin_area_km2_of_view(factor)) for _, factor in dataset.views))
        add("  vector files         %s; at least %d colour numbers" % (" and ".join("GeoParquet" if f == "geoparquet" else "GeoPackage" for f in self.vector_formats), self.colours))
        if "fd3" in self.steps:
            add("  attributes           %s on %s" % (", ".join(self.attributes), self.capacity_name))
        add("")
        add("outputs   under %s/" % layout.root)
        root = layout.root + "/"

        def relative(path):
            return path[len(root):] if path.startswith(root) else path

        def fate(variable):
            if variable not in self.drop:
                return "kept"
            readers = [label for label in self.consumers(variable) if label in self.planned]
            return "dropped after %s" % max(readers, key=lambda label: [s.label for s in steps].index(label)) if readers else "dropped at once"

        # name, what, resolution, path, fate; the path column is not padded, so a line stays as short as it can
        rows = []
        for variable, what in (("dir", "recoded flow directions"), ("acc", "upstream pixel count"),
                               ("aca", "upstream area (%s)" % dataset.aca_unit), ("str", "channel mask"),
                               ("l3", "predefined groups on the grid"), ("bsn", "basin of every pixel")):
            if variable == "l3" and not dataset.has_given_groups:
                continue
            if variable in layout.external:
                rows.append((variable, what, fine, "given, not written", ""))
                continue
            rows.append((variable, what, fine, relative(layout.raster(variable)), fate(variable)))
        rows.append(("tables", "outlets, basins, groups", fine, relative(os.path.join(layout.root, "global", "table")) + "/", ""))
        rows.append(("views", "basin and group views, rasters and polygons", views, "global/coarseresolution/, global/vector/", ""))
        for name, blocks, _ in dataset.capacities:
            rows.append((name, "regions, pieces, views", fine, relative(layout.partition(blocks, self.grouping)) + "/",
                         "rgn %s, pieces %s" % (fate("rgn"), fate("pieces"))))
        if "fd3" in self.steps:
            blocks = self.blocks_of[self.capacity_name]
            for code in self.attributes:
                rows.append((code, fd3.ATTRIBUTES[code]["name"], fine, relative(layout.attribute(code, blocks, self.grouping)), fate(code)))
        if dataset.figures and "figures" in self.steps:
            rows.append(("figures", "Figures 3, 4 and 5", first_view, "figures/", ""))
        rows.append(("logs", "markers, summary" + (", timings: input, compute, output" if self.timing else ""), "", "_logs/" + (", _logs/timings_%s.csv" % dataset.name if self.timing else ""), ""))
        name_width = max(len(row[0]) for row in rows)
        what_width = max(len(row[1]) for row in rows)
        resolution_width = max(len(row[2]) for row in rows if row[2] == fine or row[2] == first_view)
        for name, what, resolution, path, what_becomes in rows:
            add(("  %-*s  %-*s  %-*s  %s  %s" % (name_width, name, what_width, what, resolution_width, resolution, path, what_becomes)).rstrip())
        text = "\n".join(lines)
        with open(os.path.join(layout.root, "_logs", "run_summary_%s.txt" % dataset.name), "w", encoding="utf8") as handle:
            handle.write(text + "\n")
        self.note(text)

    # ---- the keep option ----
    def consumers(self, variable):
        """the labels of the steps that read the variable (the figures keep the fine masks until they are
        drawn, although they draw from the views)"""
        dataset = self.dataset
        capacities = [self.partition_label(name) for name in dataset.capacity_names]
        chosen = self.partition_label(self.capacity_name)
        labels = set()
        if variable == "acc":
            labels |= {"fd1.2"} | {"fd1.5_cut_%s" % name for name in capacities}
        elif variable == "aca":
            labels |= {"fd1.2"} | {"fd1.5_cut_%s" % name for name in capacities}
            if "hck" in self.attributes:
                labels.add("fd3_hck_%s" % chosen)
        elif variable == "str":
            labels |= {"fd1.5_cut_%s" % name for name in capacities}
            labels |= {"fd3_%s_%s" % (code, chosen) for code in self.attributes if code in ("shv", "hck", "ord")}
        elif variable == "l3":
            labels |= {"fd1.4_groups_l3"}
        elif variable == "bsn":
            labels |= {"fd1.4_groups_l3", "fd1.4_groups_hilbert", "fd2_basin_views", "figures"} | {"fd2_group_views_%s" % g for g in ("l3", "l2", "l1", "hilbert")}
            labels |= {"fd1.5_cut_%s" % name for name in capacities} | {"fd1.5_regions_%s" % name for name in capacities}
            labels |= {"fd2_region_views_%s" % name for name in capacities}          # the region view reads the basin mask beside the region mask
        elif variable == "rgn":
            labels |= {"fd2_region_views_%s" % name for name in capacities} | {"figures"}
        elif variable == "pieces":
            labels |= {"fd1.5_regions_%s" % name for name in capacities} | {"fd2_piece_views", "figures"}
        return labels

    def partition_label(self, capacity_name):
        """the word that names a partition in the step labels: the capacity, with the grouping in front
        when it is not the Level-03 one"""
        return capacity_name if self.grouping == "l3" else "%s_%s" % (self.grouping, capacity_name)

    def still_needed(self, variable):
        """does a planned step that has still to execute read the variable"""
        return any(label in self.planned and self.planned[label].will_execute and label not in self.done for label in self.consumers(variable))

    def drop_what_is_no_longer_read(self):
        """delete every dropped variable that no step still to execute reads"""
        partitions = [(blocks, self.grouping) for _, blocks, _ in self.dataset.capacities]
        for variable in sorted(self.drop):
            if self.still_needed(variable):
                continue
            for path in self.layout.files_of_variable(variable, partitions):
                if os.path.exists(path):
                    os.remove(path)
                    for side in (".report.json", ".objects.csv"):
                        if os.path.exists(path + side):
                            os.remove(path + side)
                    self.note("dropped %s (%s): no step still to run reads it" % (variable, path))

    # ---- the steps ----
    def run(self):
        dataset = self.dataset
        layout = self.layout
        self.note("==== flowdivide %s %s %s steps=%s attributes=%s capacity=%s grouping=%s drop=%s ====" % (
            VERSION, dataset.name, time.strftime("%Y-%m-%d %H:%M"), ",".join(self.steps), ",".join(self.attributes), self.capacity_name, self.grouping, ",".join(sorted(self.drop))))
        Step = self.Step
        # the recode comes first on its own: the grid is known only after it (the file is named from the native grid)
        native = dataset.raw_dir
        if not os.path.exists(native):
            raise FlowDivideError("the flow directions %s are not there" % native)
        if os.path.isdir(native):
            candidates = sorted(glob.glob(os.path.join(native, "*.tif")))
            if not candidates:
                raise FlowDivideError("no .tif tile in %s" % native)
            layout.set_grid(fd1.Grid(candidates[0], periodic=dataset.periodic, block_pixels=dataset.block_pixels, earth_model=dataset.earth_model))
        else:
            candidates = [native]
            layout.set_grid(fd1.Grid(native, periodic=dataset.periodic, block_pixels=dataset.block_pixels, earth_model=dataset.earth_model))
        # the native file's size and time are part of what identifies the recode, so a replaced input is seen
        identity = ";".join("%s:%s" % (os.path.basename(p), file_identity(p)) for p in candidates)
        dir_path = layout.raster("dir")
        if "fd1" in self.steps:
            # the coding the recode is given is part of what identifies it: a grid of one's own whose
            # codes, sink, nodata or mouth are named on the command line would otherwise keep the raster
            # an earlier run wrote under another coding.  Only a coding that was actually given is named,
            # so that a grid that takes its convention's own codes keeps the signature it has without one
            coding = ""
            if dataset.direction_codes or dataset.sink_code is not None or dataset.nodata_code is not None or dataset.mouth_code is not None:
                # the codes as the convention reads them, not as they were typed, so that two spellings
                # of one coding are one signature (--direction-codes arrives as the text "E=1,SE=2,...")
                convention = fd1.direction_convention(dataset.convention, codes=dataset.direction_codes,
                                                      sink=dataset.sink_code, nodata=dataset.nodata_code,
                                                      mouth=dataset.mouth_code)
                coding = " codes=%s sink=%s nodata=%s mouth=%s" % (
                    ",".join("%s=%s" % (name, code) for name, code in sorted(convention["codes"].items())),
                    convention["sink"], convention["nodata"], convention["mouth"])
            first = self.plan([Step("fd1.0", "recode %s -> %s convention=%s%s periodic=%s input=%s" % (native, dir_path, dataset.convention, coding, dataset.periodic, identity),
                                    self.step_fd1_0, (dir_path,), outputs=[dir_path], kind="fd1.0")])
            self.grid = layout.grid
            # the footprint the tiles assemble to (the whole rectangle they span), for the disk estimate; the
            # assembled native file and the recoded one lie on the disk together
            left = bottom = float("inf")
            right = top = float("-inf")
            for path in candidates:
                with fd1.rasterio.open(path) as tile:
                    left, bottom = min(left, tile.bounds.left), min(bottom, tile.bounds.bottom)
                    right, top = max(right, tile.bounds.right), max(top, tile.bounds.top)
            self.native_pixels = int(round((right - left) / abs(layout.grid.pixel_width)) * round((top - bottom) / abs(layout.grid.pixel_height))) * (2 if len(candidates) > 1 else 1)
            if not self.summary_only:
                self.execute(first[0])
        if self.summary_only and not os.path.exists(dir_path):
            # the page before anything ran: the grid of the recode is that of the native file (of the rectangle
            # the tiles span, when there are several)
            self.grid = layout.grid
            if len(candidates) > 1:
                left = bottom = float("inf")
                right = top = float("-inf")
                for path in candidates:
                    with fd1.rasterio.open(path) as tile:
                        left, bottom = min(left, tile.bounds.left), min(bottom, tile.bounds.bottom)
                        right, top = max(right, tile.bounds.right), max(top, tile.bounds.top)
                self.grid.ncol = int(round((right - left) / abs(self.grid.pixel_width)))
                self.grid.nrow = int(round((top - bottom) / abs(self.grid.pixel_height)))
        else:
            if not os.path.exists(dir_path):
                raise FlowDivideError("the recoded flow directions %s are not there; run fd1 first" % dir_path)
            self.grid = fd1.Grid(dir_path, periodic=dataset.periodic, block_pixels=dataset.block_pixels, earth_model=dataset.earth_model)
        layout.set_grid(self.grid)
        grid = self.grid
        self.note("grid %d x %d pixels, block %d pixels, %s, %s, tag %s" % (grid.ncol, grid.nrow, grid.block_pixels, "geographic" if grid.geographic else "projected",
                                                                             "periodic in longitude" if grid.periodic else "not periodic", layout.tag))
        views_text = ",".join("%s:%d" % (n, f) for n, f in dataset.views) + " vector=%s colours=%d rule=%d" % ("+".join(self.vector_formats), self.colours, fd2.VIEW_RULE)
        first_view = dataset.views[0][0]
        grouping = self.grouping
        steps = []
        basin_views_needed = "fd2" in self.steps or ("fd1" in self.steps and dataset.hilbert)
        if "fd1" in self.steps:
            if dataset.has_given_groups:
                layers_text = ";".join("%s:%s:%s" % (p, q or "", file_identity(p)) for p, q in dataset.group_layers)
                # burn=gdal: GDAL's RasterizeLayer on the whole output; a raster burned otherwise is made again
                steps.append(Step("fd1.4_level3", "layers=%s burn=gdal" % layers_text, fd1.fd1_4_rasterise_level3, (dir_path, layout.raster("l3"), grid, dataset.group_layers),
                                  products={"l3": layout.raster("l3")}, depends_on=["fd1.0"]))
            if dataset.acc_path is None:
                # the earth model and the numbering rule go in the arguments only when they are not the
                # default, so that a grid whose products were made under the default keeps its markers:
                # naming the default in the string would invalidate every 30 m fd1.1 and fd1.2 marker and
                # recompute hours of work that would come out the same
                steps.append(Step("fd1.1", "tile=%d threshold=%r%s channel_rules=%d" % (self.tile, float(self.channel_threshold_km2), earth_in_the_arguments(dataset), CHANNEL_RULES_VERSION), fd1.fd1_1_flow_accumulation,
                                  (dir_path, layout.raster("acc"), layout.raster("aca"), layout.raster("str"), grid, self.tile, self.channel_threshold_km2),
                                  products={"acc": layout.raster("acc"), "aca": layout.raster("aca"), "str": layout.raster("str")}, depends_on=["fd1.0"]))
            else:
                # the upstream count and area are given: only the channel mask is made from the area
                # the unit the given area is read in decides the mask, so a change of --aca-unit has to
                # make this step run again; it is named only when it is not km2, the unit the step
                # assumes, so that products made in km2 keep their markers
                unit_text = "" if dataset.aca_to_km2 == 1.0 else " aca_to_km2=%r" % dataset.aca_to_km2
                steps.append(Step("fd1.1", "channel mask from %s (%s) threshold=%r%s acc=%s channel_rules=%d" % (dataset.aca_path, file_identity(dataset.aca_path), float(self.channel_threshold_km2), unit_text, file_identity(dataset.acc_path), CHANNEL_RULES_VERSION),
                                  fd1.fd1_1_channel_mask_from_area, (layout.raster("aca"), layout.raster("str"), grid, self.channel_threshold_km2, dataset.aca_to_km2),
                                  products={"str": layout.raster("str")}, inputs=[layout.raster("acc"), layout.raster("aca")], depends_on=["fd1.0"]))
            # one basin table: basin_table_fine_<run>.csv, begun by fd1.2 and filled in
            # place by fd1.3, fd1.4 and fd1.6; global_basin_id is basin_id when the grid is the globe
            steps.append(Step("fd1.2", "basin table cut_cap=%d aca_to_km2=%r%s%s" % (dataset.cut_cap_pixels, dataset.aca_to_km2, order_in_the_arguments(dataset), earth_in_the_arguments(dataset)), fd1.fd1_2_outlet_indexing,
                              (dir_path, layout.raster("acc"), layout.raster("aca"), layout.basin_table(), grid, dataset.cut_cap_pixels, dataset.aca_to_km2, 2048, "fd1.2", dataset.basin_order, dataset.basin_id_table,
                               dataset.continent == "global", dataset.aca_single_pixel_tolerance),
                              outputs=[layout.basin_table()], inputs=[layout.raster("acc"), layout.raster("aca")], depends_on=["fd1.1"]))
            steps.append(Step("fd1.3", "tile=%d basin table in place" % self.tile, fd1.fd1_3_watershed_delineation, (dir_path, layout.basin_table(), layout.raster("bsn"), grid, self.tile),
                              outputs=[layout.basin_table()], products={"bsn": layout.raster("bsn")}, depends_on=["fd1.2"]))
        if basin_views_needed:
            basin_view_files = [layout.basin_view(n) for n, _ in dataset.views] + [layout.basin_view(n) + ".objects.csv" for n, _ in dataset.views]
            if dataset.vectorise_basins:
                for n, _ in dataset.views:
                    basin_view_files += layout.vector_files(layout.vector("bsn", n), self.vector_formats) + layout.vector_files(layout.vector("bnd", n), self.vector_formats)
            vector_rules_text = " vector_rules=%d" % FD2_VECTOR_RULES_VERSION if dataset.vectorise_basins else ""
            steps.append(Step("fd2_basin_views", "views=%s vectorise=%s%s" % (views_text, dataset.vectorise_basins, vector_rules_text), self.step_fd2_basin_views, (), outputs=basin_view_files,
                              inputs=[layout.raster("bsn")], depends_on=["fd1.3"], kind="fd2"))
        if "fd1" in self.steps:
            if dataset.has_given_groups:
                steps.append(Step("fd1.4_groups_l3", "reach=%d max=%d block=%d level1=%s levels=l3,l2,l1" % (dataset.island_reach_blocks, dataset.island_max_blocks, grid.block_pixels, dataset.level1), fd1.fd1_4_group_whole_basins,
                                  (layout.raster("bsn"), layout.raster("l3"), layout.basin_table(), grid, dataset.island_reach_blocks, dataset.island_max_blocks, dataset.level1),
                                  outputs=[layout.group_table("l3"), layout.group_table("l2"), layout.group_table("l1"), layout.basin_table()],
                                  inputs=[layout.raster("bsn"), layout.raster("l3")], depends_on=["fd1.3", "fd1.4_level3"], kind="fd1.4_groups"))
            if dataset.hilbert:
                first_name, first_blocks, _ = dataset.capacities[0]
                capacity_pixels = first_blocks * grid.block_pixels * grid.block_pixels
                steps.append(Step("fd1.4_groups_hilbert", "capacity=%d block=%d view=%s" % (capacity_pixels, grid.block_pixels, first_view), fd1.fd1_4_group_by_hilbert_curve,
                                  (layout.basin_table(), layout.basin_view(first_view), grid, capacity_pixels, "hilbert"),
                                  outputs=[layout.group_table("hilbert"), layout.basin_group_table("hilbert")], inputs=[layout.basin_view(first_view)], depends_on=["fd1.3", "fd2_basin_views"], kind="fd1.4_groups"))
            for name, blocks, levels in dataset.capacities:
                capacity_pixels = blocks * grid.block_pixels * grid.block_pixels
                depth = self.cut_levels_of(name)
                label = self.partition_label(name)
                cut_files, cut_products = self.cut_step_files(blocks, grouping)
                # the earth model is in these arguments because the cut reads the pixel area too: it turns
                # the channel threshold into a pixel count (fd1_partition.BasinCut.channel_pixels_max)
                # So is the margin of that count (fd1_partition.CHANNEL_BOUND_MARGIN), so that a cut made under
                # another bound, which a provider's smaller pixel could slip past, is made again and not taken
                # as done
                steps.append(Step("fd1.5_cut_%s" % label, "capacity=%d block=%d tile=%d levels=%s threshold=%r bound_margin=%r grouping=%s%s" % (capacity_pixels, grid.block_pixels, self.tile, depth, float(self.channel_threshold_km2), fd1.CHANNEL_BOUND_MARGIN, grouping, earth_in_the_arguments(dataset)),
                                  self.step_fd1_5_cut, (capacity_pixels, blocks, depth),
                                  outputs=cut_files, products=cut_products,
                                  inputs=[layout.raster("bsn"), layout.raster("str"), layout.raster("acc"), layout.group_table(grouping)],
                                  depends_on=["fd1.1", "fd1.3", "fd1.4_groups_%s" % grouping], kind="fd1.5_cut",
                                  refresh=lambda blocks=blocks, grouping=grouping: self.cut_step_files(blocks, grouping)))
                merge_text = dataset.merge_policy if dataset.merge_policy == "none" else "%s(%s)" % (dataset.merge_policy, fd1.MERGE_RULE)
                # regions_rules (REGIONS_RULES_VERSION): the order of the joins of merge=level2 and partition_config.txt
                # -- in the arguments so that a partition made under other rules is made again, not reused
                steps.append(Step("fd1.5_regions_%s" % label, "capacity=%d block=%d grouping=%s merge=%s land_cut_above=%s land_merge_below=%s ids=%s regions_rules=%d" % (
                                  capacity_pixels, grid.block_pixels, grouping, merge_text, dataset.land_cut_above, dataset.land_merge_below,
                                  fd1.REGION_ID_RULE, REGIONS_RULES_VERSION),
                                  self.step_fd1_5_regions, (capacity_pixels, blocks),
                                  outputs=[layout.partition_table(blocks, grouping, t) for t in ("region_fine", "piece_fine", "basin_region", "region_merges")],
                                  products={"rgn": layout.partition_raster(blocks, grouping, "rgn")},
                                  inputs=[layout.raster("bsn")], depends_on=["fd1.4_groups_%s" % grouping, "fd1.5_cut_%s" % label], kind="fd1.5_regions"))
            # fd1.6: the places of every basin in its Level-01/02/03 groups and in its region, and the
            # region of the partition whose windows fit 2^31 pixels (the first capacity of the dataset: 160 square degrees
            # on the 30 m grids, 1400 on MERIT; for a basin that capacity cut, the region its outlet piece is in).  It
            # belongs to FD1: running FD1 alone must produce it
            # region_id is the Level-03 partition of 2^31 and no other: a run on the
            # Hilbert groups does not build that partition, and its fd1.6 is left to a run on the Level-03 groups
            if dataset.has_given_groups and dataset.capacities and grouping != "l3":
                self.note("fd1.6 fills region_id from the Level-03 partition of 2^31, which a run with --groups %s does not build; "
                          "run with --groups given for it" % grouping)
            if dataset.has_given_groups and dataset.capacities and grouping == "l3":
                # the capacity of 2^31 wherever it stands in the list
                wanted_blocks = capacity_blocks_of("2^31", grid.block_pixels)
                classic = [capacity for capacity in dataset.capacities if capacity[1] == wanted_blocks]
                if not classic:
                    raise FlowDivideError("%s has no partition of 2^31 (%d blocks), which region_id is written for"
                                          % (dataset.name, wanted_blocks))
                classic_name, classic_blocks, _ = classic[0]
                classic_pixels = classic_blocks * grid.block_pixels * grid.block_pixels
                region_map_key = fd_tables.region_map_key(classic_pixels, grid.block_pixels, "l3")
                steps.append(Step("fd1.6_basin_table", "places and region_id of %s" % region_map_key,
                                  fd1.fd1_6_basin_table,
                                  (layout.basin_table(),
                                   layout.partition_table(classic_blocks, grouping, "basin_region"),
                                   layout.partition_table(classic_blocks, grouping, "region_fine"),
                                   layout.partition_table(classic_blocks, grouping, "piece_fine"),
                                   region_map_key),
                                  outputs=[layout.basin_table()],
                                  inputs=[layout.partition_table(classic_blocks, grouping, "basin_region"),
                                          layout.partition_table(classic_blocks, grouping, "region_fine")],
                                  depends_on=["fd1.4_groups_l3", "fd1.5_regions_%s" % self.partition_label(classic_name)],
                                  kind="fd1.6"))
        if "fd2" in self.steps and dataset.figures and not grid.geographic:
            dataset.figures = False
        if "fd2" in self.steps:
            for group in (("l3", "l2", "l1") if dataset.has_given_groups else ()) + (("hilbert",) if dataset.hilbert else ()):
                steps.append(Step("fd2_group_views_%s" % group, "views=%s vector_rules=%d" % (views_text, FD2_VECTOR_RULES_VERSION), self.step_fd2_group_views, (group,),
                                  outputs=[layout.group_view(group, n) for n, _ in dataset.views] + [layout.group_view(group, n) + ".objects.csv" for n, _ in dataset.views] +
                                  sum((layout.vector_files(layout.vector("grp_%s" % group, n), self.vector_formats) for n, _ in dataset.views), []),
                                  inputs=[layout.raster("bsn"), layout.group_table(group)] + [layout.basin_view(n) for n, _ in dataset.views],
                                  depends_on=["fd1.4_groups_%s" % ("l3" if group in ("l2", "l1") else group), "fd2_basin_views"], kind="fd2"))
            for name, blocks, _ in dataset.capacities:
                label = self.partition_label(name)
                steps.append(Step("fd2_region_views_%s" % label, "views=%s grouping=%s vector_rules=%d" % (views_text, grouping, FD2_VECTOR_RULES_VERSION), self.step_fd2_region_views, (blocks,),
                                  outputs=[layout.region_view(blocks, grouping, n) for n, _ in dataset.views] + [layout.region_view(blocks, grouping, n) + ".objects.csv" for n, _ in dataset.views] +
                                  sum((layout.vector_files(layout.region_view(blocks, grouping, n, "vector"), self.vector_formats) for n, _ in dataset.views), []),
                                  inputs=[layout.partition_raster(blocks, grouping, "rgn"), layout.raster("bsn")] + [layout.basin_view(n) for n, _ in dataset.views],
                                  depends_on=["fd1.5_regions_%s" % label, "fd2_basin_views"], kind="fd2"))
            if dataset.figures:
                first_name, first_blocks, _ = dataset.capacities[0]
                piece_view = layout.piece_preview(first_blocks, grouping, dataset.figure_basin_id)
                steps.append(Step("fd2_piece_views", "basin=%d preview=%d grouping=%s" % (dataset.figure_basin_id, PREVIEW_STEP, grouping), self.step_fd2_piece_views, (),
                                  outputs=[piece_view] if dataset.figure_basin_id in layout.cut_basins(first_blocks, grouping) else [], depends_on=["fd1.5_cut_%s" % self.partition_label(first_name)], kind="fd2"))
        if "figures" in self.steps and dataset.figures and not grid.geographic:
            self.note("the figures are drawn for geographic grids only; none for this projected grid")
        if "figures" in self.steps and dataset.figures and grid.geographic:
            steps.append(Step("figures", "basin=%d name=%s view=%s ticks=%s sizes=%s,%s hilbert=%s grouping=%s" % (
                              dataset.figure_basin_id, dataset.figure_basin_name, dataset.figure_view, dataset.figure_ticks, dataset.figure3_size, dataset.figure5_size, dataset.hilbert, grouping),
                              self.step_figures, (), outputs=[os.path.join(layout.root, "figures", "figures_manifest.txt")],
                              depends_on=["fd2_basin_views", "fd2_piece_views"] + (["fd2_group_views_l3"] if dataset.has_given_groups else []) + (["fd2_group_views_hilbert"] if dataset.hilbert else []) +
                              ["fd2_region_views_%s" % self.partition_label(name) for name in dataset.capacity_names], kind="figures"))
        if "fd3" in self.steps:
            blocks = self.blocks_of[self.capacity_name]
            label = self.partition_label(self.capacity_name)
            # shv, hck and ord read the channel mask fd1.1 left on disk.  When fd1 is not in this run, that
            # mask is at the threshold fd1.1's marker records, and fd3's own signature takes fd1.1's from that
            # marker too, so a run of fd3 alone after an fd1 at 1 km2 would compute them at 1 km2 and report this
            # run's threshold.  It stops here instead, before anything is computed
            if "fd1" not in self.steps and any(code in ("shv", "hck", "ord") for code in self.attributes):
                made_at = self.marker_channel_threshold("fd1.1")
                if made_at is None:
                    # no marker says which threshold made the mask on disk: it is not taken on trust
                    raise FlowDivideError("shv, hck and ord read the channel mask on disk, but no marker of fd1.1 says which threshold made it: "
                                          "run fd1 first (--steps fd1,...), or leave shv, hck and ord out of --attributes")
                # and made under this version's test of a channel pixel
                if "channel_rules=%d" % CHANNEL_RULES_VERSION not in (self.marker_arguments("fd1.1") or "").split():
                    raise FlowDivideError("the channel mask on disk was made under an older test of a channel pixel (the marker of fd1.1 "
                                          "does not say channel_rules=%d): run fd1 again (--steps fd1,...)" % CHANNEL_RULES_VERSION)
                if made_at != float(self.channel_threshold_km2):
                    raise FlowDivideError("the channel mask on disk was made at %g km2 (the marker of fd1.1) and this run asks for %g km2: "
                                          "run fd1 again at this threshold, or pass --channel-threshold-km2 %g" % (made_at, self.channel_threshold_km2, made_at))
            for code in self.attributes:
                needed = [layout.raster("dir")] + ([layout.raster("str")] if code in ("shv", "hck", "ord") else []) + ([layout.raster("aca")] if code == "hck" else [])
                outputs = [layout.attribute_table(code, blocks, grouping), layout.attribute_member_table(code, blocks, grouping)]
                products = {code: [layout.attribute(code, blocks, grouping)]}
                depends = ["fd1.5_regions_%s" % label, "fd1.1"] + (["fd3_ldn_%s" % label] if code == "lfp" else [])
                # the rules of stage 3 carry a version of their own: a marker left under other rules of
                # stage 3 (the tie between two donors of equal area, the distance raster's nodata) must
                # not let the new run skip the step.  It is named here and not in RULES_VERSION so that
                # stages 1 and 2, which such a change does not touch, keep their markers
                steps.append(Step("fd3_%s_%s" % (code, label), "capacity=%s grouping=%s min_area=%r fd3_rules=%d" % (self.capacity_name, grouping, float(self.min_basin_area_km2), FD3_RULES_VERSION), self.step_fd3, (code, blocks),
                                  outputs=outputs, products=products, inputs=needed, depends_on=depends, kind="fd3_%s" % code))
        self.plan(steps)
        if "fd1" not in self.steps and dataset.hilbert and any(s.label == "fd2_basin_views" and s.will_execute for s in steps) and os.path.exists(self.marker("fd1.4_groups_hilbert")):
            # the automatic groups were counted on the 400th basin view; fd2 alone remakes that view and leaves the groups as they were
            self.note("note: the basin views are made again but fd1 is not in --steps, so the automatic (Hilbert) groups of the earlier views stay; run fd1,fd2 to remake them")
        self.write_run_summary(steps)
        self.note("plan: %s" % ", ".join(step.label for step in steps if step.will_execute) if any(s.will_execute for s in steps) else "plan: nothing to run, everything is done")
        if self.summary_only:
            self.note("summary only: nothing run")
            return
        for step in steps:
            self.execute(step)
            self.drop_what_is_no_longer_read()
        self.note("==== finished %s %s ====" % (dataset.name, time.strftime("%Y-%m-%d %H:%M")))

    def cut_step_files(self, blocks, grouping):
        """the files of the cut at one capacity: the listing and, for every basin it lists, the piece table, the main
        stem and the codes (outputs), and the piece rasters (products)"""
        layout = self.layout
        cut_files = [layout.cut_listing(blocks, grouping)]
        for basin_id in layout.cut_basins(blocks, grouping):
            cut_files += [layout.piece_table(blocks, grouping, basin_id), layout.mainstem_table(blocks, grouping, basin_id),
                          layout.piece_codes(blocks, grouping, basin_id)]
        return cut_files, {"pieces": [raster for _, raster in layout.cuts(blocks, grouping).values()]}

    def cut_levels_of(self, capacity_name):
        """how deep the Pfafstetter cut goes at a capacity: the dataset's depth for it (3, 3, 4 for the
        built-in grids), or what --levels says for every capacity: a number, or auto"""
        if self.levels is None:
            return self.levels_of[capacity_name]
        return self.levels

    def step_fd1_0(self, dir_path):
        dataset = self.dataset
        native = dataset.raw_dir
        if os.path.isdir(native):
            tiles = sorted(glob.glob(os.path.join(native, "*.tif")))
            nodata = fd1.direction_convention(dataset.convention, codes=dataset.direction_codes, sink=dataset.sink_code,
                                              nodata=dataset.nodata_code, mouth=dataset.mouth_code)["nodata"]
            native = fd1.assemble_tiles(tiles, os.path.join(self.layout.root, "global", "fineresolution", "dir", "dir_%s_native.tif" % dataset.continent), nodata)
        return fd1.fd1_0_recode(native, dir_path, dataset.convention, periodic=dataset.periodic, direction_codes=dataset.direction_codes,
                                sink_code=dataset.sink_code, nodata_code=dataset.nodata_code, mouth_code=dataset.mouth_code)

    def basins_over_the_capacity(self, capacity_pixels):
        """the basins whose own block window exceeds the capacity, from the basin table"""
        basins = fd_tables.read_basin_table(self.layout.basin_table())
        block = self.grid.block_pixels
        rectangles = fd1.basin_rectangles(basins)
        window_rows = (rectangles[:, 1] // block + 1) * block - rectangles[:, 0] // block * block
        window_cols = (rectangles[:, 3] // block + 1) * block - rectangles[:, 2] // block * block
        windows = window_rows * window_cols
        return basins, [int(b) for b in basins.loc[windows > capacity_pixels, "basin_id"]]

    def step_fd1_5_cut(self, capacity_pixels, blocks, depth):
        layout = self.layout
        basins, over = self.basins_over_the_capacity(capacity_pixels)
        self.note("  %d basins whose window exceeds the capacity: %s" % (len(over), over[:20]))
        levels = None if depth in (None, "auto") else int(depth)
        group_of_basin = fd_tables.group_of_basin(layout.root, layout.name, self.grouping, basins)
        groups = fd_tables.read_group_table(layout.group_table(self.grouping), 0)
        code_of_group = dict(zip(groups["group_id"].astype(int), groups["level_code"].astype(int)))
        group_of = {basin_id: code_of_group[int(group_of_basin[basin_id])] for basin_id in over}
        for basin_id in over:
            fd1.fd1_5_pfafstetter_cut(layout.raster("dir"), layout.raster("bsn"), layout.raster("str"), layout.raster("acc"), basins.iloc[basin_id - 1], self.grid, capacity_pixels,
                                      layout.piece_table(blocks, self.grouping, basin_id), layout.piece_raster(blocks, self.grouping, basin_id), layout.mainstem_table(blocks, self.grouping, basin_id),
                                      codes_path=layout.piece_codes(blocks, self.grouping, basin_id), region_code=group_of[basin_id], levels=levels, aca_path=layout.raster("aca"),
                                      aca_to_km2=self.dataset.aca_to_km2, channel_threshold_km2=self.channel_threshold_km2, tile=self.tile)
        with open(layout.cut_listing(blocks, self.grouping), "w") as handle:
            handle.write("\n".join(str(b) for b in over) + ("\n" if over else ""))
        return over

    def step_fd1_5_regions(self, capacity_pixels, blocks):
        layout = self.layout
        grouping = self.grouping
        outputs = {"regions": layout.partition_table(blocks, grouping, "region_fine"), "pieces": layout.partition_table(blocks, grouping, "piece_fine"),
                   "basin_region": layout.partition_table(blocks, grouping, "basin_region"), "merges": layout.partition_table(blocks, grouping, "region_merges"),
                   "rgn": layout.partition_raster(blocks, grouping, "rgn")}
        # one line: the capacity, the block, the groups (l3, or for
        # the automatic groups: "hilbert" alone at 160 square degrees on blocks of one degree made on
        # the 400th view, the partition of the paper's appendix, else hilbert_<capacity>[_<view>]), and 32-bit indices while the
        # capacity fits them; the merge policy is recorded in the step's arguments and markers.  The directory of that
        # appendix partition is named hilbert_160deg2 here.  The
        # built-in grids keep their block, so _block<B> does not arise on them; a grid of one's own gets _block<B> from capacity_directory_name and its views are named by their cell size
        capacity_text = layout.capacity_directory_name(blocks, "l3")
        view_of_the_groups = self.dataset.views[0][0]   # the view fd1.4_groups_hilbert reads
        view_suffix = "" if view_of_the_groups == "400th" else "_" + view_of_the_groups
        groups_text = grouping if grouping == "l3" else ("hilbert" if capacity_text == "160deg2" and not view_suffix else
                                                         "hilbert_" + capacity_text + view_suffix)
        index_bits = 32 if capacity_pixels <= 2 ** 31 - 1 else 64
        with open(os.path.join(layout.partition(blocks, grouping), "partition_config.txt"), "w") as handle:
            handle.write("capacity_px=%d block_px=%d groups=%s index_bits=%d\n" % (capacity_pixels, self.grid.block_pixels, groups_text, index_bits))
        return fd1.fd1_5_regions_final(layout.basin_table(), grouping, layout.raster("bsn"),
                                       layout.cuts(blocks, grouping), self.grid, capacity_pixels, outputs, self.dataset.continent, merge_policy=self.dataset.merge_policy,
                                       land_cut_above=self.dataset.land_cut_above, land_merge_below=self.dataset.land_merge_below, level1=self.dataset.level1)

    def cells_per_degree_of_view(self, factor):
        """the cells to a degree of a view on a geographic grid (400 for 1/400 degree); 0 on a projected grid"""
        grid = self.grid
        return int(round(1.0 / (abs(grid.pixel_width) * factor))) if grid.geographic else 0

    def min_basin_area_km2_of_view(self, factor):
        """the smallest basin of a view: a quarter of a cell, at the equator on a geographic grid
        (basin_min_area_km2; the view is the basins of at least this area, the ids 1..N)"""
        grid = self.grid
        side_km = abs(grid.pixel_width) * factor * METRES_PER_DEGREE / 1000.0 if grid.geographic else abs(grid.pixel_width) * factor / 1000.0
        return fd2.min_basin_area_km2_of(side_km)

    def view_resolution_text(self, factor):
        return resolution_text(self.grid.pixel_width, self.grid.geographic, factor)

    def step_fd2_basin_views(self):
        """the basin mask folded at every factor in one pass (the rule: a view holds the basins of at least a
        quarter of a cell, the ids 1..N; a cell of a quarter land goes to the largest of them, the smallest id, holding
        a quarter of that land), then every view vectorised as one layer of N rows"""
        layout = self.layout
        basins = fd_tables.read_basin_table(layout.basin_table())
        limit = int(basins["basin_id"].max()) + 1
        jobs = []
        in_the_view = {}
        for name, factor in self.dataset.views:
            min_area = self.min_basin_area_km2_of_view(factor)
            in_the_view[name] = fd2.basins_of_the_view(basins, min_area, layout.basin_table())
            jobs.append((layout.basin_view(name), factor, in_the_view[name], None, min_area))
            self.note("  the %s view holds the %d basins of at least %.4f km2 (ids 1..%d)" % (name, in_the_view[name], min_area, in_the_view[name]))
        fd2.fold_mask_at_factors(layout.raster("bsn"), jobs, object_count_limit=limit, rule="largest_basin")
        if self.dataset.vectorise_basins:
            for name, factor in self.dataset.views:
                fd2.vectorise_mask(layout.basin_view(name), basins, "basin_id", layout.vector("bsn", name), "basins", in_the_view[name], self.cells_per_degree_of_view(factor),
                                   min_basin_area_km2=self.min_basin_area_km2_of_view(factor), formats=self.vector_formats, boundaries_stem=layout.vector("bnd", name),
                                   periodic=self.grid.periodic, colours_at_least=self.colours, resolution_text=self.view_resolution_text(factor))

    def step_fd2_group_views(self, group):
        """the basin mask read as group ids through the basin-group table and folded, following the basin view of the
        same resolution; then vectorised, every row of the group table"""
        layout = self.layout
        groups = fd_tables.read_group_table(layout.group_table(group), 0)
        lookup = fd_tables.group_of_basin(layout.root, layout.name, group)
        fd2.fold_mask_at_factors(layout.raster("bsn"), [(layout.group_view(group, name), factor, None, layout.basin_view(name)) for name, factor in self.dataset.views],
                                 lookup=lookup, object_count_limit=int(groups["group_id"].max()) + 1, rule="region")
        for name, factor in self.dataset.views:
            fd2.vectorise_mask(layout.group_view(group, name), groups, "group_id", layout.vector("grp_%s" % group, name), "basin_groups", len(groups), self.cells_per_degree_of_view(factor),
                               min_basin_area_km2=self.min_basin_area_km2_of_view(factor), formats=self.vector_formats, periodic=self.grid.periodic, colours_at_least=self.colours,
                               resolution_text=self.view_resolution_text(factor))

    def step_fd2_region_views(self, blocks):
        """the region mask folded, following the basin view of the same resolution with the basin mask read beside
        it; then vectorised, every row of the region table"""
        layout = self.layout
        grouping = self.grouping
        regions = fd_tables.read_region_table(layout.partition_table(blocks, grouping, "region_fine"))
        fd2.fold_mask_at_factors(layout.partition_raster(blocks, grouping, "rgn"), [(layout.region_view(blocks, grouping, name), factor, None, layout.basin_view(name)) for name, factor in self.dataset.views],
                                 object_count_limit=int(regions["region_id"].max()) + 1, rule="region", basin_fine_path=layout.raster("bsn"))
        for name, factor in self.dataset.views:
            fd2.vectorise_mask(layout.region_view(blocks, grouping, name), regions, "region_id", layout.region_view(blocks, grouping, name, "vector"), "regions", len(regions),
                               self.cells_per_degree_of_view(factor), min_basin_area_km2=self.min_basin_area_km2_of_view(factor), formats=self.vector_formats,
                               periodic=self.grid.periodic, colours_at_least=self.colours, resolution_text=self.view_resolution_text(factor))

    def step_fd2_piece_views(self):
        """the preview of the pieces of the figure basin at the first (largest) capacity: one cell per 30 x 30 pixels
        of the basin's rectangle, the pixel at the corner of each cell, UInt16"""
        layout = self.layout
        name, blocks, _ = self.dataset.capacities[0]
        basin_id = self.dataset.figure_basin_id
        piece_raster = layout.piece_raster(blocks, self.grouping, basin_id)
        piece_view = layout.piece_preview(blocks, self.grouping, basin_id)
        if basin_id not in layout.cut_basins(blocks, self.grouping):
            self.note("  basin %d was not cut at %s; no piece preview and no Figure 4" % (basin_id, name))
            if os.path.exists(piece_view):
                os.remove(piece_view)                   # a preview of an earlier cut that no longer exists
            return
        if not os.path.exists(piece_raster):
            raise FlowDivideError("the piece raster of basin %d at %s was dropped; run fd1 again to make it" % (basin_id, name))
        with rasterio.open(piece_raster) as pieces:
            sampled = pieces.read(1)[::PREVIEW_STEP, ::PREVIEW_STEP].astype(np.uint16)
            profile = {"driver": "GTiff", "dtype": "uint16", "count": 1, "width": sampled.shape[1], "height": sampled.shape[0], "crs": pieces.crs,
                       "transform": pieces.transform * rasterio.Affine.scale(PREVIEW_STEP), "nodata": 0, "tiled": True, "blockxsize": fd1.RASTER_BLOCK,
                       "blockysize": fd1.RASTER_BLOCK, "compress": "DEFLATE"}
        temporary = piece_view + ".partial.tif"
        with rasterio.open(temporary, "w", **profile) as out:
            out.write(sampled, 1)
        fd1.publish(temporary, piece_view)
        self.note("  written %s: %d x %d cells, one per %d x %d pixels" % (piece_view, sampled.shape[1], sampled.shape[0], PREVIEW_STEP, PREVIEW_STEP))

    def step_figures(self):
        dataset = self.dataset
        layout = self.layout
        grid = self.grid
        grouping = self.grouping
        view_name, factor = dataset.views[0]
        if dataset.figure_view is None:
            box = grid.pixel_box_lon_lat(0, grid.nrow - 1, 0, grid.ncol - 1)
            dataset.figure_view = (box[0], box[2], box[1], box[3])
        if dataset.figure_ticks is None:
            dataset.figure_ticks = (fd2._tick_values(dataset.figure_view[0], dataset.figure_view[1]), fd2._tick_values(dataset.figure_view[2], dataset.figure_view[3]))
        first_name, first_blocks, _ = dataset.capacities[0]
        capacity_pixels = first_blocks * grid.block_pixels * grid.block_pixels
        short_label = {"2^31": "2³¹", "2^30": "2³⁰", "2^29": "2²⁹"}.get(first_name, first_name)
        out = os.path.join(layout.root, "figures")
        made = []
        group_views = []
        if dataset.has_given_groups:
            group_views.append(("Basin groups by Level-03 unit", layout.group_view("l3", view_name), layout.group_table("l3"), fd_tables.group_of_basin(layout.root, layout.name, "l3")))
        if dataset.hilbert:
            group_views.append(("Basin groups by Hilbert curve", layout.group_view("hilbert", view_name), layout.group_table("hilbert"), fd_tables.group_of_basin(layout.root, layout.name, "hilbert")))
        made.append(fd2.figure3_delineation_and_groups(layout.basin_view(view_name), layout.basin_table(), group_views, capacity_pixels, grid, out, dataset.figure_view,
                                                       dataset.figure_ticks[0], dataset.figure_ticks[1], short_label, figure_size=dataset.figure3_size, stacked=dataset.figure3_stacked))
        piece_view = layout.piece_preview(first_blocks, grouping, dataset.figure_basin_id)
        if dataset.figure_basin_id in layout.cut_basins(first_blocks, grouping) and os.path.exists(piece_view):
            basins = fd_tables.read_basin_table(layout.basin_table())
            made.append(fd2.figure4_one_basin_through_the_cut(piece_view, layout.piece_table(first_blocks, grouping, dataset.figure_basin_id), layout.mainstem_table(first_blocks, grouping, dataset.figure_basin_id),
                                                              layout.piece_codes(first_blocks, grouping, dataset.figure_basin_id), basins.iloc[dataset.figure_basin_id - 1], capacity_pixels, grid, out,
                                                              "%s (%s)" % (short_label, layout.capacity_directory_name(first_blocks).replace("deg2", " deg²")), dataset.figure_basin_name))
        region_views = []
        region_gpkgs = []
        region_tables = []
        labels = []
        cut_lists = []
        for name, blocks, _ in dataset.capacities:
            region_views.append(layout.region_view(blocks, grouping, view_name))
            region_gpkgs.append(layout.region_view(blocks, grouping, view_name, "vector"))
            region_tables.append(layout.partition_table(blocks, grouping, "region_fine"))
            labels.append("%s (%s)" % ({"2^31": "2³¹", "2^30": "2³⁰", "2^29": "2²⁹"}.get(name, name), layout.capacity_directory_name(blocks).replace("deg2", " deg²")))
            cut_lists.append(layout.cut_basins(blocks, grouping))
        made.append(fd2.figure5_three_capacities(region_views, region_gpkgs, region_tables, layout.basin_view(view_name), labels, cut_lists, out, dataset.figure_view,
                                                 dataset.figure_ticks[0], dataset.figure_ticks[1], figure_size=dataset.figure5_size, vector_formats=self.vector_formats))
        with open(os.path.join(out, "figures_manifest.txt"), "w") as handle:
            handle.write("\n".join(made) + "\n")
        if self.open_figures and platform.system() == "Darwin":
            for path in made:
                subprocess.call(["open", path])
        return made

    def step_fd3(self, code, blocks):
        layout = self.layout
        grouping = self.grouping
        # the distance to the outlet and the upstream flow length are computed for every
        # basin however small (the raster), their tables list the basins of min_basin_area_km2 and more; the other four
        # classes read the basins of min_basin_area_km2 and more for both
        partition = fd3.Partition(layout.basin_table(), layout.partition_table(blocks, grouping, "region_fine"), layout.partition_table(blocks, grouping, "piece_fine"),
                                  layout.partition_table(blocks, grouping, "basin_region"), self.min_basin_area_km2, self.grid,
                                  expect={"continent": self.dataset.continent, "capacity_px": blocks * self.grid.block_pixels * self.grid.block_pixels, "block_px": self.grid.block_pixels,
                                          "groups": grouping},
                                  raster_min_basin_area_km2=0.0 if code in ("ldn", "lup") else None)
        ldn_members = layout.attribute_member_table("ldn", blocks, grouping)
        return fd3.derive_attribute(code, partition, layout.raster("dir"), layout.attribute(code, blocks, grouping), layout.attribute_table(code, blocks, grouping),
                                    layout.attribute_member_table(code, blocks, grouping), channel_path=layout.raster("str"), area_path=layout.raster("aca"),
                                    ldn_member_table=ldn_members if os.path.exists(ldn_members) else None)


# =============================================================================
#  [4] The command line
# =============================================================================

def parse_arguments(argv):
    parser = argparse.ArgumentParser(prog="flowdivide.py", description="FlowDivide: a partition of a flow-direction grid that fits a memory capacity, and the attributes computed on it")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="run the chain, or part of it, on a dataset", description="Run the FlowDivide chain, or part of it, on one dataset.")
    what = run.add_argument_group("what to run")
    what.add_argument("dataset", help="south-america, north-america, merit-global, or a name of your own with --dir and --l3")
    what.add_argument("--out-root", default=os.environ.get("FLOWDIVIDE_ROOT", os.path.join(os.getcwd(), "flowdivide_output")), help="the products go to <out-root>/<dataset>/")
    what.add_argument("--steps", default="fd1,fd2,figures", help="comma-separated among fd1 (the partition), fd2 (the views), figures, fd3 (the attributes); default fd1,fd2,figures")
    what.add_argument("--attributes", default=",".join(ALL_ATTRIBUTES), help="the attributes of fd3, comma-separated among %s (default all six)" % ",".join(ALL_ATTRIBUTES))
    what.add_argument("--capacity", default=None, help="the capacity the attributes are computed on (default the largest of the dataset)")
    partition = run.add_argument_group("the partition")
    partition.add_argument("--groups", default=None, choices=GROUPINGS + ["given"], help="the basin groups the partition is built on: given (the groups of --group-vectors, HydroBASINS Level-03 for the built-in grids; also spelled l3, which is the word in the file names) or hilbert (the automatic groups, the default when no vectors are given)")
    partition.add_argument("--levels", default=None, help="how deep the Pfafstetter cut goes at every capacity: 2, 3, 4, or auto (default: the dataset's depth per capacity, 3, 3, 4)")
    partition.add_argument("--channel-threshold-km2", type=float, default=None, help="a channel pixel has at least this upstream area (default: the grid's own, 1 on the two HydroSHEDS grids and on a grid of your own, 10 on MERIT)")
    partition.add_argument("--min-basin-area-km2", type=float, default=1.0, help="the attributes are computed for the basins of at least this area (default 1)")
    partition.add_argument("--tile", type=int, default=fd1.WORK_TILE_DEFAULT, help="the square work tile of fd1.1, fd1.3 and fd1.5 in pixels, a multiple of 512 (default %d)" % fd1.WORK_TILE_DEFAULT)
    keep = run.add_argument_group("the keep option")
    keep.add_argument("--keep", default=None, help="the variables kept on disk, comma-separated among %s; every other one is deleted as soon as the last step that reads it is done (dir, the tables and the polygon files are always kept).  The other way round is --drop" % ",".join(DROPPABLE))
    keep.add_argument("--drop", default="", help="the variables deleted as soon as nothing left in the run reads them, comma-separated among %s (the other way round is --keep)" % ",".join(DROPPABLE))
    views = run.add_argument_group("the views and the figures")
    views.add_argument("--vector", default="geoparquet", choices=["geoparquet", "gpkg", "both"], help="the format of the polygon files: geoparquet (the default; GeoParquet 1.1, zstd), gpkg (GeoPackage), or both, with the same rows and columns")
    views.add_argument("--colours", type=int, default=fd2.COLOURS_AT_LEAST, help="the colouring uses at least this many colour numbers, so that a colour does not come back two objects apart (default %d)" % fd2.COLOURS_AT_LEAST)
    views.add_argument("--no-open", action="store_true", help="do not open the figures when they are drawn")
    own = run.add_argument_group("a grid of your own")
    own.add_argument("--dir", default=None, help="the native flow directions, one file or a directory of tiles")
    own.add_argument("--convention", default="merit", choices=sorted(fd1.DIRECTION_CONVENTIONS) + ["custom"],
                     help="how --dir codes its flow directions: merit and esri (1 E, 2 SE ... 128 NE), hydrosheds (the same codes, 0 an inland sink, 255 no data), taudem (1 E, then counter-clockwise), grass (1 NE ... 8 E), degrees (45 NE ... 360 E), or custom with --direction-codes")
    own.add_argument("--direction-codes", default=None, help="the eight codes of a convention this package does not know by name, e.g. E=1,SE=2,S=4,SW=8,W=16,NW=32,N=64,NE=128")
    own.add_argument("--sink-code", type=int, default=None, help="the value of an inland sink in --dir (default: the convention's own; none means a sink is a direction pointing into nodata)")
    own.add_argument("--nodata-code", type=int, default=None, help="the value of no data in --dir (default: the convention's own)")
    own.add_argument("--mouth-code", type=int, default=None, help="the value of a river mouth in --dir (default: the convention's own; without one a direction pointing into nodata or off the grid is the mouth)")
    own.add_argument("--group-vectors", "--l3", dest="group_vectors", default=[], action="append",
                     help="a polygon file whose units are whole basins, on which the partition's basin groups are built (repeat for several; add ':<pandas query>' for a filter); HydroBASINS Level-03 for the built-in grids, any such file for a grid of your own; none: the basins are grouped automatically along a Hilbert curve")
    own.add_argument("--acc", default=None, help="an upstream pixel count raster on the grid of --dir, given instead of computing it (none: fd1.1 computes it); with a built-in grid, it replaces the grid's own count raster (the HydroSHEDS ACC mosaics can be replaced by a count made from the flow directions)")
    own.add_argument("--aca", default=None, help="an upstream area raster on the grid of --dir, given with --acc (none: fd1.1 computes it)")
    own.add_argument("--aca-unit", default="km2", choices=["km2", "m2"], help="the unit of --aca (default km2, as HydroSHEDS and MERIT Hydro ship it)")
    own.add_argument("--periodic", action="store_true", help="the grid spans 360 degrees of longitude")
    own.add_argument("--block", type=int, default=None, help="the block of the windows, in pixels (default one degree)")
    own.add_argument("--capacities", default="2^31:3,2^30:3,2^29:4", help="the capacities and the depth of the cut at each, name:levels pairs, the largest first (a name is 2^k or a number of blocks such as 160deg2)")
    own.add_argument("--merge", default="none", choices=["none", "within_parent", "any", "balanced", "level2"], help="the merge policy of fd1.5 for the groups under the capacity (level2: a region is a Level-02 unit, a unit that does not fit is divided along its Level-03 units)")
    own.add_argument("--island-reach", type=int, default=6, help="an island group gathers the uncoded basins within this many blocks of its seed (default 6; 18 on the 3 arc-second grid)")
    own.add_argument("--island-max", type=int, default=36, help="an island group's window stays below this many blocks (default 36; 1296 on the 3 arc-second grid)")
    own.add_argument("--no-hilbert", action="store_true", help="do not make the automatic (Hilbert) groups")
    own.add_argument("--figures", action="store_true", help="draw the three figures too, with --figure-basin through the cut")
    own.add_argument("--figure-basin", type=int, default=1, help="the basin of Figure 4 (default 1, the largest)")
    own.add_argument("--figure-name", default="Basin", help="the name of that basin in the file name")
    own.add_argument("--hybas-root", default=HYBAS_ROOT_DEFAULT, help="the directory of the HydroBASINS Level-03 shapefiles of the built-in datasets (default: the environment variable FLOWDIVIDE_HYBAS)")
    how = run.add_argument_group("how it runs")
    how.add_argument("--in-process", action="store_true", help="run every step in this process instead of a child process each (the peak memory is then that of the whole run)")
    how.add_argument("--only", default=None, help="run only the steps named here, comma-separated labels as the plan prints them (e.g. fd1.5_regions_1400deg2,fd2_region_views_1400deg2); every other step is left as it stands, and the log lists the planned steps held back.  A step still refuses to run when a file it reads is missing")
    how.add_argument("--summary-only", action="store_true", help="print and save the run summary (what would be read, decided and written) and the plan, and run nothing")
    how.add_argument("--timing", default="on", choices=["on", "off"], help="on (the default): every step's time is recorded split into input, compute and output, in its marker and in _logs/timings_<dataset>.csv; off: the total only")
    commands.add_parser("list", help="list the built-in datasets")
    global_ids = commands.add_parser("global-ids", help="fd1.7: the global basin id over several runs of the 30 m grid",
                                     description="fd1.7: number the basins of several runs (continents) together, largest area first; "
                                                 "every run's basin table gets global_basin_id and the global table is written.")
    global_ids.add_argument("datasets", nargs="+", help="the runs, each finished through fd1.6 (e.g. south-america north-america)")
    global_ids.add_argument("--out-root", default=os.environ.get("FLOWDIVIDE_ROOT", os.path.join(os.getcwd(), "flowdivide_output")),
                            help="the root the runs were written under (<out-root>/<dataset>/)")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if arguments.command == "list":
        for name, dataset in builtin_datasets().items():
            print("%-15s %s  %s  capacities %s  views %s" % (name, dataset.convention, dataset.raw_dir, dataset.capacities, dataset.views))
        return 0
    if arguments.command == "global-ids":
        datasets = builtin_datasets()
        runs = []
        for name in arguments.datasets:
            # the 30 m continents only: MERIT's grid is the globe and its global_basin_id is its basin_id (fd1.2)
            if name not in datasets or not datasets[name].level1 or datasets[name].continent == "global":
                raise FlowDivideError("'%s' is not a HydroSHEDS continent with a Level-01 code" % name)
            layout_root = os.path.join(arguments.out_root, name)
            runs.append((datasets[name].continent, int(datasets[name].level1), fd_tables.basin_table_path(layout_root, datasets[name].continent)))
        # the global table sits where the whole grid's own table would: <out-root>/global/global/table/
        global_table = fd_tables.basin_table_path(os.path.join(arguments.out_root, "global"), "global")
        fd1.fd1_7_global_basin_id(runs, global_table)
        return 0
    if arguments.command != "run":
        print(__doc__)
        return 2
    datasets = builtin_datasets(hybas_root=arguments.hybas_root)
    if arguments.dir:
        layers = []
        for entry in arguments.group_vectors:
            path, _, query = entry.partition(":")
            layers.append((path, query or None))
        probe = fd1.Grid(arguments.dir if not os.path.isdir(arguments.dir) else sorted(glob.glob(os.path.join(arguments.dir, "*.tif")))[0], periodic=arguments.periodic, block_pixels=arguments.block)
        capacities = []
        for pair in arguments.capacities.split(","):
            if not pair:
                continue
            name, _, levels = pair.partition(":")
            capacities.append((name, capacity_blocks_of(name, probe.block_pixels), None if levels in ("", "auto") else int(levels)))
        # The views of a grid of your own.  On a geographic grid whose block is one degree (1 and 3 arc-second
        # grids, and any other with a whole number of pixels to the degree) the views are named as the paper's:
        # 400, 200, 120 and 20 cells to the degree.  On any other grid -- projected, 10 m, 1 m, a block that is
        # not a degree -- a view is named by the size of its cell, which is what a reader of the file name needs
        # (any baseline hydrography is supported): coarse_<name>, e.g. 100m or 1km.
        if probe.geographic and abs(probe.block_pixels * abs(probe.pixel_width) - 1.0) < 1e-6 and all(probe.block_pixels % cells == 0 for cells in (400, 200, 120, 20)):
            views = [("400th", probe.block_pixels // 400), ("200th", probe.block_pixels // 200), ("120th", probe.block_pixels // 120), ("20th", probe.block_pixels // 20)]
        else:
            views = views_by_cell_size(probe)
        dataset = Dataset(arguments.dataset, arguments.dir, arguments.convention, layers, periodic=arguments.periodic, block_pixels=arguments.block, capacities=capacities,
                          direction_codes=arguments.direction_codes, sink_code=arguments.sink_code, nodata_code=arguments.nodata_code, mouth_code=arguments.mouth_code,
                          merge_policy=arguments.merge, island_reach_blocks=arguments.island_reach, island_max_blocks=arguments.island_max, views=views, hilbert=not arguments.no_hilbert or not layers,
                          figures=arguments.figures, figure_basin_id=arguments.figure_basin, figure_basin_name=arguments.figure_name,
                          acc_path=arguments.acc, aca_path=arguments.aca, aca_unit=arguments.aca_unit)
    elif arguments.dataset in datasets:
        dataset = datasets[arguments.dataset]
        # the upstream pixel count of a built-in grid may be given from elsewhere.  The count is a whole
        # number the flow directions fix, so a count made from them
        # stands for the provider's ACC mosaic where that mosaic is not on the machine.  Only a
        # grid that reads a given count takes it; the area stays the provider's, the downloaded upa
        if arguments.acc:
            if dataset.acc_path is None:
                raise FlowDivideError("--acc replaces the count raster of a built-in grid that reads one; %s computes its own" % dataset.name)
            dataset.acc_path = arguments.acc
    else:
        raise FlowDivideError("unknown dataset '%s'; the built-in ones are %s, or give --dir and --l3" % (arguments.dataset, ", ".join(datasets)))
    steps = [s.strip() for s in arguments.steps.split(",") if s.strip()]
    for step in steps:
        if step not in ("fd1", "fd2", "figures", "fd3"):
            raise FlowDivideError("unknown step '%s'" % step)
    attributes = [a.strip() for a in arguments.attributes.split(",") if a.strip()]
    if not arguments.dir and arguments.dataset in datasets:
        unset = unset_inputs(arguments.dataset, steps, arguments.hybas_root, acc_given=bool(arguments.acc),
                             attributes=attributes)
        if unset:
            raise FlowDivideError("%s reads its inputs from environment variables, and these are not set: %s"
                                  % (arguments.dataset, "; ".join(unset)))
    for code in attributes:
        if code not in ALL_ATTRIBUTES:
            raise FlowDivideError("unknown attribute '%s'" % code)
    if "lfp" in attributes and "ldn" not in attributes:
        attributes.append("ldn")                       # the longest flow path reads the distance table, which must be current
    attributes = sorted(attributes, key=lambda a: (a == "lfp", ALL_ATTRIBUTES.index(a)))      # the distance table before the path
    # --keep and --drop are the two ways of saying the same thing: which of the intermediates stay on disk.
    # A reader finds "keep these" easier to read than "drop those", and the chain works on the dropped list.
    if arguments.keep is not None and arguments.drop:
        raise FlowDivideError("--keep and --drop say the same thing in two ways; give one of them")
    if arguments.keep is not None:
        kept = [k.strip() for k in arguments.keep.split(",") if k.strip()]
        for variable in kept:
            if variable == "dir":
                continue                               # always kept; a reader may still name it
            if variable not in DROPPABLE:
                raise FlowDivideError("'%s' is not a variable of a run; choose among dir, %s" % (variable, ", ".join(DROPPABLE)))
        drop = [variable for variable in DROPPABLE if variable not in kept]
    else:
        drop = [d.strip() for d in arguments.drop.split(",") if d.strip()]
        for variable in drop:
            if variable not in DROPPABLE:
                raise FlowDivideError("'%s' is not a variable that can be dropped; choose among %s" % (variable, ", ".join(DROPPABLE)))
    levels = None
    if arguments.levels is not None:
        levels = "auto" if arguments.levels == "auto" else int(arguments.levels)
        if levels != "auto" and not 2 <= levels <= 4:
            raise FlowDivideError("--levels must be 2, 3, 4 or auto")
    capacity_name = arguments.capacity or dataset.capacities[0][0]
    grouping = {"given": "l3"}.get(arguments.groups, arguments.groups) or ("l3" if dataset.has_given_groups else "hilbert")
    if arguments.colours < 1:
        raise FlowDivideError("--colours must be at least 1")
    vector_formats = ("geoparquet", "gpkg") if arguments.vector == "both" else (arguments.vector,)
    if "gpkg" in vector_formats and arguments.colours > len(fd2.COLOR_ID_STYLE_PALETTE):
        raise FlowDivideError("--colours %d with a GeoPackage: the default style of a GeoPackage view has %d colours"
                              % (arguments.colours, len(fd2.COLOR_ID_STYLE_PALETTE)))
    channel_threshold_km2 = arguments.channel_threshold_km2 if arguments.channel_threshold_km2 is not None else dataset.channel_threshold_km2
    if not (math.isfinite(channel_threshold_km2) and channel_threshold_km2 > 0.0):
        raise FlowDivideError("--channel-threshold-km2 must be a positive number of square kilometres")
    chain = Chain(dataset, arguments.out_root, steps, attributes, capacity_name, drop, arguments.tile, arguments.min_basin_area_km2, channel_threshold_km2, not arguments.no_open,
                  grouping=grouping, levels=levels, vector_formats=vector_formats, colours=arguments.colours, separate_processes=not arguments.in_process)
    chain.timing = arguments.timing == "on"
    if arguments.only:
        chain.only = set(label.strip() for label in arguments.only.split(",") if label.strip())
        if chain.drop:
            # a variable is dropped when nothing left in the run reads it; with --only most of the run
            # is not in it, so dropping would take files the steps held back still need
            chain.note("--only: nothing is dropped, although %s would be in a whole run" % ", ".join(sorted(chain.drop)))
            chain.drop = set()
    if arguments.summary_only:
        chain.summary_only = True
        chain.summary = os.devnull                    # the chain log of a real run is left alone
    chain.run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FlowDivideError as error:
        print("flowdivide: %s" % error, file=sys.stderr)
        sys.exit(1)
