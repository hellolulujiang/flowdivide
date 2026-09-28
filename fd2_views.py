"""fd2_views.py -- FlowDivide stage 2, the views delivered with the partition (FD2 of Figure 2), and the
figures of the manuscript drawn from them.

Most users meet the partition beside data on coarser grids, so the basin and region masks are folded
by an integer factor to a coarse grid, one object per cell, turned into polygons carrying their
rows of the tables, and coloured so that no two neighbours share a number:

    resampling     a coarse cell covers factor x factor pixels, and the view is an upscaling: the
                   large basins are kept first, a small basin is read at a finer view.  A view holds the basins of at least an area, a
                   quarter of a cell at the equator (0.0194, 0.0775, 0.2151 and 7.7450 km2 at 1/400,
                   1/200, 1/120 and 1/20 degree), and since fd1.2 numbers the basins from the largest
                   down, those are the ids 1..N: the table of the view is the first N rows of the basin
                   table, and one index, basin_id - 1, reaches a basin in the fine table and in every
                   view.  Basin view: a cell that is at least a quarter land (pixels of any basin, the
                   small ones included) is land, less and it stays water; a land cell goes to the largest
                   -- the smallest id -- of the basins of the view that hold at least a quarter of ITS
                   LAND.  Region and group views FOLLOW the basin view of the same resolution: a cell
                   the basin view gave a basin takes the region of that basin (a cut basin lies in
                   several regions: the one holding most of its pixels in the cell) or the group of
                   that basin from the table; a cell the basin view left empty goes by majority,
                   the region or group holding most of the cell, the cell being at least a quarter
                   land, so the regions still tile the land.  The piece preview of Figure 4 is a sample
                   of the piece raster (fd2_piece_views).  Every view carries the rule as the metadata
                   item FLOWDIVIDE_VIEW_RULE, and a basin view the area of its smallest basin as
                   FLOWDIVIDE_VIEW_MIN_BASIN_AREA_KM2, the same text.
    vectorisation  one layer per view, its rows those of the table (basins: rows 1..N, fid = basin_id;
                   regions and groups: every row, fid = row): every object with a cell becomes one
                   multipolygon (holes and parts kept) from 4-connected cells, OGC-valid: two cells that
                   meet only at a corner are two polygons of it (an 8-connected
                   ring would touch itself there); the boundary lines of a basin view are the
                   outer rings of the 8-connected polygons, a D8 basin's outline as a map draws it; an object of the view without a cell at this resolution KEEPS ITS ROW, with
                   no geometry, 0 cells and -9999 in the columns only a cell can give, never dropped in
                   silence.  The file carries few columns: the identifier, the size
                   and the place of the object, then what the view adds (the cell count, cells_per_degree,
                   the rectangle of its cells, color_id); everything else is in the fine table at the row
                   the identifier names.  Written as GeoParquet (the default: ISO WKB, zstd, the "geo"
                   entry of GeoParquet 1.1 without a crs entry, so OGC:CRS84), as GeoPackage, or both,
                   with the same rows and columns (--vector).
    colouring      color_id, an integer such that two objects that touch on the coarse mask (corners
                   included) never share it: the greedy colouring of the adjacency graph in the
                   smallest-last order of Matula and Beck (1983), choosing among at least eight numbers
                   the one used least so far and preferring one absent from the neighbours' neighbours;
                   the palette that paints the numbers is the map's choice (PALETTES
                   below)
    default style  a GeoPackage view carries a QGIS categorized fill on color_id as its default style, in
                   the file's layer_styles table, so that QGIS colours the layer the moment it is opened
                   (a GeoParquet file carries no style)

The three figures of the paper that show the partition (Figure 3: delineation and grouping;
Figure 4: one basin through the cut; Figure 5: the three capacities) are drawn from the coarse
views at the end of this file.
"""
import contextlib
import json
import os
import time
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import rasterio
from numba import njit
from rasterio.windows import Window

from fd1_partition import (FlowDivideError, GDAL_WRITE_THREADS, RASTER_BLOCK, checking, timing, log, publish, read_block, read_table, write_json, write_table, ensure_directory,
                           basin_rectangles, group_windows_in_pixels)
import fd_tables

FLOWDIVIDE_VERSION = "1.0.0"            # named in the metadata of every vector file (flowdivide.py’s VERSION says the same)

CELL_MIN_LAND_PERCENT = 25              # a cell with less land than this stays water
CELL_MIN_WINNER_PERCENT = 25            # a candidate basin holds at least this much of the cell's land
VIEW_RULE = 3                           # the rule of the views: 3, the basins of at least an area
VIEW_RULE_TAG = ("v9.63: the view holds the basins of at least its minimum area, the ids 1..N; a cell of a quarter land goes to the "
                 "largest of them (the smallest id) holding a quarter of that land; the region view follows the basin view")   # written into every view
# the rule of the polygons and the boundary lines, written last among the
# metadata items of every vector file (the polygons given to users must be OGC-valid)
VECTOR_RULE_TAG = ("v9.90: the polygons are GDAL Polygonize's with 4-connected cells, OGC-valid (two cells that meet only at a corner are two "
                   "polygons of the MultiPolygon); the boundary lines are the outer rings of the 8-connected polygons")
EQUATOR_KM = 40075.017                  # the WGS84 equator, for the area of a cell
NO_CELL_AT_THIS_RESOLUTION = -9999      # the cell rectangle and the colour of a row of a view that holds no cell
VIEW_ADDED_COLUMNS = ["coarse_grid_count", "cells_per_degree", "cell_row_min", "cell_row_max", "cell_col_min", "cell_col_max", "color_id"]
# the columns of the fine table that a view carries, by the identifier of its table (a vector
# file keeps only the columns it needs, the table keeps everything): the identifier, the size and the place of the object; a name the
# table does not have (level2_code in a Level-03 table, level1_code in either) is passed over; the boundary file
# keeps the identifier alone
VIEW_TABLE_COLUMNS = {
    "basin_id": ["basin_id", "basin_area_km2", "outlet_lon", "outlet_lat"],
    "region_id": ["region_id", "cut_basin_id", "topological_level", "region_grid_count"],
    "group_id": ["group_id", "group_level", "group_kind", "level_code", "level3_count", "basin_count", "land_grid_count"],
}
VECTOR_FORMATS = ("geoparquet", "gpkg")  # --vector: one of them, or both
GEOPARQUET_ROW_GROUP_ROWS = 65536         # the row groups of a GeoParquet file

# The palettes of the figures.  Which hue a color_id number gets is a choice of the map, not of the data:
# the colouring gives every object a number 1 .. K (at least COLOURS_AT_LEAST numbers are used, so that
# the same colour does not come back two objects apart); "okabe-ito" is the palette of the style
# (the seven colours of Okabe and Ito and a purple, then the reserves), "pastel" the soft colours of the
# figures.  The GeoPackage views carry COLOR_ID_STYLE_PALETTE as their default style (embed_color_id_style).
PALETTES = {
    "okabe-ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8064A2", "#A6D854", "#8C6D31", "#80CDC1"],
    "pastel": ["#9fbdd4", "#d8c79c", "#aecfa6", "#e0b7a4", "#bcb3d4", "#c9d894", "#e6c4d1", "#9dc9c2", "#d6cbae", "#b2c2da", "#cfb9c9", "#a7d3dc"],
    "tableau": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#8CD17D", "#D4A6C8"],
    "greys": ["#f2f2f2", "#d9d9d9", "#bfbfbf", "#a6a6a6", "#8c8c8c", "#737373", "#595959", "#404040", "#e6e6e6", "#cccccc", "#b3b3b3", "#999999"],
}
COLOR_ID_PALETTE = PALETTES["okabe-ito"]
# The default style of a GeoPackage view: its palette
# (eight colours and three reserves; the grey the older table ended with read as "no value" on a map and was taken
# out), one fill per color_id number, outlines of 0.1 mm in COLOR_ID_STYLE_OUTLINE.  A color_id past the last entry
# stops the step: the palette is extended by hand, a colour is never used for two numbers.
COLOR_ID_STYLE_PALETTE = [("#E69F00", "orange"), ("#56B4E9", "sky blue"), ("#009E73", "bluish green"), ("#F0E442", "yellow"),
                          ("#0072B2", "blue"), ("#D55E00", "vermillion"), ("#CC79A7", "reddish purple"), ("#8064A2", "purple"),
                          ("#A6D854", "light green (reserve)"), ("#8C6D31", "brown (reserve)"), ("#80CDC1", "teal (reserve)")]
COLOR_ID_STYLE_OUTLINE = "#4d4d4d"
COLOR_ID_STYLE_NAME = "color_id"
COLOURS_AT_LEAST = 8


def min_basin_area_km2_of(cell_side_km):
    """the smallest basin of a view: a quarter of a cell of that side (at the equator on a geographic grid),
    rounded to four decimals half up (0.0194 / 0.0775 / 0.2151 / 7.7450 km2 for
    1/400, 1/200, 1/120 and 1/20 degree)"""
    quarter = Decimal(repr(float(cell_side_km))) ** 2 / Decimal(4)
    return float(quarter.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def integral_ids(column, name):
    """a column of identifiers as int64, refused when a value is not a finite whole number (a table read with
    a fraction would otherwise be truncated in silence)"""
    values = np.asarray(column.to_numpy())
    if values.dtype == object:
        # a column read as objects (mixed, Decimal, text): every value must be an integer already
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in values):
            raise FlowDivideError("the column %s holds a value that is not an integer" % name)
        values = np.array([int(value) for value in values], dtype=object)
        if values.size and (max(values) > np.iinfo(np.int64).max or min(values) < np.iinfo(np.int64).min):
            raise FlowDivideError("the column %s holds an identifier beyond 64 bits" % name)
        return values.astype(np.int64)
    if not np.issubdtype(values.dtype, np.integer):
        as_float = values.astype(np.float64)
        if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.floor(as_float)) or np.any(np.abs(as_float) >= 2.0 ** 63):
            raise FlowDivideError("the column %s holds a value that is not a whole number within 64 bits" % name)
    elif values.dtype == np.uint64 and values.size and values.max() > np.iinfo(np.int64).max:
        raise FlowDivideError("the column %s holds an identifier beyond 63 bits" % name)
    return values.astype(np.int64)


def basins_of_the_view(basins, min_area_km2, table_path=""):
    """N, the basins of the view: the table must hold the ids 1..n in row order with finite positive areas
    that do not increase down the rows, so the basins
    of at least min_area_km2 are its first N rows"""
    ids = integral_ids(basins["basin_id"], "basin_id")
    areas = basins["basin_area_km2"].to_numpy(np.float64)
    if ids.size == 0:
        raise FlowDivideError("the basin table %s has no rows" % table_path)
    if not np.array_equal(ids, np.arange(1, ids.size + 1)):
        raise FlowDivideError("the basin table %s does not hold the ids 1..n in row order" % table_path)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise FlowDivideError("a basin of %s has no finite positive area" % table_path)
    if np.any(np.diff(areas) > 0):
        raise FlowDivideError("the basin table %s is not in order of area: the basins of at least an area would not be the ids 1..N" % table_path)
    in_the_view = int(np.count_nonzero(areas >= min_area_km2))
    if in_the_view == 0:
        raise FlowDivideError("no basin of %s has the %.4f km2 of the view" % (table_path, min_area_km2))
    return in_the_view


# =============================================================================
#  [1] Resampling: which object a coarse cell belongs to
# =============================================================================

@njit(cache=True)
def _fold_strip_largest_basin(fine, factor, basin_count_in_the_view, out_row, min_land_pixels, count_of_basin, touched,
                              cell_count, cell_row_min, cell_row_max, cell_col_min, cell_col_max, coarse_row, flags):
    """the basin view of one row of coarse cells: a cell of
    at least min_land_pixels pixels of any basin (the basins beyond the view included) goes to the largest, that is
    the smallest id, of the basins of the view (ids 1..basin_count_in_the_view) that hold at least
    CELL_MIN_WINNER_PERCENT of the cell's land.  count_of_basin is one counter per basin id, zero between cells;
    touched the ids seen in the cell.  A basin id beyond the table sets flags[0]."""
    fine_nrow, fine_ncol = fine.shape
    coarse_ncol = out_row.size
    limit = min(count_of_basin.size, cell_count.size)
    for coarse_col in range(coarse_ncol):
        touched_count = 0
        land_pixels = 0
        for row in range(fine_nrow):
            for col in range(coarse_col * factor, min((coarse_col + 1) * factor, fine_ncol)):
                basin_id = fine[row, col]
                if basin_id == 0:
                    continue
                if basin_id < 0 or basin_id >= limit:        # a negative id is refused as one past the table
                    flags[0] = 1
                    continue
                land_pixels += 1
                if basin_id > basin_count_in_the_view:
                    continue
                if count_of_basin[basin_id] == 0:
                    touched[touched_count] = basin_id
                    touched_count += 1
                count_of_basin[basin_id] += 1
        cell_is_land = land_pixels >= min_land_pixels
        winner = np.uint32(0)
        for k in range(touched_count):
            basin_id = touched[k]
            count_in_the_cell = count_of_basin[basin_id]
            count_of_basin[basin_id] = 0
            if not cell_is_land or count_in_the_cell * 100 < CELL_MIN_WINNER_PERCENT * land_pixels:
                continue
            if winner == 0 or basin_id < winner:
                winner = basin_id
        out_row[coarse_col] = winner
        if winner != 0:
            cell_count[winner] += 1
            if coarse_row < cell_row_min[winner]:
                cell_row_min[winner] = coarse_row
            if coarse_row > cell_row_max[winner]:
                cell_row_max[winner] = coarse_row
            if coarse_col < cell_col_min[winner]:
                cell_col_min[winner] = coarse_col
            if coarse_col > cell_col_max[winner]:
                cell_col_max[winner] = coarse_col


@njit(cache=True)
def _fold_strip_region(fine, basin_fine, basin_view_row, factor, lookup, out_row, min_land_pixels, scratch_id, scratch_count,
                       cell_count, cell_row_min, cell_row_max, cell_col_min, cell_col_max, coarse_row, flags):
    """the region (or group) view of one row of coarse cells, following the basin view.  fine is the region mask of a capacity partition (lookup empty) or the basin mask
    read as groups (lookup: basin id -> group id); basin_fine the basin mask; basin_view_row the basin view's cells
    of this row.  A cell with a basin takes, for a partition, the region holding most of that basin's pixels in
    the cell (the smaller id breaks a tie), for a group the group of that basin.  A cell without one takes the
    region or group holding most of it when at least min_land_pixels of its pixels lie in a region (a group).
    flags[0]: a basin id beyond the table; flags[1]: the masks are not of one run (the cell's basin has no pixel in
    the cell, or a pixel of it lies in no region)."""
    fine_nrow, fine_ncol = fine.shape
    coarse_ncol = out_row.size
    for coarse_col in range(coarse_ncol):
        col0 = coarse_col * factor
        col1 = min((coarse_col + 1) * factor, fine_ncol)
        basin_of_the_cell = basin_view_row[coarse_col]
        winner = np.uint32(0)
        if basin_of_the_cell != 0 and lookup.size > 0:
            # a group view: the group of the cell's basin, which must have a pixel in the cell
            if basin_of_the_cell < 0 or basin_of_the_cell >= lookup.size or lookup[basin_of_the_cell] == 0:
                flags[0] = 1
            else:
                present = False
                for row in range(fine_nrow):
                    for col in range(col0, col1):
                        if basin_fine[row, col] == basin_of_the_cell:
                            present = True
                            break
                    if present:
                        break
                if present:
                    winner = lookup[basin_of_the_cell]
                else:
                    flags[1] = 1
        elif basin_of_the_cell != 0:
            # a capacity partition: the region holding most of the cell's basin's pixels in the cell
            distinct = 0
            for row in range(fine_nrow):
                for col in range(col0, col1):
                    if basin_fine[row, col] != basin_of_the_cell:
                        continue
                    region_id = fine[row, col]
                    if region_id == 0:
                        flags[1] = 1
                        continue
                    slot = 0
                    while slot < distinct and scratch_id[slot] != region_id:
                        slot += 1
                    if slot == distinct:
                        scratch_id[distinct] = region_id
                        scratch_count[distinct] = 0
                        distinct += 1
                    scratch_count[slot] += 1
            winner_count = 0
            for slot in range(distinct):
                if scratch_count[slot] > winner_count or (scratch_count[slot] == winner_count and scratch_id[slot] < winner):
                    winner = scratch_id[slot]
                    winner_count = scratch_count[slot]
            if distinct == 0:
                flags[1] = 1
        else:
            # the majority rule, for a cell the basin view left empty: the object holding most of the cell, the
            # cell being at least a quarter land (pixels in a region, or in a basin of a group)
            distinct = 0
            for row in range(fine_nrow):
                for col in range(col0, col1):
                    value = fine[row, col]
                    if value == 0:
                        continue
                    if lookup.size > 0:
                        if value >= lookup.size:
                            flags[0] = 1
                            continue
                        object_id = lookup[value]
                        if object_id == 0:
                            flags[0] = 1
                            continue
                    else:
                        object_id = value
                    slot = 0
                    while slot < distinct and scratch_id[slot] != object_id:
                        slot += 1
                    if slot == distinct:
                        scratch_id[distinct] = object_id
                        scratch_count[distinct] = 0
                        distinct += 1
                    scratch_count[slot] += 1
            winner_count = 0
            land_pixels = 0
            for slot in range(distinct):
                land_pixels += scratch_count[slot]
                if scratch_count[slot] > winner_count or (scratch_count[slot] == winner_count and scratch_id[slot] < winner):
                    winner = scratch_id[slot]
                    winner_count = scratch_count[slot]
            if land_pixels < min_land_pixels:
                winner = np.uint32(0)
        if winner >= cell_count.size:
            flags[0] = 1                                # a region or group id the table does not name
            winner = np.uint32(0)
        out_row[coarse_col] = winner
        if winner != 0:
            cell_count[winner] += 1
            if coarse_row < cell_row_min[winner]:
                cell_row_min[winner] = coarse_row
            if coarse_row > cell_row_max[winner]:
                cell_row_max[winner] = coarse_row
            if coarse_col < cell_col_min[winner]:
                cell_col_min[winner] = coarse_col
            if coarse_col > cell_col_max[winner]:
                cell_col_max[winner] = coarse_col


def min_land_pixels_of(factor):
    """a cell is land when at least CELL_MIN_LAND_PERCENT of its factor x factor pixels are
    (land_pixels * 100 >= CELL_MIN_LAND_PERCENT * factor * factor)"""
    return -(-CELL_MIN_LAND_PERCENT * factor * factor // 100)


def fold_mask_at_factors(fine_path, jobs, lookup=None, object_count_limit=None, tag="fd2.1", rule="largest_basin", basin_fine_path=None):
    """the fine mask folded at several factors in ONE pass over it: jobs is a list of (out_path, factor,
    basins_in_the_view or None, basin_view_path or None, min_basin_area_km2 or None); a strip of fine rows is a
    multiple of every factor, so every job folds whole coarse rows from it.  rule: "largest_basin", the basin
    view (the job names N, the basins of the view, ids 1..N, and the area of its smallest basin, which
    goes into the metadata); "region", the region or group view following the basin view named in the job
    (lookup: basin id -> group id for a group view, and the fine mask is the basin mask; for a capacity
    partition the fine mask is the region mask and basin_fine_path the basin mask read beside it; the area
    of the basin view is carried over).  Every view carries the rule in its metadata (FLOWDIVIDE_VIEW_RULE)
    and the area as FLOWDIVIDE_VIEW_MIN_BASIN_AREA_KM2.  A grid that is not a
    multiple of the factor keeps a last row and column of partial cells, judged over factor x factor pixels
    as if the missing ones were water.  Returns the objects tables
    of the jobs, in order."""
    import math
    started = time.time()
    jobs = [tuple(job) + (None,) * (5 - len(job)) for job in jobs]
    factors = [factor for _, factor, _, _, _ in jobs]
    if any(int(factor) != factor or factor < 1 for factor in factors):
        raise FlowDivideError("a fold factor must be a positive integer")
    if rule not in ("largest_basin", "region"):
        raise FlowDivideError("unknown fold rule '%s'" % rule)
    if rule == "largest_basin" and any(in_the_view is None or in_the_view < 1 or min_area is None for _, _, in_the_view, _, min_area in jobs):
        raise FlowDivideError("a basin view names the basins of the view (ids 1..N) and the area of its smallest basin")
    if rule == "region" and any(view is None for _, _, _, view, _ in jobs):
        raise FlowDivideError("a region view follows the basin view of its resolution, which must be named")
    if rule == "region" and lookup is None and basin_fine_path is None:
        raise FlowDivideError("a region view of a capacity partition reads the basin mask beside the region mask")
    common = 1
    for factor in factors:
        common = common * factor // math.gcd(common, factor)
    with contextlib.ExitStack() as stack:
        fine = stack.enter_context(rasterio.open(fine_path))
        # the masks are unsigned; a signed one could hand the kernels a negative id, which Numba would take as an
        # index without a bounds check
        if fine.dtypes[0] not in ("uint8", "uint16", "uint32"):
            raise FlowDivideError("%s is %s; the fine masks are unsigned integers" % (fine_path, fine.dtypes[0]))
        fine_nrow = fine.height
        fine_ncol = fine.width
        if common * fine_ncol * 4 > 2000000000 and len(jobs) > 1:
            # the factors have a common multiple too large for one strip: every factor gets its own pass
            log(tag, "the factors %s have no small common multiple; one pass per factor" % factors)
            return [fold_mask_at_factors(fine_path, [job], lookup=lookup, object_count_limit=object_count_limit, tag=tag, rule=rule,
                                         basin_fine_path=basin_fine_path)[0] for job in jobs]
        if object_count_limit is None:
            object_count_limit = int(lookup.max()) + 1 if lookup is not None and lookup.size else None
        if object_count_limit is None:
            largest = 0
            for row0 in range(0, fine_nrow, 4096):
                largest = max(largest, int(fine.read(1, window=Window(0, row0, fine_ncol, min(4096, fine_nrow - row0))).max()))
            object_count_limit = largest + 1
        lookup_array = lookup.astype(np.uint32) if lookup is not None else np.zeros(0, np.uint32)
        for _, _, in_the_view, _, _ in jobs:
            if in_the_view is not None and in_the_view >= object_count_limit:
                raise FlowDivideError("the view holds %d basins, the fold covers %d ids" % (in_the_view, object_count_limit))
        basin_fine = None
        if rule == "region" and lookup is None:
            basin_fine = stack.enter_context(rasterio.open(basin_fine_path))
            # the basin mask on the grid of the fine mask, and in its CRS
            if (basin_fine.width, basin_fine.height) != (fine_ncol, fine_nrow) or basin_fine.transform != fine.transform or \
                    basin_fine.crs != fine.crs:
                raise FlowDivideError("the basin mask %s is not on the grid of %s" % (basin_fine_path, fine_path))
            if basin_fine.dtypes[0] not in ("uint8", "uint16", "uint32"):          # unsigned, as the mask above
                raise FlowDivideError("%s is %s; the fine masks are unsigned integers" % (basin_fine_path, basin_fine.dtypes[0]))
        states = []
        for out_path, factor, in_the_view, basin_view_path, min_area_km2 in jobs:
            coarse_nrow = -(-fine_nrow // factor)
            coarse_ncol = -(-fine_ncol // factor)
            transform = fine.transform * rasterio.Affine.scale(factor)
            profile = {"driver": "GTiff", "dtype": "uint32", "count": 1, "width": coarse_ncol, "height": coarse_nrow, "crs": fine.crs,
                       "transform": transform, "nodata": 0, "tiled": True, "blockxsize": RASTER_BLOCK,
                       "blockysize": RASTER_BLOCK, "compress": "DEFLATE", "predictor": 2, "BIGTIFF": "YES", "NUM_THREADS": GDAL_WRITE_THREADS}
            basin_view = None
            if rule == "region":
                basin_view = stack.enter_context(rasterio.open(basin_view_path))
                if (basin_view.width, basin_view.height) != (coarse_ncol, coarse_nrow) or basin_view.transform != transform or \
                        basin_view.crs != fine.crs:
                    raise FlowDivideError("the basin view %s is not the fold of %s at factor %d" % (basin_view_path, fine_path, factor))
                if basin_view.dtypes[0] not in ("uint8", "uint16", "uint32"):          # unsigned, as the masks
                    raise FlowDivideError("%s is %s; the basin views are unsigned integers" % (basin_view_path, basin_view.dtypes[0]))
                if basin_view.tags().get("FLOWDIVIDE_VIEW_RULE") != VIEW_RULE_TAG:
                    raise FlowDivideError("the basin view %s does not carry the rule of this version; make the basin views first" % basin_view_path)
                min_area_text = basin_view.tags().get("FLOWDIVIDE_VIEW_MIN_BASIN_AREA_KM2")     # a region view carries the area of the basin view it follows
                if min_area_text is None:
                    raise FlowDivideError("the basin view %s carries no FLOWDIVIDE_VIEW_MIN_BASIN_AREA_KM2" % basin_view_path)
            else:
                min_area_text = "%.4f" % min_area_km2
            dataset = stack.enter_context(rasterio.open(out_path + ".partial.tif", "w", **profile))
            dataset.update_tags(FLOWDIVIDE_VIEW_RULE=VIEW_RULE_TAG, FLOWDIVIDE_VIEW_MIN_BASIN_AREA_KM2=min_area_text)
            states.append({
                "out_path": out_path, "factor": factor, "coarse_nrow": coarse_nrow, "coarse_ncol": coarse_ncol,
                "in_the_view": int(in_the_view) if in_the_view is not None else 0,
                "cell_count": np.zeros(object_count_limit, np.int64), "row_min": np.full(object_count_limit, np.iinfo(np.int32).max, np.int32),
                "row_max": np.full(object_count_limit, -1, np.int32), "col_min": np.full(object_count_limit, np.iinfo(np.int32).max, np.int32),
                "col_max": np.full(object_count_limit, -1, np.int32), "scratch_id": np.zeros(factor * factor + 1, np.uint32),
                "scratch_count": np.zeros(factor * factor + 1, np.int64),
                "min_land": min_land_pixels_of(factor), "count_of_basin": np.zeros(object_count_limit if rule == "largest_basin" else 0, np.int64),
                "flags": np.zeros(2, np.int64), "basin_view": basin_view, "dataset": dataset})
        strip_multiples = max(1, 200000000 // (common * fine_ncol))
        strip_rows = common * strip_multiples
        for fine_row0 in range(0, fine_nrow, strip_rows):
            fine_count = min(strip_rows, fine_nrow - fine_row0)
            strip = fine.read(1, window=Window(0, fine_row0, fine_ncol, fine_count))
            basin_strip = strip if basin_fine is None else basin_fine.read(1, window=Window(0, fine_row0, fine_ncol, fine_count))
            for state in states:
                factor = state["factor"]
                coarse_row0 = fine_row0 // factor
                coarse_count = -(-fine_count // factor)
                out_rows = np.zeros((coarse_count, state["coarse_ncol"]), np.uint32)
                basin_view_rows = None
                if rule == "region":
                    basin_view_rows = state["basin_view"].read(1, window=Window(0, coarse_row0, state["coarse_ncol"], coarse_count))
                for k in range(coarse_count):
                    block = strip[k * factor:min((k + 1) * factor, fine_count)]
                    if rule == "largest_basin":
                        _fold_strip_largest_basin(block, factor, state["in_the_view"], out_rows[k], state["min_land"], state["count_of_basin"], state["scratch_id"],
                                                  state["cell_count"], state["row_min"], state["row_max"], state["col_min"], state["col_max"], coarse_row0 + k, state["flags"])
                    else:
                        basin_block = basin_strip[k * factor:min((k + 1) * factor, fine_count)]
                        _fold_strip_region(block, basin_block, basin_view_rows[k], factor, lookup_array, out_rows[k], state["min_land"], state["scratch_id"], state["scratch_count"],
                                           state["cell_count"], state["row_min"], state["row_max"], state["col_min"], state["col_max"], coarse_row0 + k, state["flags"])
                state["dataset"].write(out_rows, 1, window=Window(0, coarse_row0, state["coarse_ncol"], coarse_count))
            if (fine_row0 // strip_rows) % 10 == 0:
                log(tag, "folded rows %d .. %d of %d at factors %s" % (fine_row0, fine_row0 + fine_count - 1, fine_nrow, factors))
    for state in states:
        if state["flags"][0]:
            raise FlowDivideError("%s: an id the table does not name met in the fold at factor %d" % (fine_path, state["factor"]))
        if state["flags"][1]:
            raise FlowDivideError("%s: the masks and the basin view at factor %d are not of one run (a cell's basin without a pixel in the cell, or a pixel of it in no region)" % (fine_path, state["factor"]))
    tables = []
    for state in states:
        publish(state["out_path"] + ".partial.tif", state["out_path"])
        present = np.nonzero(state["cell_count"] > 0)[0]
        objects = pd.DataFrame({"object_id": present, "coarse_grid_count": state["cell_count"][present], "cell_row_min": state["row_min"][present],
                                "cell_row_max": state["row_max"][present], "cell_col_min": state["col_min"][present], "cell_col_max": state["col_max"][present]})
        write_table(objects, state["out_path"] + ".objects.csv")
        tables.append(objects)
        log(tag, "written %s: %d x %d cells at factor %d, %d objects with a cell (rule %s)" % (state["out_path"], state["coarse_ncol"], state["coarse_nrow"], state["factor"], len(objects), rule))
    log(tag, "one pass over %s for %d factors, %d seconds" % (fine_path, len(jobs), int(time.time() - started)))
    return tables


# =============================================================================
#  [2] The colour number of every object, no two neighbours alike
# =============================================================================

@njit(cache=True)
def _adjacent_pairs_of_strip(strip, next_row, has_next_row, periodic, pair_a, pair_b, start):
    """every cell against the cells to its right, below-left, below and below-right (8 neighbours, each
    pair of cells looked at once; on a periodic grid the last column's right-hand neighbours are in the
    first column); a pair of two different objects is appended.  Returns the count."""
    nrow, ncol = strip.shape
    count = start
    for row in range(nrow):
        for col in range(ncol):
            here = strip[row, col]
            if here == 0:
                continue
            if col + 1 < ncol or periodic:
                other = strip[row, (col + 1) % ncol]
                if other != 0 and other != here:
                    if count >= pair_a.size:
                        return -1
                    pair_a[count] = here
                    pair_b[count] = other
                    count += 1
            below_available = row + 1 < nrow or has_next_row
            if not below_available:
                continue
            for dcol in (-1, 0, 1):
                n_col = col + dcol
                if periodic:
                    n_col = n_col % ncol
                elif n_col < 0 or n_col >= ncol:
                    continue
                if row + 1 < nrow:
                    other = strip[row + 1, n_col]
                else:
                    other = next_row[n_col]
                if other != 0 and other != here:
                    if count >= pair_a.size:
                        return -1
                    pair_a[count] = here
                    pair_b[count] = other
                    count += 1
    return count


def adjacent_pairs(coarse_path, periodic=False, strip_rows=None):
    """the pairs of objects that touch anywhere on the coarse mask, each once, as an (n, 2) array of ids"""
    pairs = []
    with rasterio.open(coarse_path) as coarse:
        nrow = coarse.height
        ncol = coarse.width
        if strip_rows is None:
            strip_rows = max(1, min(2048, 10000000 // max(ncol, 1)))
        for row0 in range(0, nrow, strip_rows):
            count_rows = min(strip_rows, nrow - row0)
            strip = coarse.read(1, window=Window(0, row0, ncol, count_rows))
            has_next = row0 + count_rows < nrow
            next_row = coarse.read(1, window=Window(0, row0 + count_rows, ncol, 1))[0] if has_next else np.zeros(ncol, np.uint32)
            capacity = 4 * strip.size + 8
            pair_a = np.empty(capacity, np.uint32)
            pair_b = np.empty(capacity, np.uint32)
            count = _adjacent_pairs_of_strip(strip, next_row, has_next, periodic, pair_a, pair_b, 0)
            if count < 0:
                raise FlowDivideError("more neighbouring pairs than cells times four; not possible")
            if count:
                a = pair_a[:count].astype(np.int64)
                b = pair_b[:count].astype(np.int64)
                low = np.minimum(a, b)
                high = np.maximum(a, b)
                pairs.append(np.unique(low * (2 ** 32) + high))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    keys = np.unique(np.concatenate(pairs))
    return np.column_stack([keys // (2 ** 32), keys % (2 ** 32)])


def colour_objects(row_ids, present_ids, pairs, at_least=8):
    """color_id for every object with cells, fixed by the tie rules below (an order that takes ties to
    the smallest id through a heap gives other colours on 7920 of the 9145 basins of South America's 20th view):
    the objects are numbered by their row in the table (row_ids, the identifiers in row
    order; present_ids, those with cells); the smallest-last order of Matula and Beck (1983) is taken with
    buckets, every object put at the head of its bucket in falling row order (the smallest row at the head), and a
    neighbour whose count falls moved to the head of the bucket below, the neighbours in rising row order; then the
    greedy colouring through that order backwards: the numbers on a coloured neighbour are barred; among 1 .. the
    numbers open (at_least at first) the rest are candidates, those on no neighbour of a neighbour preferred; of
    these the one given to the fewest objects so far, the smaller number on a tie; no candidate opens one number
    more.  Returns a dict object_id -> color_id, the largest number used and the degeneracy."""
    row_ids = np.asarray(row_ids, np.int64)
    object_count = row_ids.size
    row_of_id = {int(identifier): row for row, identifier in enumerate(row_ids.tolist())}
    has_cells = np.zeros(object_count, bool)
    for identifier in present_ids:
        has_cells[row_of_id[int(identifier)]] = True
    neighbours = [[] for _ in range(object_count)]
    for first_id, second_id in pairs:
        first_row = row_of_id.get(int(first_id))
        second_row = row_of_id.get(int(second_id))
        if first_row is None or second_row is None:
            continue
        neighbours[first_row].append(second_row)
        neighbours[second_row].append(first_row)
    for adjacent in neighbours:
        adjacent.sort()
    # the smallest-last order with buckets
    degree = [len(adjacent) for adjacent in neighbours]
    next_in_bucket = [-1] * object_count
    previous_in_bucket = [-1] * object_count
    taken_out = [False] * object_count
    largest_degree = max(degree) if degree else 0
    vertex_count = int(has_cells.sum())
    bucket_head = [-1] * (largest_degree + 1)
    for row in range(object_count - 1, -1, -1):
        if not has_cells[row]:
            continue
        next_in_bucket[row] = bucket_head[degree[row]]
        if next_in_bucket[row] != -1:
            previous_in_bucket[next_in_bucket[row]] = row
        bucket_head[degree[row]] = row
    order = []
    smallest_degree = 0
    degeneracy = 0
    while len(order) < vertex_count:
        while smallest_degree <= largest_degree and bucket_head[smallest_degree] == -1:
            smallest_degree += 1
        if smallest_degree > largest_degree:
            raise FlowDivideError("the smallest-last order ran out of objects before the %d with cells" % vertex_count)
        row = bucket_head[smallest_degree]
        bucket_head[smallest_degree] = next_in_bucket[row]
        if next_in_bucket[row] != -1:
            previous_in_bucket[next_in_bucket[row]] = -1
        taken_out[row] = True
        order.append(row)
        degeneracy = max(degeneracy, smallest_degree)
        for neighbour in neighbours[row]:
            if taken_out[neighbour]:
                continue
            if previous_in_bucket[neighbour] != -1:
                next_in_bucket[previous_in_bucket[neighbour]] = next_in_bucket[neighbour]
            else:
                bucket_head[degree[neighbour]] = next_in_bucket[neighbour]
            if next_in_bucket[neighbour] != -1:
                previous_in_bucket[next_in_bucket[neighbour]] = previous_in_bucket[neighbour]
            degree[neighbour] -= 1
            previous_in_bucket[neighbour] = -1
            next_in_bucket[neighbour] = bucket_head[degree[neighbour]]
            if next_in_bucket[neighbour] != -1:
                previous_in_bucket[next_in_bucket[neighbour]] = neighbour
            bucket_head[degree[neighbour]] = neighbour
        if smallest_degree > 0:
            smallest_degree -= 1
    # the greedy colouring through the order backwards
    number_limit = max(degeneracy + 1, at_least)
    colour = [0] * object_count
    objects_of_number = [0] * (number_limit + 1)
    numbers_open = at_least
    for row in reversed(order):
        barred = set()
        second_ring = set()
        for neighbour in neighbours[row]:
            if colour[neighbour] > 0:
                barred.add(colour[neighbour])
            for second_neighbour in neighbours[neighbour]:
                if second_neighbour != row and colour[second_neighbour] > 0:
                    second_ring.add(colour[second_neighbour])
        chosen = 0
        for ring in (0, 1):
            if chosen:
                break
            for number in range(1, numbers_open + 1):
                if number in barred or (ring == 0 and number in second_ring):
                    continue
                if chosen == 0 or objects_of_number[number] < objects_of_number[chosen]:
                    chosen = number
        if chosen == 0:
            if numbers_open >= number_limit:
                raise FlowDivideError("an object bars all %d colour numbers, more than the degeneracy %d allows" % (numbers_open, degeneracy))
            numbers_open += 1
            chosen = numbers_open
        colour[row] = chosen
        objects_of_number[chosen] += 1
    largest = max((number for number in range(1, numbers_open + 1) if objects_of_number[number] > 0), default=0)
    return {int(row_ids[row]): colour[row] for row in range(object_count) if has_cells[row]}, largest, degeneracy


def verify_colouring(coarse_path, colour_of, present_ids, periodic=False):
    """read the mask once more: no two neighbouring cells of different objects with one colour, every
    object of the mask with a colour above zero.  Returns (bad pairs, objects without a colour)."""
    pairs = adjacent_pairs(coarse_path, periodic)
    bad = 0
    for a, b in pairs:
        if colour_of.get(int(a), 0) == colour_of.get(int(b), 0):
            bad += 1
    uncoloured = sum(1 for object_id in present_ids if colour_of.get(int(object_id), 0) <= 0)
    return bad, uncoloured


# The HydroBASINS levels a group table can be folded to.  A Level-03 code is three digits: the first the
# Level-01 region, the first two the Level-02 unit.  Level-01 is the nine regions HydroBASINS draws, the
# Arctic (8) among them, apart from North America (7).


# The rule the fold follows, in the step's arguments for the same reason as MERGE_RULE.
# "nine-regions": Level-01 is the nine regions HydroBASINS draws, the Arctic (8) its own.
LEVEL_FOLD_RULE = "nine-regions"
GROUP_ID_LIMIT = 1000000            # the largest group id the tables carry


# =============================================================================
#  [3] Vectorisation: one object, one multipolygon, its row of the table
# =============================================================================

def _multipolygon_as_gdal_gives_it(polygons, cell_count, cell_area):
    """the polygons of one object as GDAL's Polygonize gives them with 4-connected cells, in its order and with its
    rings (the polygons given to users must be OGC-valid).  With 8-connected cells, two cells that meet only at a
    corner lie in one ring that touches itself there, which OGC calls invalid (747 of the 9145 basins of South
    America's 20th view), and such a ring is not split with make_valid here.  With 4-connected cells they are two polygons of the MultiPolygon, which may touch at a point.  GDAL does not promise
    validity, so it is checked, and the area the rings enclose against the cells to half a cell: a
    ring that missed or took a cell is off by a whole one, the rounding of the coordinates far less."""
    from shapely.geometry import MultiPolygon
    result = MultiPolygon(polygons)
    area = sum(polygon.area for polygon in polygons)
    expected = cell_count * cell_area
    if not result.is_valid or abs(area - expected) > 0.5 * cell_area:
        raise FlowDivideError("the polygons of an object are not OGC-valid (%s) or enclose %.12g against its %d cells' %.12g"
                              % (result.is_valid, area, cell_count, expected))
    return result


def columns_the_view_keeps(table, id_column, identifier_only=False):
    """the columns of the fine table that the vector file carries, in the order of VIEW_TABLE_COLUMNS:
    the names the table has; the identifier must be among them"""
    wanted = [id_column] if identifier_only else VIEW_TABLE_COLUMNS[id_column]
    kept = [name for name in wanted if name in table.columns]
    if id_column not in kept:
        raise FlowDivideError("the table has no column %s" % id_column)
    return kept


def _is_wgs84_longitude_latitude(crs):
    """EPSG:4326 or OGC:CRS84 (the same coordinates as GeoParquet's default, whatever the axis order the
    authority names)"""
    import pyproj
    crs = pyproj.CRS(crs)
    return crs.is_geographic and (crs.equals(pyproj.CRS("EPSG:4326"), ignore_axis_order=True) or crs.equals(pyproj.CRS("OGC:CRS84"), ignore_axis_order=True))


def _geo_metadata(geometry_type, bbox, crs):
    """the "geo" entry of GeoParquet 1.1: the geometry column, ISO WKB, the geometry
    type, the bounding box when a geometry was written; no crs entry when the grid is longitude and latitude on
    WGS84 (OGC:CRS84 by omission, what the GeoPackage says with EPSG:4326), else the grid's crs as PROJJSON (a
    grid of one's own may be projected)"""
    column = {"encoding": "WKB", "geometry_types": [geometry_type]}
    if bbox is not None:
        column["bbox"] = [float(v) for v in bbox]
    if crs is not None and not _is_wgs84_longitude_latitude(crs):
        column["crs"] = crs.to_json_dict()
    return json.dumps({"version": "1.1.0", "primary_column": "geometry", "columns": {"geometry": column}}, separators=(",", ":"))


def write_geoparquet(frame, path, geometry_type, metadata):
    """`frame` (a GeoDataFrame whose geometry may hold None) as a GeoParquet file: the attribute columns in their
    order and a last column "geometry" of ISO WKB (little endian) or NULL, zstd, row groups of
    GEOPARQUET_ROW_GROUP_ROWS, the file metadata `metadata` (FLOWDIVIDE_* items) and the "geo" entry """
    import pyarrow
    import pyarrow.parquet
    import shapely
    geometries = frame.geometry
    has_geometry = geometries.notna().to_numpy()
    wkb = [None] * len(frame)
    bbox = None
    if has_geometry.any():
        present = geometries[has_geometry]
        blobs = shapely.to_wkb(present.to_numpy(), byte_order=1, flavor="iso")
        for position, blob in zip(np.nonzero(has_geometry)[0], blobs):
            wkb[position] = blob
        bounds = present.total_bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    columns = [name for name in frame.columns if name != frame.geometry.name]
    arrays = []
    fields = []
    for name in columns:
        values = frame[name].to_numpy()
        if values.dtype == np.int32:
            arrays.append(pyarrow.array(values, type=pyarrow.int32()))
            fields.append(pyarrow.field(name, pyarrow.int32(), nullable=False))
        elif np.issubdtype(values.dtype, np.integer):
            arrays.append(pyarrow.array(values.astype(np.int64), type=pyarrow.int64()))
            fields.append(pyarrow.field(name, pyarrow.int64(), nullable=False))
        else:
            arrays.append(pyarrow.array(values.astype(np.float64), type=pyarrow.float64()))
            fields.append(pyarrow.field(name, pyarrow.float64(), nullable=False))
    arrays.append(pyarrow.array(wkb, type=pyarrow.binary()))
    fields.append(pyarrow.field("geometry", pyarrow.binary(), nullable=True))
    file_metadata = {key: str(value) for key, value in metadata.items()}
    file_metadata["geo"] = _geo_metadata(geometry_type, bbox, frame.crs)
    table = pyarrow.Table.from_arrays(arrays, schema=pyarrow.schema(fields, metadata=file_metadata))
    pyarrow.parquet.write_table(table, path, compression="zstd", row_group_size=GEOPARQUET_ROW_GROUP_ROWS)


def read_view_vector(stem, formats=("geoparquet",), layer=None):
    """the vector file of a view read back as a GeoDataFrame in the first of `formats` (the run's --vector, so
    that a file of another format left from an earlier run is never read instead)"""
    import geopandas
    if formats[0] == "geoparquet":
        return geopandas.read_parquet(stem + ".parquet")
    return geopandas.read_file(stem + ".gpkg", layer=layer, engine="pyogrio")


def _rgba_of(hex_colour):
    return "%d,%d,%d,255" % (int(hex_colour[1:3], 16), int(hex_colour[3:5], 16), int(hex_colour[5:7], 16))


def color_id_style_qml():
    """the QGIS categorized renderer on color_id, one fill symbol per entry of COLOR_ID_STYLE_PALETTE (QGIS 3.x)"""
    from xml.sax.saxutils import escape
    categories = []
    symbols = []
    for index, (hex_colour, name) in enumerate(COLOR_ID_STYLE_PALETTE):
        number = index + 1
        label = escape(("%d %s" % (number, name)).strip())
        categories.append('      <category value="%d" symbol="%d" label="%s" render="true" type="long"/>' % (number, index, label))
        symbols.append("\n".join([
            '      <symbol type="fill" name="%d" alpha="1" clip_to_extent="1" force_rhr="0" is_animated="0" frame_rate="10">' % index,
            '        <layer class="SimpleFill" enabled="1" locked="0" pass="0" id="color_id_%d">' % number,
            '          <Option type="Map">',
            '            <Option name="color" type="QString" value="%s"/>' % _rgba_of(hex_colour),
            '            <Option name="outline_color" type="QString" value="%s"/>' % _rgba_of(COLOR_ID_STYLE_OUTLINE),
            '            <Option name="outline_style" type="QString" value="solid"/>',
            '            <Option name="outline_width" type="QString" value="0.1"/>',
            '            <Option name="outline_width_unit" type="QString" value="MM"/>',
            '            <Option name="style" type="QString" value="solid"/>',
            '          </Option>',
            '        </layer>',
            '      </symbol>']))
    return "\n".join([
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>",
        '<qgis version="3.40" styleCategories="Symbology">',
        '  <renderer-v2 type="categorizedSymbol" attr="color_id" forceraster="0" symbollevels="0" enableorderby="0" referencescale="-1">',
        '    <categories>',
        "\n".join(categories),
        '    </categories>',
        '    <symbols>',
        "\n".join(symbols),
        '    </symbols>',
        '  </renderer-v2>',
        '  <layerGeometryType>2</layerGeometryType>',
        '</qgis>',
        ''])


def embed_color_id_style(gpkg_path, layer, largest_colour):
    """the default style of a GeoPackage view: the categorized fill on color_id written into the file's layer_styles
    table with useAsDefault.  The table is made as QGIS 3.40 makes
    it, and named in gpkg_contents: QGIS reads it through GDAL, which lists only the tables gpkg_contents names."""
    import sqlite3
    if largest_colour > len(COLOR_ID_STYLE_PALETTE):
        raise FlowDivideError("%s: color_id goes up to %d, the palette of the default style has %d entries (extend "
                              "COLOR_ID_STYLE_PALETTE; a colour is never used for two numbers)" % (gpkg_path, largest_colour, len(COLOR_ID_STYLE_PALETTE)))
    connection = sqlite3.connect(gpkg_path)
    try:
        geometry = connection.execute("SELECT column_name, geometry_type_name FROM gpkg_geometry_columns WHERE table_name = ?", (layer,)).fetchone()
        if geometry is None or geometry[1].upper() not in ("POLYGON", "MULTIPOLYGON"):
            raise FlowDivideError("%s: no polygon layer %s to style" % (gpkg_path, layer))
        connection.execute(
            "CREATE TABLE IF NOT EXISTS layer_styles ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
            " f_table_catalog TEXT(256), f_table_schema TEXT(256), f_table_name TEXT(256), f_geometry_column TEXT(256),"
            " styleName TEXT(30), styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN, description TEXT,"
            " owner TEXT(30), ui TEXT(30),"
            " update_time DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))")
        connection.execute(
            "INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change)"
            " SELECT 'layer_styles', 'attributes', 'layer_styles', '', strftime('%Y-%m-%dT%H:%M:%fZ','now')"
            " WHERE NOT EXISTS (SELECT 1 FROM gpkg_contents WHERE table_name = 'layer_styles')")
        connection.execute("DELETE FROM layer_styles WHERE f_table_name = ? AND styleName = ?", (layer, COLOR_ID_STYLE_NAME))
        connection.execute(
            "INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, styleName,"
            " styleQML, styleSLD, useAsDefault, description, owner, ui) VALUES ('', '', ?, ?, ?, ?, '', 1, ?, '', NULL)",
            (layer, geometry[0], COLOR_ID_STYLE_NAME, color_id_style_qml(),
             "categorized fill on color_id, %d entries of the palette; the layer uses 1..%d" % (len(COLOR_ID_STYLE_PALETTE), largest_colour)))
        connection.commit()
    finally:
        connection.close()


def check_color_id_style(gpkg_path, layer):
    """the read-back of embed_color_id_style: one default style on the layer, the one this module writes, and
    layer_styles named in gpkg_contents"""
    import sqlite3
    connection = sqlite3.connect(gpkg_path)
    try:
        rows = connection.execute("SELECT styleQML FROM layer_styles WHERE f_table_name = ? AND useAsDefault = 1", (layer,)).fetchall()
        listed = connection.execute("SELECT COUNT(*) FROM gpkg_contents WHERE table_name = 'layer_styles'").fetchone()[0]
    finally:
        connection.close()
    if len(rows) != 1 or rows[0][0] != color_id_style_qml() or listed != 1:
        raise FlowDivideError("%s: the default style of layer %s does not read back as written" % (gpkg_path, layer))


def _check_geoparquet_against_gpkg(parquet_path, gpkg_path, layer, feature_ids=None):
    """when both formats are written: the same rows in the same order, the same values, the same WKB.  feature_ids: the fids the GeoPackage must carry,
    1..n when None (the boundaries carry their table rows)"""
    import geopandas
    import pyarrow.parquet
    import shapely
    table = pyarrow.parquet.read_table(parquet_path)
    package = geopandas.read_file(gpkg_path, layer=layer, fid_as_index=True, engine="pyogrio")
    if table.num_rows != len(package):
        raise FlowDivideError("%s holds %d rows, %s %d" % (parquet_path, table.num_rows, gpkg_path, len(package)))
    expected_ids = list(range(1, len(package) + 1)) if feature_ids is None else [int(value) for value in feature_ids]
    if list(package.index) != expected_ids:
        raise FlowDivideError("%s: the feature ids are not the row numbers" % gpkg_path)
    columns = [name for name in table.column_names if name != "geometry"]
    if columns != [name for name in package.columns if name != "geometry"]:
        raise FlowDivideError("the columns of %s and %s differ" % (parquet_path, gpkg_path))
    for name in columns:
        if not np.array_equal(table.column(name).to_numpy(), package[name].to_numpy()):
            raise FlowDivideError("column %s differs between %s and %s" % (name, parquet_path, gpkg_path))
    wkb = table.column("geometry").to_pylist()
    package_wkb = [None if geometry is None else shapely.to_wkb(geometry, byte_order=1, flavor="iso") for geometry in package.geometry]
    if wkb != package_wkb:
        raise FlowDivideError("the geometries of %s and %s differ" % (parquet_path, gpkg_path))


def vectorise_mask(coarse_path, table, id_column, out_stem, layer, rows_in_the_view, cells_per_degree, min_basin_area_km2=None, formats=("geoparquet",),
                   tag="fd2.2", boundaries_stem=None, memory_limit_bytes=8 * 2 ** 30, periodic=False, colours_at_least=COLOURS_AT_LEAST, resolution_text=""):
    """One view as one layer: the rows 1..rows_in_the_view of `table` (a DataFrame with the id column; basins:
    the basins of the view, ids 1..N; regions and groups: every row), in the table's order, with the columns the
    view keeps (VIEW_TABLE_COLUMNS) and the columns the view adds (VIEW_ADDED_COLUMNS).  A row whose object holds
    cells on the coarse mask gets its multipolygon, its cell count, the rectangle of its cells and its color_id;
    a row without a cell at this resolution keeps its row with no geometry, 0 cells and
    NO_CELL_AT_THIS_RESOLUTION in the rectangle and the colour.  Written as <out_stem>.parquet (GeoParquet 1.1,
    zstd, ISO WKB), <out_stem>.gpkg (layer `layer`, fid = row number), or both, as `formats` says.  With
    boundaries_stem the outer rings of every polygon are written there as lines, with the identifier and the
    first two added columns only (a basin's divide as a map draws it).  The mask is held in memory when it is
    smaller than memory_limit_bytes, otherwise every object's rectangle is read from the file.  The colouring
    uses at least colours_at_least numbers."""
    import geopandas
    from rasterio import features
    from shapely.geometry import shape, MultiLineString, LineString
    started = time.time()
    formats = tuple(formats)
    if not formats or any(name not in VECTOR_FORMATS for name in formats):
        raise FlowDivideError("the vector formats are among %s" % ", ".join(VECTOR_FORMATS))
    objects = read_table(coarse_path + ".objects.csv")
    present_ids = integral_ids(objects["object_id"], "object_id") if len(objects) else np.zeros(0, np.int64)
    rows_in_the_view = int(rows_in_the_view)
    if rows_in_the_view < 1 or rows_in_the_view > len(table):
        raise FlowDivideError("the view holds %d rows, the table %d" % (rows_in_the_view, len(table)))
    view_table = table.iloc[:rows_in_the_view]
    ids_of_the_view = integral_ids(view_table[id_column], id_column)
    if id_column == "basin_id" and not np.array_equal(ids_of_the_view, np.arange(1, rows_in_the_view + 1)):
        raise FlowDivideError("the basins of the view are not the ids 1..%d in row order" % rows_in_the_view)
    if np.unique(ids_of_the_view).size != ids_of_the_view.size:
        raise FlowDivideError("an identifier of the table repeats")
    row_of_id = dict(zip(ids_of_the_view.tolist(), range(rows_in_the_view)))
    if not set(present_ids.tolist()) <= set(row_of_id):
        raise FlowDivideError("%s holds objects beyond the %d rows of this view: the mask was not made by the rule of this version" % (coarse_path, rows_in_the_view))
    # the objects list checked against the mask itself: every id with a cell is listed, with its count.  A list
    # that had lost a row would draw that object as an empty geometry of 0 cells and publish it
    with rasterio.open(coarse_path) as coarse_dataset:
        counted = np.zeros(max(rows_in_the_view, int(present_ids.max()) if present_ids.size else 0) + 1, np.int64)
        for strip_row0 in range(0, coarse_dataset.height, 4096):
            strip_count = min(4096, coarse_dataset.height - strip_row0)
            strip = coarse_dataset.read(1, window=Window(0, strip_row0, coarse_dataset.width, strip_count)).ravel()
            strip = strip[strip != 0].astype(np.int64)
            if strip.size and int(strip.max()) >= counted.size:
                raise FlowDivideError("%s holds object %d, beyond the %d rows of this view" % (coarse_path, int(strip.max()), rows_in_the_view))
            counted += np.bincount(strip, minlength=counted.size)
    if np.unique(present_ids).size != present_ids.size:
        raise FlowDivideError("%s.objects.csv lists an object twice" % coarse_path)
    listed = np.zeros(counted.size, np.int64)
    if len(objects):
        listed[present_ids] = objects["coarse_grid_count"].to_numpy(np.int64)
    if not np.array_equal(listed, counted):
        first = int(np.nonzero(listed != counted)[0][0])
        raise FlowDivideError("%s.objects.csv does not list the mask's objects: object %d has %d cells in the mask, %d in the list"
                              % (coarse_path, first, int(counted[first]), int(listed[first])))
    if present_ids.size:
        pairs = adjacent_pairs(coarse_path, periodic)
        colour_of, largest_colour, degeneracy = colour_objects(ids_of_the_view, present_ids, pairs, at_least=colours_at_least)
    else:
        # no object holds a cell at this resolution: every row is written without a geometry
        pairs, colour_of, largest_colour, degeneracy = np.zeros((0, 2), np.int64), {}, 0, 0
    log(tag, "%s: %d objects with cells of the %d rows, %d neighbouring pairs, degeneracy %d, %d colours" % (coarse_path, present_ids.size, rows_in_the_view, pairs.shape[0], degeneracy, largest_colour))
    kept = columns_the_view_keeps(table, id_column)
    boundary_kept = columns_the_view_keeps(table, id_column, identifier_only=True)
    with rasterio.open(coarse_path) as coarse:
        transform = coarse.transform
        crs = coarse.crs
        in_memory = coarse.width * coarse.height * 4 <= memory_limit_bytes
        whole = coarse.read(1) if in_memory else None
        geometries = [None] * rows_in_the_view
        boundary_geometries = []
        boundary_rows = []
        cell_count = np.zeros(rows_in_the_view, np.int64)
        cell_box = np.full((rows_in_the_view, 4), NO_CELL_AT_THIS_RESOLUTION, np.int32)       # row_min, row_max, col_min, col_max
        colour = np.full(rows_in_the_view, NO_CELL_AT_THIS_RESOLUTION, np.int32)
        for record in objects.itertuples(index=False):
            object_id = int(record.object_id)
            row_index = row_of_id[object_id]
            row0 = int(record.cell_row_min)
            row1 = int(record.cell_row_max)
            col0 = int(record.cell_col_min)
            col1 = int(record.cell_col_max)
            if in_memory:
                block = whole[row0:row1 + 1, col0:col1 + 1]
            else:
                block = coarse.read(1, window=Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1))
            mask = (block == object_id).astype(np.uint8)
            if int(mask.sum()) != int(record.coarse_grid_count):
                raise FlowDivideError("object %d holds %d cells in its rectangle, the fold counted %d" % (object_id, int(mask.sum()), int(record.coarse_grid_count)))
            window_transform = transform * rasterio.Affine.translation(col0, row0)
            polygons = [shape(geometry) for geometry, value in features.shapes(mask, mask=mask.astype(bool), connectivity=4, transform=window_transform) if value == 1]
            geometry = _multipolygon_as_gdal_gives_it(polygons, int(record.coarse_grid_count), abs(transform.a * transform.e))
            geometries[row_index] = geometry
            cell_count[row_index] = int(record.coarse_grid_count)
            cell_box[row_index] = (row0, row1, col0, col1)
            colour[row_index] = colour_of[object_id]
            if boundaries_stem is not None:
                # the outline from the 8-connected polygons: a cell touching the
                # rest at a corner only is not a ring of its own (it would show as a dot inside the Mississippi)
                outline = [shape(ring) for ring, value in features.shapes(mask, mask=mask.astype(bool), connectivity=8, transform=window_transform) if value == 1]
                boundary_geometries.append(MultiLineString([LineString(polygon.exterior.coords) for polygon in outline]))
                boundary_rows.append(row_index)
    attributes = pd.DataFrame({name: view_table[name].to_numpy() for name in kept})
    attributes["coarse_grid_count"] = cell_count
    attributes["cells_per_degree"] = np.full(rows_in_the_view, int(cells_per_degree), np.int32)
    attributes["cell_row_min"] = cell_box[:, 0]
    attributes["cell_row_max"] = cell_box[:, 1]
    attributes["cell_col_min"] = cell_box[:, 2]
    attributes["cell_col_max"] = cell_box[:, 3]
    attributes["color_id"] = colour
    frame = geopandas.GeoDataFrame(attributes, geometry=geopandas.GeoSeries(geometries, crs=crs), crs=crs)
    rows_text = ("the basins of the view, ids 1..N, row k is basin k + 1 (row basin_id - 1 of the fine table), as the GeoPackage's fid = basin_id"
                 if id_column == "basin_id" else "every row of the fine table in its order, as the GeoPackage's fid = row number")
    metadata = {"FLOWDIVIDE_VERSION": FLOWDIVIDE_VERSION, "FLOWDIVIDE_VIEW_RULE": VIEW_RULE_TAG, "FLOWDIVIDE_RESOLUTION": resolution_text, "FLOWDIVIDE_ROWS": rows_text}
    if min_basin_area_km2 is not None:
        metadata["FLOWDIVIDE_VIEW_MIN_BASIN_AREA_KM2"] = "%.4f" % min_basin_area_km2
    metadata["FLOWDIVIDE_VECTOR_RULE"] = VECTOR_RULE_TAG
    drawn = int(len(objects))
    without_geometry = rows_in_the_view - drawn
    written = {}
    if "geoparquet" in formats:
        temporary = out_stem + ".partial.parquet"
        with timing("output"):
            write_geoparquet(frame, temporary, "MultiPolygon", metadata)
        written["geoparquet"] = temporary
    if "gpkg" in formats:
        temporary = out_stem + ".partial.gpkg"
        if os.path.exists(temporary):
            os.remove(temporary)
        with timing("output"):
            # the layer is declared MultiPolygon, as the GeoParquet file is: a view in which no object holds a cell
            # has no geometry to take the type from, and would be written as a layer of unknown type
            frame.to_file(temporary, driver="GPKG", layer=layer, engine="pyogrio", geometry_type="MultiPolygon", dataset_metadata=metadata)
            embed_color_id_style(temporary, layer, largest_colour)
        written["gpkg"] = temporary
    # read back and check (a check, timed on its own): every attribute column as it was meant, the geometry of
    # exactly the rows with cells, -9999 / 0 where there is none, the colours against the mask; both formats when
    # both are written
    with checking():
        for format_name, path in written.items():
            check = geopandas.read_parquet(path) if format_name == "geoparquet" else geopandas.read_file(path, layer=layer, fid_as_index=True, engine="pyogrio")
            if format_name == "gpkg" and list(check.index) != list(range(1, rows_in_the_view + 1)):
                raise FlowDivideError("%s: the feature ids are not the row numbers 1..%d" % (path, rows_in_the_view))
            if format_name == "gpkg":
                check_color_id_style(path, layer)
            if list(check.columns) != list(attributes.columns) + ["geometry"]:
                raise FlowDivideError("%s does not hold the columns %s" % (path, list(attributes.columns)))
            for name in attributes.columns:
                if not np.array_equal(check[name].to_numpy(), attributes[name].to_numpy()):
                    raise FlowDivideError("%s: the column %s does not read back as written" % (path, name))
            read_ids = check[id_column].to_numpy(np.int64)
            has_geometry = check.geometry.notna().to_numpy()
            read_colours = check["color_id"].to_numpy(np.int64)
            read_counts = check["coarse_grid_count"].to_numpy(np.int64)
            read_box = check[["cell_row_min", "cell_row_max", "cell_col_min", "cell_col_max"]].to_numpy(np.int64)
            with_cells = np.isin(read_ids, present_ids)
            if not np.array_equal(has_geometry, with_cells):
                raise FlowDivideError("%s: a geometry is not where the cells are" % path)
            if not (np.all(read_counts[~with_cells] == 0) and np.all(read_colours[~with_cells] == NO_CELL_AT_THIS_RESOLUTION) and np.all(read_box[~with_cells] == NO_CELL_AT_THIS_RESOLUTION)):
                raise FlowDivideError("%s: a row without a cell does not carry 0 / -9999" % path)
            if not (np.all(read_counts[with_cells] > 0) and np.all(read_colours[with_cells] >= 1) and np.all(read_box[with_cells] >= 0)):
                raise FlowDivideError("%s: a row with cells lacks its count, rectangle or colour" % path)
            present_geometry = check.geometry[has_geometry]
            if not bool((~present_geometry.is_empty).all() and present_geometry.is_valid.all()):     # 4-connected, valid
                raise FlowDivideError("%s: an empty or OGC-invalid geometry" % path)
            bad_pairs, uncoloured = verify_colouring(coarse_path, dict(zip(read_ids[with_cells].tolist(), read_colours[with_cells].tolist())), present_ids, periodic)
            if bad_pairs or uncoloured:
                raise FlowDivideError("%s fails the read-back check: %d neighbouring pairs with one colour, %d objects without a colour" % (path, bad_pairs, uncoloured))
        if len(written) == 2:
            _check_geoparquet_against_gpkg(written["geoparquet"], written["gpkg"], layer)
    for format_name, path in written.items():
        publish(path, out_stem + (".parquet" if format_name == "geoparquet" else ".gpkg"))
    if boundaries_stem is not None:
        boundary_attributes = pd.DataFrame({name: view_table[name].to_numpy()[boundary_rows] for name in boundary_kept})
        boundary_attributes["coarse_grid_count"] = cell_count[boundary_rows]
        boundary_attributes["cells_per_degree"] = np.full(len(boundary_rows), int(cells_per_degree), np.int32)
        boundary_frame = geopandas.GeoDataFrame(boundary_attributes, geometry=geopandas.GeoSeries(boundary_geometries, crs=crs), crs=crs)
        boundary_written = {}
        if "geoparquet" in formats:
            temporary = boundaries_stem + ".partial.parquet"
            with timing("output"):
                write_geoparquet(boundary_frame, temporary, "MultiLineString", metadata)
            boundary_written["geoparquet"] = temporary
        if "gpkg" in formats:
            temporary = boundaries_stem + ".partial.gpkg"
            if os.path.exists(temporary):
                os.remove(temporary)
            # the fid of a boundary is its table row (row + 1), not 1..k in
            # the order written: with a basin without a cell in the view the two numberings part
            boundary_frame_with_fid = boundary_frame.set_index(pd.Index(np.asarray(boundary_rows, np.int64) + 1, name="fid"))
            with timing("output"):
                boundary_frame_with_fid.to_file(temporary, driver="GPKG", layer="boundaries", engine="pyogrio", index=True,
                                                dataset_metadata=metadata)
            del boundary_frame_with_fid
            boundary_written["gpkg"] = temporary
        with checking():
            for format_name, path in boundary_written.items():
                check = geopandas.read_parquet(path) if format_name == "geoparquet" else geopandas.read_file(path, layer="boundaries", fid_as_index=True, engine="pyogrio")
                if list(check.columns) != list(boundary_attributes.columns) + ["geometry"] or len(check) != len(boundary_rows):
                    raise FlowDivideError("%s does not hold the %d boundary rows with their columns" % (path, len(boundary_rows)))
                if format_name == "gpkg" and not np.array_equal(check.index.to_numpy(np.int64), np.asarray(boundary_rows, np.int64) + 1):
                    raise FlowDivideError("%s: the fid of the boundaries is not their table row" % path)
                for name in boundary_attributes.columns:
                    if not np.array_equal(check[name].to_numpy(), boundary_attributes[name].to_numpy()):
                        raise FlowDivideError("%s: the column %s does not read back as written" % (path, name))
                if not bool(check.geometry.notna().all() and (~check.geometry.is_empty).all()):
                    raise FlowDivideError("%s: a boundary is missing or empty" % path)
            if len(boundary_written) == 2:
                _check_geoparquet_against_gpkg(boundary_written["geoparquet"], boundary_written["gpkg"], "boundaries",
                                               feature_ids=np.asarray(boundary_rows, np.int64) + 1)
        for format_name, path in boundary_written.items():
            publish(path, boundaries_stem + (".parquet" if format_name == "geoparquet" else ".gpkg"))
    report = {"rows": rows_in_the_view, "drawn": drawn, "without_geometry": without_geometry, "columns": list(attributes.columns), "formats": list(formats),
              "pairs": int(pairs.shape[0]), "degeneracy": int(degeneracy), "colours": int(largest_colour), "colours_at_least": int(colours_at_least),
              "seconds": round(time.time() - started, 1)}
    write_json(out_stem + ".report.json", report)
    log(tag, "written %s (%s): %d rows, %d drawn, %d without a cell at this resolution (rows kept, no geometry), %d columns, %d colours, %d seconds" % (
        out_stem, " and ".join(formats), rows_in_the_view, drawn, without_geometry, len(attributes.columns), largest_colour, int(report["seconds"])))
    return report


# =============================================================================
#  [4] The figures of the manuscript, drawn from the coarse views
# =============================================================================
#
#  Figure 3  (a) the basins shaded by drainage area, the basins over the capacity outlined;
#            (b) the basin groups along the Level-03 units, (c) the automatic groups along the Hilbert
#            curve, both hatched where over the capacity
#  Figure 4  one cut basin: (a) the nine pieces of the first level, the pieces over the capacity
#            hatched with their windows; (b) the pieces used, all within the capacity; (c) the regions
#            merged in code order; (d) the region graph, every arrow pointing to a larger number
#  Figure 5  the partition at every capacity: regions coloured, the cut basins outlined
#
#  Every figure is drawn at the print width of a two-column journal figure (7.48 inch) so that the point
#  sizes in the file are the point sizes on paper; nothing goes below 7 pt.

FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]
SIZE_LETTER = 8.5
SIZE_TITLE = 8.0
SIZE_FOOT = 7.2
SIZE_TICK = 7.5
SIZE_LABEL = 7.0
INK = "#12161c"
TEXT_COLOUR = "#2b3038"
OCEAN = "#f4f2ee"
MAP_PALETTE = PALETTES["pastel"]          # the figures are drawn with the soft colours
OVER_CAPACITY_COLOUR = "#96301a"
CUT_BASIN_OUTLINE = "#8c1d0b"
RIVER_COLOUR = (0.05, 0.16, 0.45, 1.0)
BOUNDARY_LINE = (0.16, 0.18, 0.21, 1.0)
OUTLET_MARKER = dict(marker="v", markerfacecolor="#d7301f", markeredgecolor="#12161c", markeredgewidth=0.6, markersize=6.0, linestyle="none")


def _apply_style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY, "font.size": SIZE_TITLE, "axes.titlesize": SIZE_TITLE,
        "axes.labelsize": SIZE_TITLE, "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "hatch.linewidth": 0.3, "hatch.color": OVER_CAPACITY_COLOUR,
    })


def _panel_heading(axis, letter, title):
    figure = axis.figure
    renderer = figure.canvas.get_renderer()
    probe = axis.text(0.0, 1.0, letter + " ", transform=axis.transAxes, fontsize=SIZE_LETTER, fontweight="bold", va="bottom", ha="left")
    letter_width = probe.get_window_extent(renderer=renderer).width * 72.0 / figure.dpi
    probe.remove()
    axis.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, 4.0), textcoords="offset points", fontsize=SIZE_LETTER,
                  fontweight="bold", va="bottom", ha="left", color=INK, annotation_clip=False)
    axis.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(letter_width, 4.0), textcoords="offset points", fontsize=SIZE_TITLE,
                  va="bottom", ha="left", color=INK, annotation_clip=False)


def _panel_foot(axis, text, y=-0.085):
    axis.text(0.5, y, text, transform=axis.transAxes, fontsize=SIZE_FOOT, ha="center", va="top", color=TEXT_COLOUR)


def _map_axis(axis, view, longitude_ticks, latitude_ticks, hide_latitude_labels=False):
    axis.set_xlim(view[0], view[1])
    axis.set_ylim(view[2], view[3])
    axis.set_xticks(longitude_ticks)
    axis.set_yticks(latitude_ticks)
    axis.set_xticklabels(["%g°%s" % (abs(v), "W" if v < 0 else "E") if v != 0 else "0°" for v in longitude_ticks])
    axis.set_yticklabels([] if hide_latitude_labels else ["%g°%s" % (abs(v), "S" if v < 0 else "N") if v != 0 else "0°" for v in latitude_ticks])
    axis.tick_params(labelsize=SIZE_TICK, length=2.2, pad=1.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#8a8f98")


def _tick_values(low, high, at_most=5):
    """round tick values in degrees for a map spanning low .. high, at most `at_most` of them"""
    for step in (0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60):
        first = np.ceil(low / step) * step
        values = [float(v) for v in np.arange(first, high + 1e-9, step)]
        if len(values) <= at_most:
            return [int(v) if float(v).is_integer() else v for v in values]
    return []


def _extent_of(path):
    with rasterio.open(path) as dataset:
        t = dataset.transform
        return (t.c, t.c + dataset.width * t.a, t.f + dataset.height * t.e, t.f)


def _read_view(path, view=None, max_pixels=4000):
    """a coarse raster (or its part inside the view in degrees) with its extent, read at most max_pixels
    wide or high: a larger one is read decimated (every k-th cell), which is all a printed panel can show"""
    from rasterio.enums import Resampling
    with rasterio.open(path) as dataset:
        t = dataset.transform
        if view is None:
            col0, row0, col1, row1 = 0, 0, dataset.width, dataset.height
        else:
            col0 = max(0, int((view[0] - t.c) / t.a))
            col1 = min(dataset.width, int(np.ceil((view[1] - t.c) / t.a)))
            row0 = max(0, int((view[3] - t.f) / t.e))
            row1 = min(dataset.height, int(np.ceil((view[2] - t.f) / t.e)))
        step = max(1, int(np.ceil(max(col1 - col0, row1 - row0) / max_pixels)))
        window = Window(col0, row0, col1 - col0, row1 - row0)
        out_shape = ((row1 - row0) // step, (col1 - col0) // step)
        array = dataset.read(1, window=window, out_shape=out_shape, resampling=Resampling.nearest)
        return array, (t.c + col0 * t.a, t.c + col1 * t.a, t.f + row1 * t.e, t.f + row0 * t.e)


def _greedy_map_colours(label_raster, label_list):
    """a colour index for every label such that touching labels differ (for the maps only; the
    color_id of the products is made by colour_objects)"""
    neighbours = {label: set() for label in label_list}
    for one, other in ((label_raster[:, :-1], label_raster[:, 1:]), (label_raster[:-1, :], label_raster[1:, :])):
        differ = (one != other) & (one > 0) & (other > 0)
        for a, b in set(zip(one[differ].tolist(), other[differ].tolist())):
            if a in neighbours and b in neighbours:
                neighbours[a].add(b)
                neighbours[b].add(a)
    colour_of = {}
    for label in sorted(label_list, key=lambda l: -len(neighbours[l])):
        taken = {colour_of[other] for other in neighbours[label] if other in colour_of}
        colour_of[label] = next((index for index in range(len(MAP_PALETTE)) if index not in taken), 0)
    return colour_of


def _draw_labelled_raster(axis, label_raster, extent, label_text_of, hatched=(), label_size=SIZE_LABEL, colour_of=None, dots_below=40, dots_for_every_label=False):
    """every label filled with its colour, the lines between labels, hatching for some, a number at the
    middle of each label's cells; a label too small to see gets a dot (dots_for_every_label: also the
    labels that get no text).  Returns the text objects."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    label_list = sorted(int(v) for v in np.unique(label_raster) if v > 0)
    if colour_of is None:
        colour_of = _greedy_map_colours(label_raster, label_list)
    painted = np.zeros(label_raster.shape, np.int16)
    for label in label_list:
        painted[label_raster == label] = colour_of[label] % len(MAP_PALETTE) + 1
    axis.imshow(painted, cmap=ListedColormap([OCEAN] + MAP_PALETTE), vmin=0, vmax=len(MAP_PALETTE), extent=extent, interpolation="nearest", origin="upper")
    boundary = np.zeros(label_raster.shape, bool)
    boundary[:, :-1] |= (label_raster[:, :-1] != label_raster[:, 1:]) & (label_raster[:, :-1] > 0) & (label_raster[:, 1:] > 0)
    boundary[:-1, :] |= (label_raster[:-1, :] != label_raster[1:, :]) & (label_raster[:-1, :] > 0) & (label_raster[1:, :] > 0)
    layer = np.zeros(label_raster.shape + (4,), float)
    layer[boundary] = BOUNDARY_LINE
    axis.imshow(layer, extent=extent, interpolation="nearest", origin="upper")
    rows, cols = label_raster.shape
    longitude = extent[0] + (np.arange(cols) + 0.5) * (extent[1] - extent[0]) / cols
    latitude = extent[3] - (np.arange(rows) + 0.5) * (extent[3] - extent[2]) / rows
    if hatched:
        mask = np.isin(label_raster, list(hatched)).astype(float)
        axis.contourf(longitude, latitude, mask, levels=[0.5, 1.5], colors="none", hatches=["/////"])
    texts = []
    for label in label_list:
        if label not in label_text_of and not dots_for_every_label:
            continue
        row_index, col_index = np.nonzero(label_raster == label)
        centre = (float(longitude[col_index].mean()), float(latitude[row_index].mean()))
        if row_index.size < dots_below:
            axis.plot([centre[0]], [centre[1]], marker="o", markersize=2.4, markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.3, linestyle="none", zorder=7)
        if label not in label_text_of:
            continue
        texts.append(axis.text(centre[0], centre[1] - (0.0 if row_index.size >= dots_below else 0.02 * (extent[3] - extent[2])), label_text_of[label],
                               fontsize=label_size, ha="center", va="center", color=INK, zorder=6, clip_on=True,
                               bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.72)))
    return texts


def _separate_labels(figure, axis, texts, rounds=200):
    """push apart labels drawn on top of one another, in display units, keeping every box in the frame"""
    if not texts:
        return
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer=renderer)
    for _ in range(rounds):
        boxes = [text.get_window_extent(renderer=renderer) for text in texts]
        shift_x = [0.0] * len(texts)
        shift_y = [0.0] * len(texts)
        moved = False
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                a = boxes[first]
                b = boxes[second]
                overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
                overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue
                moved = True
                if overlap_y <= overlap_x:
                    step = 0.5 * (overlap_y + 1.0)
                    sign = -1.0 if a.y0 + a.y1 <= b.y0 + b.y1 else 1.0
                    shift_y[first] += sign * step
                    shift_y[second] -= sign * step
                else:
                    step = 0.5 * (overlap_x + 1.0)
                    sign = -1.0 if a.x0 + a.x1 <= b.x0 + b.x1 else 1.0
                    shift_x[first] += sign * step
                    shift_x[second] -= sign * step
        if not moved:
            break
        for index, text in enumerate(texts):
            if shift_x[index] == 0.0 and shift_y[index] == 0.0:
                continue
            box = boxes[index]
            wanted_x = min(max(shift_x[index], frame.x0 + 1.5 - box.x0), frame.x1 - 1.5 - box.x1)
            wanted_y = min(max(shift_y[index], frame.y0 + 1.5 - box.y0), frame.y1 - 1.5 - box.y1)
            position = axis.transData.transform(text.get_position())
            new = axis.transData.inverted().transform((position[0] + wanted_x, position[1] + wanted_y))
            text.set_position((float(new[0]), float(new[1])))


def _next_version_path(directory, stem):
    import glob
    import re
    existing = glob.glob(os.path.join(directory, stem + "_v*.png"))
    version = 1 + max([int(re.search(r"_v(\d+)\.png$", p).group(1)) for p in existing] + [0])
    return os.path.join(directory, "%s_v%d.png" % (stem, version))


def figure3_delineation_and_groups(basin_view_path, basin_table_path, group_views, capacity_pixels, grid, out_directory,
                                   view, longitude_ticks, latitude_ticks, capacity_label, figure_size=(7.48, 4.3), stacked=False, tag="figure"):
    """Figure 3: (a) the basins shaded by drainage area, the basins whose own window exceeds the capacity
    outlined in dark red; then one panel per grouping in group_views, a list of (title, group view
    raster, group table, basin-group table): (b) the basin groups by Level-03 unit, (c) by Hilbert
    curve, hatched where over the capacity.  stacked: the panels one above the other (a tall
    continent) instead of side by side."""
    import matplotlib.pyplot as plt
    _apply_style()
    basins = fd_tables.read_basin_table(basin_table_path)[["basin_id", "basin_area_km2", "basin_row_min", "basin_row_max", "basin_col_min", "basin_col_max"]]
    rectangles = basin_rectangles(basins)
    windows = np.asarray([grid.window_pixels(int(a), int(b), int(c), int(d)) for a, b, c, d in rectangles], np.int64)
    over = basins.loc[windows > capacity_pixels, "basin_id"].astype(int).tolist()
    basin_raster, extent = _read_view(basin_view_path, view)
    panel_count = 1 + len(group_views)
    if stacked:
        figure, axes = plt.subplots(panel_count, 1, figsize=figure_size)
        plt.subplots_adjust(left=0.062, right=0.988, top=0.975, bottom=0.062, hspace=0.20)
        foot_y = -0.095
    else:
        figure, axes = plt.subplots(1, panel_count, figsize=figure_size)
        plt.subplots_adjust(left=0.058, right=0.988, top=0.945, bottom=0.205, wspace=0.06)
        foot_y = -0.085
    axes = list(np.atleast_1d(axes))
    # (a) drainage area
    area = np.zeros(int(basins["basin_id"].max()) + 1, np.float64)
    area[basins["basin_id"].to_numpy(np.int64)] = basins["basin_area_km2"].to_numpy()
    painted = np.full(basin_raster.shape, np.nan)
    land = basin_raster > 0
    painted[land] = np.log10(np.maximum(area[basin_raster[land]], 0.01))
    colour_map = plt.get_cmap("YlGnBu").copy()
    colour_map.set_bad(OCEAN)
    image = axes[0].imshow(painted, cmap=colour_map, vmin=-1.0, vmax=7.0, extent=extent, interpolation="nearest", origin="upper")
    rows, cols = basin_raster.shape
    longitude = extent[0] + (np.arange(cols) + 0.5) * (extent[1] - extent[0]) / cols
    latitude = extent[3] - (np.arange(rows) + 0.5) * (extent[3] - extent[2]) / rows
    for basin_id in over:
        one = (basin_raster == basin_id).astype(float)
        if one.any():
            axes[0].contour(longitude, latitude, one, levels=[0.5], colors=[CUT_BASIN_OUTLINE], linewidths=1.0)
    bar_axis = axes[0].inset_axes([0.07, 0.075, 0.34, 0.028])
    colour_bar = figure.colorbar(image, cax=bar_axis, orientation="horizontal", ticks=[-1, 1, 3, 5, 7])
    colour_bar.ax.set_xticklabels(["≤0.1", "10", "10³", "10⁵", "10⁷"], fontfamily="DejaVu Sans")
    colour_bar.ax.tick_params(labelsize=SIZE_LABEL, length=1.8, pad=1.2, width=0.5)
    colour_bar.outline.set_linewidth(0.5)
    colour_bar.ax.set_title("drainage area (km²)", fontsize=SIZE_LABEL, pad=2.0, color=TEXT_COLOUR)
    _panel_heading(axes[0], "(a)", "Watershed delineation")
    axes[0].text(0.965, 0.965, "%s complete basins\n%s of at least 1 km²" % (format(len(basins), ","), format(int((basins["basin_area_km2"] >= 1.0).sum()), ",")),
                 transform=axes[0].transAxes, fontsize=SIZE_LABEL, ha="right", va="top", color=TEXT_COLOUR)
    _panel_foot(axes[0], "dark red: the %d basins whose window\nexceeds %s pixels, the capacity" % (len(over), capacity_label), y=foot_y)
    # the groups: one panel per grouping, drawn from the basin view through the basin-group table, so the
    # panels differ only in how the basins were grouped
    over_set = set(over)
    texts_of_axis = {}
    for index, (title, group_view_path, group_table_path, group_of_basin) in enumerate(group_views, start=1):
        # group_of_basin: the group id of every basin by basin_id, element 0 unused (fd_tables.group_of_basin)
        groups = fd_tables.read_group_table(group_table_path, 0)
        lookup = np.asarray(group_of_basin, np.int64)
        group_raster = np.where(basin_raster > 0, lookup[np.minimum(basin_raster, lookup.size - 1)], 0)
        over_groups = groups.loc[groups["window_grid_count"] > capacity_pixels, "group_id"].astype(int).tolist()
        groups_of_an_oversized_basin = set(int(lookup[b]) for b in over_set if b < lookup.size)
        forced = sum(1 for g in over_groups if g in groups_of_an_oversized_basin)
        label_text = {int(g): str(int(g)) for g in groups["group_id"]}
        texts_of_axis[index] = _draw_labelled_raster(axes[index], group_raster, extent, label_text, hatched=over_groups)
        _panel_heading(axes[index], "(%s)" % "abcdef"[index], title)
        _panel_foot(axes[index], "%d basin groups, %d over the capacity (hatched);\n%d of them by a basin over on its own" % (len(groups), len(over_groups), forced), y=foot_y)
    for index, axis in enumerate(axes):
        _map_axis(axis, view, longitude_ticks, latitude_ticks, hide_latitude_labels=(index > 0 and not stacked))
    for index, texts in texts_of_axis.items():
        _separate_labels(figure, axes[index], texts)
    ensure_directory(out_directory)
    path = _next_version_path(out_directory, "Fig03_Delineation_And_Groups")
    figure.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    log(tag, "written %s" % path)
    return path


def figure4_one_basin_through_the_cut(piece_view_path, piece_table_path, mainstem_path, codes_path, basin_row, capacity_pixels, grid, out_directory, capacity_label,
                                      basin_name, figure_size=(7.48, 3.7), tag="figure"):
    """Figure 4: one cut basin: (a) the nine pieces of the first level, those over the capacity hatched with
    their windows; (b) the pieces used; (c) the regions; (d) the region graph.  The dark blue line is
    the main stem of the first walk; the red triangle the outlet."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse
    _apply_style()
    table = read_table(piece_table_path)
    deepest, extent = _read_view(piece_view_path)
    stem = read_table(mainstem_path)
    by_id = {int(row.piece): row for row in table.itertuples(index=False)}
    # the pieces are numbered in the order they were made; their Pfafstetter codes come from the codes table
    codes = read_table(codes_path)
    code_of = dict(zip(codes["piece"].astype(int), codes["code"].astype(int)))
    # the deepest piece of every cell up to its level-1 piece and its used ancestor
    used_ids = set(int(c) for c in table.loc[table["used"] == 1, "piece"])
    cut_ids = set(int(c) for c in table["sub_of"])            # the pieces cut again never appear in the raster
    id_limit = int(table["piece"].max()) + 1
    level1_of = np.zeros(id_limit, np.int32)
    used_of = np.zeros(id_limit, np.int32)
    region_of = np.zeros(id_limit, np.int32)
    for piece_id in table["piece"].astype(int):
        if piece_id in cut_ids:
            continue
        walk = piece_id
        while int(by_id[walk].sub_of):
            walk = int(by_id[walk].sub_of)
        level1_of[piece_id] = code_of[walk]
        walk = piece_id
        while walk not in used_ids:
            walk = int(by_id[walk].sub_of)
            if walk == 0:
                raise FlowDivideError("piece %d has no used ancestor" % piece_id)
        used_of[piece_id] = code_of[walk]
        region_of[piece_id] = int(by_id[walk].region) % 100
    level1_raster = np.where(deepest > 0, level1_of[deepest], 0)
    used_raster = np.where(deepest > 0, used_of[deepest], 0)
    region_raster = np.where(deepest > 0, region_of[deepest], 0)
    by_code = {code_of[piece_id]: row for piece_id, row in by_id.items()}
    used_codes = set(code_of[piece_id] for piece_id in used_ids)
    over_level1 = [code_of[int(row.piece)] for row in table.itertuples(index=False) if int(row.level) == 1 and int(row.window_grid_count) > capacity_pixels]
    figure = plt.figure(figsize=figure_size)
    grid_spec = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.30], left=0.058, right=0.988, top=0.935, bottom=0.03, wspace=0.06, hspace=0.62)
    axes = [figure.add_subplot(grid_spec[0, k]) for k in range(3)]
    graph_axis = figure.add_subplot(grid_spec[1, :])
    view = (extent[0], extent[1], extent[2], extent[3])
    outlet = (float(basin_row["outlet_lon"]), float(basin_row["outlet_lat"]))

    def river_and_outlet(axis):
        axis.plot(stem["lon"], stem["lat"], color=RIVER_COLOUR, linewidth=0.9, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        axis.plot([outlet[0]], [outlet[1]], zorder=7, **OUTLET_MARKER)

    texts_a = _draw_labelled_raster(axes[0], level1_raster, extent, {code: str(code) for code in range(1, 10)}, hatched=over_level1)
    for code in over_level1:
        row = by_code[code]
        window = grid.window_of_rectangle(int(row.row_min), int(row.row_max), int(row.col_min), int(row.col_max))
        box = grid.pixel_box_lon_lat(*window)
        axes[0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, linestyle=(0, (3, 2)), linewidth=0.7, edgecolor=OVER_CAPACITY_COLOUR, zorder=5))
    for code in range(2, 10, 2):
        if code in by_code:
            row = by_code[code]
            axes[0].plot([grid.longitude_of_col(int(row.outlet_col))], [grid.latitude_of_row(int(row.outlet_row))], marker="o", markersize=2.6, markerfacecolor=INK,
                         markeredgecolor="white", markeredgewidth=0.4, linestyle="none", zorder=7)
    river_and_outlet(axes[0])
    _panel_heading(axes[0], "(a)", "Nine Pfafstetter pieces")
    _panel_foot(axes[0], "pieces %s exceed the capacity, %s (hatched)" % (" and ".join(str(c) for c in over_level1), capacity_label) if over_level1 else "every piece within the capacity", y=-0.105)
    texts_b = _draw_labelled_raster(axes[1], used_raster, extent, {code: str(code) for code in used_codes})
    river_and_outlet(axes[1])
    _panel_heading(axes[1], "(b)", "Pieces over the capacity cut again")
    _panel_foot(axes[1], "%d pieces, all within the capacity" % len(used_codes), y=-0.105)
    region_numbers = sorted(set(int(by_code[c].region) % 100 for c in used_codes))
    texts_c = _draw_labelled_raster(axes[2], region_raster, extent, {number: "%02d" % number for number in region_numbers})
    river_and_outlet(axes[2])
    _panel_heading(axes[2], "(c)", "Merged in code order into regions")
    _panel_foot(axes[2], "%d regions, numbered from upstream" % len(region_numbers), y=-0.105)
    longitude_ticks = _tick_values(view[0], view[1])
    latitude_ticks = _tick_values(view[2], view[3])
    for index, axis in enumerate(axes):
        _map_axis(axis, view, longitude_ticks, latitude_ticks, hide_latitude_labels=(index > 0))
    for axis, texts in zip(axes, (texts_a, texts_b, texts_c)):
        _separate_labels(figure, axis, texts)
    # (d) the region graph
    links = set()
    for code in used_codes:
        row = by_code[code]
        if int(row.flows_into):
            source = int(row.region) % 100
            target = int(by_id[int(row.flows_into)].region) % 100
            if source != target:
                links.add((source, target))
    x_of = {number: float(k + 1) for k, number in enumerate(region_numbers)}
    graph_axis.set_xlim(0.3, len(region_numbers) + 0.7)
    graph_axis.set_ylim(-1.05, 1.55)
    figure.canvas.draw()
    box = graph_axis.get_window_extent()
    # a circle on the page: its diameter in pixels is at most 0.62 of the spacing of the nodes and 0.8 of the panel's height
    pixels_per_x_unit = box.width / (len(region_numbers) + 0.4)
    pixels_per_y_unit = box.height / 2.6
    diameter_pixels = min(0.62 * pixels_per_x_unit, 0.8 * box.height)
    node_width = diameter_pixels / pixels_per_x_unit
    node_height = diameter_pixels / pixels_per_y_unit
    graph_axis.axis("off")
    for number in region_numbers:
        graph_axis.add_patch(Ellipse((x_of[number], 0.0), node_width, node_height, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=5))
        graph_axis.text(x_of[number], 0.0, "%02d" % number, fontsize=SIZE_LABEL, ha="center", va="center", color=INK, zorder=6)
    for upstream, downstream in sorted(links):
        span = x_of[downstream] - x_of[upstream]
        shrink = 0.5 * diameter_pixels * 72.0 / figure.dpi + 1.0
        # the arc rises above the row of nodes; its height is kept inside the panel whatever the node spacing
        rad = min(0.55 / max(span, 1.0) ** 0.5 if span > 1.0 else 0.9, 2.2 * pixels_per_y_unit / (span * pixels_per_x_unit))
        graph_axis.add_patch(FancyArrowPatch((x_of[upstream], 0.0), (x_of[downstream], 0.0), arrowstyle="-|>", mutation_scale=6, linewidth=0.7, color=INK,
                                             shrinkA=shrink, shrinkB=shrink, connectionstyle="arc3,rad=%.3f" % (-rad), zorder=4))
    _panel_heading(graph_axis, "(d)", "Region graph: every arrow points downstream, to a larger number")
    ensure_directory(out_directory)
    path = _next_version_path(out_directory, "Fig04_%s_Pfafstetter_Regions" % basin_name.replace(" ", ""))
    figure.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    log(tag, "written %s" % path)
    return path


def figure5_three_capacities(region_view_paths, region_vector_stems, region_table_paths, basin_view_path, capacity_labels, cut_basin_ids_per_capacity, out_directory,
                             view, longitude_ticks, latitude_ticks, figure_size=(7.48, 4.3), tag="figure", vector_formats=("geoparquet",)):
    """Figure 5: the partition at every capacity, side by side: each region in its color_id colour (from
    the region polygons), every cut basin outlined in dark red, a dot for a region too small to see."""
    import matplotlib.pyplot as plt
    _apply_style()
    count = len(region_view_paths)
    figure, axes = plt.subplots(1, count, figsize=figure_size)
    if count == 1:
        axes = [axes]
    plt.subplots_adjust(left=0.058, right=0.988, top=0.93, bottom=0.2, wspace=0.06)
    basin_raster, extent = _read_view(basin_view_path, view)
    rows, cols = basin_raster.shape
    longitude = extent[0] + (np.arange(cols) + 0.5) * (extent[1] - extent[0]) / cols
    latitude = extent[3] - (np.arange(rows) + 0.5) * (extent[3] - extent[2]) / rows
    for index, axis in enumerate(axes):
        region_raster, region_extent = _read_view(region_view_paths[index], view)
        colour_of = None
        vector_file = (region_vector_stems[index] + (".parquet" if vector_formats[0] == "geoparquet" else ".gpkg")) if region_vector_stems[index] else None
        if vector_file and os.path.exists(vector_file):
            polygons = read_view_vector(region_vector_stems[index], vector_formats, layer="regions")
            with_cells = polygons["coarse_grid_count"].to_numpy() > 0
            colour_of = {int(r): int(c) - 1 for r, c in zip(polygons["region_id"][with_cells], polygons["color_id"][with_cells])}
        regions = fd_tables.read_region_table(region_table_paths[index])
        _draw_labelled_raster(axis, region_raster, region_extent, {}, colour_of=colour_of, dots_for_every_label=True)
        for basin_id in cut_basin_ids_per_capacity[index]:
            one = (basin_raster == basin_id).astype(float)
            if one.any():
                axis.contour(longitude, latitude, one, levels=[0.5], colors=[CUT_BASIN_OUTLINE], linewidths=1.0)
        _panel_heading(axis, "(%s)" % "abcdef"[index], "Capacity %s" % capacity_labels[index])
        pieces_count = int(regions["piece_count"].sum())
        _panel_foot(axis, "%d regions; %d basins (dark red outline) cut into %d" % (len(regions), len(cut_basin_ids_per_capacity[index]), pieces_count))
        _map_axis(axis, view, longitude_ticks, latitude_ticks, hide_latitude_labels=(index > 0))
    ensure_directory(out_directory)
    path = _next_version_path(out_directory, "Fig05_Regions_Capacities")
    figure.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    log(tag, "written %s" % path)
    return path
