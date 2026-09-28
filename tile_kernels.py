"""tile_kernels.py -- the four kernels on REGULAR TILES, out of core: what a program does when it knows
nothing about where the basins are.

The grid is cut into the tiles it is distributed in (HydroSHEDS v2: 10° × 10°, corners at whole multiples
of ten degrees, named by the south-west corner; MERIT Hydro: 30° packages), ONE TILE IS HELD AT A TIME,
and what crosses a tile edge is passed through the exit graph.  Nothing here holds the whole grid: the
arrays that live across the passes are one entry per exit pixel and one per inlet pixel, not one per pixel.

    upa   accumulation    the upstream area.  Two passes over the grid: pass 1 gives every exit what drains
                          to it from inside its tile, the exit graph gives every inlet what arrives there,
                          pass 2 accumulates every tile again.  The three kernels of the accumulation are
                          FD1.1's own (this file drives them over the distribution tiles; FD1.1 drives them
                          over its work tiles, which are a multiple of the raster block and at most 32768
                          pixels a side, and a 10° tile at 1 arc-second is 36,000)
    ldn   labelling       the distance to the outlet along the flow path.  Two passes over the grid: pass 1
                          gives every exit the step across its edge and every inlet the along-path distance
                          to the exit it drains to; the exit graph, downstream first, gives every exit its
                          distance to the terminal; pass 2 writes distance-to-drain plus that.
    lup   maximum         the longest path from a divide down to the pixel, and the pixel it starts at.
                          Two passes: pass 1 the local maximum at every exit; the exit graph, upstream
                          first, carries (length, source pixel) with the tie to the smaller pixel index;
                          pass 2 seeds every inlet with what arrives and writes the raster.
    ord   stream order    the Strahler order.  This one cannot be done in two passes: an order is not a
                          sum, so an order arriving late at a tile changes the orders downstream of it and
                          the tile has to be computed again.  The tiles are therefore SWEPT until nothing
                          changes, and the number of sweeps is reported -- the cost a cut that does not
                          respect the basins pays, and the reason the partition of FlowDivide visits every
                          region once.  A tile is swept again only when an order arriving at one of its
                          inlets has changed.

What one tile costs.  Nothing here holds the grid, but one tile with its working arrays is held, and the
arrays are: the flow directions of the tile and its halo (1 byte a pixel), the order (4), the in-degree of
the order (1), the exit each pixel drains to (4), the distance along the path (8, ldn and lup), the length
and its source pixel (8 and 8, lup), the order (1, ord), and the output written for the tile (4 or 8).  So
about 29 bytes a pixel for ldn, 41 for lup, 11 for ord, 21 for upa, the outputs and the casts counted.
A 10° tile of a 1 arc-second grid is
36,000 × 36,000 = 1.3 × 10^9 pixels, so one tile is 38 GB for ldn and 53 for lup: the tiles the data come in
are themselves large, which is the comparison Figure 7 makes and not a fault of this file.  A smaller tile
is given with --tile-degrees; a tile of more than 2^31 - 1 pixels is refused, because the pixels inside a
tile are numbered with 32-bit integers.

Which pixels get a value: every land pixel here (every channel pixel for ord).  fd3_attributes.py writes the
distance to the outlet and the upstream flow length (ldn, lup) for every basin however small,
and lists only the basins of min_basin_area_km2 and more in their tables; its other attributes
(shv, hck, ord, lfp) write the basins of min_basin_area_km2 and more and leave the rest at nodata.  The two are
the same quantity on
the pixels both compute, and the timing of Figure 7 is of the work, not of the
set of pixels.  The tie of two equally long paths goes to the smaller pixel index of the whole grid, which
no cut can change; on a periodic grid that index is the one of the column inside the grid, so the tie is
settled the same way by every tile but the index is not an unrolled longitude.  Two paths of the same
mathematical length are compared as the sums of their steps in metres, and the sums are taken in a different
order here than over the whole grid, so a tie could in principle fall the other way; the
whole-domain check found the same source pixel on all 6.9 million pixels of the test basin, and an exact
whole-centimetre computation found the same on basin 7.  The periodic path and the
tie across a seam are not exercised by test_tile_kernels.py.

Every kernel starts from the recoded flow directions and nothing else (ord also reads the channel mask, as
every tool that computes the Strahler order of a network does).  The distances are metres on the earth
model of fd1_partition, the same as the package's own fd3 attributes, so an answer of this file and an
answer of fd3 are the same quantity and can be compared pixel for pixel.

    python tile_kernels.py <dir.tif> upa <upa.tif> <str.tif> [--channel-threshold-km2 1]
    python tile_kernels.py <dir.tif> ldn <ldn.tif>
    python tile_kernels.py <dir.tif> lup <lup.tif> [--source <lup_source.tif>]
    python tile_kernels.py <dir.tif> ord <ord.tif> --channel <str.tif>
    common: [--tile-degrees 10] [--tile-metres 100000] [--periodic]

test_tile_kernels.py checks every answer against the whole-domain one of three_ways.py on a small grid.
"""
import argparse
import math
import contextlib
import os
import time

import numpy as np
import rasterio
from numba import njit
from rasterio.windows import Window

from fd1_partition import (DROW, DCOL, IS_LAND, MERIT_NODATA, FlowDivideError, Grid, log, publish,
                           raster_profile, read_halo_tile, tile_exits_and_drains, tile_inlets, tile_topological_order,
                           write_json, _accumulate_over_exit_graph_area_only, _link_exits_to_inlets,
                           _pass_one_local_totals_area_only, _pass_two_accumulate_area_only)

PIXEL_NONE = -1
TILE_DEGREES_DEFAULT = 10.0          # HydroSHEDS v2 is distributed in 10 x 10 degree tiles
TILE_METRES_DEFAULT = 100000.0       # a projected grid: 100 km tiles


# =============================================================================
#  [1] The tiles the data are distributed in
# =============================================================================

def tiles_of_distribution(grid, tile_degrees=TILE_DEGREES_DEFAULT, tile_metres=TILE_METRES_DEFAULT):
    """The tiles of the producer's own lattice that the grid touches, as (row0, nrow, col0, ncol) clipped
    to the grid: on a geographic grid the corners lie at whole multiples of tile_degrees (HydroSHEDS v2's
    10° × 10° tiles, named by their south-west corner, e.g. n40w080), on a projected grid at multiples of
    tile_metres.  The tiles do not move with the grid's own corner, which is the point: they are where the
    producer cut, not where the data happen to start.  A tile that the grid's edge cuts short is kept as it
    is (a window of a continent begins in the middle of a tile); on a periodic grid a tile that straddles
    the seam comes back as the two rectangles either side of it, so the seam's pixels are exits and inlets
    of two tiles and every answer is still right."""
    if grid.transform.b != 0.0 or grid.transform.d != 0.0:
        raise FlowDivideError("the lattice of the producer's tiles is read off the grid's own corner; a rotated raster is refused")
    if not (grid.transform.a > 0.0 and grid.transform.e < 0.0):
        raise FlowDivideError("the grid must run east and south from its corner (pixel width %g, height %g)" % (grid.transform.a, grid.transform.e))
    side_units = tile_degrees if grid.geographic else tile_metres
    if not (side_units > 0.0) or not math.isfinite(side_units):
        raise FlowDivideError("the tile must be a positive size, %g was given" % side_units)
    pixels_across = side_units / abs(grid.pixel_width)
    pixels_down = side_units / abs(grid.pixel_height)
    if abs(pixels_across - round(pixels_across)) > 1e-6 or abs(pixels_down - round(pixels_down)) > 1e-6:
        raise FlowDivideError("a tile of %g units is %.6f pixels across and %.6f down on this grid; a tile must be a whole number of pixels both ways"
                              % (side_units, pixels_across, pixels_down))
    if abs(pixels_across - pixels_down) > 1e-6:
        raise FlowDivideError("the pixels are %g by %g units, so a square tile is not a square number of pixels" % (abs(grid.pixel_width), abs(grid.pixel_height)))
    pixels_per_tile = int(round(pixels_across))
    if pixels_per_tile * pixels_per_tile > 2 ** 31 - 1:
        raise FlowDivideError("a tile of %g units is %d x %d pixels; the pixels inside a tile are numbered with 32-bit integers, so a tile holds at most 2^31 - 1"
                              % (side_units, pixels_per_tile, pixels_per_tile))
    # How far the grid's own corner lies inside a tile of the producer: from the lattice line west of its
    # west edge, and the line north of its north edge.  The first row and the first column of tiles are cut
    # short by that, which is what puts every other tile edge on a lattice line.  (A truncating int() in the
    # columns or a stray modulo in the rows puts the tiles on the grid's own corner instead of the producer's
    # lattice.)
    west = grid.transform.c
    north = grid.transform.f
    pixels_east_of_lattice = (west - math.floor(west / side_units + 1e-9) * side_units) / abs(grid.pixel_width)
    pixels_south_of_lattice = (math.ceil(north / side_units - 1e-9) * side_units - north) / abs(grid.pixel_height)
    column_offset = int(round(pixels_east_of_lattice)) % pixels_per_tile
    row_offset = int(round(pixels_south_of_lattice)) % pixels_per_tile
    tiles = []
    row0 = -row_offset
    while row0 < grid.nrow:
        row_start = max(row0, 0)
        nrow = min(row0 + pixels_per_tile, grid.nrow) - row_start
        col0 = -column_offset
        while col0 < grid.ncol:
            col_start = max(col0, 0)
            ncol = min(col0 + pixels_per_tile, grid.ncol) - col_start
            if nrow > 0 and ncol > 0:
                tiles.append((row_start, nrow, col_start, ncol))
            col0 += pixels_per_tile
        row0 += pixels_per_tile
    return tiles, pixels_per_tile


# =============================================================================
#  [2] What every kernel's pass 1 shares: the exits, the inlets, the distances along the path
# =============================================================================

def check_no_flow_leaves_the_grid(off_grid_total, into_nodata_total):
    """FD1.0 turns a direction that points into nodata or off the grid into a river mouth, so a grid that is
    ready for these kernels has none left.  A grid that still has them is in another convention, and every
    area and every distance below such a pixel would be wrong without a word (fd1.1 refuses the same way)."""
    if off_grid_total != 0 or into_nodata_total != 0:
        raise FlowDivideError("%d pixels flow off the grid and %d into nodata; the grid is not in the MERIT convention (run FD1.0)"
                              % (off_grid_total, into_nodata_total))


@njit(cache=True)
def _step_length_of(lengths, row, drow, dcol):
    """the length of one step from a pixel in `row`: the five lengths of a row are east-west, north,
    south, north-diagonal, south-diagonal, as Grid.row_step_lengths_m writes them"""
    if drow == 0:
        return lengths[row, 0]
    if dcol == 0:
        if drow < 0:
            return lengths[row, 1]
        return lengths[row, 2]
    if drow < 0:
        return lengths[row, 3]
    return lengths[row, 4]


@njit(cache=True)
def _distance_to_drain_of_tile(halo_dir, order, lengths, distance_to_drain):
    """The along-path distance from every land pixel of the tile to the exit (or the terminal) it drains
    to, read off the tile's own order backwards: a pixel's distance is its downstream pixel's plus the
    step between them, and an exit pixel's is zero.  The step across the tile edge is NOT counted here;
    it is counted once, in the exit graph."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        row_offset = DROW[code]
        column_offset = DCOL[code]
        if row_offset == 0 and column_offset == 0:
            distance_to_drain[pixel] = 0.0
            continue
        downstream_row = row + row_offset
        downstream_column = col + column_offset
        inside = downstream_row >= 0 and downstream_row < nrow and downstream_column >= 0 and downstream_column < ncol
        if not inside or IS_LAND[halo_dir[downstream_row + 1, downstream_column + 1]] == 0:
            distance_to_drain[pixel] = 0.0                     # this pixel is the exit, or its path ends here
            continue
        downstream_pixel = downstream_row * ncol + downstream_column
        step = _step_length_of(lengths, row, row_offset, column_offset)
        distance_to_drain[pixel] = distance_to_drain[downstream_pixel] + step


@njit(cache=True)
def _step_across_of_exits(halo_dir, exit_local, lengths):
    """the step from every exit pixel to the inlet pixel it flows into, one number per exit"""
    halo_nrow, halo_ncol = halo_dir.shape
    ncol = halo_ncol - 2
    step_across = np.empty(exit_local.size, np.float64)
    for index in range(exit_local.size):
        pixel = exit_local[index]
        if pixel < 0:
            return step_across[:0]                    # an exit without a pixel: the caller refuses
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        step_across[index] = _step_length_of(lengths, row, DROW[code], DCOL[code])
    return step_across


@njit(cache=True)
def _exit_pixel_of_each_exit(halo_dir, order, drain, exit_count):
    """the exit pixel of every exit: the land pixel that drains to that exit and whose downstream pixel
    lies outside the tile.  (drain[] is the exit of the whole path, so the exit pixel is the last one.)"""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    exit_pixel = np.full(exit_count, PIXEL_NONE, np.int64)
    for position in range(order.size):
        pixel = order[position]
        index = drain[pixel]
        if index < 0:
            continue
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        row_offset = DROW[code]
        column_offset = DCOL[code]
        if row_offset == 0 and column_offset == 0:
            continue
        downstream_row = row + row_offset
        downstream_column = col + column_offset
        if downstream_row < 0 or downstream_row >= nrow or downstream_column < 0 or downstream_column >= ncol:
            exit_pixel[index] = pixel
    return exit_pixel


class TileEdgeRecords:
    """what the passes over the tiles leave behind, one entry per exit pixel and one per inlet pixel of
    every tile -- never one per pixel of the grid, which is what keeps this out of core"""

    def __init__(self):
        self.exit_global = []
        self.exit_destination = []
        self.exit_step_across = []
        self.exit_value = []
        self.exit_source = []
        self.inlet_local = []
        self.inlet_global = []
        self.inlet_exit_global = []
        self.inlet_distance_to_drain = []
        self.inlet_drain_exit = []
        self.tile_exit_offsets = [0]
        self.tile_inlet_offsets = [0]

    def concatenate(self):
        def joined(parts, dtype):
            if len(parts) == 0:
                return np.zeros(0, dtype)
            return np.concatenate(parts).astype(dtype, copy=False)
        self.exit_global = joined(self.exit_global, np.int64)
        self.exit_destination = joined(self.exit_destination, np.int64)
        self.exit_step_across = joined(self.exit_step_across, np.float64)
        self.exit_value = joined(self.exit_value, np.float64)
        self.exit_source = joined(self.exit_source, np.int64)
        self.inlet_local = joined(self.inlet_local, np.int32)
        self.inlet_global = joined(self.inlet_global, np.int64)
        self.inlet_exit_global = joined(self.inlet_exit_global, np.int64)
        self.inlet_distance_to_drain = joined(self.inlet_distance_to_drain, np.float64)
        self.inlet_drain_exit = joined(self.inlet_drain_exit, np.int64)
        self.tile_exit_offsets = np.asarray(self.tile_exit_offsets, np.int64)
        self.tile_inlet_offsets = np.asarray(self.tile_inlet_offsets, np.int64)


def _pass_one_over_tiles(dir_path, grid, tiles, tag, want_local_maximum=False):
    """Pass 1 of ldn and lup: every tile read once, and for every exit and every inlet of it the few
    numbers the exit graph needs.  With want_local_maximum the local longest path at every exit pixel is
    computed as well (that is lup's pass 1).  Returns the records and the number of land pixels."""
    records = TileEdgeRecords()
    land_total = 0
    off_grid_total = 0
    into_nodata_total = 0
    with rasterio.open(dir_path) as dir_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            if taken != land_count:
                raise FlowDivideError("the flow directions hold a cycle inside the tile at row %d col %d" % (row0, col0))
            land_total += land_count
            lengths = grid.row_step_lengths_m(row0, nrow)
            drain = np.full(nrow * ncol, -1, np.int32)
            largest_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(largest_edge, np.int64)
            exit_destination = np.empty(largest_edge, np.int64)
            exit_count, off_grid, into_nodata = tile_exits_and_drains(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, drain, exit_global, exit_destination)
            off_grid_total += off_grid
            into_nodata_total += into_nodata
            distance_to_drain = np.zeros(nrow * ncol, np.float64)
            _distance_to_drain_of_tile(halo, order, lengths, distance_to_drain)
            exit_pixel = _exit_pixel_of_each_exit(halo, order, drain, exit_count)
            step_across = _step_across_of_exits(halo, exit_pixel, lengths)
            if step_across.size != exit_count:
                raise FlowDivideError("an exit of the tile at row %d col %d has no pixel; the exits and the drains do not agree" % (row0, col0))
            inlet_local = np.empty(largest_edge, np.int32)
            inlet_global = np.empty(largest_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            inlet_drain_exit = np.empty(inlet_count, np.int64)
            inlet_distance_to_drain = np.empty(inlet_count, np.float64)
            for index in range(inlet_count):
                pixel = inlet_local[index]
                exit_index = drain[pixel]
                inlet_exit_global[index] = exit_global[exit_index] if exit_index >= 0 else -1
                inlet_drain_exit[index] = exit_index
                inlet_distance_to_drain[index] = distance_to_drain[pixel]
            if want_local_maximum:
                local_value = np.zeros(nrow * ncol, np.float64)
                local_source = np.full(nrow * ncol, PIXEL_NONE, np.int64)
                _local_maximum_of_tile(halo, order, lengths, row0, col0, grid.ncol, grid.periodic,
                                       np.zeros(0, np.int32), np.zeros(0, np.float64), np.zeros(0, np.int64), local_value, local_source)
                exit_value = np.array([local_value[pixel] if pixel >= 0 else 0.0 for pixel in exit_pixel], np.float64)
                exit_source = np.array([local_source[pixel] if pixel >= 0 else -1 for pixel in exit_pixel], np.int64)
                del local_value, local_source
            else:
                exit_value = np.zeros(exit_count, np.float64)
                exit_source = np.full(exit_count, -1, np.int64)
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.exit_step_across.append(step_across)
            records.exit_value.append(exit_value)
            records.exit_source.append(exit_source)
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.inlet_distance_to_drain.append(inlet_distance_to_drain)
            records.inlet_drain_exit.append(inlet_drain_exit)
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            del halo, order, drain, distance_to_drain
            log(tag, "pass 1 tile %d of %d (row %d col %d): %d land pixels, %d exits, %d inlets" % (tile_index + 1, len(tiles), row0, col0, land_count, exit_count, inlet_count))
    records.concatenate()
    check_no_flow_leaves_the_grid(off_grid_total, into_nodata_total)
    return records, land_total


def _exit_graph_of(records, tag):
    """the exit graph: for every exit the inlet it flows into and the exit that inlet drains to"""
    status, inlet_of_exit, next_exit = _link_exits_to_inlets(records.exit_global, records.exit_destination, records.inlet_global, records.inlet_exit_global)
    if status != 0:
        raise FlowDivideError("the exit graph cannot be linked (status %d): a destination is no inlet, or an inlet names no exit" % status)
    log(tag, "exit graph: %d exits, %d inlets" % (records.exit_global.size, records.inlet_global.size))
    return inlet_of_exit, next_exit


@njit(cache=True)
def _exit_topological_order(next_exit):
    """the exits upstream first (every exit after the exits that flow into it); the second value is False
    when the graph holds a cycle, which flow directions cannot give"""
    exit_count = next_exit.size
    upstream_count = np.zeros(exit_count, np.int32)
    for index in range(exit_count):
        if next_exit[index] >= 0:
            upstream_count[next_exit[index]] += 1
    order = np.empty(exit_count, np.int64)
    tail = 0
    for index in range(exit_count):
        if upstream_count[index] == 0:
            order[tail] = index
            tail += 1
    head = 0
    while head < tail:
        index = order[head]
        head += 1
        target = next_exit[index]
        if target >= 0:
            upstream_count[target] -= 1
            if upstream_count[target] == 0:
                order[tail] = target
                tail += 1
    return order, head == exit_count


# =============================================================================
#  [3] upa: the upstream area, two passes
# =============================================================================

def upa_on_tiles(dir_path, out_path, channel_path, grid, tiles, channel_threshold_km2=1.0, tag="tiles.upa"):
    """the upstream area, one tile at a time on the producer's tiles: two passes over the grid, with FD1.1's
    own three kernels (the local totals, the exit graph, the second accumulation).

    channel_path=None writes the upstream area alone.  Figure 7 measures one variable per process, and the
    channel mask is a prepared input there, given to every tool alike, so the bar of the upstream area must
    not carry the writing of a second raster."""
    started = time.time()
    records = TileEdgeRecords()
    land_total = 0
    off_grid_total = 0
    into_nodata_total = 0
    with rasterio.open(dir_path) as dir_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            if taken != land_count:
                raise FlowDivideError("the flow directions hold a cycle inside the tile at row %d col %d" % (row0, col0))
            land_total += land_count
            drain = np.full(nrow * ncol, -1, np.int32)
            largest_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(largest_edge, np.int64)
            exit_destination = np.empty(largest_edge, np.int64)
            exit_count, off_grid, into_nodata = tile_exits_and_drains(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, drain, exit_global, exit_destination)
            off_grid_total += off_grid
            into_nodata_total += into_nodata
            local_area = _pass_one_local_totals_area_only(drain, exit_count, ncol, grid.row_pixel_areas_m2(row0, nrow))
            inlet_local = np.empty(largest_edge, np.int32)
            inlet_global = np.empty(largest_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            for index in range(inlet_count):
                exit_index = drain[inlet_local[index]]
                inlet_exit_global[index] = exit_global[exit_index] if exit_index >= 0 else -1
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.exit_step_across.append(np.zeros(exit_count, np.float64))
            records.exit_value.append(local_area)
            records.exit_source.append(np.full(exit_count, -1, np.int64))
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.inlet_distance_to_drain.append(np.zeros(inlet_count, np.float64))
            records.inlet_drain_exit.append(np.full(inlet_count, -1, np.int64))
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            del halo, order, drain
            log(tag, "pass 1 tile %d of %d (row %d col %d): %d land pixels, %d exits, %d inlets" % (tile_index + 1, len(tiles), row0, col0, land_count, exit_count, inlet_count))
    records.concatenate()
    check_no_flow_leaves_the_grid(off_grid_total, into_nodata_total)
    inlet_of_exit, next_exit = _exit_graph_of(records, tag)
    status, arriving_area = _accumulate_over_exit_graph_area_only(next_exit, inlet_of_exit, records.inlet_global.size, records.exit_value)
    if status != 0:
        raise FlowDivideError("the exit graph of the tiles has a cycle")
    ended_area_total = 0.0
    # the threshold in float32 and in the raster's unit, compared with the area as float32, as FD3 does
    # compares them; a double threshold against the float32 area comes out otherwise at 1.00000001 km2
    threshold_m2 = np.float32(channel_threshold_km2 * 1e6)
    temporary = out_path + ".partial.tif"
    channel_temporary = channel_path + ".partial.tif" if channel_path else None
    with contextlib.ExitStack() as open_files:
        dir_dataset = open_files.enter_context(rasterio.open(dir_path))
        out_dataset = open_files.enter_context(rasterio.open(temporary, "w", **raster_profile(grid, "int64", 0)))
        channel_dataset = (open_files.enter_context(
            rasterio.open(channel_temporary, "w", **raster_profile(grid, "uint8", 0, predictor=False)))
            if channel_temporary else None)
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            first = records.tile_inlet_offsets[tile_index]
            last = records.tile_inlet_offsets[tile_index + 1]
            area_out = np.zeros((nrow, ncol), np.float64)
            ended_area = _pass_two_accumulate_area_only(halo, order, records.inlet_local[first:last], arriving_area[first:last],
                                                        grid.row_pixel_areas_m2(row0, nrow), area_out)
            ended_area_total += ended_area
            area_m2 = np.floor(area_out + 0.5).astype(np.int64)             # to the nearest square metre, halves up
            out_dataset.write(area_m2, 1, window=Window(col0, row0, ncol, nrow))
            if channel_dataset is not None:
                channel = (area_m2.astype(np.float32) >= threshold_m2).astype(np.uint8)
                channel_dataset.write(channel, 1, window=Window(col0, row0, ncol, nrow))
                del channel
            del halo, order, area_out, area_m2
            log(tag, "pass 2 tile %d of %d written" % (tile_index + 1, len(tiles)))
    publish(temporary, out_path)
    if channel_temporary:
        publish(channel_temporary, channel_path)
    report = {"kernel": "upa", "tiles": len(tiles), "land_pixels": int(land_total), "land_area_km2": ended_area_total / 1e6,
              "exits": int(records.exit_global.size), "inlets": int(records.inlet_global.size), "passes_over_the_grid": 2,
              "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s: %d land pixels, %.1f km2, %.1f s%s"
        % (out_path, land_total, report["land_area_km2"], report["seconds"],
           " (and the channel mask %s)" % channel_path if channel_path else " (the area alone)"))
    return report


# =============================================================================
#  [4] ldn: the distance to the outlet, two passes
# =============================================================================

@njit(cache=True)
def _distance_of_exits_to_terminal(exit_order, next_exit, inlet_of_exit, exit_step_across, inlet_distance_to_drain):
    """The exit graph, DOWNSTREAM FIRST: an exit's distance to the terminal is the step across its own
    edge, plus the along-path distance from the inlet pixel it flows into to that tile's own exit, plus
    that exit's distance.  The along-path distance inside a tile is counted exactly once, here."""
    exit_count = next_exit.size
    distance_to_terminal = np.zeros(exit_count, np.float64)
    for position in range(exit_count - 1, -1, -1):
        index = exit_order[position]
        inlet = inlet_of_exit[index]
        below = 0.0
        if inlet >= 0:
            below = inlet_distance_to_drain[inlet]
            target = next_exit[index]
            if target >= 0:
                below += distance_to_terminal[target]
        distance_to_terminal[index] = exit_step_across[index] + below
    return distance_to_terminal


@njit(cache=True)
def _ldn_write_tile(halo_dir, order, drain, distance_to_drain, exit_distance_to_terminal, out):
    """every land pixel: its distance to the exit it drains to, plus that exit's distance to the terminal"""
    halo_nrow, halo_ncol = halo_dir.shape
    ncol = halo_ncol - 2
    for position in range(order.size):
        pixel = order[position]
        row = pixel // ncol
        col = pixel - row * ncol
        index = drain[pixel]
        value = distance_to_drain[pixel]
        if index >= 0:
            value += exit_distance_to_terminal[index]
        out[row, col] = value


def ldn_on_tiles(dir_path, out_path, grid, tiles, tag="tiles.ldn", nodata=-9999.0):
    """the distance to the outlet, one tile at a time on the producer's tiles: two passes over the grid"""
    started = time.time()
    records, land_total = _pass_one_over_tiles(dir_path, grid, tiles, tag)
    inlet_of_exit, next_exit = _exit_graph_of(records, tag)
    exit_order, acyclic = _exit_topological_order(next_exit)
    if not acyclic:
        raise FlowDivideError("the exit graph of the tiles holds a cycle: the flow directions are not a forest")
    exit_distance = _distance_of_exits_to_terminal(exit_order, next_exit, inlet_of_exit, records.exit_step_across, records.inlet_distance_to_drain)
    temporary = out_path + ".partial.tif"
    seconds_reading = seconds_computing = seconds_writing = 0.0
    # both datasets closed on an error too
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(temporary, "w", **raster_profile(grid, "float32", nodata)) as out_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            clock = time.time()
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            seconds_reading += time.time() - clock
            clock = time.time()
            order, land_count, taken = tile_topological_order(halo)
            lengths = grid.row_step_lengths_m(row0, nrow)
            drain = np.full(nrow * ncol, -1, np.int32)
            largest_edge = 2 * (nrow + ncol) + 8
            exit_count, _, _ = tile_exits_and_drains(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, drain,
                                                     np.empty(largest_edge, np.int64), np.empty(largest_edge, np.int64))
            distance_to_drain = np.zeros(nrow * ncol, np.float64)
            _distance_to_drain_of_tile(halo, order, lengths, distance_to_drain)
            first = records.tile_exit_offsets[tile_index]
            last = records.tile_exit_offsets[tile_index + 1]
            out = np.full((nrow, ncol), nodata, np.float64)
            _ldn_write_tile(halo, order, drain, distance_to_drain, exit_distance[first:last], out)
            seconds_computing += time.time() - clock
            clock = time.time()
            out_dataset.write(out.astype(np.float32), 1, window=Window(col0, row0, ncol, nrow))
            seconds_writing += time.time() - clock
            del halo, order, drain, distance_to_drain, out
            log(tag, "pass 2 tile %d of %d written" % (tile_index + 1, len(tiles)))
        clock = time.time()
    log(tag, "pass 2: read %.1f s, computed %.1f s, wrote %.1f s, closed %.1f s"
        % (seconds_reading, seconds_computing, seconds_writing, time.time() - clock))
    publish(temporary, out_path)
    report = {"kernel": "ldn", "tiles": len(tiles), "land_pixels": int(land_total), "exits": int(records.exit_global.size),
              "inlets": int(records.inlet_global.size), "passes_over_the_grid": 2, "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s: %d land pixels, %d exits, %.1f s" % (out_path, land_total, records.exit_global.size, report["seconds"]))
    return report


# =============================================================================
#  [5] lup: the longest path upstream and where it starts, two passes
# =============================================================================

@njit(cache=True)
def _local_maximum_of_tile(halo_dir, order, lengths, tile_row0, tile_col0, grid_ncol, periodic,
                           inlet_local, arriving_value, arriving_source, value, source):
    """The longest path down to every pixel of the tile, counting only what lies inside the tile and what
    arrives at its inlets.  A pixel with no inflow starts at zero and is its own source; a tie between two
    equally long paths goes to the smaller GLOBAL pixel index, so the answer cannot depend on the cut."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    for position in range(order.size):
        pixel = order[position]
        row = pixel // ncol
        col = pixel - row * ncol
        if periodic:
            source[pixel] = (tile_row0 + row) * grid_ncol + (tile_col0 + col) % grid_ncol
        else:
            source[pixel] = (tile_row0 + row) * grid_ncol + tile_col0 + col
        value[pixel] = 0.0
    for index in range(inlet_local.size):
        pixel = inlet_local[index]
        if arriving_source[index] >= 0:
            value[pixel] = arriving_value[index]
            source[pixel] = arriving_source[index]
    for position in range(order.size):
        pixel = order[position]
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        row_offset = DROW[code]
        column_offset = DCOL[code]
        if row_offset == 0 and column_offset == 0:
            continue
        downstream_row = row + row_offset
        downstream_column = col + column_offset
        if downstream_row < 0 or downstream_row >= nrow or downstream_column < 0 or downstream_column >= ncol:
            continue
        if IS_LAND[halo_dir[downstream_row + 1, downstream_column + 1]] == 0:
            continue
        downstream_pixel = downstream_row * ncol + downstream_column
        step = _step_length_of(lengths, row, row_offset, column_offset)
        candidate = value[pixel] + step
        if candidate > value[downstream_pixel]:
            value[downstream_pixel] = candidate
            source[downstream_pixel] = source[pixel]
        elif candidate == value[downstream_pixel] and source[pixel] < source[downstream_pixel]:
            source[downstream_pixel] = source[pixel]


@njit(cache=True)
def _maximum_over_exit_graph(exit_order, next_exit, inlet_of_exit, exit_step_across, inlet_distance_to_drain,
                             exit_value, exit_source):
    """The exit graph, UPSTREAM FIRST: what leaves an exit reaches the inlet below it one step later and
    then travels that tile's own along-path distance to its exit.  Returns what arrives at every inlet."""
    exit_count = next_exit.size
    value = exit_value.copy()
    source = exit_source.copy()
    for position in range(exit_count):
        index = exit_order[position]
        target = next_exit[index]
        if target < 0:
            continue
        inlet = inlet_of_exit[index]
        candidate = value[index] + exit_step_across[index]
        if inlet >= 0:
            candidate += inlet_distance_to_drain[inlet]
        if candidate > value[target]:
            value[target] = candidate
            source[target] = source[index]
        elif candidate == value[target] and source[index] < source[target]:
            source[target] = source[index]
    arriving_value = np.zeros(inlet_distance_to_drain.size, np.float64)
    arriving_source = np.full(inlet_distance_to_drain.size, PIXEL_NONE, np.int64)
    for index in range(exit_count):
        inlet = inlet_of_exit[index]
        if inlet < 0:
            continue
        candidate = value[index] + exit_step_across[index]
        if arriving_source[inlet] == PIXEL_NONE or candidate > arriving_value[inlet]:
            arriving_value[inlet] = candidate
            arriving_source[inlet] = source[index]
        elif candidate == arriving_value[inlet] and source[index] < arriving_source[inlet]:
            arriving_source[inlet] = source[index]
    return arriving_value, arriving_source


def lup_on_tiles(dir_path, out_path, grid, tiles, source_path=None, tag="tiles.lup", nodata=-9999.0):
    """the longest path from a divide down to every pixel, one tile at a time: two passes over the grid"""
    started = time.time()
    records, land_total = _pass_one_over_tiles(dir_path, grid, tiles, tag, want_local_maximum=True)
    inlet_of_exit, next_exit = _exit_graph_of(records, tag)
    exit_order, acyclic = _exit_topological_order(next_exit)
    if not acyclic:
        raise FlowDivideError("the exit graph of the tiles holds a cycle: the flow directions are not a forest")
    arriving_value, arriving_source = _maximum_over_exit_graph(exit_order, next_exit, inlet_of_exit, records.exit_step_across,
                                                               records.inlet_distance_to_drain, records.exit_value, records.exit_source)
    temporary = out_path + ".partial.tif"
    source_temporary = (source_path + ".partial.tif") if source_path else None
    # the source raster in the same with, so that it is closed on an error too
    with contextlib.ExitStack() as stack:
        source_dataset = stack.enter_context(rasterio.open(source_temporary, "w", **raster_profile(grid, "int64", -1))) if source_path else None
        dir_dataset = stack.enter_context(rasterio.open(dir_path))
        out_dataset = stack.enter_context(rasterio.open(temporary, "w", **raster_profile(grid, "float32", nodata)))
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            lengths = grid.row_step_lengths_m(row0, nrow)
            first = records.tile_inlet_offsets[tile_index]
            last = records.tile_inlet_offsets[tile_index + 1]
            value = np.zeros(nrow * ncol, np.float64)
            source = np.full(nrow * ncol, PIXEL_NONE, np.int64)
            _local_maximum_of_tile(halo, order, lengths, row0, col0, grid.ncol, grid.periodic,
                                   records.inlet_local[first:last], arriving_value[first:last], arriving_source[first:last], value, source)
            out = np.full((nrow, ncol), nodata, np.float64)
            source_out = np.full((nrow, ncol), -1, np.int64)
            for position in range(order.size):
                pixel = int(order[position])
                row = pixel // ncol
                col = pixel - row * ncol
                out[row, col] = value[pixel]
                source_out[row, col] = source[pixel]
            out_dataset.write(out.astype(np.float32), 1, window=Window(col0, row0, ncol, nrow))
            if source_dataset is not None:
                source_dataset.write(source_out, 1, window=Window(col0, row0, ncol, nrow))
            del halo, order, value, source, out, source_out
            log(tag, "pass 2 tile %d of %d written" % (tile_index + 1, len(tiles)))
    if source_dataset is not None:
        publish(source_temporary, source_path)
    publish(temporary, out_path)
    report = {"kernel": "lup", "tiles": len(tiles), "land_pixels": int(land_total), "exits": int(records.exit_global.size),
              "inlets": int(records.inlet_global.size), "passes_over_the_grid": 2, "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s: %d land pixels, %.1f s" % (out_path, land_total, report["seconds"]))
    return report


# =============================================================================
#  [6] ord: the Strahler order, swept over the tiles until nothing changes
# =============================================================================
#
#  The order of a pixel is the largest order among the pixels flowing into it, plus one when two or more
#  of them carry that largest order.  That summary merges in any arrival order, so an order crosses a tile
#  edge as one byte; what it cannot do is be folded into a tile that has already been computed, because
#  the summary is not invertible.  So a tile whose arriving orders change is computed again, and the
#  sweeps are counted.  The exit graph is a forest, but the TILE graph need not be: a river that leaves a
#  tile and comes back puts a cycle in it, and then no order of the tiles exists at all.

@njit(cache=True)
def _strahler_order_of_tile(halo_dir, order, channel, inlet_local, arriving_largest, arriving_second,
                            exit_pixel_of_exit, exit_order_out, value):
    """The Strahler order of every channel pixel of the tile, given the largest and second largest order
    arriving at each of its inlets.  Writes the order of every pixel into `value` (0 off the channel) and
    the order at every exit pixel into `exit_order_out`.  Returns 1 when an order does not fit a byte, 2 when a
    channel pixel flows into a land pixel off the channel inside the tile, 3 when an order arrives from another tile at
    an inlet off the channel (the channel is broken there; the whole-domain FD3 stops on it too, and carrying on
    would count the pixel below the break as a source)."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    largest = np.zeros(nrow * ncol, np.uint8)
    second = np.zeros(nrow * ncol, np.uint8)
    for index in range(inlet_local.size):
        pixel = inlet_local[index]
        if arriving_largest[index] > 0 and channel[pixel // ncol, pixel - (pixel // ncol) * ncol] == 0:
            return 3
        largest[pixel] = arriving_largest[index]
        second[pixel] = arriving_second[index]
    for position in range(order.size):
        pixel = order[position]
        row = pixel // ncol
        col = pixel - row * ncol
        if channel[row, col] == 0:
            value[pixel] = 0
            continue
        if largest[pixel] == 0:
            own = 1
        elif second[pixel] + 1 > largest[pixel]:
            own = second[pixel] + 1
        else:
            own = largest[pixel]
        if own > 255:
            return 1
        value[pixel] = own
        code = halo_dir[row + 1, col + 1]
        row_offset = DROW[code]
        column_offset = DCOL[code]
        if row_offset == 0 and column_offset == 0:
            continue
        downstream_row = row + row_offset
        downstream_column = col + column_offset
        if downstream_row < 0 or downstream_row >= nrow or downstream_column < 0 or downstream_column >= ncol:
            continue
        if IS_LAND[halo_dir[downstream_row + 1, downstream_column + 1]] == 0:
            continue
        if channel[downstream_row, downstream_column] == 0:
            return 2
        downstream_pixel = downstream_row * ncol + downstream_column
        if own >= largest[downstream_pixel]:
            second[downstream_pixel] = largest[downstream_pixel]
            largest[downstream_pixel] = own
        elif own > second[downstream_pixel]:
            second[downstream_pixel] = own
    for index in range(exit_pixel_of_exit.size):
        pixel = exit_pixel_of_exit[index]
        exit_order_out[index] = value[pixel] if pixel >= 0 else 0
    return 0


def ord_on_tiles(dir_path, channel_path, out_path, grid, tiles, tag="tiles.ord"):
    """The Strahler order of the channel network, one tile at a time on the producer's tiles: the tiles are
    swept until no order arriving at an inlet changes any more, and the number of sweeps is reported."""
    started = time.time()
    tile_count = len(tiles)
    with rasterio.open(channel_path) as channel_dataset:
        # the channel mask on the grid of the flow directions, and in the grid's CRS
        if channel_dataset.height != grid.nrow or channel_dataset.width != grid.ncol or channel_dataset.transform != grid.transform or \
                channel_dataset.crs != grid.crs:
            raise FlowDivideError("the channel mask %s is not the grid of %s" % (channel_path, dir_path))
    # the edges of every tile, from one pass that reads the flow directions and the channel mask
    records = TileEdgeRecords()
    exit_pixel_per_tile = []
    land_total = 0
    off_grid_total = 0
    into_nodata_total = 0
    with rasterio.open(dir_path) as dir_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            if taken != land_count:
                raise FlowDivideError("the flow directions hold a cycle inside the tile at row %d col %d" % (row0, col0))
            land_total += land_count
            drain = np.full(nrow * ncol, -1, np.int32)
            largest_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(largest_edge, np.int64)
            exit_destination = np.empty(largest_edge, np.int64)
            exit_count, off_grid, into_nodata = tile_exits_and_drains(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, drain, exit_global, exit_destination)
            off_grid_total += off_grid
            into_nodata_total += into_nodata
            exit_pixel = _exit_pixel_of_each_exit(halo, order, drain, exit_count)
            if exit_count > 0 and int(exit_pixel.min()) < 0:
                raise FlowDivideError("an exit of the tile at row %d col %d has no pixel; the exits and the drains do not agree" % (row0, col0))
            inlet_local = np.empty(largest_edge, np.int32)
            inlet_global = np.empty(largest_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            for index in range(inlet_count):
                exit_index = drain[inlet_local[index]]
                inlet_exit_global[index] = exit_global[exit_index] if exit_index >= 0 else -1
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.exit_step_across.append(np.zeros(exit_count, np.float64))
            records.exit_value.append(np.zeros(exit_count, np.float64))
            records.exit_source.append(np.full(exit_count, -1, np.int64))
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.inlet_distance_to_drain.append(np.zeros(inlet_count, np.float64))
            records.inlet_drain_exit.append(np.full(inlet_count, -1, np.int64))
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            exit_pixel_per_tile.append(exit_pixel)
            del halo, order, drain
            log(tag, "edges of tile %d of %d: %d exits, %d inlets" % (tile_index + 1, tile_count, exit_count, inlet_count))
    records.concatenate()
    check_no_flow_leaves_the_grid(off_grid_total, into_nodata_total)
    inlet_of_exit, next_exit = _exit_graph_of(records, tag)
    # a cycle through several tiles, each tile without one of its own, passes the check of every tile and
    # the sweeps settle on it (two pixels flowing into each other, one a tile, come out Strahler 1); the exits
    # must form a forest, as the other kernels here check
    _, exits_form_a_forest = _exit_topological_order(next_exit)
    if not exits_form_a_forest:
        raise FlowDivideError("the exits of the tiles flow in a cycle; the flow directions are not a forest")
    # which exit flows into which inlet, and the tile of every inlet, so that a changed order marks the
    # tile below it for another sweep
    tile_of_inlet = np.zeros(records.inlet_global.size, np.int64)
    for tile_index in range(tile_count):
        first = records.tile_inlet_offsets[tile_index]
        last = records.tile_inlet_offsets[tile_index + 1]
        tile_of_inlet[first:last] = tile_index
    exit_order_value = np.zeros(records.exit_global.size, np.uint8)
    arriving_largest = np.zeros(records.inlet_global.size, np.uint8)
    arriving_second = np.zeros(records.inlet_global.size, np.uint8)
    to_sweep = np.ones(tile_count, np.bool_)
    sweeps = 0
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(channel_path) as channel_dataset:
        while to_sweep.any():
            sweeps += 1
            # A sweep either changes an order at an exit or is the last one.  An order only ever rises and no
            # order passes 255, so the changes are at most 255 per exit and that bounds the sweeps -- not the
            # number of tiles: a river that leaves a tile and comes back makes the tile graph cyclic and then
            # one sweep can settle one hop (4 x tiles + 16 is not a bound, nor is one per exit; 255 per exit
            # is).
            if sweeps > 255 * records.exit_global.size + tile_count + 2:
                raise FlowDivideError("the orders do not settle after %d sweeps over %d tiles and %d exits; the flow directions are not a forest"
                                      % (sweeps, tile_count, records.exit_global.size))
            sweeping = np.flatnonzero(to_sweep)
            to_sweep_next = np.zeros(tile_count, np.bool_)
            for tile_index in sweeping:
                row0, nrow, col0, ncol = tiles[tile_index]
                halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
                order, land_count, taken = tile_topological_order(halo)
                channel = channel_dataset.read(1, window=Window(col0, row0, ncol, nrow))
                first_inlet = records.tile_inlet_offsets[tile_index]
                last_inlet = records.tile_inlet_offsets[tile_index + 1]
                first_exit = records.tile_exit_offsets[tile_index]
                last_exit = records.tile_exit_offsets[tile_index + 1]
                value = np.zeros(nrow * ncol, np.uint8)
                new_exit_order = np.zeros(last_exit - first_exit, np.uint8)
                status = _strahler_order_of_tile(halo, order, channel, records.inlet_local[first_inlet:last_inlet],
                                                 arriving_largest[first_inlet:last_inlet], arriving_second[first_inlet:last_inlet],
                                                 exit_pixel_per_tile[tile_index], new_exit_order, value)
                if status == 1:
                    raise FlowDivideError("a Strahler order does not fit a byte in the tile at row %d col %d" % (row0, col0))
                if status != 0:
                    raise FlowDivideError("the channel mask is broken in the tile at row %d col %d: a channel pixel flows into a "
                                          "land pixel off the channel (status %d)" % (row0, col0, status))
                for index in range(first_exit, last_exit):
                    if new_exit_order[index - first_exit] == exit_order_value[index]:
                        continue
                    exit_order_value[index] = new_exit_order[index - first_exit]
                    inlet = inlet_of_exit[index]
                    if inlet >= 0:
                        to_sweep_next[tile_of_inlet[inlet]] = True
                del halo, order, channel, value
            # the orders arriving at every inlet, from the exits as they now stand
            arriving_largest[:] = 0
            arriving_second[:] = 0
            for index in range(records.exit_global.size):
                inlet = inlet_of_exit[index]
                if inlet < 0:
                    continue
                order_here = exit_order_value[index]
                if order_here >= arriving_largest[inlet]:
                    arriving_second[inlet] = arriving_largest[inlet]
                    arriving_largest[inlet] = order_here
                elif order_here > arriving_second[inlet]:
                    arriving_second[inlet] = order_here
            to_sweep = to_sweep_next
            log(tag, "sweep %d: %d tiles computed, %d marked for the next" % (sweeps, sweeping.size, int(to_sweep.sum())))
    # the last sweep with the orders as they stand, written out
    temporary = out_path + ".partial.tif"
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(channel_path) as channel_dataset, \
            rasterio.open(temporary, "w", **raster_profile(grid, "uint8", 0, predictor=False)) as out_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            channel = channel_dataset.read(1, window=Window(col0, row0, ncol, nrow))
            first_inlet = records.tile_inlet_offsets[tile_index]
            last_inlet = records.tile_inlet_offsets[tile_index + 1]
            first_exit = records.tile_exit_offsets[tile_index]
            last_exit = records.tile_exit_offsets[tile_index + 1]
            value = np.zeros(nrow * ncol, np.uint8)
            settled_exit_order = np.zeros(last_exit - first_exit, np.uint8)
            status = _strahler_order_of_tile(halo, order, channel, records.inlet_local[first_inlet:last_inlet],
                                             arriving_largest[first_inlet:last_inlet], arriving_second[first_inlet:last_inlet],
                                             exit_pixel_per_tile[tile_index], settled_exit_order, value)
            if status != 0:
                raise FlowDivideError("the Strahler order of the tile at row %d col %d stopped with status %d (1 an order past "
                                      "a byte, 2 and 3 a broken channel mask)" % (row0, col0, status))
            if not np.array_equal(settled_exit_order, exit_order_value[first_exit:last_exit]):
                raise FlowDivideError("the orders of the tile at row %d col %d are not the ones the sweeps settled on" % (row0, col0))
            out_dataset.write(value.reshape(nrow, ncol), 1, window=Window(col0, row0, ncol, nrow))
            del halo, order, channel, value
            log(tag, "written tile %d of %d" % (tile_index + 1, tile_count))
    publish(temporary, out_path)
    report = {"kernel": "ord", "tiles": tile_count, "land_pixels": int(land_total), "exits": int(records.exit_global.size),
              "inlets": int(records.inlet_global.size), "sweeps": sweeps, "passes_over_the_grid": sweeps + 2,
              "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s: %d sweeps over %d tiles, %.1f s" % (out_path, sweeps, tile_count, report["seconds"]))
    return report


# =============================================================================
#  [7] The command line
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="the four kernels on the tiles the data are distributed in, one tile at a time")
    parser.add_argument("dir", help="the recoded flow directions (FD1.0's output: 1 E ... 128 NE, 0 a mouth, 255 a sink, 247 no data)")
    parser.add_argument("kernel", choices=["upa", "ldn", "lup", "ord"])
    parser.add_argument("out", help="the raster to write")
    parser.add_argument("second_out", nargs="?", default=None, help="upa: the channel mask to write")
    parser.add_argument("--channel", default=None, help="ord: the channel mask (0 off the network)")
    parser.add_argument("--source", default=None, help="lup: the raster of the pixel every longest path starts at")
    parser.add_argument("--tile-degrees", type=float, default=TILE_DEGREES_DEFAULT, help="the tiles of a geographic grid (default 10, as HydroSHEDS v2 is distributed)")
    parser.add_argument("--tile-metres", type=float, default=TILE_METRES_DEFAULT, help="the tiles of a projected grid (default 100000)")
    parser.add_argument("--channel-threshold-km2", type=float, default=1.0, help="upa: a channel pixel drains at least this")
    parser.add_argument("--periodic", action="store_true", help="the grid spans 360 degrees of longitude")
    arguments = parser.parse_args()
    grid = Grid(arguments.dir, periodic=arguments.periodic)
    tiles, pixels_per_tile = tiles_of_distribution(grid, arguments.tile_degrees, arguments.tile_metres)
    log("tiles", "%d tiles of %d x %d pixels over a grid of %d x %d" % (len(tiles), pixels_per_tile, pixels_per_tile, grid.nrow, grid.ncol))
    if arguments.kernel == "upa":
        # no second output: the upstream area alone, which is what Figure 7 charges this kernel with
        report = upa_on_tiles(arguments.dir, arguments.out, arguments.second_out, grid, tiles,
                              channel_threshold_km2=arguments.channel_threshold_km2)
    elif arguments.kernel == "ldn":
        report = ldn_on_tiles(arguments.dir, arguments.out, grid, tiles)
    elif arguments.kernel == "lup":
        report = lup_on_tiles(arguments.dir, arguments.out, grid, tiles, source_path=arguments.source)
    else:
        if not arguments.channel:
            raise SystemExit("ord needs --channel: the Strahler order is the order of a network")
        report = ord_on_tiles(arguments.dir, arguments.channel, arguments.out, grid, tiles)
    print(report)


if __name__ == "__main__":
    main()
