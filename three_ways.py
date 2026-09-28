"""three_ways.py -- the same attribute computed three ways on one basin, so that a reader can see for
themselves that a cut does not change the answer.

    whole_int64   the whole rectangle in memory, every pixel addressed by a 64-bit index; no boundary
                  at all, the reference answer
    by_part       one part at a time plus a graph of the pixels that leave a part (the exit graph);
                  part_of_pixel is the only input that separates the two cuts:
                      square_tiles_of_hydrosheds  the 10 x 10 degree tiles HydroSHEDS v2 is distributed
                                                  in, corners at whole multiples of ten degrees
                                                  (hydrosheds.org: "named with a 7-digit identifier based
                                                  on the coordinates of their lower-left corner, e.g.
                                                  n40w080")
                      regions_of_the_cut          the regions of FlowDivide's published partition

Seven variables, two functions each (the whole domain and one part at a time):

    upg / upa   accumulate_whole_int64 / accumulate_by_part   (the pixel count, or the area with a weight)
    shv         shreve_whole_int64 / shreve_by_part
    ldn         distance_to_outlet_whole_int64 / distance_to_outlet_by_part      (whole centimetres)
    lup         upstream_flow_length_whole_int64 / upstream_flow_length_by_part  (whole centimetres, and
                                                                                   the farthest source pixel)
    ord         strahler_whole_int64 / strahler_by_part
    hck         hack_whole_int64 / hack_by_part                                  (needs the upstream area)

One part at a time is the ORDER OF WORK, not the memory: the whole rectangle stays resident whichever way
runs, so that the answers can be compared pixel for pixel.  Bytes per rectangle pixel, as the arrays are
sized: the whole domain 29 (upa 37 with its weight, lup 37 with the source pixel); by-part 65 (upa and hck
73, lup 89 with the source and the arriving arrays).  On basin 7 of South America, the Parnaiba, 7.5e8
pixels in its rectangle, that is 22 GB for the whole domain and 45-62 GB for by-part, which macOS holds
on a 64 GB laptop by compressing.  The production run of the
package (fd3_attributes.py) holds one region at a time; this file is for checking, not for producing.

Every length is a sum of whole centimetres held in a double, which is exact, so the same path adds up to
the same number however the domain was cut and the comparison is plain equality; a tie between two
equally long paths goes to the smaller pixel index.  The Strahler order of a pixel comes from the largest
and second largest order among its inflows, which merges in any arrival order; when the parts cannot be
put in an order (a river that leaves a tile and comes back), the parts are swept again until nothing
changes, and the number of sweeps is reported.

test_three_ways_on_one_basin.py runs all of this on one basin and compares.
"""
import math
import os

import numpy as np
from numba import njit

from fd1_partition import (EARTH_MODEL_WGS84_ZONE, MERIT_NODATA, MERIT_SINK, earth_distance_m,
                           pixel_area_m2)

LIBRARY_NODATA = 255       # the flow directions this file works on: 255 no data, 0 (or a pit) ends a path
PIXEL_NONE = -1
PART_NONE = np.uint32(0xFFFFFFFF)
TILE_DEGREES = 10.0


# =============================================================================
#  [1] The small things every section uses: the D8 step, the downstream pixel, the topological order
# =============================================================================

@njit(cache=True)
def _d8_row_column_offset(flow_direction_code):
    """the eight ESRI codes as (found, drow, dcol); 0, 255 and anything else end a path"""
    if flow_direction_code == 1:
        return True, 0, 1
    if flow_direction_code == 2:
        return True, 1, 1
    if flow_direction_code == 4:
        return True, 1, 0
    if flow_direction_code == 8:
        return True, 1, -1
    if flow_direction_code == 16:
        return True, 0, -1
    if flow_direction_code == 32:
        return True, -1, -1
    if flow_direction_code == 64:
        return True, -1, 0
    if flow_direction_code == 128:
        return True, -1, 1
    return False, 0, 0


@njit(cache=True)
def _downstream_pixel_of(flow_direction, nrow, ncol, pixel):
    found, row_offset, column_offset = _d8_row_column_offset(flow_direction[pixel])
    if not found:
        return PIXEL_NONE
    row = pixel // ncol
    column = pixel % ncol
    downstream_row = row + row_offset
    downstream_column = column + column_offset
    if downstream_row < 0 or downstream_row >= nrow or downstream_column < 0 or downstream_column >= ncol:
        return PIXEL_NONE
    downstream_pixel = downstream_row * ncol + downstream_column
    if flow_direction[downstream_pixel] == LIBRARY_NODATA:
        return PIXEL_NONE
    return downstream_pixel


@njit(cache=True)
def build_downstream_pixel_array(flow_direction, nrow, ncol):
    """the downstream pixel of every pixel (int64), PIXEL_NONE where the path ends"""
    pixel_count = nrow * ncol
    downstream_of_pixel = np.empty(pixel_count, np.int64)
    for pixel in range(pixel_count):
        downstream_of_pixel[pixel] = _downstream_pixel_of(flow_direction, nrow, ncol, pixel)
    return downstream_of_pixel


@njit(cache=True)
def step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, pixel, downstream_pixel):
    """the length of one D8 step in whole centimetres (rounded half away from zero)"""
    row = pixel // ncol
    column = pixel % ncol
    downstream_row = downstream_pixel // ncol
    downstream_column = downstream_pixel % ncol
    north_south = step_metres_north_south[row]
    east_west = step_metres_east_west[row]
    if downstream_row != row and downstream_column != column:
        return math.floor(math.sqrt(north_south * north_south + east_west * east_west) * 100.0 + 0.5)
    if downstream_row != row:
        return math.floor(north_south * 100.0 + 0.5)
    if downstream_column != column:
        return math.floor(east_west * 100.0 + 0.5)
    return 0.0


@njit(cache=True)
def topological_order_whole(flow_direction, downstream_of_pixel):
    """every land pixel, upstream first (a pixel comes after every pixel that flows into it); the second
    value is False when the directions hold a cycle"""
    pixel_count = flow_direction.size
    upstream_count = np.zeros(pixel_count, np.uint32)
    valid_pixel_count = 0
    for pixel in range(pixel_count):
        if flow_direction[pixel] == LIBRARY_NODATA:
            continue
        valid_pixel_count += 1
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel != PIXEL_NONE:
            upstream_count[downstream_pixel] += 1
    order = np.empty(valid_pixel_count, np.int64)
    order_count = 0
    for pixel in range(pixel_count):
        if flow_direction[pixel] == LIBRARY_NODATA:
            continue
        if upstream_count[pixel] == 0:
            order[order_count] = pixel
            order_count += 1
    position = 0
    while position < order_count:
        pixel = order[position]
        position += 1
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        upstream_count[downstream_pixel] -= 1
        if upstream_count[downstream_pixel] == 0:
            order[order_count] = downstream_pixel
            order_count += 1
    return order, order_count == valid_pixel_count


@njit(cache=True)
def _strahler_merge_upstream_order(upstream_order, largest, second_largest):
    """fold one upstream order into (largest, second largest): what lets the order cross a boundary in
    any arrival order (a1 and a2 of TauDEM's gridnet.cpp)"""
    if upstream_order >= largest:
        return upstream_order, largest
    if upstream_order > second_largest:
        return largest, upstream_order
    return largest, second_largest


@njit(cache=True)
def _strahler_order_from_summary(largest, second_largest):
    if largest == 0:
        return 1
    if second_largest + 1 > largest:
        return second_largest + 1
    return largest


# =============================================================================
#  [2] The parts: the two cuts, and what the by-part forms share (the exit graph)
# =============================================================================

@njit(cache=True)
def square_tiles_of_hydrosheds(flow_direction, nrow, ncol, west, north, pixel_width, pixel_height):
    """a part id per pixel from HydroSHEDS v2's own tiling: 10 x 10 degree tiles with their corners at
    whole multiples of ten degrees, whatever the rectangle is; the ids count the tiles the rectangle
    touches row by row.  Returns (part_of_pixel, part_count, tile_columns, tile_rows)."""
    tile_column_of_west = math.floor((west + 0.5 * pixel_width) / TILE_DEGREES)
    tile_row_of_north = math.floor((north + 0.5 * pixel_height) / TILE_DEGREES)
    tile_column_of_col = np.empty(ncol, np.int64)
    tile_row_of_row = np.empty(nrow, np.int64)
    tile_columns = 0
    tile_rows = 0
    for col in range(ncol):
        longitude = west + (col + 0.5) * pixel_width
        tile_column_of_col[col] = math.floor(longitude / TILE_DEGREES) - tile_column_of_west
        if tile_column_of_col[col] + 1 > tile_columns:
            tile_columns = tile_column_of_col[col] + 1
    for row in range(nrow):
        latitude = north + (row + 0.5) * pixel_height
        tile_row_of_row[row] = tile_row_of_north - math.floor(latitude / TILE_DEGREES)
        if tile_row_of_row[row] + 1 > tile_rows:
            tile_rows = tile_row_of_row[row] + 1
    part_of_pixel = np.empty(nrow * ncol, np.uint32)
    for row in range(nrow):
        part_of_this_row = tile_row_of_row[row] * tile_columns
        for col in range(ncol):
            pixel = row * ncol + col
            if flow_direction[pixel] == LIBRARY_NODATA:
                part_of_pixel[pixel] = PART_NONE
            else:
                part_of_pixel[pixel] = part_of_this_row + tile_column_of_col[col]
    return part_of_pixel, int(tile_columns * tile_rows), int(tile_columns), int(tile_rows)


@njit(cache=True)
def regions_of_the_cut(flow_direction, region_of_pixel):
    """a part id per pixel from the region ids of the published partition (0 outside the basin),
    renumbered from zero in the order the ids are met.  Returns (part_of_pixel, part_count); part_count
    is -1 when a land pixel has no region."""
    pixel_count = flow_direction.size
    largest_region_id = 0
    for pixel in range(pixel_count):
        if flow_direction[pixel] != LIBRARY_NODATA and region_of_pixel[pixel] > largest_region_id:
            largest_region_id = region_of_pixel[pixel]
    part_of_region_id = np.full(largest_region_id + 1, PART_NONE, np.uint32)
    part_of_pixel = np.empty(pixel_count, np.uint32)
    part_count = 0
    for pixel in range(pixel_count):
        if flow_direction[pixel] == LIBRARY_NODATA:
            part_of_pixel[pixel] = PART_NONE
            continue
        region_id = region_of_pixel[pixel]
        if region_id == 0:
            return part_of_pixel, -1
        if part_of_region_id[region_id] == PART_NONE:
            part_of_region_id[region_id] = part_count
            part_count += 1
        part_of_pixel[pixel] = part_of_region_id[region_id]
    return part_of_pixel, part_count


class Cut:
    """everything the by-part variables share, built once from part_of_pixel by cut_build: the pixels
    of each part in the part's own topological order, the exits (pixels whose downstream pixel is in
    another part), which exit every pixel drains to and how far it is along the path, the exit graph
    (exit x flows into a pixel d of another part, d drains to that part's exit y, so x -> y) in an order
    upstream first, and whether the parts themselves can be ordered"""

    def __init__(self):
        self.part_count = 0
        self.valid_pixel_count = 0
        self.largest_part_pixels = 0
        self.exit_count = 0
        self.inlet_count = 0
        self.part_graph_edge_count = 0
        self.part_graph_is_acyclic = False
        self.downstream_of_pixel = None
        self.part_start = None                       # part p holds part_local_order[part_start[p]:part_start[p+1]]
        self.part_local_order = None                 # every land pixel, its part's pixels contiguous, upstream first inside a part
        self.exit_index_of_pixel = None              # the exit number of an exit pixel, PIXEL_NONE otherwise
        self.drain_exit_of_pixel = None              # the exit a pixel drains to inside its part, PIXEL_NONE if it reaches a terminal
        self.distance_to_drain_centimetres = None    # along the path, to that exit or terminal
        self.exit_pixel_of_exit = None
        self.next_exit_of_exit = None                # the exit the inlet pixel below an exit drains to
        self.exit_topological_order = None           # the exits, upstream first
        self.part_topological_order = None           # the parts, upstream first; None when the part graph has a cycle


@njit(cache=True)
def _cut_parts_and_local_orders(flow_direction, downstream_of_pixel, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west, ncol):
    """the pixels of every part in the part's own topological order, the exits, and for every pixel the
    exit it drains to and the distance along the path to it"""
    pixel_count = flow_direction.size
    part_start = np.zeros(part_count + 2, np.int64)
    exit_index_of_pixel = np.full(pixel_count, PIXEL_NONE, np.int64)
    drain_exit_of_pixel = np.full(pixel_count, PIXEL_NONE, np.int64)
    distance_to_drain_centimetres = np.zeros(pixel_count, np.float64)
    valid_pixel_count = 0
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        part_start[part_of_pixel[pixel] + 1] += 1
        valid_pixel_count += 1
    largest_part_pixels = 0
    for part in range(part_count):
        if part_start[part + 1] > largest_part_pixels:
            largest_part_pixels = part_start[part + 1]
        part_start[part + 1] += part_start[part]
    part_pixels = np.empty(valid_pixel_count, np.int64)
    fill_position = part_start[:part_count].copy()
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        part_pixels[fill_position[part_of_pixel[pixel]]] = pixel
        fill_position[part_of_pixel[pixel]] += 1
    exit_count = 0
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        if part_of_pixel[downstream_pixel] == part_of_pixel[pixel]:
            continue
        exit_index_of_pixel[pixel] = exit_count
        exit_count += 1
    exit_pixel_of_exit = np.empty(exit_count + 1, np.int64)
    for pixel in range(pixel_count):
        if exit_index_of_pixel[pixel] != PIXEL_NONE:
            exit_pixel_of_exit[exit_index_of_pixel[pixel]] = pixel
    in_part_upstream_count = np.zeros(pixel_count, np.uint32)
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        if part_of_pixel[downstream_pixel] != part_of_pixel[pixel]:
            continue
        in_part_upstream_count[downstream_pixel] += 1
    part_local_order = np.empty(valid_pixel_count, np.int64)
    every_part_is_a_forest = True
    for part in range(part_count):
        part_first = part_start[part]
        part_last = part_start[part + 1]
        order_count = part_first
        for position in range(part_first, part_last):
            pixel = part_pixels[position]
            if in_part_upstream_count[pixel] == 0:
                part_local_order[order_count] = pixel
                order_count += 1
        position = part_first
        while position < order_count:
            pixel = part_local_order[position]
            position += 1
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                continue
            if part_of_pixel[downstream_pixel] != part_of_pixel[pixel]:
                continue
            in_part_upstream_count[downstream_pixel] -= 1
            if in_part_upstream_count[downstream_pixel] == 0:
                part_local_order[order_count] = downstream_pixel
                order_count += 1
        if order_count != part_last:
            every_part_is_a_forest = False
            break
        # backwards through the same order: a pixel drains to what its downstream pixel drains to
        for position in range(part_last - 1, part_first - 1, -1):
            pixel = part_local_order[position]
            if exit_index_of_pixel[pixel] != PIXEL_NONE:
                drain_exit_of_pixel[pixel] = exit_index_of_pixel[pixel]
                distance_to_drain_centimetres[pixel] = 0.0
                continue
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                drain_exit_of_pixel[pixel] = PIXEL_NONE
                distance_to_drain_centimetres[pixel] = 0.0
                continue
            drain_exit_of_pixel[pixel] = drain_exit_of_pixel[downstream_pixel]
            distance_to_drain_centimetres[pixel] = (distance_to_drain_centimetres[downstream_pixel]
                                                    + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, pixel, downstream_pixel))
    return (part_start, part_local_order, exit_index_of_pixel, drain_exit_of_pixel, distance_to_drain_centimetres,
            exit_pixel_of_exit, exit_count, valid_pixel_count, largest_part_pixels, every_part_is_a_forest)


@njit(cache=True)
def _cut_exit_graph(downstream_of_pixel, part_of_pixel, drain_exit_of_pixel, exit_pixel_of_exit, exit_count, pixel_count):
    """the exit graph (x -> y), how many pixels other parts flow into, the exits in an order upstream
    first (the second flag False when the exit graph has a cycle, which a forest cannot give)"""
    next_exit_of_exit = np.empty(exit_count + 1, np.int64)
    pixel_is_inlet = np.zeros(pixel_count, np.uint8)
    for exit_index in range(exit_count):
        exit_pixel = exit_pixel_of_exit[exit_index]
        inlet_pixel = downstream_of_pixel[exit_pixel]
        next_exit_of_exit[exit_index] = drain_exit_of_pixel[inlet_pixel]
        pixel_is_inlet[inlet_pixel] = 1
    inlet_count = 0
    for pixel in range(pixel_count):
        inlet_count += pixel_is_inlet[pixel]
    upstream_exit_count = np.zeros(exit_count + 1, np.uint32)
    for exit_index in range(exit_count):
        next_exit = next_exit_of_exit[exit_index]
        if next_exit != PIXEL_NONE:
            upstream_exit_count[next_exit] += 1
    exit_topological_order = np.empty(exit_count + 1, np.int64)
    order_count = 0
    for exit_index in range(exit_count):
        if upstream_exit_count[exit_index] == 0:
            exit_topological_order[order_count] = exit_index
            order_count += 1
    position = 0
    while position < order_count:
        exit_index = exit_topological_order[position]
        position += 1
        next_exit = next_exit_of_exit[exit_index]
        if next_exit == PIXEL_NONE:
            continue
        upstream_exit_count[next_exit] -= 1
        if upstream_exit_count[next_exit] == 0:
            exit_topological_order[order_count] = next_exit
            order_count += 1
    return next_exit_of_exit, exit_topological_order, inlet_count, order_count == exit_count


@njit(cache=True)
def _cut_part_graph_order(downstream_of_pixel, part_of_pixel, exit_pixel_of_exit, exit_count, part_count):
    """can the parts themselves be ordered?  The edges (upstream part, downstream part) of the exits,
    made unique, then a topological order; (order, edge_count, acyclic)"""
    packed_edge = np.empty(exit_count, np.uint64)
    packed_count = 0
    for exit_index in range(exit_count):
        exit_pixel = exit_pixel_of_exit[exit_index]
        downstream_pixel = downstream_of_pixel[exit_pixel]
        upstream_part = np.uint64(part_of_pixel[exit_pixel])
        downstream_part = np.uint64(part_of_pixel[downstream_pixel])
        if downstream_part == np.uint64(PART_NONE):
            continue
        packed_edge[packed_count] = (upstream_part << np.uint64(32)) | downstream_part
        packed_count += 1
    packed_edge = np.sort(packed_edge[:packed_count])
    unique_count = 0
    for edge_index in range(packed_count):
        if edge_index > 0 and packed_edge[edge_index] == packed_edge[edge_index - 1]:
            continue
        packed_edge[unique_count] = packed_edge[edge_index]
        unique_count += 1
    upstream_part_count = np.zeros(part_count, np.uint32)
    part_edge_start = np.zeros(part_count + 1, np.int64)
    edge_index = 0
    for part in range(part_count + 1):
        part_edge_start[part] = edge_index
        while edge_index < unique_count and (packed_edge[edge_index] >> np.uint64(32)) == np.uint64(part):
            edge_index += 1
    for edge_index in range(unique_count):
        downstream_part = int(packed_edge[edge_index] & np.uint64(0xFFFFFFFF))
        upstream_part_count[downstream_part] += 1
    part_order = np.empty(part_count, np.int64)
    order_count = 0
    for part in range(part_count):
        if upstream_part_count[part] == 0:
            part_order[order_count] = part
            order_count += 1
    position = 0
    while position < order_count:
        part = part_order[position]
        position += 1
        for edge_index in range(part_edge_start[part], part_edge_start[part + 1]):
            downstream_part = int(packed_edge[edge_index] & np.uint64(0xFFFFFFFF))
            upstream_part_count[downstream_part] -= 1
            if upstream_part_count[downstream_part] == 0:
                part_order[order_count] = downstream_part
                order_count += 1
    return part_order, unique_count, order_count == part_count


@njit(cache=True)
def _every_land_pixel_has_a_part(flow_direction, part_of_pixel, part_count):
    """the invariant the by-part forms rest on: a land pixel has a part below part_count, a nodata pixel
    has none; an exit whose inlet had no part would otherwise be taken for a terminal in silence"""
    for pixel in range(flow_direction.size):
        if flow_direction[pixel] == LIBRARY_NODATA:
            if part_of_pixel[pixel] != PART_NONE:
                return False
        elif part_of_pixel[pixel] >= part_count:
            return False
    return True


def cut_build(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west):
    """everything the by-part variables share, from one part array; raises on a cycle or a pixel without a part"""
    if part_of_pixel.dtype != np.uint32:
        raise ValueError("part_of_pixel must be uint32 (PART_NONE is 0xFFFFFFFF)")
    # the sizes before any kernel indexes with them (Numba does not check bounds)
    if flow_direction.size != int(nrow) * int(ncol) or part_of_pixel.size != flow_direction.size or \
            np.asarray(step_metres_north_south).size != int(nrow) or np.asarray(step_metres_east_west).size != int(nrow):
        raise ValueError("the flow directions, the parts and the step lengths do not all fit a %d x %d grid" % (nrow, ncol))
    if not _every_land_pixel_has_a_part(flow_direction, part_of_pixel, np.uint32(part_count)):
        raise ValueError("a land pixel has no part, or a nodata pixel has one")
    cut = Cut()
    cut.part_count = int(part_count)
    cut.downstream_of_pixel = build_downstream_pixel_array(flow_direction, nrow, ncol)
    (cut.part_start, cut.part_local_order, cut.exit_index_of_pixel, cut.drain_exit_of_pixel, cut.distance_to_drain_centimetres,
     cut.exit_pixel_of_exit, cut.exit_count, cut.valid_pixel_count, cut.largest_part_pixels, every_part_is_a_forest) = \
        _cut_parts_and_local_orders(flow_direction, cut.downstream_of_pixel, part_of_pixel, cut.part_count, step_metres_north_south, step_metres_east_west, ncol)
    if not every_part_is_a_forest:
        raise ValueError("a part's own flow directions hold a cycle")
    cut.next_exit_of_exit, cut.exit_topological_order, cut.inlet_count, exit_graph_is_acyclic = \
        _cut_exit_graph(cut.downstream_of_pixel, part_of_pixel, cut.drain_exit_of_pixel, cut.exit_pixel_of_exit, cut.exit_count, flow_direction.size)
    if not exit_graph_is_acyclic:
        raise ValueError("the exit graph holds a cycle: the flow directions are not a forest")
    part_order, cut.part_graph_edge_count, cut.part_graph_is_acyclic = \
        _cut_part_graph_order(cut.downstream_of_pixel, part_of_pixel, cut.exit_pixel_of_exit, cut.exit_count, cut.part_count)
    cut.part_topological_order = part_order if cut.part_graph_is_acyclic else None
    return cut


def cost_of_cut(cut, payload_bytes_per_exit, sweeps=0):
    return {"part_count": cut.part_count, "exit_pixel_count": cut.exit_count, "inlet_pixel_count": cut.inlet_count,
            "payload_bytes_per_exit": payload_bytes_per_exit, "boundary_bytes_total": payload_bytes_per_exit * cut.exit_count,
            "part_graph_edge_count": cut.part_graph_edge_count, "part_graph_is_acyclic": int(cut.part_graph_is_acyclic),
            "largest_part_pixels": cut.largest_part_pixels, "strahler_sweep_count": sweeps}


def cost_of_whole(pixel_count, payload_bytes_per_exit):
    return {"part_count": 1, "exit_pixel_count": 0, "inlet_pixel_count": 0, "payload_bytes_per_exit": payload_bytes_per_exit,
            "boundary_bytes_total": 0, "part_graph_edge_count": 0, "part_graph_is_acyclic": 1,
            "largest_part_pixels": pixel_count, "strahler_sweep_count": 0}


# =============================================================================
#  [3] Upstream sums: the pixel count, the area, the Shreve magnitude
# =============================================================================

@njit(cache=True)
def _accumulate_whole(flow_direction, downstream_of_pixel, order, pixel_weight, has_weight):
    pixel_count = flow_direction.size
    accumulated = np.zeros(pixel_count, np.float64)
    for pixel in range(pixel_count):
        if flow_direction[pixel] == LIBRARY_NODATA:
            continue
        accumulated[pixel] = pixel_weight[pixel] if has_weight else 1.0
    for position in range(order.size):
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        accumulated[downstream_pixel] += accumulated[pixel]
    return accumulated


def _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south=None, step_metres_east_west=None, **per_pixel):
    """the sizes every public entry works with, checked before a kernel indexes with them (Numba does not check
    bounds): the flow directions one value a pixel, the step lengths one a row, and every
    other array given one a pixel"""
    if np.asarray(flow_direction).size != int(nrow) * int(ncol):
        raise ValueError("the flow directions do not fit a %d x %d grid" % (nrow, ncol))
    for name, steps in (("north-south steps", step_metres_north_south), ("east-west steps", step_metres_east_west)):
        if steps is not None and np.asarray(steps).size != int(nrow):
            raise ValueError("the %s are not one a row of the %d rows" % (name, nrow))
    for name, values in per_pixel.items():
        if values is not None and np.asarray(values).size != int(nrow) * int(ncol):
            raise ValueError("%s is not one value a pixel of the %d x %d grid" % (name, nrow, ncol))


def accumulate_whole_int64(flow_direction, nrow, ncol, pixel_weight=None):
    """a per-pixel weight (1 when None) summed downstream over the whole domain; (values, cost)"""
    _check_the_arrays(flow_direction, nrow, ncol, pixel_weight=pixel_weight)
    downstream_of_pixel = build_downstream_pixel_array(flow_direction, nrow, ncol)
    order, is_forest = topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise ValueError("the flow directions hold a cycle")
    has_weight = pixel_weight is not None
    weight = pixel_weight if has_weight else np.zeros(1, np.float64)
    accumulated = _accumulate_whole(flow_direction, downstream_of_pixel, order, weight, has_weight)
    return accumulated, cost_of_whole(flow_direction.size, 8)


@njit(cache=True)
def _accumulate_by_part(part_of_pixel, part_count, part_start, part_local_order, downstream_of_pixel,
                        exit_pixel_of_exit, next_exit_of_exit, exit_topological_order, exit_count, pixel_weight, has_weight):
    pixel_count = part_of_pixel.size
    accumulated = np.zeros(pixel_count, np.float64)
    # pass 1: every part on its own
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] != PART_NONE:
            accumulated[pixel] = pixel_weight[pixel] if has_weight else 1.0
    for part in range(part_count):
        for position in range(part_start[part], part_start[part + 1]):
            pixel = part_local_order[position]
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                continue
            if part_of_pixel[downstream_pixel] != part_of_pixel[pixel]:
                continue
            accumulated[downstream_pixel] += accumulated[pixel]
    exit_total = np.zeros(exit_count + 1, np.float64)
    for exit_index in range(exit_count):
        exit_total[exit_index] = accumulated[exit_pixel_of_exit[exit_index]]
    # the exit graph, upstream first
    for position in range(exit_count):
        exit_index = exit_topological_order[position]
        next_exit = next_exit_of_exit[exit_index]
        if next_exit == PIXEL_NONE:
            continue
        exit_total[next_exit] += exit_total[exit_index]
    # pass 2: every part again, with what arrives at its inlet pixels
    for pixel in range(pixel_count):
        accumulated[pixel] = 0.0
        if part_of_pixel[pixel] != PART_NONE:
            accumulated[pixel] = pixel_weight[pixel] if has_weight else 1.0
    for exit_index in range(exit_count):
        inlet_pixel = downstream_of_pixel[exit_pixel_of_exit[exit_index]]
        accumulated[inlet_pixel] += exit_total[exit_index]
    for part in range(part_count):
        for position in range(part_start[part], part_start[part + 1]):
            pixel = part_local_order[position]
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                continue
            if part_of_pixel[downstream_pixel] != part_of_pixel[pixel]:
                continue
            accumulated[downstream_pixel] += accumulated[pixel]
    return accumulated


def accumulate_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west, pixel_weight=None):
    """the same sum one part at a time: every part on its own, the exit graph, every part again with
    what arrives at its inlets.  One number crosses each exit, combined by addition, so the parts need no
    order; (values, cost)"""
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west, part_of_pixel=part_of_pixel, pixel_weight=pixel_weight)
    cut = cut_build(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west)
    has_weight = pixel_weight is not None
    weight = pixel_weight if has_weight else np.zeros(1, np.float64)
    accumulated = _accumulate_by_part(part_of_pixel, cut.part_count, cut.part_start, cut.part_local_order, cut.downstream_of_pixel,
                                      cut.exit_pixel_of_exit, cut.next_exit_of_exit, cut.exit_topological_order, cut.exit_count, weight, has_weight)
    return accumulated, cost_of_cut(cut, 8)


@njit(cache=True)
def shreve_source_weights(flow_direction, nrow, ncol):
    """1.0 at every source pixel (no pixel flows into it), 0 elsewhere: the Shreve magnitude is the
    accumulation of these"""
    pixel_count = flow_direction.size
    a_pixel_flows_in = np.zeros(pixel_count, np.uint8)
    for pixel in range(pixel_count):
        if flow_direction[pixel] == LIBRARY_NODATA:
            continue
        downstream_pixel = _downstream_pixel_of(flow_direction, nrow, ncol, pixel)
        if downstream_pixel != PIXEL_NONE:
            a_pixel_flows_in[downstream_pixel] = 1
    source_weight = np.zeros(pixel_count, np.float64)
    for pixel in range(pixel_count):
        if flow_direction[pixel] != LIBRARY_NODATA and a_pixel_flows_in[pixel] == 0:
            source_weight[pixel] = 1.0
    return source_weight


def _shreve_as_uint32(accumulated):
    """the source counts as uint32; a count past UINT32_MAX is an error, not a wrapped value"""
    if accumulated.size and float(np.nanmax(accumulated)) > 4294967295.0:
        raise ValueError("a Shreve magnitude passes what a uint32 holds")
    return accumulated.astype(np.uint32)


def shreve_whole_int64(flow_direction, nrow, ncol):
    """the number of sources above a pixel, itself included; (uint32 values, cost)"""
    _check_the_arrays(flow_direction, nrow, ncol)
    accumulated, cost = accumulate_whole_int64(flow_direction, nrow, ncol, shreve_source_weights(flow_direction, nrow, ncol))
    cost["payload_bytes_per_exit"] = 4
    return _shreve_as_uint32(accumulated), cost


def shreve_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west):
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west, part_of_pixel=part_of_pixel)
    accumulated, cost = accumulate_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west,
                                           shreve_source_weights(flow_direction, nrow, ncol))
    cost["payload_bytes_per_exit"] = 4
    cost["boundary_bytes_total"] = 4 * cost["exit_pixel_count"]
    return _shreve_as_uint32(accumulated), cost


# =============================================================================
#  [4] Values handed up from the terminal and added to on the way: the distance to the outlet, the Hack order
# =============================================================================

@njit(cache=True)
def _distance_to_outlet_whole(flow_direction, downstream_of_pixel, order, step_metres_north_south, step_metres_east_west, ncol):
    pixel_count = flow_direction.size
    downstream_centimetres = np.zeros(pixel_count, np.float64)
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        downstream_centimetres[pixel] = (downstream_centimetres[downstream_pixel]
                                         + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, pixel, downstream_pixel))
    return downstream_centimetres


def distance_to_outlet_whole_int64(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west):
    """the distance along the flow path to the terminal, whole centimetres, over the whole domain"""
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west)
    downstream_of_pixel = build_downstream_pixel_array(flow_direction, nrow, ncol)
    order, is_forest = topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise ValueError("the flow directions hold a cycle")
    values = _distance_to_outlet_whole(flow_direction, downstream_of_pixel, order, step_metres_north_south, step_metres_east_west, ncol)
    return values, cost_of_whole(flow_direction.size, 8)


@njit(cache=True)
def _distance_to_outlet_by_part(part_of_pixel, downstream_of_pixel, drain_exit_of_pixel, distance_to_drain_centimetres,
                                exit_pixel_of_exit, exit_topological_order, exit_count, step_metres_north_south, step_metres_east_west, ncol):
    # pass 1 is already done: cut_build filled distance_to_drain_centimetres.  The exit graph, downstream first:
    # an exit's distance is its step across the cut, the inlet pixel's distance to its own drain, and what
    # that drain carries -- the part's inlet-to-exit distance counted exactly once
    exit_centimetres = np.zeros(exit_count + 1, np.float64)
    for position in range(exit_count - 1, -1, -1):
        exit_index = exit_topological_order[position]
        exit_pixel = exit_pixel_of_exit[exit_index]
        inlet_pixel = downstream_of_pixel[exit_pixel]
        inlet_drain_exit = drain_exit_of_pixel[inlet_pixel]
        centimetres_below_inlet = 0.0 if inlet_drain_exit == PIXEL_NONE else exit_centimetres[inlet_drain_exit]
        exit_centimetres[exit_index] = (step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, exit_pixel, inlet_pixel)
                                        + distance_to_drain_centimetres[inlet_pixel] + centimetres_below_inlet)
    pixel_count = part_of_pixel.size
    downstream_centimetres = np.zeros(pixel_count, np.float64)
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        drain_exit = drain_exit_of_pixel[pixel]
        downstream_centimetres[pixel] = distance_to_drain_centimetres[pixel] + (0.0 if drain_exit == PIXEL_NONE else exit_centimetres[drain_exit])
    return downstream_centimetres


def distance_to_outlet_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west):
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west, part_of_pixel=part_of_pixel)
    cut = cut_build(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west)
    values = _distance_to_outlet_by_part(part_of_pixel, cut.downstream_of_pixel, cut.drain_exit_of_pixel, cut.distance_to_drain_centimetres,
                                         cut.exit_pixel_of_exit, cut.exit_topological_order, cut.exit_count, step_metres_north_south, step_metres_east_west, ncol)
    return values, cost_of_cut(cut, 8)


@njit(cache=True)
def _hack_pixel_is_the_main_donor(downstream_of_pixel, upstream_area, nrow, ncol, pixel, downstream_pixel):
    """the main donor of a confluence has the largest upstream area; a tie goes to the smaller pixel
    index.  Only the eight neighbours of the downstream pixel are read, so every cut makes the same choice."""
    downstream_row = downstream_pixel // ncol
    downstream_column = downstream_pixel % ncol
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            if row_offset == 0 and column_offset == 0:
                continue
            neighbour_row = downstream_row + row_offset
            neighbour_column = downstream_column + column_offset
            if neighbour_row < 0 or neighbour_row >= nrow or neighbour_column < 0 or neighbour_column >= ncol:
                continue
            neighbour = neighbour_row * ncol + neighbour_column
            if neighbour == pixel or downstream_of_pixel[neighbour] != downstream_pixel:
                continue
            if upstream_area[neighbour] > upstream_area[pixel]:
                return False
            if upstream_area[neighbour] == upstream_area[pixel] and neighbour < pixel:
                return False
    return True


@njit(cache=True)
def _hack_whole(flow_direction, downstream_of_pixel, order, upstream_area, nrow, ncol):
    pixel_count = flow_direction.size
    hack_order = np.zeros(pixel_count, np.uint8)
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            hack_order[pixel] = 1
            continue
        step = 0 if _hack_pixel_is_the_main_donor(downstream_of_pixel, upstream_area, nrow, ncol, pixel, downstream_pixel) else 1
        if hack_order[downstream_pixel] + step > 255:
            return hack_order, False
        hack_order[pixel] = hack_order[downstream_pixel] + step
    return hack_order, True


def hack_whole_int64(flow_direction, nrow, ncol, upstream_area):
    """Hack (1957): 1 at the terminal, the main donor keeps the order of the pixel below, every other
    donor takes one more; over the whole domain"""
    _check_the_arrays(flow_direction, nrow, ncol, upstream_area=upstream_area)
    downstream_of_pixel = build_downstream_pixel_array(flow_direction, nrow, ncol)
    order, is_forest = topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise ValueError("the flow directions hold a cycle")
    hack_order, fits = _hack_whole(flow_direction, downstream_of_pixel, order, upstream_area, nrow, ncol)
    if not fits:
        raise ValueError("a Hack order does not fit a byte")
    return hack_order, cost_of_whole(flow_direction.size, 1)


@njit(cache=True)
def _hack_by_part(part_of_pixel, part_count, part_start, part_local_order, downstream_of_pixel, exit_index_of_pixel, drain_exit_of_pixel,
                  exit_pixel_of_exit, exit_topological_order, exit_count, upstream_area, nrow, ncol):
    pixel_count = part_of_pixel.size
    hack_order = np.zeros(pixel_count, np.uint8)
    # pass 1: inside each part, the steps of one between a pixel and the exit or terminal it drains to;
    # an exit's own step across the cut is counted once, below, when the exit is settled
    steps_to_drain = np.zeros(pixel_count, np.uint8)
    for part in range(part_count):
        for position in range(part_start[part + 1] - 1, part_start[part] - 1, -1):
            pixel = part_local_order[position]
            if exit_index_of_pixel[pixel] != PIXEL_NONE:
                steps_to_drain[pixel] = 0
                continue
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                steps_to_drain[pixel] = 0
                continue
            step = 0 if _hack_pixel_is_the_main_donor(downstream_of_pixel, upstream_area, nrow, ncol, pixel, downstream_pixel) else 1
            if steps_to_drain[downstream_pixel] + step > 255:
                return hack_order, False
            steps_to_drain[pixel] = steps_to_drain[downstream_pixel] + step
    # the exit graph, downstream first: an exit's order is the order at the pixel it flows into plus its own step
    exit_order = np.zeros(exit_count + 1, np.uint8)
    for position in range(exit_count - 1, -1, -1):
        exit_index = exit_topological_order[position]
        exit_pixel = exit_pixel_of_exit[exit_index]
        inlet_pixel = downstream_of_pixel[exit_pixel]
        inlet_drain_exit = drain_exit_of_pixel[inlet_pixel]
        order_below_inlet = 1 if inlet_drain_exit == PIXEL_NONE else exit_order[inlet_drain_exit]
        step = 0 if _hack_pixel_is_the_main_donor(downstream_of_pixel, upstream_area, nrow, ncol, exit_pixel, inlet_pixel) else 1
        order_here = order_below_inlet + steps_to_drain[inlet_pixel] + step
        if order_here > 255:
            return hack_order, False
        exit_order[exit_index] = order_here
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        drain_exit = drain_exit_of_pixel[pixel]
        order_at_drain = 1 if drain_exit == PIXEL_NONE else exit_order[drain_exit]
        order_here = order_at_drain + steps_to_drain[pixel]
        if order_here > 255:
            return hack_order, False
        hack_order[pixel] = order_here
    return hack_order, True


def hack_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west, upstream_area):
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west, part_of_pixel=part_of_pixel, upstream_area=upstream_area)
    cut = cut_build(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west)
    hack_order, fits = _hack_by_part(part_of_pixel, cut.part_count, cut.part_start, cut.part_local_order, cut.downstream_of_pixel,
                                     cut.exit_index_of_pixel, cut.drain_exit_of_pixel, cut.exit_pixel_of_exit, cut.exit_topological_order,
                                     cut.exit_count, upstream_area, nrow, ncol)
    if not fits:
        raise ValueError("a Hack order does not fit a byte")
    return hack_order, cost_of_cut(cut, 1)


# =============================================================================
#  [5] The longest path upstream, with its source pixel
# =============================================================================

@njit(cache=True)
def _upstream_flow_length_whole(flow_direction, downstream_of_pixel, order, step_metres_north_south, step_metres_east_west, ncol):
    pixel_count = flow_direction.size
    longest_upstream_centimetres = np.zeros(pixel_count, np.float64)
    farthest_source_pixel = np.arange(pixel_count, dtype=np.int64)
    for position in range(order.size):
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        candidate_centimetres = (longest_upstream_centimetres[pixel]
                                 + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, pixel, downstream_pixel))
        candidate_source = farthest_source_pixel[pixel]
        if candidate_centimetres > longest_upstream_centimetres[downstream_pixel]:
            longest_upstream_centimetres[downstream_pixel] = candidate_centimetres
            farthest_source_pixel[downstream_pixel] = candidate_source
            continue
        if candidate_centimetres == longest_upstream_centimetres[downstream_pixel] and candidate_source < farthest_source_pixel[downstream_pixel]:
            farthest_source_pixel[downstream_pixel] = candidate_source
    return longest_upstream_centimetres, farthest_source_pixel


def upstream_flow_length_whole_int64(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west):
    """the longest path from a source down to the pixel, whole centimetres, and the source it comes from;
    a tie goes to the smaller source pixel index"""
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west)
    downstream_of_pixel = build_downstream_pixel_array(flow_direction, nrow, ncol)
    order, is_forest = topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise ValueError("the flow directions hold a cycle")
    length, source = _upstream_flow_length_whole(flow_direction, downstream_of_pixel, order, step_metres_north_south, step_metres_east_west, ncol)
    return length, source, cost_of_whole(flow_direction.size, 16)


@njit(cache=True)
def _upstream_flow_length_by_part(part_of_pixel, part_count, part_start, part_local_order, downstream_of_pixel, distance_to_drain_centimetres,
                                  exit_pixel_of_exit, next_exit_of_exit, exit_topological_order, exit_count, step_metres_north_south, step_metres_east_west, ncol):
    pixel_count = part_of_pixel.size
    longest_upstream_centimetres = np.zeros(pixel_count, np.float64)
    farthest_source_pixel = np.arange(pixel_count, dtype=np.int64)
    # pass 1: every part on its own, its inlet pixels taken as sources
    for part in range(part_count):
        for position in range(part_start[part], part_start[part + 1]):
            pixel = part_local_order[position]
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                continue
            if part_of_pixel[downstream_pixel] != part_of_pixel[pixel]:
                continue
            candidate_centimetres = (longest_upstream_centimetres[pixel]
                                     + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, pixel, downstream_pixel))
            if candidate_centimetres > longest_upstream_centimetres[downstream_pixel]:
                longest_upstream_centimetres[downstream_pixel] = candidate_centimetres
                farthest_source_pixel[downstream_pixel] = farthest_source_pixel[pixel]
                continue
            if candidate_centimetres == longest_upstream_centimetres[downstream_pixel] and farthest_source_pixel[pixel] < farthest_source_pixel[downstream_pixel]:
                farthest_source_pixel[downstream_pixel] = farthest_source_pixel[pixel]
    exit_centimetres = np.zeros(exit_count + 1, np.float64)
    exit_source = np.empty(exit_count + 1, np.int64)
    for exit_index in range(exit_count):
        exit_pixel = exit_pixel_of_exit[exit_index]
        exit_centimetres[exit_index] = longest_upstream_centimetres[exit_pixel]
        exit_source[exit_index] = farthest_source_pixel[exit_pixel]
    # the exit graph, upstream first; the part's own inlet-to-exit distance is added here and nowhere else
    for position in range(exit_count):
        exit_index = exit_topological_order[position]
        next_exit = next_exit_of_exit[exit_index]
        if next_exit == PIXEL_NONE:
            continue
        exit_pixel = exit_pixel_of_exit[exit_index]
        inlet_pixel = downstream_of_pixel[exit_pixel]
        candidate_centimetres = (exit_centimetres[exit_index]
                                 + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, exit_pixel, inlet_pixel)
                                 + distance_to_drain_centimetres[inlet_pixel])
        if candidate_centimetres > exit_centimetres[next_exit]:
            exit_centimetres[next_exit] = candidate_centimetres
            exit_source[next_exit] = exit_source[exit_index]
            continue
        if candidate_centimetres == exit_centimetres[next_exit] and exit_source[exit_index] < exit_source[next_exit]:
            exit_source[next_exit] = exit_source[exit_index]
    # what arrives at each inlet pixel, then every part again
    arriving_centimetres = np.zeros(pixel_count, np.float64)
    arriving_source = np.full(pixel_count, PIXEL_NONE, np.int64)
    for exit_index in range(exit_count):
        exit_pixel = exit_pixel_of_exit[exit_index]
        inlet_pixel = downstream_of_pixel[exit_pixel]
        candidate_centimetres = (exit_centimetres[exit_index]
                                 + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, exit_pixel, inlet_pixel))
        if arriving_source[inlet_pixel] == PIXEL_NONE or candidate_centimetres > arriving_centimetres[inlet_pixel]:
            arriving_centimetres[inlet_pixel] = candidate_centimetres
            arriving_source[inlet_pixel] = exit_source[exit_index]
            continue
        if candidate_centimetres == arriving_centimetres[inlet_pixel] and exit_source[exit_index] < arriving_source[inlet_pixel]:
            arriving_source[inlet_pixel] = exit_source[exit_index]
    for pixel in range(pixel_count):
        longest_upstream_centimetres[pixel] = 0.0
        farthest_source_pixel[pixel] = pixel
        if arriving_source[pixel] != PIXEL_NONE:
            longest_upstream_centimetres[pixel] = arriving_centimetres[pixel]
            farthest_source_pixel[pixel] = arriving_source[pixel]
    for part in range(part_count):
        for position in range(part_start[part], part_start[part + 1]):
            pixel = part_local_order[position]
            downstream_pixel = downstream_of_pixel[pixel]
            if downstream_pixel == PIXEL_NONE:
                continue
            if part_of_pixel[downstream_pixel] != part_of_pixel[pixel]:
                continue
            candidate_centimetres = (longest_upstream_centimetres[pixel]
                                     + step_length_centimetres_of(step_metres_north_south, step_metres_east_west, ncol, pixel, downstream_pixel))
            if candidate_centimetres > longest_upstream_centimetres[downstream_pixel]:
                longest_upstream_centimetres[downstream_pixel] = candidate_centimetres
                farthest_source_pixel[downstream_pixel] = farthest_source_pixel[pixel]
                continue
            if candidate_centimetres == longest_upstream_centimetres[downstream_pixel] and farthest_source_pixel[pixel] < farthest_source_pixel[downstream_pixel]:
                farthest_source_pixel[downstream_pixel] = farthest_source_pixel[pixel]
    return longest_upstream_centimetres, farthest_source_pixel


def upstream_flow_length_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west):
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west, part_of_pixel=part_of_pixel)
    cut = cut_build(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west)
    length, source = _upstream_flow_length_by_part(part_of_pixel, cut.part_count, cut.part_start, cut.part_local_order, cut.downstream_of_pixel,
                                                   cut.distance_to_drain_centimetres, cut.exit_pixel_of_exit, cut.next_exit_of_exit,
                                                   cut.exit_topological_order, cut.exit_count, step_metres_north_south, step_metres_east_west, ncol)
    return length, source, cost_of_cut(cut, 16)


# =============================================================================
#  [6] The largest and second largest upstream order: Strahler
# =============================================================================

@njit(cache=True)
def _strahler_whole(flow_direction, downstream_of_pixel, order):
    pixel_count = flow_direction.size
    strahler_order = np.zeros(pixel_count, np.uint8)
    largest_upstream_order = np.zeros(pixel_count, np.uint8)
    second_largest_upstream_order = np.zeros(pixel_count, np.uint8)
    for position in range(order.size):
        pixel = order[position]
        strahler_order[pixel] = _strahler_order_from_summary(largest_upstream_order[pixel], second_largest_upstream_order[pixel])
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel == PIXEL_NONE:
            continue
        largest_upstream_order[downstream_pixel], second_largest_upstream_order[downstream_pixel] = \
            _strahler_merge_upstream_order(strahler_order[pixel], largest_upstream_order[downstream_pixel], second_largest_upstream_order[downstream_pixel])
    return strahler_order


def strahler_whole_int64(flow_direction, nrow, ncol):
    """the Strahler order of every land pixel (every source a first-order stream), over the whole domain"""
    _check_the_arrays(flow_direction, nrow, ncol)
    downstream_of_pixel = build_downstream_pixel_array(flow_direction, nrow, ncol)
    order, is_forest = topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise ValueError("the flow directions hold a cycle")
    return _strahler_whole(flow_direction, downstream_of_pixel, order), cost_of_whole(flow_direction.size, 2)


@njit(cache=True)
def _strahler_by_part(part_of_pixel, part_count, part_start, part_local_order, downstream_of_pixel, valid_pixel_count,
                      part_topological_order, parts_are_ordered):
    """every part in part order when there is one, else in id order, swept again until every pixel is
    settled: a pixel is settled once every pixel flowing into it is; (order, sweeps, all settled)"""
    pixel_count = part_of_pixel.size
    strahler_order = np.zeros(pixel_count, np.uint8)
    unsettled_upstream_count = np.zeros(pixel_count, np.uint32)
    largest_upstream_order = np.zeros(pixel_count, np.uint8)
    second_largest_upstream_order = np.zeros(pixel_count, np.uint8)
    pixel_is_settled = np.zeros(pixel_count, np.uint8)
    for pixel in range(pixel_count):
        if part_of_pixel[pixel] == PART_NONE:
            continue
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel != PIXEL_NONE:
            unsettled_upstream_count[downstream_pixel] += 1
    settled_pixel_count = 0
    sweep_count = 0
    while settled_pixel_count < valid_pixel_count:
        settled_before_sweep = settled_pixel_count
        for part_position in range(part_count):
            part = part_topological_order[part_position] if parts_are_ordered else part_position
            for position in range(part_start[part], part_start[part + 1]):
                pixel = part_local_order[position]
                if pixel_is_settled[pixel] != 0:
                    continue
                if unsettled_upstream_count[pixel] > 0:
                    continue
                strahler_order[pixel] = _strahler_order_from_summary(largest_upstream_order[pixel], second_largest_upstream_order[pixel])
                pixel_is_settled[pixel] = 1
                settled_pixel_count += 1
                downstream_pixel = downstream_of_pixel[pixel]
                if downstream_pixel == PIXEL_NONE:
                    continue
                largest_upstream_order[downstream_pixel], second_largest_upstream_order[downstream_pixel] = \
                    _strahler_merge_upstream_order(strahler_order[pixel], largest_upstream_order[downstream_pixel], second_largest_upstream_order[downstream_pixel])
                unsettled_upstream_count[downstream_pixel] -= 1
        sweep_count += 1
        if settled_pixel_count == settled_before_sweep:
            break
    return strahler_order, sweep_count, settled_pixel_count == valid_pixel_count


def strahler_by_part(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west):
    _check_the_arrays(flow_direction, nrow, ncol, step_metres_north_south, step_metres_east_west, part_of_pixel=part_of_pixel)
    cut = cut_build(flow_direction, nrow, ncol, part_of_pixel, part_count, step_metres_north_south, step_metres_east_west)
    parts_are_ordered = cut.part_topological_order is not None
    part_order = cut.part_topological_order if parts_are_ordered else np.zeros(1, np.int64)
    strahler_order, sweeps, all_settled = _strahler_by_part(part_of_pixel, cut.part_count, cut.part_start, cut.part_local_order,
                                                            cut.downstream_of_pixel, cut.valid_pixel_count, part_order, parts_are_ordered)
    if not all_settled:
        raise ValueError("the flow directions are not a forest: a sweep settled nothing")
    return strahler_order, cost_of_cut(cut, 2, sweeps)


# =============================================================================
#  [7] A case on disk: one basin's rectangle as prepare_case leaves it
# =============================================================================

def flow_directions_to_library_convention(merit_directions):
    """MERIT convention (247 no data, 0 the mouth, 255 a pit) to the convention of this file (255 no data,
    0 ends a path), in place"""
    is_nodata = merit_directions == MERIT_NODATA
    is_pit = merit_directions == MERIT_SINK
    merit_directions[is_pit] = 0
    merit_directions[is_nodata] = LIBRARY_NODATA
    return merit_directions


def row_steps_and_areas(nrow, west, north, pixel_width, pixel_height, earth_model=EARTH_MODEL_WGS84_ZONE):
    """the length of a north-south and of an east-west step and the pixel area, one of each per row
    (the earth_distance_m and the pixel area of fd1_partition under earth_model -- the exact area on the
    WGS84 ellipsoid unless the caller asks for MERIT Hydro's own).  The
    area is whole square metres, as FD1.1 keeps it: a sum of whole numbers in a double is exact, so
    the upstream area adds up to the same number however the domain was cut; with fractional weights
    the by-part sum, taken in another order, differs in the last bits on a few thousand pixels."""
    step_metres_north_south = np.empty(nrow, np.float64)
    step_metres_east_west = np.empty(nrow, np.float64)
    pixel_area_m2_by_row = np.empty(nrow, np.float64)
    longitude = west + 0.5 * pixel_width
    for row in range(nrow):
        latitude = north + (row + 0.5) * pixel_height
        step_metres_north_south[row] = earth_distance_m(latitude, longitude, latitude + pixel_height, longitude)
        step_metres_east_west[row] = earth_distance_m(latitude, longitude, latitude, longitude + pixel_width)
        pixel_area_m2_by_row[row] = math.floor(pixel_area_m2(latitude, abs(pixel_width), abs(pixel_height), earth_model) + 0.5)
    return step_metres_north_south, step_metres_east_west, pixel_area_m2_by_row


def read_case_manifest(case_directory):
    """case.txt as a dict of strings"""
    manifest = {}
    with open(os.path.join(case_directory, "case.txt")) as file:
        for line in file:
            key, _, value = line.rstrip("\n").partition(" ")
            manifest[key] = value
    return manifest
