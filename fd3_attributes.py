"""fd3_attributes.py -- FlowDivide stage 3, the attributes computed region by region (FD3 of Figure 2).

Every basic hydrographic attribute is a recurrence along the flow directions, and there are four
kinds: an accumulation adds a value downstream (the Shreve magnitude), a label copies a value upstream
from the outlet (the distance to the outlet, the Hack order), a maximum carries the largest value
downstream (the upstream flow length, the longest flow path), and the Strahler order takes the largest
order among the inflows plus one when two share it.  On the partition of stage 1 every one of them
is computed once per region, exactly, with one value crossing each cut:

    the regions are visited in an order in which every region a region needs comes first: the regions
    downstream first for a label (the piece below hands the value up), the regions upstream first for
    an accumulation, a maximum and the Strahler order (the pieces above hand their values down);

    a region is read as one window, the rectangle its members cover plus one pixel; the flow
    directions are the mask, since an upstream walk from a basin's outlet cannot leave the basin;

    every member (a whole basin, or a piece of a cut basin) is walked from its outlet upstream, depth
    first, with an explicit stack; the walk never enters a child piece (its outlet pixel is blocked),
    but reads the child's outlet as one more donor, and the value that crosses the cut is the value
    written at that pixel by the region visited before;

    the output window is read before the walk and written back after it, so that the pixels of other
    regions inside the same rectangle go back as they came.

Six attributes are provided, and a user adds one by writing its rule as a Numba kernel of the same
shape as the six below (see the end of this file).

    shv  Shreve magnitude          UInt32   channel pixels; 1 at a source, the sum of the inflows below
    ldn  distance to the outlet    Float32  metres along the flow path; 0 at the outlet, -1 elsewhere
    hck  Hack order                Byte     channel pixels; 1 along the main stem (the inflow with the
                                            largest upstream area keeps the order), one more on a tributary
    lup  upstream flow length      Float32  metres of the longest path from a divide down to the pixel
    lfp  longest flow path         UInt32   the basin id on the pixels of the basin's longest path, and the
                                            paths as lines in a GeoPackage
    ord  Strahler order            Byte     channel pixels

A channel pixel is a pixel of the channel mask of FD1.1 (upstream area at least the threshold: by
default 1 km2 on the HydroSHEDS grids and 10 km2 on MERIT).  The attributes
are computed for the basins of at least min_basin_area_km2 (1 km2 by default) and the pieces of the
cut basins; every other pixel keeps the nodata value.
"""
import math
import os
import time

import numpy as np
import pandas as pd
import rasterio
from numba import njit
from rasterio.windows import Window

import fd_tables
from fd1_partition import (DROW, DCOL, IS_LAND, IS_TERMINAL, MERIT_NODATA, FlowDivideError, Grid, RASTER_BLOCK, check_merit_flow_directions, earth_distance_m, log, publish,
                           raster_profile, read_block, read_table, timing, write_block, write_json, write_table, ensure_directory)

PATH_FRAMES = 1 << 22         # pixels of the longest flow path a member can hold, the one walk that is a path
SWEPT_ATTRIBUTES = ("shv", "ldn", "hck", "lup", "ord")   # every class but the longest flow path, which
                                                         # walks one path and not a whole member


# =============================================================================
#  [1] The partition as the driver reads it: kept basins, members, regions, the visiting order
# =============================================================================

class Member:
    """a whole basin, or a piece of a cut basin: what the walk of one member needs"""

    def __init__(self, member_id, basin_id, parent_member_id, region_id, outlet_row, outlet_col, parent_inlet_row, parent_inlet_col,
                 rectangle, pixel_count, downstream_depth):
        self.member_id = int(member_id)
        self.basin_id = int(basin_id)
        self.parent_member_id = int(parent_member_id)
        self.region_id = int(region_id)
        self.outlet_row = int(outlet_row)
        self.outlet_col = int(outlet_col)                 # a grid column (folded back on a periodic grid)
        self.parent_inlet_row = int(parent_inlet_row)
        self.parent_inlet_col = int(parent_inlet_col)
        self.rectangle = tuple(int(v) for v in rectangle)  # unrolled on a periodic grid
        self.pixel_count = int(pixel_count)
        self.downstream_depth = int(downstream_depth)
        self.children = []


class Partition:
    """the tables of one partition read once: the kept basins, their members, the members gathered
    by region, and the order the regions may be visited in"""

    def __init__(self, basin_table_path, region_table_path, piece_table_path, basin_region_table_path, min_basin_area_km2, grid, expect=None,
                 raster_min_basin_area_km2=None):
        """expect: {"continent", "capacity_px", "block_px", "groups"} the region map must have been written for: its run
        and its key (fd_tables.region_map_key), checked against its header.  The tables are those of fd_tables (the
        code's), their .done markers demanded.

        raster_min_basin_area_km2: the basins of at least this area are computed into the
        raster, those of at least min_basin_area_km2 are listed in the tables.  The distance to the outlet and the
        upstream flow length cover every basin however small (0), their tables the basins of 1 km2 and more; the other
        classes take None, the same area for both.  A basin below min_basin_area_km2 is never cut (a basin is cut only
        when its window exceeds the capacity), so it is one whole member in one region: such basins are kept as arrays
        gathered by region (small_*), not as Member objects, because North America holds some thirty million of them."""
        self.grid = grid
        basins = fd_tables.read_basin_table(basin_table_path)
        regions = fd_tables.read_region_table(region_table_path)
        pieces = fd_tables.read_piece_table(piece_table_path)
        if expect is None:
            with open(basin_region_table_path) as handle:
                fields = dict(word.split("=", 1) for word in handle.readline().split()[2:] if "=" in word)
            run, key = fields.get("run"), fields.get("key")
        else:
            run = expect["continent"]
            key = fd_tables.region_map_key(expect["capacity_px"], expect["block_px"], expect.get("groups", "l3"))
        region_of_basin, in_the_map = fd_tables.read_basin_map(basin_region_table_path, run, "region", key)
        if in_the_map != len(basins):
            raise FlowDivideError("the basin-to-region map lists %d basins, the basin table %d" % (in_the_map, len(basins)))
        region_of_basin = region_of_basin.astype(np.int64)
        # which basins the raster covers and which the tables list.  A basin below the tables' area that the
        # raster covers is kept as an object only when it is cut (possible when min_basin_area_km2 is set above a cut
        # basin's area); the whole ones are the arrays below
        self.table_min_basin_area_km2 = float(min_basin_area_km2)
        area = basins["basin_area_km2"].to_numpy(np.float64)
        in_the_tables = area >= min_basin_area_km2
        small_whole = np.zeros(len(basins), bool)
        small_cut = np.zeros(len(basins), bool)
        if raster_min_basin_area_km2 is not None and raster_min_basin_area_km2 < min_basin_area_km2:
            below = (area >= raster_min_basin_area_km2) & ~in_the_tables
            region_of_row = region_of_basin[basins["basin_id"].to_numpy(np.int64)]
            small_whole = below & (region_of_row != 0)
            small_cut = below & (region_of_row == 0)
            del below, region_of_row
        kept = basins[in_the_tables | small_cut]
        self.basins = {}
        self.members = {}
        for row in kept.itertuples(index=False):
            basin_id = int(row.basin_id)
            self.basins[basin_id] = {"outlet_row": int(row.outlet_row), "outlet_col": int(row.outlet_col), "area_km2": float(row.basin_area_km2),
                                     "pixel_count": int(row.basin_grid_count), "outlet_lon": float(row.outlet_lon), "outlet_lat": float(row.outlet_lat), "members": []}
            region_id = int(region_of_basin[basin_id])
            if region_id != 0:
                member = Member(basin_id, basin_id, 0, region_id, row.outlet_row, row.outlet_col, -1, -1,
                                (row.basin_row_min, row.basin_row_max, row.basin_col_min, row.basin_col_max), row.basin_grid_count, 0)
                self.members[member.member_id] = member
                self.basins[basin_id]["members"].append(member)
        basin_count = len(basins)
        # every piece of the table, kept basin or not, is checked: a piece
        # id above the basin ids and listed once, pixels and an area, and a basin the map leaves without a region
        piece_ids = pieces["piece_id"].to_numpy(np.int64)
        if len(piece_ids) and ((piece_ids <= basin_count).any() or len(np.unique(piece_ids)) != len(piece_ids)):
            raise FlowDivideError("%s holds a piece id that is a basin id or is listed twice" % piece_table_path)
        if len(pieces):
            piece_area = pieces["piece_area_km2"].to_numpy(np.float64)
            piece_basin = pieces["basin_id"].to_numpy(np.int64)
            if (pieces["piece_grid_count"].to_numpy(np.int64) <= 0).any() or not ((piece_area > 0) & (piece_area < 1e30)).all():
                raise FlowDivideError("%s holds a piece without pixels or without a positive area" % piece_table_path)
            if ((piece_basin < 1) | (piece_basin > basin_count)).any() or (region_of_basin[piece_basin] != 0).any():
                raise FlowDivideError("%s holds a piece of a basin that is not cut (the map gives it a region)" % piece_table_path)
        for row in pieces.itertuples(index=False):
            if int(row.basin_id) not in self.basins:
                continue                                    # the basin is below the area kept, and so are its pieces
            member = Member(row.piece_id, row.basin_id, row.parent_piece_id, row.region_id, row.outlet_row, row.outlet_col, row.parent_inlet_row, row.parent_inlet_col,
                            (row.row_min, row.row_max, row.col_min, row.col_max), row.piece_grid_count, row.downstream_depth)
            self.members[member.member_id] = member
            self.basins[int(row.basin_id)]["members"].append(member)
        # a basin not in one region is cut: at least two pieces, and their pixels are the basin's
        for basin_id, basin in self.basins.items():
            if int(region_of_basin[basin_id]) != 0:
                continue
            pieces_of_basin = basin["members"]
            if len(pieces_of_basin) < 2 or sum(int(piece.pixel_count) for piece in pieces_of_basin) != basin["pixel_count"]:
                raise FlowDivideError("basin %d is cut, but its %d pieces do not hold its %d pixels"
                                      % (basin_id, len(pieces_of_basin), basin["pixel_count"]))
        for member in self.members.values():
            if member.parent_member_id:
                if member.parent_member_id not in self.members or self.members[member.parent_member_id].basin_id != member.basin_id:
                    raise FlowDivideError("piece %d flows into piece %d, which is not a piece of its basin %d"
                                          % (member.member_id, member.parent_member_id, member.basin_id))
                self.members[member.parent_member_id].children.append(member)
        # the parents make a tree: walking down from any member reaches a member without a parent (a
        # cycle; without the walk, two pieces parent to each other would be taken)
        for member in self.members.values():
            steps = 0
            walker = member
            while walker.parent_member_id:
                walker = self.members[walker.parent_member_id]
                steps += 1
                if steps > len(self.members):
                    raise FlowDivideError("the pieces of basin %d flow into each other in a cycle" % member.basin_id)
        self.regions = {int(row.region_id): row for row in regions.itertuples(index=False)}
        # the basins below the table's area that the raster still covers, as arrays sorted by region
        self.small_by_region = {}
        self.small_basin_id = np.zeros(0, np.int64)
        self.small_outlet_row = np.zeros(0, np.int64)
        self.small_outlet_col = np.zeros(0, np.int64)
        self.small_pixel_count = np.zeros(0, np.int64)
        self.raster_covers_every_basin = raster_min_basin_area_km2 is not None and raster_min_basin_area_km2 <= 0.0
        if small_whole.any():
            # only the four columns the sweep needs, taken column by column: a copy of the whole rows of thirty million
            # basins would hold some 13 GB
            small_basin_id = basins["basin_id"].to_numpy(np.int64)[small_whole]
            small_region_id = region_of_basin[small_basin_id]
            sort = np.argsort(small_region_id, kind="stable")
            self.small_basin_id = small_basin_id[sort]
            self.small_outlet_row = basins["outlet_row"].to_numpy(np.int64)[small_whole][sort]
            self.small_outlet_col = basins["outlet_col"].to_numpy(np.int64)[small_whole][sort]
            self.small_pixel_count = basins["basin_grid_count"].to_numpy(np.int64)[small_whole][sort]
            small_region_id = small_region_id[sort]
            region_ids, first, count = np.unique(small_region_id, return_index=True, return_counts=True)
            self.small_by_region = {int(region_id): (int(start), int(start + n)) for region_id, start, n in zip(region_ids, first, count)}
            del small_basin_id, small_region_id, sort
        del area, in_the_tables, small_whole, small_cut
        unlisted = sorted((set(member.region_id for member in self.members.values()) | set(self.small_by_region)) - set(self.regions))
        if unlisted:
            raise FlowDivideError("members lie in regions %s, which the region table does not list" % unlisted[:5])
        self.members_by_region = {}
        for member in self.members.values():
            self.members_by_region.setdefault(member.region_id, []).append(member)
        # the dependencies: region A depends on region B when a member of A has its parent in B (A upstream of B)
        self.depends_on = {}
        for member in self.members.values():
            if member.parent_member_id:
                parent_region = self.members[member.parent_member_id].region_id
                if parent_region != member.region_id:
                    self.depends_on.setdefault(member.region_id, set()).add(parent_region)

    def region_order(self, downstream_first):
        """every region with members, in an order in which every region a region needs comes first;
        whenever more than one could go next, the smallest id goes.  downstream_first: the regions a
        region depends on (its downstream neighbours) come before it; otherwise the reverse relation."""
        import heapq
        region_ids = sorted(set(self.members_by_region) | set(self.small_by_region))   # a region of small basins only too
        after = {r: set() for r in region_ids}
        needed = {r: 0 for r in region_ids}
        for upstream, targets in self.depends_on.items():
            for downstream in targets:
                if upstream not in after or downstream not in after:
                    continue
                if downstream_first:
                    after[downstream].add(upstream)      # the downstream region first, then the upstream one
                    needed[upstream] += 1
                else:
                    after[upstream].add(downstream)
                    needed[downstream] += 1
        ready = [r for r in region_ids if needed[r] == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            region = heapq.heappop(ready)
            order.append(region)
            for other in after[region]:
                needed[other] -= 1
                if needed[other] == 0:
                    heapq.heappush(ready, other)
        if len(order) != len(region_ids):
            raise FlowDivideError("the region dependencies hold a cycle")
        return order

    def region_rectangle(self, region_id):
        """the rectangle the region's pixels lie in (the bounding box of the region table, unrolled on a
        periodic grid), with one pixel of margin, clipped to the rows of the grid.  The members' own
        rectangles lie inside it; on a periodic grid they may be given in the grid's frame, so the
        table's box, made in the region's frame, is what the window is read from."""
        region = self.regions[region_id]
        rows_min = int(region.bbox_row_min) - 1
        rows_max = int(region.bbox_row_max) + 1
        cols_min = int(region.bbox_col_min) - 1
        cols_max = int(region.bbox_col_max) + 1
        for member in self.members_by_region.get(region_id, []):
            rows_min = min(rows_min, member.rectangle[0] - 1)
            rows_max = max(rows_max, member.rectangle[1] + 1)
            if not self.grid.periodic:
                cols_min = min(cols_min, member.rectangle[2] - 1)
                cols_max = max(cols_max, member.rectangle[3] + 1)
        if not self.grid.periodic:
            cols_min = max(cols_min, 0)
            cols_max = min(cols_max, self.grid.ncol - 1)
        elif cols_max - cols_min + 1 > self.grid.ncol:
            raise FlowDivideError("the window of region %d is wider than the grid; a periodic window may not hold a column twice" % region_id)
        return (max(rows_min, 0), min(rows_max, self.grid.nrow - 1), cols_min, cols_max)



def _table_lon_lat(grid, row, col):
    """the centre of a pixel in longitude and latitude for the tables: on a geographic grid the grid's own
    longitude and latitude (origin + (index + 0.5) * pixel size); on a
    projected grid the centre transformed to WGS84 (not the grid's own metres)"""
    if grid.geographic:
        return grid.longitude_of_col(col), grid.latitude_of_row(row)
    lon, lat = grid.pixel_centre_lon_lat(row, col)
    return float(lon), float(lat)


def local_position(grid, rectangle, row, col):
    """a grid pixel's row and column inside the window of a rectangle; on a periodic grid the column is
    taken modulo the grid's width, so that a window starting before the first column or running past the
    last one finds its pixels"""
    local_row = row - rectangle[0]
    local_col = col - rectangle[2]
    if grid.periodic:
        local_col = local_col % grid.ncol
    return local_row, local_col


# =============================================================================
#  [2] What every rule needs of a pixel: the step, the neighbour, the cut
# =============================================================================
#
#  A rule reads two things of a pixel: the length of the step to the pixel it flows into, which depends
#  only on the row, and whether the pixel is one of the outlets of a child piece, which is where a state
#  crosses a cut.  The traversal itself is in section [2b]: one visiting order per window, swept once for
#  each rule.  One method serves every rule: a walk with an explicit stack reads the eight neighbours of
#  every pixel and jumps about the window, and on basin 4 it cost twice to four times what the sweep
#  costs.

@njit(cache=True)
def _is_blocked(blocked_sorted, pixel):
    position = np.searchsorted(blocked_sorted, pixel)
    return position < blocked_sorted.size and blocked_sorted[position] == pixel


@njit(cache=True)
def _step_length(lengths, row, drow, dcol):
    """the length of the step from a pixel in `row` to its neighbour at (drow, dcol): the five lengths of
    the row are east-west, north, south, north-diagonal, south-diagonal"""
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
def _neighbour_offset(cursor):
    """the eight neighbours in a fixed order: the cursor 0 .. 7 to (drow, dcol), the centre skipped"""
    index = cursor if cursor < 4 else cursor + 1
    return index // 3 - 1, index % 3 - 1


@njit(cache=True)
def paint_path_downstream(dir_window, out_window, lengths, start_row, start_col, end_row, end_col, value, path_rows, path_cols):
    """the flow path from (start) down to (end), both inside the window, painted with `value` into
    out_window; the pixels are listed in path_rows / path_cols.  Returns (status, pixel count, length
    in metres): status 2 when the walk leaves the window or ends before reaching the end pixel."""
    nrow, ncol = dir_window.shape
    row = start_row
    col = start_col
    count = 0
    length = 0.0
    while True:
        if count >= path_rows.size:
            return 1, count, length
        path_rows[count] = row
        path_cols[count] = col
        count += 1
        out_window[row, col] = value
        if row == end_row and col == end_col:
            return 0, count, length
        code = dir_window[row, col]
        drow = DROW[code]
        dcol = DCOL[code]
        if drow == 0 and dcol == 0:
            return 2, count, length
        length += _step_length(lengths, row, drow, dcol)
        row += drow
        col += dcol
        if row < 0 or row >= nrow or col < 0 or col >= ncol:
            return 2, count, length


# =============================================================================
#  [2b] The ordered sweeps: the same rules, over a visiting order built once
# =============================================================================
#
#  The walks of section [2] do the arithmetic inside a depth-first traversal, so every pixel pays a push,
#  a pop and eight neighbour tests, and the traversal jumps about the window.  The sweeps here do the same
#  arithmetic over an order built once for the whole window in three linear passes; the loops then read
#  two arrays and nothing else.  Measured on basin 4 (903 million land pixels, one process): the walk
#  costs 236 ns a pixel for the accumulation, the tile kernels, which sweep an order, 118; the method is
#  chosen for speed, not for the least memory.  The order costs 13 bytes for every pixel of the window, which
#  the capacity keeps well inside the memory the partition was made for.
#
#  Three arrays describe the window once, and all four classes then share them:
#      downstream  int32, the pixel each land pixel flows into inside the window; -1 when the flow leaves
#                  the window, when the pixel is a terminal, or when the code is not a land code
#      order       int32, every land pixel of the window, each one after every pixel that flows into it
#      member      int32, the member a pixel belongs to as an index into the region's own members, and -1
#                  for a pixel that belongs to another region (the pixels above a cut into this region)
#
#  Inside a region a cut between two of its own members needs no state at all: both sides are in the same
#  window and in the same order, so the value crosses pixel by pixel.  A state is read only where a member
#  of this region receives from a member of an earlier region, and written only at a member's outlet whose
#  parent is in a later one -- the same two places as in section [2].


@njit(cache=True)
def window_flow_structure(dir_window, mask, use_mask):
    """The window read once into the arrays every class then sweeps: the pixel each pixel flows into, and
    a visiting order in which a pixel comes after every pixel that flows into it.  With use_mask the order
    is built over the pixels `mask` marks and no others, which is how the three classes that live on the
    channel network are computed: the upstream area only grows along the flow, so a channel pixel's
    downstream pixel is a channel pixel as well, and the masked pixels are the network itself.  The
    channel network is about two pixels in a hundred, and building the order over every land pixel instead
    cost those classes twice what the walk they replaced cost (measured).
    Returns (downstream, order, taken, broken); taken is -2 when the directions hold a cycle, and
    broken is the first pixel of the mask whose downstream pixel is land inside the window and not of
    the mask, or -1 when there is none.  Such a pixel would lose everything above it: the link is not
    made, so the pixels above never enter the order and never get a member, and the sweep would write
    the nodata over them and report success.  The upstream area grows along the flow, so a mask made
    from it cannot break; a mask made from a provider's area raster with a nodata pixel inside the
    network can, and the whole step is refused there."""
    nrow, ncol = dir_window.shape
    pixel_count = nrow * ncol
    broken = -1
    downstream = np.full(pixel_count, -1, np.int32)
    inflow = np.zeros(pixel_count, np.uint8)
    taken_already = np.zeros(pixel_count, np.uint8)      # a pixel the peel has appended: the scan must not
    order = np.empty(pixel_count, np.int32)              # start it again once its inflow has reached zero
    land_count = 0
    # pass 1: the downstream pixel of every land pixel, and how many pixels flow into each
    for row in range(nrow):
        for col in range(ncol):
            code = dir_window[row, col]
            if IS_LAND[code] == 0:
                continue
            if use_mask and mask[row, col] == 0:
                continue
            land_count += 1
            pixel = row * ncol + col
            row_offset = DROW[code]
            column_offset = DCOL[code]
            if row_offset == 0 and column_offset == 0:
                continue                                   # a terminal: it flows nowhere
            downstream_row = row + row_offset
            downstream_column = col + column_offset
            if downstream_row < 0 or downstream_row >= nrow or downstream_column < 0 or downstream_column >= ncol:
                continue                                   # the flow leaves the window
            if IS_LAND[dir_window[downstream_row, downstream_column]] == 0:
                continue                                   # the flow leaves the land
            if use_mask and mask[downstream_row, downstream_column] == 0:
                if broken < 0:
                    broken = pixel                         # the mask breaks along the flow: see above
                continue
            downstream_pixel = downstream_row * ncol + downstream_column
            downstream[pixel] = downstream_pixel
            if inflow[downstream_pixel] < 255:
                inflow[downstream_pixel] += 1
    # pass 2: the order, by taking the pixels nothing flows into and walking down from each of them
    taken = 0
    for row in range(nrow):
        for col in range(ncol):
            pixel = row * ncol + col
            if IS_LAND[dir_window[row, col]] == 0 or inflow[pixel] != 0 or taken_already[pixel] != 0:
                continue
            if use_mask and mask[row, col] == 0:
                continue
            walking = pixel
            while walking >= 0:
                order[taken] = walking
                taken_already[walking] = 1
                taken += 1
                next_pixel = downstream[walking]
                if next_pixel < 0:
                    break
                inflow[next_pixel] -= 1
                if inflow[next_pixel] != 0:
                    break                                  # its other donors are not done yet
                walking = next_pixel
    if taken != land_count:
        return downstream, order[:taken], -2, broken       # the directions hold a cycle
    return downstream, order[:taken], taken, broken


@njit(cache=True)
def channel_mask_break_of_ours(dir_window, channel_window, is_outlet, blocked_sorted, ncol):
    """The first pixel of the channel mask that breaks along the flow AND drains to a member of this
    region, or -1 when there is none.

    A break is a channel pixel whose downstream pixel is inside the window, is land, and is not of the
    mask: the link is not made, so the pixels above it never enter the visiting order and the sweep
    would leave them at the nodata.  Whose break it is, is answered by following the flow down from it
    with the directions alone, mask or no mask: the pixel belongs to this region when the walk reaches
    the outlet of one of its members, and to somebody else when it leaves the window, reaches a
    terminal, or reaches the outlet of a child piece that another region works (which is where this
    region's business ends).  The walk is only ever made when a break was found, which the upstream
    area forbids on a mask made from it."""
    nrow, ncol_local = dir_window.shape
    pixel_count = nrow * ncol_local
    for row in range(nrow):
        for col in range(ncol_local):
            if channel_window[row, col] == 0:
                continue
            code = dir_window[row, col]
            if IS_LAND[code] == 0:
                continue
            row_offset = DROW[code]
            column_offset = DCOL[code]
            if row_offset == 0 and column_offset == 0:
                continue
            downstream_row = row + row_offset
            downstream_column = col + column_offset
            if downstream_row < 0 or downstream_row >= nrow or downstream_column < 0 or downstream_column >= ncol_local:
                continue
            if IS_LAND[dir_window[downstream_row, downstream_column]] == 0:
                continue
            if channel_window[downstream_row, downstream_column] != 0:
                continue
            # a break: follow the flow down and see whose it is
            walking_row = row
            walking_column = col
            for _ in range(pixel_count):
                pixel = walking_row * ncol_local + walking_column
                if is_outlet[pixel] != 0:
                    return row * ncol_local + col                  # it drains to a member of ours
                if _is_blocked(blocked_sorted, pixel):
                    break                                          # another region's piece starts here
                walking_code = dir_window[walking_row, walking_column]
                if IS_LAND[walking_code] == 0:
                    break
                next_row = walking_row + DROW[walking_code]
                next_column = walking_column + DCOL[walking_code]
                if next_row == walking_row and next_column == walking_column:
                    break                                          # a terminal
                if next_row < 0 or next_row >= nrow or next_column < 0 or next_column >= ncol_local:
                    break                                          # it leaves the window
                walking_row = next_row
                walking_column = next_column
    return -1


@njit(cache=True)
def window_member_of_pixel(order, downstream, outlet_pixels, blocked_sorted, pixel_count):
    """Which member each pixel of the window belongs to: the outlet of member i carries i, a pixel carries
    what its downstream pixel carries, and a pixel that drains through a cut into another region carries
    -1.  One sweep of the order downstream first, so a pixel is reached after the pixel it flows into."""
    member = np.full(pixel_count, -1, np.int32)
    for index in range(outlet_pixels.size):
        member[outlet_pixels[index]] = index
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        if member[pixel] >= 0:
            continue                                       # an outlet of ours: it was set above
        if _is_blocked(blocked_sorted, pixel):
            continue                                       # the outlet of a member of another region
        downstream_pixel = downstream[pixel]
        if downstream_pixel >= 0:
            member[pixel] = member[downstream_pixel]
    return member


@njit(cache=True)
def check_owned_edges(order, downstream, member, dir_window, ncol, is_outlet):
    """Every pixel of this region either flows into a pixel inside the window, or is a terminal, or is the
    outlet of one of its members, where the flow leaves for the region below.  Anything else means the
    window does not hold what the tables say it holds.  Returns the pixel at fault, or -1."""
    for position in range(order.size):
        pixel = order[position]
        if member[pixel] < 0 or downstream[pixel] >= 0 or is_outlet[pixel] != 0:
            continue
        row = pixel // ncol
        code = dir_window[row, pixel - row * ncol]
        if IS_TERMINAL[code] == 0:
            return pixel
    return -1


@njit(cache=True)
def sweep_accumulate_area(order, downstream, member, areas_of_row, value, ncol, visited_of_member,
                          value_at_outlet_of_member, outlet_pixels):
    """The accumulation class: every pixel of ours starts with its own area and hands its total to the
    pixel it flows into.  The order is swept upstream first, so a pixel is complete when it is reached."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        value[pixel] += areas_of_row[row]
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel >= 0 and member[downstream_pixel] >= 0:
            value[downstream_pixel] += value[pixel]
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


@njit(cache=True)
def sweep_distance_to_outlet(order, downstream, member, lengths, value, ncol,
                             visited_of_member, farthest_of_member, head_pixel_of_member):
    """The labelling class: a pixel's distance is the distance of the pixel it flows into plus the step
    between them.  The order is swept downstream first, so the value a pixel needs is already there; an
    outlet of ours whose parent is in another region was given its value before the sweep."""
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        downstream_pixel = downstream[pixel]
        row = pixel // ncol
        if downstream_pixel >= 0 and member[downstream_pixel] >= 0:
            downstream_row = downstream_pixel // ncol
            step = _step_length(lengths, row, downstream_row - row,
                                (downstream_pixel - downstream_row * ncol) - (pixel - row * ncol))
            value[pixel] = value[downstream_pixel] + step
        visited_of_member[mine] += 1
        # the farthest pixel of the member, and on a tie the smaller row and then the smaller column,
        # which is the smaller index in this window: the rule compares the pixel's place in the rectangle
        # it read, and on a periodic grid that rectangle is unrolled, so folding the column back onto
        # the grid would pick the other pixel at the seam.  A member
        # of one pixel takes that pixel as its head
        if value[pixel] > farthest_of_member[mine] or head_pixel_of_member[mine] < 0:
            farthest_of_member[mine] = value[pixel]
            head_pixel_of_member[mine] = pixel
        elif value[pixel] == farthest_of_member[mine] and pixel < head_pixel_of_member[mine]:
            head_pixel_of_member[mine] = pixel
    return 0


@njit(cache=True)
def sweep_maximum_length(order, downstream, member, lengths, value, ncol, visited_of_member,
                         value_at_outlet_of_member, outlet_pixels):
    """The maximum class: a pixel hands its own value plus the step to the pixel it flows into, which keeps
    the largest of what arrives.  Swept upstream first; a divide keeps the zero it starts with."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        row = pixel // ncol
        downstream_row = downstream_pixel // ncol
        step = _step_length(lengths, row, downstream_row - row,
                            (downstream_pixel - downstream_row * ncol) - (pixel - row * ncol))
        candidate = value[pixel] + step
        if candidate > value[downstream_pixel]:
            value[downstream_pixel] = candidate
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


@njit(cache=True)
def sweep_shreve_magnitude(order, downstream, member, channel_window, value, ncol, is_outlet,
                           visited_of_member, sources_of_member, value_at_outlet_of_member, outlet_pixels):
    """The accumulation class on the channel network: a channel head counts 1 and every other channel
    pixel counts what arrives at it.  Swept upstream first, so what arrives at a pixel is complete when
    the pixel is reached; `value` carries the arriving sum until the pixel is finished and its own
    magnitude afterwards."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        if value[pixel] == 0:
            value[pixel] = 1                                # a channel head
            sources_of_member[mine] += 1
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        downstream_row = downstream_pixel // ncol
        if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] == 0:
            if is_outlet[pixel] == 0:
                return 3                                    # the channel mask breaks along the flow
            continue
        if value[downstream_pixel] > 4294967295 - value[pixel]:
            return 1
        value[downstream_pixel] += value[pixel]
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


@njit(cache=True)
def _first_channel_pixel_without_area(channel_window, area_window):
    """the flat index of the first channel pixel whose upstream area is not a finite positive number, -1
    when there is none (a NaN would become the best donor's area and every later comparison would be false)"""
    nrow, ncol = channel_window.shape
    for row in range(nrow):
        for column in range(ncol):
            if channel_window[row, column] != 0:
                value = area_window[row, column]
                if not (np.isfinite(value) and value > 0):
                    return row * ncol + column
    return -1


@njit(cache=True)
def sweep_main_stem_donor(order, downstream, member, channel_window, area_window, ncol,
                          best_donor, best_area):
    """Which channel pixel that flows into a channel pixel carries the most upstream area, and so keeps
    the Hack order of the pixel below.  A tie goes to the smaller index in this window, which is the
    the rule (it compares the donor's place in the rectangle it read; for two neighbours of one pixel
    the row decides, and the rectangle is at least three columns wide).  One sweep upstream first;
    nothing depends on the order here, but the sweep is the cheapest way over the member's pixels."""
    for position in range(order.size):
        pixel = order[position]
        if member[pixel] < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        downstream_row = downstream_pixel // ncol
        if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] == 0:
            continue
        area_here = area_window[row, column]
        if best_donor[downstream_pixel] < 0 or area_here > best_area[downstream_pixel]:
            best_area[downstream_pixel] = area_here
            best_donor[downstream_pixel] = pixel
        elif area_here == best_area[downstream_pixel] and pixel < best_donor[downstream_pixel]:
            best_donor[downstream_pixel] = pixel
    return 0


@njit(cache=True)
def sweep_hack_order(order, downstream, member, channel_window, value, ncol, best_donor, visited_of_member,
                     largest_of_member):
    """The labelling class on the channel network: the pixel that keeps the most upstream area keeps the
    order of the pixel below it, every other channel pixel takes one more.  Swept downstream first, so the
    order a pixel reads is already final; a member's outlet was given its order before the sweep."""
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        downstream_pixel = downstream[pixel]
        if downstream_pixel >= 0 and member[downstream_pixel] >= 0:
            downstream_row = downstream_pixel // ncol
            if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] != 0:
                if best_donor[downstream_pixel] == pixel:
                    value[pixel] = value[downstream_pixel]
                elif value[downstream_pixel] >= 255:
                    return 1                                # an order the raster cannot carry
                else:
                    value[pixel] = value[downstream_pixel] + 1
        if value[pixel] == 0:
            return 2                                        # nothing gave this channel pixel an order
        visited_of_member[mine] += 1
        if value[pixel] > largest_of_member[mine]:
            largest_of_member[mine] = value[pixel]
    return 0


@njit(cache=True)
def sweep_strahler_order(order, downstream, member, channel_window, value, ncol, is_outlet,
                         arriving_largest, arriving_second, visited_of_member, sources_of_member,
                         value_at_outlet_of_member, outlet_pixels):
    """The stream-order class, on the channel pixels only: a pixel's order is the largest order that
    arrives, plus one when that largest arrives at least twice; a channel head is 1.  Swept upstream
    first, so every order that arrives at a pixel is final when the pixel is reached."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        largest = arriving_largest[pixel]
        if largest == 0:
            here = 1                                        # a channel head
            sources_of_member[mine] += 1
        elif largest == arriving_second[pixel]:
            here = largest + 1
            if here > 255:
                return 1                                    # an order the raster cannot carry
        else:
            here = largest
        value[pixel] = here
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        downstream_row = downstream_pixel // ncol
        if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] == 0:
            if is_outlet[pixel] == 0:
                return 3                                    # the channel mask breaks along the flow
            continue
        if here > arriving_largest[downstream_pixel]:
            arriving_second[downstream_pixel] = arriving_largest[downstream_pixel]
            arriving_largest[downstream_pixel] = here
        elif here > arriving_second[downstream_pixel]:
            arriving_second[downstream_pixel] = here
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


def derive_attribute_by_sweep(attribute, partition, dir_path, out_path, table_path, member_table_path,
                              channel_path=None, area_path=None, tag=None, started=None):
    """One attribute of the labelling, maximum or stream-order class over the whole partition, computed by
    sweeping a visiting order instead of walking with a stack (section [2b]).  The products, the tables and
    the states across the cuts are those of derive_attribute; only the traversal differs."""
    spec = ATTRIBUTES[attribute]
    tag = tag or "fd3." + attribute
    started = started if started is not None else time.time()
    grid = partition.grid
    regions_in_order = partition.region_order(spec["downstream_first"])
    temporary = out_path + ".partial.tif"
    profile = raster_profile(grid, spec["dtype"], spec["nodata"])
    member_rows = []
    states = {}                      # member_id -> the value that crosses its cut, in full precision
    # which visit writes each block of the raster last: a block no later region will touch goes out as
    # soon as this region has filled it, the others are held in memory until their last region has
    last_writer = np.full(((grid.nrow + RASTER_BLOCK - 1) // RASTER_BLOCK,
                           (grid.ncol + RASTER_BLOCK - 1) // RASTER_BLOCK), -1, np.int32)
    for visit, region_id in enumerate(regions_in_order):
        rectangle = partition.region_rectangle(region_id)
        # the same walk write_back does, so that the block a region is charged with is the block it
        # writes: the columns cut at the seam, the blocks taken inside the grid
        remaining = rectangle[3] - rectangle[2] + 1
        column = rectangle[2]
        while remaining > 0:
            grid_column = (column % grid.ncol) if grid.periodic else column
            length = min(remaining, grid.ncol - grid_column)
            for block_col in range(grid_column // RASTER_BLOCK,
                                   (grid_column + length - 1) // RASTER_BLOCK + 1):
                for block_row in range(rectangle[0] // RASTER_BLOCK, rectangle[1] // RASTER_BLOCK + 1):
                    last_writer[block_row, block_col] = visit
            column += length
            remaining -= length
    held = {}                        # the blocks two regions share, until the last of them has written
    # every dataset opened is closed on any error too, not on the normal path only, and
    # the channel mask and the upstream area are checked to lie on the grid of the flow directions before they are
    # read by row and column
    opened = []
    try:
        dir_dataset = rasterio.open(dir_path)
        opened.append(dir_dataset)
        out_dataset = rasterio.open(temporary, "w", **dict(profile, BIGTIFF="YES"))
        opened.append(out_dataset)
        channel_dataset = rasterio.open(channel_path) if spec["channel"] else None
        if channel_dataset is not None:
            opened.append(channel_dataset)
        area_dataset = rasterio.open(area_path) if spec["area"] else None
        if area_dataset is not None:
            opened.append(area_dataset)
        _check_on_the_grid_of_the_flow_directions(dir_dataset, dir_path, channel_dataset, area_dataset)
    except BaseException:
        for dataset in opened:
            dataset.close()
        raise
    try:
        for visit, region_id in enumerate(regions_in_order):
            members = list(partition.members_by_region.get(region_id, []))
            rectangle = partition.region_rectangle(region_id)
            work_dtype = "float64" if spec["dtype"] == "float32" else spec["dtype"]
            # the pixels an earlier region may have written in this window, and nothing else is read back
            # from the output raster.  Which pixels they are depends on the direction the class travels:
            # a class handed down from the divides (shv, ord) finds the value at the outlet of a child piece
            # that lies in another region; a class handed up from the outlet (hck) finds it at the outlet of
            # one of this region's own members, written there by the region of its parent.  ldn, lup and the
            # accumulation carry their state in memory and read nothing.
            in_this_region = {member.member_id for member in members}
            # the structure indexes the window with int32; a window past that would wrap
            # round silently, and the test comes before the rectangle is read so that nothing large is
            # allocated first
            if (rectangle[1] - rectangle[0] + 1) * (rectangle[3] - rectangle[2] + 1) > 2147483647:
                raise FlowDivideError("%s: the window of region %d holds %d pixels, more than the int32 the "
                                      "visiting order is indexed with; build the partition at a smaller capacity"
                                      % (attribute, region_id,
                                         (rectangle[1] - rectangle[0] + 1) * (rectangle[3] - rectangle[2] + 1)))
            clock = time.time()
            window = RegionWindow(grid, rectangle, dir_dataset, out_dataset, work_dtype, spec["nodata"],
                                  channel_dataset, area_dataset)
            seconds_reading = time.time() - clock
            clock = time.time()
            on_the_channel = attribute in ("shv", "hck", "ord")
            downstream, order, taken, broken = window_flow_structure(
                window.dir, window.channel if on_the_channel else np.zeros((1, 1), np.uint8), on_the_channel)
            seconds_building = time.time() - clock
            if taken < 0:
                raise FlowDivideError("%s: the flow directions of region %d hold a cycle" % (attribute, region_id))
            mask_breaks_somewhere = broken >= 0
            pixel_count = window.nrow * window.ncol
            outlet_pixels = np.empty(len(members), np.int32)
            for index, member in enumerate(members):
                row, col = window.local(grid, member.outlet_row, member.outlet_col)
                outlet_pixels[index] = row * window.ncol + col
            # the distance and the upstream flow length cover the basins below the table's
            # area too.  Each is one whole member of this region, kept as arrays; their outlets follow the members' in
            # the same array, so the labelling and the sweep take them as members, and nothing crosses a cut from them
            small_start, small_stop = (0, 0)
            if attribute in ("ldn", "lup"):
                small_start, small_stop = partition.small_by_region.get(region_id, (0, 0))
            small_count = small_stop - small_start
            if small_count:
                small_local_row, small_local_col = local_position(grid, window.rectangle,
                                                                  partition.small_outlet_row[small_start:small_stop],
                                                                  partition.small_outlet_col[small_start:small_stop])
                outside = ((small_local_row < 0) | (small_local_row >= window.nrow) |
                           (small_local_col < 0) | (small_local_col >= window.ncol))
                if outside.any():
                    first = int(np.argmax(outside))
                    raise FlowDivideError("%s: the outlet of basin %d lies outside the window of its region %d"
                                          % (attribute, int(partition.small_basin_id[small_start + first]), region_id))
                outlet_pixels = np.concatenate([outlet_pixels,
                                                (small_local_row * window.ncol + small_local_col).astype(np.int32)])
                del small_local_row, small_local_col, outside
            # the outlets of children that lie in another region: their pixels are not ours, and the
            # value they leave is read as a state (a child in this region needs nothing, it is in the order)
            arriving = []            # (the child, its outlet pixel, the pixel it flows into)
            for member in members:
                for child in member.children:
                    if child.member_id in in_this_region:
                        continue
                    outlet_row, outlet_col = window.local(grid, child.outlet_row, child.outlet_col)
                    inlet_row, inlet_col = window.local(grid, child.parent_inlet_row, child.parent_inlet_col)
                    arriving.append((child, outlet_row * window.ncol + outlet_col, inlet_row * window.ncol + inlet_col))
            blocked_sorted = np.asarray(sorted(pixel for _, pixel, _ in arriving), np.int64)
            clock = time.time()
            member_of_pixel = window_member_of_pixel(order, downstream, outlet_pixels, blocked_sorted, pixel_count)
            seconds_labelling = time.time() - clock
            clock = time.time()
            value = window.out.reshape(pixel_count)
            visited_of_member = np.zeros(len(outlet_pixels), np.int64)     # the members, then the small basins
            ours = member_of_pixel >= 0
            is_outlet = np.zeros(pixel_count, np.uint8)
            is_outlet[outlet_pixels] = 1
            # a break in the channel mask is answered for when it drains to one of this region's
            # members; the walk that decides it is only made when the structure met a break at all
            if mask_breaks_somewhere:
                broken_of_ours = channel_mask_break_of_ours(window.dir, window.channel, is_outlet,
                                                            blocked_sorted, window.ncol)
                if broken_of_ours >= 0:
                    raise FlowDivideError("%s: the channel mask breaks along the flow at row %d column %d of region "
                                          "%d: the pixel is a channel pixel and the pixel it flows into is not, so "
                                          "everything above it would be left out"
                                          % (attribute, broken_of_ours // window.ncol + window.row0,
                                             broken_of_ours % window.ncol + window.col0, region_id))
            at_fault = check_owned_edges(order, downstream, member_of_pixel, window.dir, window.ncol, is_outlet)
            if at_fault >= 0:
                raise FlowDivideError("%s: the pixel at row %d column %d of region %d flows out of its "
                                      "window and is neither a terminal nor the outlet of a member"
                                      % (attribute, at_fault // window.ncol + window.row0,
                                         at_fault % window.ncol + window.col0, region_id))
            if attribute == "ldn":
                value[ours] = 0.0
                for index, member in enumerate(members):
                    if member.parent_member_id and member.parent_member_id not in in_this_region:
                        if member.member_id not in states:
                            raise FlowDivideError("%s: the state of member %d was not left by an earlier region"
                                                  % (attribute, member.member_id))
                        pixel = int(outlet_pixels[index])
                        row = pixel // window.ncol
                        inlet_row, inlet_col = window.local(grid, member.parent_inlet_row, member.parent_inlet_col)
                        step = _step_length(window.lengths, row, inlet_row - row,
                                            inlet_col - (pixel - row * window.ncol))
                        value[pixel] = states[member.member_id] + step
                farthest_of_member = np.zeros(len(outlet_pixels), np.float64)
                head_pixel_of_member = np.full(len(outlet_pixels), -1, np.int64)
                sweep_distance_to_outlet(order, downstream, member_of_pixel, window.lengths, value,
                                         window.ncol,
                                         visited_of_member, farthest_of_member, head_pixel_of_member)
            elif attribute == "lup":
                value[ours] = 0.0
                for child, outlet_pixel, inlet_pixel in arriving:
                    if child.member_id not in states:
                        raise FlowDivideError("%s: the state of member %d was not left by an earlier region"
                                              % (attribute, child.member_id))
                    outlet_row = outlet_pixel // window.ncol
                    inlet_row = inlet_pixel // window.ncol
                    step = _step_length(window.lengths, outlet_row, inlet_row - outlet_row,
                                        (inlet_pixel - inlet_row * window.ncol) - (outlet_pixel - outlet_row * window.ncol))
                    candidate = states[child.member_id] + step
                    if candidate > value[inlet_pixel]:
                        value[inlet_pixel] = candidate
                value_at_outlet_of_member = np.zeros(len(outlet_pixels), np.float64)
                sweep_maximum_length(order, downstream, member_of_pixel, window.lengths, value, window.ncol,
                                     visited_of_member, value_at_outlet_of_member, outlet_pixels)
            elif attribute == "ord":
                channel = window.channel.reshape(pixel_count)
                arriving_largest = np.zeros(pixel_count, np.uint8)
                arriving_second = np.zeros(pixel_count, np.uint8)
                for child, outlet_pixel, inlet_pixel in arriving:
                    if channel[outlet_pixel] == 0:
                        continue
                    if child.member_id not in states:
                        raise FlowDivideError("ord: the order of member %d was not left by an earlier "
                                              "region" % child.member_id)
                    came = np.uint8(states[child.member_id])    # what the earlier region left at the cut
                    if came == 0:
                        raise FlowDivideError("%s: the order at a cut of region %d is zero" % (attribute, region_id))
                    if came > arriving_largest[inlet_pixel]:
                        arriving_second[inlet_pixel] = arriving_largest[inlet_pixel]
                        arriving_largest[inlet_pixel] = came
                    elif came > arriving_second[inlet_pixel]:
                        arriving_second[inlet_pixel] = came
                sources_of_member = np.zeros(len(members), np.int64)
                value_at_outlet_of_member = np.zeros(len(members), np.int64)
                status = sweep_strahler_order(order, downstream, member_of_pixel, window.channel, value,
                                              window.ncol, is_outlet, arriving_largest, arriving_second,
                                              visited_of_member, sources_of_member,
                                              value_at_outlet_of_member, outlet_pixels)
                if status == 3:
                    raise FlowDivideError("ord: a channel pixel of region %d flows into a pixel of the same "
                                          "region that carries no channel" % region_id)
                if status != 0:
                    raise FlowDivideError("ord: an order of region %d does not fit the raster" % region_id)
            elif attribute == "shv":
                channel = window.channel.reshape(pixel_count)
                for child, outlet_pixel, inlet_pixel in arriving:  # what an earlier region left at a cut
                    if channel[outlet_pixel] == 0:
                        continue
                    if child.member_id not in states:
                        raise FlowDivideError("shv: the magnitude of member %d was not left by an earlier "
                                              "region" % child.member_id)
                    value[inlet_pixel] += states[child.member_id]
                sources_of_member = np.zeros(len(members), np.int64)
                value_at_outlet_of_member = np.zeros(len(members), np.int64)
                status = sweep_shreve_magnitude(order, downstream, member_of_pixel, window.channel, value,
                                                window.ncol, is_outlet, visited_of_member, sources_of_member,
                                                value_at_outlet_of_member, outlet_pixels)
                if status == 3:
                    raise FlowDivideError("shv: a channel pixel of region %d flows into a pixel of the same "
                                          "region that carries no channel" % region_id)
                if status != 0:
                    raise FlowDivideError("shv: the sweep of region %d ended with status %d" % (region_id, status))
            elif attribute == "hck":
                channel = window.channel.reshape(pixel_count)
                bad_pixel = int(_first_channel_pixel_without_area(window.channel, window.area))
                if bad_pixel >= 0:
                    raise FlowDivideError("%s: region %d: a channel pixel has an upstream area that is not a finite "
                                          "positive number" % (tag, region_id))
                area = window.area.reshape(pixel_count)
                best_donor = np.full(pixel_count, -1, np.int64)
                best_area = np.zeros(pixel_count, np.float32)
                sweep_main_stem_donor(order, downstream, member_of_pixel, window.channel, window.area,
                                      window.ncol, best_donor, best_area)
                # a child's outlet is a donor of the pixel it flows into, and may well be its main stem
                for _, outlet_pixel, inlet_pixel in arriving:
                    if channel[outlet_pixel] == 0 or channel[inlet_pixel] == 0:
                        continue
                    area_there = area[outlet_pixel]
                    if best_donor[inlet_pixel] < 0 or area_there > best_area[inlet_pixel]:
                        best_area[inlet_pixel] = area_there
                        best_donor[inlet_pixel] = outlet_pixel
                    elif area_there == best_area[inlet_pixel] and outlet_pixel < best_donor[inlet_pixel]:
                        # the same tie as the sweep above: the smaller index in this window
                        best_donor[inlet_pixel] = outlet_pixel
                for index, member in enumerate(members):           # the order a member's outlet starts from
                    pixel = int(outlet_pixels[index])
                    if channel[pixel] == 0:
                        continue
                    if member.parent_member_id == 0:
                        value[pixel] = 1                           # the outlet of a whole basin
                    elif member.parent_member_id not in in_this_region:
                        if member.member_id not in states:
                            raise FlowDivideError("hck: the order of member %d was not left by the region "
                                                  "of its parent" % member.member_id)
                        value[pixel] = states[member.member_id]
                largest_of_member = np.zeros(len(members), np.int64)
                status = sweep_hack_order(order, downstream, member_of_pixel, window.channel, value,
                                          window.ncol, best_donor, visited_of_member, largest_of_member)
                if status == 1:
                    raise FlowDivideError("hck: an order of region %d does not fit the raster" % region_id)
                if status != 0:
                    raise FlowDivideError("hck: a channel pixel of region %d was given no order" % region_id)
                for child, outlet_pixel, inlet_pixel in arriving:   # the order the child's region reads
                    if channel[outlet_pixel] == 0 or channel[inlet_pixel] == 0:
                        continue
                    states[child.member_id] = int(value[inlet_pixel] if best_donor[inlet_pixel] == outlet_pixel
                                                  else value[inlet_pixel] + 1)
            else:
                raise FlowDivideError("the ordered sweep does not carry the attribute '%s'" % attribute)
            seconds_sweeping = time.time() - clock
            clock = time.time()
            # the states this region leaves for the regions below it, and the rows of its members
            for index, member in enumerate(members):
                if attribute == "ldn":
                    for child in member.children:
                        if child.member_id in in_this_region:
                            continue
                        inlet_row, inlet_col = window.local(grid, child.parent_inlet_row, child.parent_inlet_col)
                        states[child.member_id] = float(value[inlet_row * window.ncol + inlet_col])
                elif attribute == "lup":
                    states[member.member_id] = float(value_at_outlet_of_member[index])
                elif attribute in ("shv", "ord"):
                    states[member.member_id] = int(value_at_outlet_of_member[index])
                expected = member.pixel_count if attribute in ("ldn", "lup") else None
                if expected is not None and visited_of_member[index] != expected:
                    raise FlowDivideError("%s: member %d took %d pixels of the window, the table says %d"
                                          % (attribute, member.member_id, visited_of_member[index], expected))
                # a class that lives on the channel network takes a part of the member, so its count is a
                # bound and not an equality -- checked as one
                if expected is None and visited_of_member[index] > member.pixel_count:
                    raise FlowDivideError("%s: member %d took %d pixels of the window, more than the %d the "
                                          "table gives it" % (attribute, member.member_id,
                                                              visited_of_member[index], member.pixel_count))
                row = {"member_id": member.member_id, "basin_id": member.basin_id, "region_id": region_id,
                       "pixels_visited": int(visited_of_member[index])}
                if attribute == "ldn":
                    head_pixel = int(head_pixel_of_member[index])
                    head_row = head_pixel // window.ncol + window.row0 if head_pixel >= 0 else -1
                    head_col = head_pixel - (head_pixel // window.ncol) * window.ncol + window.col0 if head_pixel >= 0 else -1
                    row.update({"farthest_metres": float(farthest_of_member[index]), "head_row": head_row,
                                "head_col": head_col})
                elif attribute == "lup":
                    row.update({"value_at_outlet": float(value_at_outlet_of_member[index])})
                elif attribute in ("ord", "shv"):
                    row.update({"sources": int(sources_of_member[index]),
                                "value_at_outlet": int(value_at_outlet_of_member[index])})
                elif attribute == "hck":
                    row.update({"largest_order": int(largest_of_member[index])})
                member_rows.append(row)
            # every small basin took exactly its pixels, and with every basin covered the members and the small
            # basins together took every pixel the region raster gives this region
            if small_count:
                wrong = np.nonzero(visited_of_member[len(members):] != partition.small_pixel_count[small_start:small_stop])[0]
                if len(wrong):
                    first = int(wrong[0])
                    raise FlowDivideError("%s: basin %d took %d pixels of the window, the table says %d"
                                          % (attribute, int(partition.small_basin_id[small_start + first]),
                                             int(visited_of_member[len(members) + first]),
                                             int(partition.small_pixel_count[small_start + first])))
            if attribute in ("ldn", "lup") and partition.raster_covers_every_basin:
                taken_in_region = int(ours.sum())
                if taken_in_region != int(partition.regions[region_id].region_grid_count):
                    raise FlowDivideError("%s: region %d: its members and basins took %d pixels, the region table gives it %d"
                                          % (attribute, region_id, taken_in_region, int(partition.regions[region_id].region_grid_count)))
            # only the block rows this region has a pixel in are touched, and each of them keeps what an
            # earlier region wrote in it.  The Hack order also writes the outlet of a child piece in another
            # region, which is not one of ours, so those pixels join the mask
            mine = ours.reshape(window.nrow, window.ncol)
            written, holding = window.write_back(grid, out_dataset, spec["dtype"], mine, last_writer,
                                                 visit, held)
            seconds_writing = time.time() - clock
            clock = time.time()
            # every array of the window goes before the next region reads its own.  `value` is a view of the
            # window's output, `channel` of its mask, and with `ours`, `mine` and `is_outlet` they would keep the
            # last window's arrays beside the next one
            del window, downstream, order, member_of_pixel, value, ours, mine, is_outlet
            channel = area = best_donor = best_area = arriving_largest = arriving_second = None
            seconds_freeing = time.time() - clock
            log(tag, "region %d (%d of %d): %d members and %d small basins, %d x %d window, %d blocks written, %d held "
                     "(%d in memory); read %.1f s, order %.1f s, members %.1f s, swept %.1f s, wrote %.1f s, freed %.1f s"
                % (region_id, visit + 1, len(regions_in_order), len(members), small_count,
                   rectangle[1] - rectangle[0] + 1, rectangle[3] - rectangle[2] + 1, written, holding, len(held),
                   seconds_reading, seconds_building, seconds_labelling, seconds_sweeping, seconds_writing,
                   seconds_freeing))
        # the rows in the order the members are worked in, which is the order the tables have always had
        depth_of_member = {member.member_id: member.downstream_depth for member in partition.members.values()}
        member_rows.sort(key=lambda row: ((depth_of_member[row["member_id"]] if ATTRIBUTES[attribute]["downstream_first"]
                                           else -depth_of_member[row["member_id"]]), row["member_id"]))
        # a block whose last region by rectangle turned out to have no pixel in it is still held, and is
        # final: nothing else will touch it.  That region writes it, so nothing should be
        # left here; the loop stays so that a block is never lost
        if held:
            for (block_row, block_col), block_values in held.items():
                values = block_values.astype(spec["dtype"]) if str(block_values.dtype) != spec["dtype"] else block_values
                write_block(out_dataset, values, block_row * RASTER_BLOCK, block_col * RASTER_BLOCK, grid.periodic)
            log(tag, "%d blocks held to the end were written there" % len(held))
            held.clear()
    except BaseException:
        for dataset in opened:
            dataset.close()
        raise
    log(tag, "the regions are done %.1f s into the run" % (time.time() - started))
    clock = time.time()
    dir_dataset.close()
    seconds_closing_inputs = time.time() - clock
    clock = time.time()
    out_dataset.close()
    seconds_closing_output = time.time() - clock
    for dataset in (channel_dataset, area_dataset):
        if hasattr(dataset, "close"):
            dataset.close()
    log(tag, "closing the flow directions took %.1f s and the output raster %.1f s"
        % (seconds_closing_inputs, seconds_closing_output))
    member_table = pd.DataFrame(member_rows, columns=MEMBER_TABLE_COLUMNS[attribute])
    basin_table = _basin_table_from_members(attribute, partition, member_table)
    publish(temporary, out_path)
    write_table(member_table, member_table_path, float_format="%.17g")
    write_basin_table_as_the_c_code_does(basin_table, table_path)
    report = {"attribute": attribute, "regions_visited": len(regions_in_order), "members": len(member_rows),
              "basins": len(basin_table), "traversal": "order", "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s and %s: %d regions, %d members, %d basins, %d seconds (swept in order)"
        % (out_path, table_path, len(regions_in_order), len(member_rows), len(basin_table), int(report["seconds"])))
    return report


# =============================================================================
#  [3] The driver: read a region, walk its members, write it back
# =============================================================================

ATTRIBUTES = {
    "shv": {"name": "Shreve magnitude", "dtype": "uint32", "nodata": 0, "downstream_first": False, "channel": True, "area": False, "table": "shreve"},
    # the distance's nodata is -1; the upstream flow length keeps -9999
    "ldn": {"name": "distance to the outlet (m)", "dtype": "float32", "nodata": -1.0, "downstream_first": True, "channel": False, "area": False, "table": "distance_to_outlet"},
    "hck": {"name": "Hack order", "dtype": "uint8", "nodata": 0, "downstream_first": True, "channel": True, "area": True, "table": "hack"},
    "lup": {"name": "upstream flow length (m)", "dtype": "float32", "nodata": -9999.0, "downstream_first": False, "channel": False, "area": False, "table": "lup"},
    "lfp": {"name": "longest flow path", "dtype": "uint32", "nodata": 0, "downstream_first": True, "channel": False, "area": False, "table": "lfp"},
    "ord": {"name": "Strahler order", "dtype": "uint8", "nodata": 0, "downstream_first": False, "channel": True, "area": False, "table": "strahler"},
}


BASE_COLUMNS = ["basin_id", "outlet_row", "outlet_col", "outlet_lon", "outlet_lat", "basin_area_km2"]
BASIN_TABLE_COLUMNS = {
    "shv": BASE_COLUMNS + ["magnitude_at_outlet", "source_grid_count", "river_grid_count"],
    "ldn": ["basin_id", "outlet_row", "outlet_col", "basin_area_km2", "measured_grid_count", "farthest_metres"],
    "hck": BASE_COLUMNS + ["largest_order", "river_grid_count"],
    "lup": BASE_COLUMNS + ["lup_at_outlet_m", "written_grid_count"],
    "lfp": BASE_COLUMNS + ["longest_flow_path_m", "path_grid_count", "head_row", "head_col", "head_lon", "head_lat"],
    "ord": BASE_COLUMNS + ["order_at_outlet", "river_grid_count"],
}
# how every column of the six basin tables is printed, the 30 m and the MERIT ones alike, so that a run gives the
# published tables byte for byte
BASIN_TABLE_FORMAT = {
    "basin_id": "%d", "outlet_row": "%d", "outlet_col": "%d", "outlet_lon": "%.7f", "outlet_lat": "%.7f", "basin_area_km2": "%.6f",
    "magnitude_at_outlet": "%d", "source_grid_count": "%d", "river_grid_count": "%d",
    "measured_grid_count": "%d", "farthest_metres": "%.3f",
    "largest_order": "%d", "order_at_outlet": "%d",
    "lup_at_outlet_m": "%.2f", "written_grid_count": "%d",
    "longest_flow_path_m": "%.3f", "path_grid_count": "%d", "head_row": "%d", "head_col": "%d", "head_lon": "%.7f", "head_lat": "%.7f",
}


def write_basin_table_as_the_c_code_does(table, path):
    """a basin table with its header and its format column by column, published when complete"""
    formats = [BASIN_TABLE_FORMAT[column] for column in table.columns]
    line_format = " ".join(formats) + "\n"
    temporary = path + ".partial"
    with timing("output"):
        with open(temporary, "w") as handle:
            handle.write(" ".join(table.columns) + "\n")
            for values in table.itertuples(index=False, name=None):
                handle.write(line_format % tuple(values))
    publish(temporary, path)


MEMBER_BASE = ["member_id", "basin_id", "region_id", "pixels_visited"]
MEMBER_TABLE_COLUMNS = {
    "shv": MEMBER_BASE + ["sources", "value_at_outlet"], "ldn": MEMBER_BASE + ["farthest_metres", "head_row", "head_col"], "hck": MEMBER_BASE + ["largest_order"],
    "lup": MEMBER_BASE + ["value_at_outlet"], "ord": MEMBER_BASE + ["sources", "value_at_outlet"], "lfp": ["member_id", "basin_id", "region_id", "path_grid_count", "length_m"],
}


class RegionWindow:
    """the arrays of one region: the flow directions, the output (read back from the raster), and the
    channel mask and upstream area when the attribute needs them.  Each array is read straight into
    its own buffer (no second copy of a window that may hold two billion pixels); on a periodic grid a
    piece of columns past the seam is read in strips of rows into its place; the output is written
    back in strips of rows, converted strip by strip."""

    def __init__(self, grid, rectangle, dir_dataset, out_dataset, out_dtype, out_nodata,
                 channel_dataset=None, area_dataset=None, read_out_at=None, read_out_whole=False):
        """read_out_at: the pixels of the output raster an earlier region may have written, as local pixel
        indices (the outlets of the child pieces that lie in another region).  Only those are read; the
        rest of the output window starts at its nodata, and only the blocks this region writes go back.
        Reading the whole output window again cost 7 GB of traffic a region on a continent and gave back
        nothing but those few pixels (measured)."""
        self.rectangle = rectangle
        self.row0 = rectangle[0]
        self.col0 = rectangle[2]
        self.nrow = rectangle[1] - rectangle[0] + 1
        self.ncol = rectangle[3] - rectangle[2] + 1
        self.grid = grid
        # the type on the file, before it is read into bytes: a Float32 1.4 read into uint8 would come out as
        # the direction 1 and pass the check of the codes
        if dir_dataset.dtypes[0] != "uint8":
            raise FlowDivideError("the flow directions are %s, not uint8" % dir_dataset.dtypes[0])
        self.dir = self._read(dir_dataset, MERIT_NODATA, "uint8")
        check_merit_flow_directions(self.dir, "the window %s" % (rectangle,))
        if read_out_whole:
            self.out = self._read(out_dataset, out_nodata, out_dtype)   # a driver that writes the whole
        else:                                                           # window back needs what is there
            self.out = np.full((self.nrow, self.ncol), out_nodata, dtype=out_dtype)
        self.out_nodata = out_nodata
        if read_out_at is not None and not read_out_whole:
            for pixel in read_out_at:
                row = int(pixel) // self.ncol
                col = int(pixel) - row * self.ncol
                column_in_grid = (self.col0 + col) % grid.ncol if grid.periodic else self.col0 + col
                one = out_dataset.read(1, window=Window(column_in_grid, self.row0 + row, 1, 1))
                self.out[row, col] = one[0, 0]
        self.channel = self._read(channel_dataset, 0, "uint8") if channel_dataset is not None else None
        self.area = self._read(area_dataset, 0, "float32") if area_dataset is not None else None      # Float32 (4 bytes a pixel)
        self.lengths = grid.row_step_lengths_m(self.row0, self.nrow)

    def _read(self, dataset, fill, dtype):
        """the window as one array of the wanted dtype: read into a preallocated buffer, in pieces of
        columns on a periodic grid (the rows always lie inside the grid)"""
        grid = self.grid
        out = np.full((self.nrow, self.ncol), fill, dtype=dtype)
        col = self.col0
        while col < self.col0 + self.ncol:
            if grid.periodic:
                wrapped = col % grid.ncol
                length = min(grid.ncol - wrapped, self.col0 + self.ncol - col)
            else:
                wrapped = col
                length = self.col0 + self.ncol - col
            target = out[:, col - self.col0:col - self.col0 + length]
            if target.flags["C_CONTIGUOUS"]:
                dataset.read(1, window=Window(wrapped, self.row0, length, self.nrow), out=target)
            else:
                strip_rows = max(1, 50000000 // max(length, 1))
                for row in range(0, self.nrow, strip_rows):
                    count = min(strip_rows, self.nrow - row)
                    target[row:row + count] = dataset.read(1, window=Window(wrapped, self.row0 + row, length, count), out_dtype=dtype)
            col += length
        return out

    def local(self, grid, row, col):
        local_row, local_col = local_position(grid, self.rectangle, row, col)
        if local_row < 0 or local_row >= self.nrow or local_col < 0 or local_col >= self.ncol:
            raise FlowDivideError("the pixel (%d, %d) lies outside the window of its region" % (row, col))
        return local_row, local_col

    def column_spans(self, grid):
        """the window's columns as pieces that each lie inside the grid: (the grid column a piece starts
        at, how many columns, where the piece starts in the window).  A window of a periodic grid may run
        past the last column and come back at column 0, and the grid's last block is narrower than the
        others (432,000 columns is not a multiple of 512), so stepping across the seam by the block width
        puts the two out of step: the columns are cut at the seam first and the blocks taken inside each
        piece (otherwise the Shreve sweep of the 90 m grid stops at the first region whose window
        wraps, writing into a target of no columns)."""
        spans = []
        remaining, column, offset = self.ncol, self.col0, 0
        while remaining > 0:
            grid_column = (column % grid.ncol) if grid.periodic else column
            length = min(remaining, grid.ncol - grid_column)
            if length <= 0:
                raise FlowDivideError("the window column %d is outside the grid" % column)
            spans.append((grid_column, length, offset))
            column += length
            offset += length
            remaining -= length
        return spans

    def blocks_of_the_window(self, grid, block=RASTER_BLOCK):
        """every block of the raster this window touches: its index and its own rows and columns in the
        grid, and where the part of the window that falls in it sits in the window and in the block.  A
        block is always taken whole (clipped only by the grid), so that two regions that share it hold and
        write the same shape."""
        for block_row in range((self.row0 // block) * block, self.row0 + self.nrow, block):
            block_rows = min(block_row + block, grid.nrow) - block_row
            row_from = max(block_row, self.row0)
            rows = min(block_row + block_rows, self.row0 + self.nrow) - row_from
            if rows <= 0:
                continue
            for grid_column, length, offset in self.column_spans(grid):
                for block_col in range((grid_column // block) * block, grid_column + length, block):
                    block_cols = min(block_col + block, grid.ncol) - block_col
                    col_from = max(block_col, grid_column)
                    columns = min(block_col + block_cols, grid_column + length) - col_from
                    if columns <= 0:
                        continue
                    yield ((block_row // block, block_col // block), block_row, block_col,
                           block_rows, block_cols,
                           row_from - self.row0, offset + (col_from - grid_column),
                           row_from - block_row, col_from - block_col, rows, columns)

    def write_back(self, grid, out_dataset, dtype, mine=None, last_writer=None, visit=None, held=None):
        """The region's pixels written into a raster that is still being created, so that nothing is ever
        read back: in create mode GDAL compresses a block inside the write call, while in update mode it
        keeps the block dirty until the close, which on basin 4 cost 59 to 79 s a region set.
        A block a later region will also write is held whole in memory until that region has put its
        pixels in; `last_writer` says which visit writes each block last.  With mine None the whole window
        goes back in one call, which is what the drivers that own their whole window do."""
        if mine is None:
            values = self.out
            write_block(out_dataset, values.astype(dtype) if str(values.dtype) != dtype else values,
                        self.row0, self.col0, grid.periodic)
            return 1, 0
        written = holding = 0
        pieces = list(self.blocks_of_the_window(grid))
        # a window that runs round a periodic grid can reach one block twice (a grid narrower than a block, or
        # a window within a block of the full circle), so a block goes out or is held at its last appearance only, and
        # is filled at every one of them (a fresh block would otherwise be written over the first)
        last_appearance = {piece[0]: position for position, piece in enumerate(pieces)}
        filling = {}
        for position, (index, block_row, block_col, block_rows, block_cols,
                       row_in_window, col_in_window, row_in_block, col_in_block, rows, columns) in enumerate(pieces):
            here = mine[row_in_window:row_in_window + rows, col_in_window:col_in_window + columns]
            block_values = filling.pop(index, None)
            if here.any():
                if block_values is None and held is not None:
                    block_values = held.pop(index, None)
                if block_values is None:
                    # the block is kept in the raster's own type, as it is written: each pixel is rounded
                    # once, when it is put in, as the conversion at the write rounded it, and a held block of the
                    # distance or the upstream flow length takes 1 MB instead of 2
                    block_values = np.full((block_rows, block_cols), self.out_nodata, dtype=dtype)
                target = block_values[row_in_block:row_in_block + rows, col_in_block:col_in_block + columns]
                if target.shape != here.shape:
                    raise FlowDivideError("block %s of the window at (%d, %d): the part that falls in it is "
                                          "%s in the window and %s in the block"
                                          % (index, self.row0, self.col0, here.shape, target.shape))
                target[here] = self.out[row_in_window:row_in_window + rows,
                                        col_in_window:col_in_window + columns][here]
            elif block_values is None:
                # no pixel of this region in the block.  A block held for this region is final all the same
                # (no later window reaches it), so it goes out now; kept to the end of the run, such blocks add up
                # (58,832 for the Shreve magnitude on North America at 2^30, and the distance to the outlet then
                # runs out of memory)
                if (held is None or last_writer is None or last_writer[index] != visit
                        or last_appearance[index] != position or index not in held):
                    continue
                block_values = held.pop(index)
            if last_appearance[index] != position:
                filling[index] = block_values                  # the window reaches this block again further on
                continue
            if last_writer is None or last_writer[index] == visit:
                write_block(out_dataset,
                            block_values.astype(dtype) if str(block_values.dtype) != dtype else block_values,
                            block_row, block_col, grid.periodic)
                written += 1
            else:
                held[index] = block_values
                holding += 1
        return written, holding


def _members_in_walk_order(members, downstream_first):
    """inside a region: the member holding the basin's outlet first for a value handed up (ascending
    depth), the most upstream piece first for a value handed down (descending depth)"""
    return sorted(members, key=lambda m: (m.downstream_depth if downstream_first else -m.downstream_depth, m.member_id))


def _blocked_pixels(grid, window, member):
    """the outlets of the member's children as sorted local pixel indices, and the children in that order"""
    pairs = []
    for child in member.children:
        row, col = window.local(grid, child.outlet_row, child.outlet_col)
        if 0 <= row < window.nrow and 0 <= col < window.ncol:
            pairs.append((row * window.ncol + col, child))
        else:
            raise FlowDivideError("the outlet of piece %d lies outside the window of its parent's region" % child.member_id)
    pairs.sort(key=lambda pair: pair[0])
    return np.asarray([pair[0] for pair in pairs], np.int64), [pair[1] for pair in pairs]


def derive_attribute(attribute, partition, dir_path, out_path, table_path, member_table_path, channel_path=None, area_path=None, tag=None, ldn_member_table=None, lines_path=None):
    """One attribute over the whole partition: the raster <out_path> on the grid of DIR, the table
    <table_path> with one row per basin (the columns of the <name>_basin table) and the table
    <member_table_path> with one row per member.  channel_path is needed for shv, hck and ord,
    area_path (the upstream area) for hck.  For lfp, the per-member heads of a ldn run (ldn_member_table)
    are used when given, otherwise the distance walk is run first without writing its raster; the
    paths as lines go to lines_path."""
    if attribute not in ATTRIBUTES:
        raise FlowDivideError("unknown attribute '%s'" % attribute)
    spec = ATTRIBUTES[attribute]
    tag = tag or "fd3." + attribute
    grid = partition.grid
    started = time.time()
    if spec["channel"] and channel_path is None:
        raise FlowDivideError("%s needs the channel mask" % attribute)
    if spec["area"] and area_path is None:
        raise FlowDivideError("%s needs the upstream area" % attribute)
    # the distance to the outlet and the upstream flow length cover the basins below the tables' area too, so
    # with only such basins there is still a raster to make (the table then holds its header alone)
    covers_small_basins = attribute in ("ldn", "lup") and partition.small_basin_id.size > 0
    if not partition.basins and not covers_small_basins:
        raise FlowDivideError("no basin reaches the area the attributes are computed for; nothing to compute")
    if attribute in SWEPT_ATTRIBUTES:
        return derive_attribute_by_sweep(attribute, partition, dir_path, out_path, table_path,
                                         member_table_path, channel_path, area_path, tag, started)
    if attribute == "lfp":
        return _derive_longest_flow_path(partition, dir_path, out_path, table_path, member_table_path, lines_path, tag, ldn_member_table, started)
    raise FlowDivideError("the attribute '%s' has no traversal: every class but the longest flow path is swept in order (section [2b])" % attribute)

class _nothing:
    """a stand-in for a raster that is not needed, usable in a with statement"""

    def __enter__(self):
        return None

    def __exit__(self, *exception):
        return False


def _basin_table_from_members(attribute, partition, member_table):
    """one row per basin from the rows of its members, with the columns of the basin tables: the
    value at the basin's outlet member, the sums and the maxima over the members"""
    rows = []
    if len(member_table) == 0:
        return pd.DataFrame(rows, columns=BASIN_TABLE_COLUMNS[attribute])
    row_of_member = {int(row.member_id): row for row in member_table.itertuples(index=False)}
    for basin_id in sorted(partition.basins):
        basin = partition.basins[basin_id]
        if basin["area_km2"] < partition.table_min_basin_area_km2:
            continue                  # a cut basin below the tables' area, in the raster only
        member_rows = [row_of_member[m.member_id] for m in basin["members"] if m.member_id in row_of_member]
        if not member_rows:
            continue
        outlet_members = [row_of_member[m.member_id] for m in basin["members"] if m.downstream_depth == 0 and m.member_id in row_of_member]
        if not outlet_members:
            continue
        outlet_member = outlet_members[0]
        # the outlet's longitude and latitude worked out, not read back from the basin table,
        # so that both print the same double: on a geographic grid origin + (index + 0.5) * pixel size, on a projected
        # one the centre transformed to WGS84 (_table_lon_lat)
        outlet_lon, outlet_lat = _table_lon_lat(partition.grid, basin["outlet_row"], basin["outlet_col"])
        row = {"basin_id": basin_id, "outlet_row": basin["outlet_row"], "outlet_col": basin["outlet_col"],
               "outlet_lon": outlet_lon, "outlet_lat": outlet_lat, "basin_area_km2": basin["area_km2"]}
        river_pixels = int(sum(r.pixels_visited for r in member_rows))
        if attribute == "ldn":
            head = max(member_rows, key=lambda r: r.farthest_metres)
            row = {"basin_id": basin_id, "outlet_row": basin["outlet_row"], "outlet_col": basin["outlet_col"], "basin_area_km2": basin["area_km2"],
                   "measured_grid_count": river_pixels, "farthest_metres": float(head.farthest_metres)}
        elif attribute == "hck":
            row.update({"largest_order": int(max(r.largest_order for r in member_rows)), "river_grid_count": river_pixels})
        elif attribute == "shv":
            sources = int(sum(r.sources for r in member_rows))
            row.update({"magnitude_at_outlet": int(outlet_member.value_at_outlet), "source_grid_count": sources, "river_grid_count": river_pixels})
            if int(outlet_member.value_at_outlet) != sources:
                raise FlowDivideError("basin %d: the magnitude at the outlet (%d) is not the number of sources (%d)" % (basin_id, int(outlet_member.value_at_outlet), sources))
        elif attribute == "ord":
            row.update({"order_at_outlet": int(outlet_member.value_at_outlet), "river_grid_count": river_pixels})
        elif attribute == "lup":
            row.update({"lup_at_outlet_m": float(outlet_member.value_at_outlet), "written_grid_count": river_pixels})
        rows.append(row)
    return pd.DataFrame(rows, columns=BASIN_TABLE_COLUMNS[attribute])


def _derive_longest_flow_path(partition, dir_path, out_path, table_path, member_table_path, lines_path, tag, ldn_member_table, started):
    """the longest flow path of every kept basin: the head is the pixel with the largest distance to the
    outlet (from the ldn member table, or from a distance walk run here without a raster), and the
    path is painted from the head down through the chain of members that hold it, one region at a
    time, as the basin id.  The lines go to lines_path (a GeoPackage), the segments per member to
    member_table_path."""
    grid = partition.grid
    if ldn_member_table is None or not os.path.exists(ldn_member_table):
        log(tag, "no distance table given: the distance walk runs first, without a raster")
        scratch = out_path + ".ldn_scratch.tif"
        ldn_member_table = table_path + ".ldn_scratch_members.csv"
        derive_attribute("ldn", partition, dir_path, scratch, table_path + ".ldn_scratch.csv", ldn_member_table, tag=tag + ".ldn")
        for path in (scratch, scratch + ".report.json", table_path + ".ldn_scratch.csv"):
            if os.path.exists(path):
                os.remove(path)
    members_ldn = read_table(ldn_member_table, exact_floats=True)
    # the distance covers every basin, so its member table also lists the pieces of a basin below the tables' area
    # that is cut (a min_basin_area_km2 above the cut's); the path is drawn for the basins of the tables only
    # reads its members at the tables' area
    members_ldn = members_ldn[np.isin(members_ldn["basin_id"].to_numpy(np.int64), np.fromiter(partition.basins, np.int64, len(partition.basins)))]
    if set(members_ldn["member_id"].astype(int)) != set(partition.members):
        raise FlowDivideError("the distance table %s does not list exactly the members of this partition; run ldn again on it" % ldn_member_table)
    # every distance a finite number, not negative: an infinity would pass the length check below (inf > inf is
    # false) and be written as the basin's longest flow path
    farthest = members_ldn["farthest_metres"].to_numpy(np.float64)
    if farthest.size and not np.all(np.isfinite(farthest) & (farthest >= 0.0)):
        raise FlowDivideError("the distance table %s holds a distance that is not a finite number of metres at least 0" % ldn_member_table)
    # the head of every basin, and the entry pixel of every member on the chain from the head down
    entry = {}
    heads = {}
    for basin_id, group in members_ldn.groupby("basin_id"):
        best = group.sort_values(["farthest_metres", "head_row", "head_col"], ascending=[False, True, True]).iloc[0]
        member = partition.members[int(best["member_id"])]
        heads[int(basin_id)] = {"member_id": member.member_id, "head_row": int(best["head_row"]), "head_col": int(best["head_col"]), "length_m": float(best["farthest_metres"])}
        entry[member.member_id] = (int(best["head_row"]), int(best["head_col"]))
        while member.parent_member_id:
            parent = partition.members[member.parent_member_id]
            entry[parent.member_id] = (member.parent_inlet_row, member.parent_inlet_col)
            member = parent
    order = partition.region_order(True)
    temporary = out_path + ".partial.tif"
    profile = raster_profile(grid, "uint32", 0)
    with rasterio.open(temporary, "w", **profile) as created:
        pass
    longest_member = max([m.pixel_count for m in partition.members.values()] + [1])
    path_rows = np.empty(max(PATH_FRAMES, longest_member + 1), np.int64)        # a path is never longer than its member
    path_cols = np.empty(path_rows.size, np.int64)
    segments = {}
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(temporary, "r+") as out_dataset:
        for visit, region_id in enumerate(order):
            members = [m for m in partition.members_by_region[region_id] if m.member_id in entry]
            if not members:
                continue
            rectangle = partition.region_rectangle(region_id)
            window = RegionWindow(grid, rectangle, dir_dataset, out_dataset, "uint32", 0, read_out_whole=True)
            for member in members:
                start_row, start_col = window.local(grid, *entry[member.member_id])
                end_row, end_col = window.local(grid, member.outlet_row, member.outlet_col)
                status, count, length = paint_path_downstream(window.dir, window.out, window.lengths, start_row, start_col, end_row, end_col, np.uint32(member.basin_id), path_rows, path_cols)
                if status != 0:
                    raise FlowDivideError("lfp: the path segment of member %d (basin %d) ended with status %d" % (member.member_id, member.basin_id, status))
                lon = grid.transform.c + (((path_cols[:count] + window.col0) % grid.ncol) + 0.5) * grid.pixel_width
                lat = grid.transform.f + (path_rows[:count] + window.row0 + 0.5) * grid.pixel_height
                segments[member.member_id] = (np.column_stack([lon, lat]), int(count), float(length))
            window.write_back(grid, out_dataset, "uint32")
            del window
            log(tag, "region %d (%d of %d): %d path segments" % (region_id, visit + 1, len(order), len(members)))
    # the segments joined from the head down, one line per basin, and the length checked
    rows = []
    geometries = []
    segment_rows = [{"member_id": member_id, "basin_id": partition.members[member_id].basin_id, "region_id": partition.members[member_id].region_id,
                     "path_grid_count": count, "length_m": length} for member_id, (_, count, length) in sorted(segments.items())]
    for basin_id in sorted(heads):
        head = heads[basin_id]
        member = partition.members[head["member_id"]]
        points = []
        count = 0
        length = 0.0
        while True:
            segment_points, segment_count, segment_length = segments[member.member_id]
            points.append(segment_points)
            count += segment_count
            length += segment_length
            if not member.parent_member_id:
                break
            parent = partition.members[member.parent_member_id]
            # the step across the cut, from this member's outlet to the parent's inlet pixel
            length += earth_distance_m(grid.latitude_of_row(member.outlet_row), grid.longitude_of_col(member.outlet_col),
                                       grid.latitude_of_row(member.parent_inlet_row), grid.longitude_of_col(member.parent_inlet_col)) if grid.geographic \
                else math.hypot((member.outlet_row - member.parent_inlet_row) * grid.pixel_height, (member.outlet_col - member.parent_inlet_col) * grid.pixel_width)
            member = parent
        if not (abs(length - head["length_m"]) <= 1e-3 * max(1.0, head["length_m"])):     # a NaN or inf fails too
            raise FlowDivideError("basin %d: the painted path measures %.1f m, the distance walk found %.1f m" % (basin_id, length, head["length_m"]))
        basin = partition.basins[basin_id]
        head_col = head["head_col"] % grid.ncol if grid.periodic else head["head_col"]
        # the longitudes and latitudes of the outlet and the head, the head's column folded onto
        # the grid first: origin + (index + 0.5) * pixel size on a geographic grid, the centre transformed to WGS84 on
        # a projected one (_table_lon_lat)
        outlet_lon, outlet_lat = _table_lon_lat(grid, basin["outlet_row"], basin["outlet_col"])
        head_lon, head_lat = _table_lon_lat(grid, head["head_row"], head_col)
        rows.append({"basin_id": basin_id, "outlet_row": basin["outlet_row"], "outlet_col": basin["outlet_col"],
                     "outlet_lon": outlet_lon, "outlet_lat": outlet_lat,
                     # the length the distance walk found (largest_distance_m); the
                     # painted length is only checked against it above (the two sum in other orders)
                     "basin_area_km2": basin["area_km2"], "longest_flow_path_m": head["length_m"], "path_grid_count": count, "head_row": head["head_row"], "head_col": head_col,
                     "head_lon": head_lon, "head_lat": head_lat})
        geometries.append(np.concatenate(points))
    table = pd.DataFrame(rows, columns=BASIN_TABLE_COLUMNS["lfp"])
    if lines_path is not None:
        _write_lines(lines_path, table, geometries, grid)     # the lines at the fine grid, only when asked for
    # the raster and the table carry .done markers; the table's marker carries the number of basins
    # ("... km2 (N basins, ..."), which the table is checked against when it is read; the old
    # markers go before anything is replaced
    for marked_path in (out_path, table_path):
        if os.path.exists(marked_path + ".done"):
            os.remove(marked_path + ".done")
    publish(temporary, out_path)
    write_table(pd.DataFrame(segment_rows, columns=MEMBER_TABLE_COLUMNS["lfp"]), member_table_path, float_format="%.6f")
    write_basin_table_as_the_c_code_does(table, table_path)
    lfp_checks = ("the longest flow path of every basin of at least %.3f km2 (%d basins, written by the Python package "
                  "flowdivide), every path ends at its basin's outlet" % (partition.table_min_basin_area_km2, len(table)))
    fd_tables.write_done_marker(out_path, lfp_checks)
    fd_tables.write_done_marker(table_path, lfp_checks)
    report = {"attribute": "lfp", "regions_visited": len(order), "basins": len(table), "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s and %s: %d basins%s" % (out_path, table_path, len(table), ", the lines to %s" % lines_path if lines_path else ""))
    return report


def _write_lines(path, table, geometries, grid):
    """the longest flow paths as one line per basin in a GeoPackage, the table's columns as attributes; a
    path across the antimeridian is split there into the parts of a MultiLineString"""
    import geopandas
    from shapely.geometry import LineString, MultiLineString
    lines = []
    for points in geometries:
        if len(points) < 2:
            points = np.vstack([points, points])
        if grid.geographic:
            points = points.copy()
            points[:, 0] = ((points[:, 0] + 180.0) % 360.0) - 180.0
        breaks = np.nonzero(np.abs(np.diff(points[:, 0])) > 180.0)[0] + 1 if grid.geographic else np.zeros(0, np.int64)
        if breaks.size == 0:
            lines.append(LineString(points))
            continue
        # every crossing gets its point on the seam on both sides, so that no step is lost
        parts = []
        start = 0
        for index in breaks:
            before = points[index - 1]
            after = points[index]
            unrolled_after_lon = after[0] + 360.0 if after[0] < before[0] else after[0] - 360.0
            seam_lon = 180.0 if unrolled_after_lon > before[0] else -180.0
            fraction = (seam_lon - before[0]) / (unrolled_after_lon - before[0])
            seam_lat = before[1] + fraction * (after[1] - before[1])
            parts.append(np.vstack([points[start:index], [[seam_lon, seam_lat]]]))
            points = points.copy()
            points[index - 1] = [-seam_lon, seam_lat]         # the next part starts on the other side of the seam
            start = index - 1
        parts.append(points[start:])
        parts = [part for part in parts if len(part) >= 2 and not np.array_equal(part[0], part[-1]) or len(part) > 2]
        lines.append(MultiLineString([LineString(part) for part in parts]) if parts else LineString(points[:2]))
    frame = geopandas.GeoDataFrame(table, geometry=lines, crs=grid.crs)
    temporary = path + ".partial.gpkg"
    if os.path.exists(temporary):
        os.remove(temporary)
    frame.to_file(temporary, driver="GPKG", layer="longest_flow_path")
    publish(temporary, path)


# =============================================================================
#  [4] A rule of the user's own
# =============================================================================
#
#  An attribute is added by writing one Numba kernel with the shape of the sweeps of section [2b] and
#  registering it here.  The kernel is called once per region and receives the visiting order of the
#  window, the pixel each pixel flows into, the member each pixel belongs to (-1 for a pixel of another
#  region), the window arrays it asked for, the row step lengths, the width of the window and an array to
#  count the pixels it took for each member; it returns a status, 0 when it is done.  A rule handed down
#  from the divides sweeps the order forwards, a rule handed up from the outlet sweeps it backwards, and
#  the value that crosses a cut is read from, and written to, the output array at the pixels the driver
#  reports.  The driver does the rest: the order of the regions (downstream_first for a rule handed up),
#  the windows, the write-back and the tables.

def register_attribute(code, name, dtype, nodata, downstream_first, channel, area, kernel):
    """make `kernel` available to derive_user_attribute under `code`; kernel(order, downstream, member,
    dir_window, out_window, channel_window, area_window, lengths, ncol, visited_of_member) -> status,
    with channel and area saying whether the kernel wants the channel mask and the upstream area"""
    ATTRIBUTES[code] = {"name": name, "dtype": dtype, "nodata": nodata, "downstream_first": downstream_first,
                        "channel": channel, "area": area, "kernel": kernel, "table": code}


def _check_on_the_grid_of_the_flow_directions(dir_dataset, dir_path, channel_dataset, area_dataset):
    """the channel mask and the upstream area on the grid of the flow directions: the same size, CRS and transform
    (one function for the provided attributes and the registered rules)"""
    for name, dataset in (("the channel mask", channel_dataset), ("the upstream area", area_dataset)):
        if dataset is not None and ((dataset.width, dataset.height) != (dir_dataset.width, dir_dataset.height)
                                    or dataset.crs != dir_dataset.crs
                                    or any(abs(a - b) > 1e-9 * max(1.0, abs(b))
                                           for a, b in zip(tuple(dataset.transform)[:6], tuple(dir_dataset.transform)[:6]))):
            raise FlowDivideError("%s is not on the grid of the flow directions %s" % (name, dir_path))


def derive_user_attribute(code, partition, dir_path, out_path, table_path, channel_path=None,
                          area_path=None, tag=None):
    """the driver for a registered rule: the same visit as derive_attribute_by_sweep, the kernel the user's"""
    spec = ATTRIBUTES[code]
    if "kernel" not in spec:
        raise FlowDivideError("'%s' is one of the six provided attributes; use derive_attribute" % code)
    tag = tag or "fd3." + code
    grid = partition.grid
    started = time.time()
    regions_in_order = partition.region_order(spec["downstream_first"])
    temporary = out_path + ".partial.tif"
    with rasterio.open(temporary, "w", **raster_profile(grid, spec["dtype"], spec["nodata"])):
        pass
    rows = []
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(temporary, "r+") as out_dataset, \
            (rasterio.open(channel_path) if spec["channel"] else _nothing()) as channel_dataset, \
            (rasterio.open(area_path) if spec["area"] else _nothing()) as area_dataset:
        # the same grid checks as the provided attributes (an ACA one pixel east would reach the registered kernel
        # pixel for pixel)
        _check_on_the_grid_of_the_flow_directions(dir_dataset, dir_path, channel_dataset, area_dataset)
        for visit, region_id in enumerate(regions_in_order):
            members = list(partition.members_by_region[region_id])
            rectangle = partition.region_rectangle(region_id)
            # the same int32 bound the swept classes are held to: the visiting order indexes the window
            # with int32, and a window past that would wrap round silently
            if (rectangle[1] - rectangle[0] + 1) * (rectangle[3] - rectangle[2] + 1) > 2147483647:
                raise FlowDivideError("%s: the window of region %d holds %d pixels, more than the int32 the "
                                      "visiting order is indexed with; build the partition at a smaller capacity"
                                      % (code, region_id,
                                         (rectangle[1] - rectangle[0] + 1) * (rectangle[3] - rectangle[2] + 1)))
            window = RegionWindow(grid, rectangle, dir_dataset, out_dataset, spec["dtype"], spec["nodata"],
                                  channel_dataset, area_dataset, read_out_whole=True)
            downstream, order, taken, _ = window_flow_structure(window.dir, np.zeros((1, 1), np.uint8), False)
            if taken < 0:
                raise FlowDivideError("%s: the flow directions of region %d hold a cycle" % (code, region_id))
            pixel_count = window.nrow * window.ncol
            outlet_pixels = np.empty(len(members), np.int32)
            for index, member in enumerate(members):
                row, col = window.local(grid, member.outlet_row, member.outlet_col)
                outlet_pixels[index] = row * window.ncol + col
            in_this_region = {member.member_id for member in members}
            blocked = []
            for member in members:
                for child in member.children:
                    if child.member_id in in_this_region:
                        continue
                    child_row, child_col = window.local(grid, child.outlet_row, child.outlet_col)
                    blocked.append(child_row * window.ncol + child_col)
            member_of_pixel = window_member_of_pixel(order, downstream, outlet_pixels,
                                                     np.asarray(sorted(blocked), np.int64), pixel_count)
            channel = window.channel if window.channel is not None else np.zeros((1, 1), np.uint8)
            area = window.area if window.area is not None else np.zeros((1, 1), np.float32)
            visited_of_member = np.zeros(len(members), np.int64)
            status = spec["kernel"](order, downstream, member_of_pixel, window.dir, window.out, channel,
                                    area, window.lengths, window.ncol, visited_of_member)
            if status != 0:
                raise FlowDivideError("%s: the sweep of region %d ended with status %d" % (code, region_id, status))
            for index, member in enumerate(members):
                rows.append({"member_id": member.member_id, "basin_id": member.basin_id,
                             "region_id": region_id, "pixels_visited": int(visited_of_member[index])})
            window.write_back(grid, out_dataset, spec["dtype"])
            del window, downstream, order, member_of_pixel, channel, area      # the views keep the window
            log(tag, "region %d (%d of %d): %d members, %d x %d window, swept in order"
                % (region_id, visit + 1, len(regions_in_order), len(members),
                   rectangle[1] - rectangle[0] + 1, rectangle[3] - rectangle[2] + 1))
    member_table = pd.DataFrame(rows, columns=MEMBER_BASE)
    publish(temporary, out_path)
    write_table(member_table, table_path)
    report = {"attribute": code, "regions_visited": len(regions_in_order), "members": len(rows),
              "seconds": round(time.time() - started, 1)}
    write_json(out_path + ".report.json", report)
    log(tag, "written %s and %s: %d regions, %d members, %d seconds"
        % (out_path, table_path, len(regions_in_order), len(rows), int(report["seconds"])))
    return report
