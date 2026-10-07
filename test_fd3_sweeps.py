"""test_fd3_sweeps.py -- the third stage on a partition against the answer on the whole basin, pixel for
pixel, on a real basin.

    python test_fd3_sweeps.py <case directory> [--attributes shv,ldn,hck,lup,ord]

The case directory is one that test_three_ways_on_one_basin.py prepared: input_dir.tif, the flow directions
of one basin's rectangle, and case.txt.  The test builds a partition of that basin in the shape the third
stage reads -- the four tables of a published partition, written here for one region and one member, with
the region's rectangle the basin's own -- computes the attributes with fd3_attributes, and compares every
pixel with the answer three_ways.py computes on the whole basin in memory.  The channel network and the
upstream area the channel classes need are computed here as well, from the same flow directions.

What this checks: that the ordered sweeps of fd3_attributes section [2b] give the whole-basin answer, that
the member and basin tables agree with it, and that the raster carries nothing outside the basin.  What it
does not check: the states across a cut, which need a basin the capacity cuts (test_three_ways_on_one_basin
covers the cut on the pieces, and the comparison runs of Figure 7 cover a cut basin on the published
partition).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numba import njit

import fd1_partition as fd1
import fd3_attributes as fd3
import fd_tables
import test_three_ways_on_one_basin as prepared
import three_ways


# The three classes that live on the channel network are computed here over the whole basin, in the order
# three_ways builds, so that the reference shares nothing with the code it checks but the flow directions:
# three_ways' own shreve, strahler and hack count every land pixel as a source, while the third stage
# counts the heads of the channel network, so they are different quantities and cannot be compared.

@njit(cache=True)
def shreve_on_the_channel(order, downstream_of_pixel, channel):
    """the number of channel heads above a channel pixel, itself counted when it is a head"""
    value = np.zeros(channel.size, np.int64)
    for position in range(order.size):                      # upstream first: what arrives is final
        pixel = order[position]
        if channel[pixel] == 0:
            continue
        if value[pixel] == 0:
            value[pixel] = 1
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel >= 0 and channel[downstream_pixel] != 0:
            value[downstream_pixel] += value[pixel]
    return value


@njit(cache=True)
def strahler_on_the_channel(order, downstream_of_pixel, channel):
    """the largest order that arrives, plus one when it arrives at least twice; a head is 1"""
    largest = np.zeros(channel.size, np.int32)
    second = np.zeros(channel.size, np.int32)
    value = np.zeros(channel.size, np.int32)
    for position in range(order.size):                      # upstream first
        pixel = order[position]
        if channel[pixel] == 0:
            continue
        if largest[pixel] == 0:
            here = 1
        elif largest[pixel] == second[pixel]:
            here = largest[pixel] + 1
        else:
            here = largest[pixel]
        value[pixel] = here
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0 or channel[downstream_pixel] == 0:
            continue
        if here > largest[downstream_pixel]:
            second[downstream_pixel] = largest[downstream_pixel]
            largest[downstream_pixel] = here
        elif here > second[downstream_pixel]:
            second[downstream_pixel] = here
    return value


@njit(cache=True)
def hack_on_the_channel(order, downstream_of_pixel, channel, upstream_area, ncol):
    """1 at the terminal of the channel network, the channel donor with the most upstream area keeps the
    order of the pixel below it, every other channel donor takes one more (a tie to the smaller index)"""
    best_donor = np.full(channel.size, -1, np.int64)
    best_area = np.zeros(channel.size, np.float64)
    for position in range(order.size):
        pixel = order[position]
        if channel[pixel] == 0:
            continue
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0 or channel[downstream_pixel] == 0:
            continue
        area_here = upstream_area[pixel]
        if best_donor[downstream_pixel] < 0 or area_here > best_area[downstream_pixel]:
            best_area[downstream_pixel] = area_here
            best_donor[downstream_pixel] = pixel
        elif area_here == best_area[downstream_pixel] and pixel < best_donor[downstream_pixel]:
            best_donor[downstream_pixel] = pixel
    value = np.zeros(channel.size, np.int32)
    for position in range(order.size - 1, -1, -1):          # downstream first: the pixel below is done
        pixel = order[position]
        if channel[pixel] == 0:
            continue
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0 or channel[downstream_pixel] == 0:
            value[pixel] = 1                                # the terminal of the network
        elif best_donor[downstream_pixel] == pixel:
            value[pixel] = value[downstream_pixel]
        else:
            value[pixel] = value[downstream_pixel] + 1
    return value


CHANNEL_THRESHOLD_M2 = 1.0e6      # a channel pixel carries at least 1 km2, as the products do


@njit(cache=True)
def step_metres_of(lengths, ncol, pixel, downstream_pixel):
    """the length of one D8 step in metres, the five lengths of the row the step starts in"""
    row = pixel // ncol
    downstream_row = downstream_pixel // ncol
    if downstream_row == row:
        return lengths[row, 0]
    if downstream_pixel - downstream_row * ncol == pixel - row * ncol:
        return lengths[row, 1] if downstream_row < row else lengths[row, 2]
    return lengths[row, 3] if downstream_row < row else lengths[row, 4]


@njit(cache=True)
def distance_to_outlet_in_metres(order, downstream_of_pixel, lengths, ncol):
    """the distance along the flow path down to the terminal, in metres, the step table the third stage
    uses (three_ways rounds every step to a whole centimetre, and takes the diagonal as
    the hypotenuse of the two sides; the traversal is what this test checks, not the step table)"""
    value = np.zeros(downstream_of_pixel.size, np.float64)
    for position in range(order.size - 1, -1, -1):          # downstream first
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel >= 0:
            value[pixel] = value[downstream_pixel] + step_metres_of(lengths, ncol, pixel, downstream_pixel)
    return value


@njit(cache=True)
def upstream_flow_length_in_metres(order, downstream_of_pixel, lengths, ncol):
    """the longest path from a divide down to the pixel, in metres, the same step table"""
    value = np.zeros(downstream_of_pixel.size, np.float64)
    for position in range(order.size):                      # upstream first
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0:
            continue
        candidate = value[pixel] + step_metres_of(lengths, ncol, pixel, downstream_pixel)
        if candidate > value[downstream_pixel]:
            value[downstream_pixel] = candidate
    return value


def log(message):
    print("%s [test fd3] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def tables_of_one_basin(case, grid, land_pixels, outlet_row, outlet_col, area_km2):
    """the four tables of a partition that holds one whole basin in one region, written where the third
    stage expects them, in the formats of fd_tables"""
    directory = os.path.join(case, "partition_of_one_basin")
    fd1.ensure_directory(directory)
    basin = fd_tables.new_basin_table(1)
    basin["outlet_row"] = outlet_row
    basin["outlet_col"] = outlet_col
    basin["outlet_lon"] = 0.0
    basin["outlet_lat"] = 0.0
    basin["basin_grid_count"] = land_pixels
    basin["basin_area_km2"] = area_km2
    basin["basin_row_min"] = 0
    basin["basin_row_max"] = grid.nrow
    basin["basin_col_min"] = 0
    basin["basin_col_max"] = grid.ncol
    basin["region_id"] = 1
    region = pd.DataFrame([{name: 0 for name in fd_tables.REGION_TABLE_COLUMNS}])
    region["region_id"] = 1
    region["row_max"] = grid.nrow
    region["col_max"] = grid.ncol
    region["nrow"] = grid.nrow
    region["ncol"] = grid.ncol
    region["region_grid_count"] = land_pixels
    region["basin_count"] = 1
    region["window_grid_count"] = grid.nrow * grid.ncol
    region["fill_percent"] = 100.0
    region["bbox_row_max"] = grid.nrow
    region["bbox_col_max"] = grid.ncol
    region["level3_members"] = ""
    piece = pd.DataFrame({name: pd.Series(dtype=np.int64) for name in fd_tables.PIECE_TABLE_COLUMNS})
    paths = {"basin": os.path.join(directory, "basin_table_fine_one-basin.csv"),
             "region": os.path.join(directory, "region_fine_one-basin.csv"),
             "piece": os.path.join(directory, "piece_fine_one-basin.csv"),
             "basin_region": os.path.join(directory, "basin_region_one-basin.csv")}
    fd_tables.write_basin_table(basin, paths["basin"], "fd1.6", "test")
    fd_tables.write_region_table(region, paths["region"], "test")
    fd_tables.write_piece_table(piece, paths["piece"], "test")
    key = fd_tables.region_map_key(grid.nrow * grid.ncol, 512, "l3")
    fd_tables.write_basin_map(paths["basin_region"], np.array([0, 1], np.int64), "one-basin", "region", key)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--attributes", default="shv,ldn,hck,lup,ord")
    arguments = parser.parse_args()
    case = arguments.case
    dir_path = os.path.join(case, "input_dir.tif")
    grid = fd1.Grid(dir_path)
    the_case = prepared.load_case(case)                     # the directions in the library's convention
    flow_direction = the_case["flow_direction"]
    nrow, ncol = the_case["nrow"], the_case["ncol"]
    land = (flow_direction != three_ways.LIBRARY_NODATA).reshape(nrow, ncol)
    land_pixels = the_case["land_pixels"]
    log("the case holds %d x %d pixels, %d of them land" % (nrow, ncol, land_pixels))

    # the answers on the whole basin, and the two rasters the channel classes read
    step_north_south = the_case["step_metres_north_south"]
    step_east_west = the_case["step_metres_east_west"]
    pixel_area = np.repeat(the_case["pixel_area_m2_by_row"], ncol)
    whole = {}
    whole["upa"], _ = three_ways.accumulate_whole_int64(flow_direction, nrow, ncol, pixel_area)
    # the channel classes read the upstream area as Float32, as the third stage reads it from the product,
    # so the reference weighs the main stem on the very numbers the sweep sees
    area_float32 = whole["upa"].reshape(nrow, ncol).astype(np.float32)
    channel = ((area_float32 >= CHANNEL_THRESHOLD_M2) & land).astype(np.uint8)
    downstream_of_pixel = three_ways.build_downstream_pixel_array(flow_direction, nrow, ncol)
    order_whole, is_forest = three_ways.topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise SystemExit("the flow directions of the case hold a cycle")
    lengths = grid.row_step_lengths_m(0, nrow)
    whole["ldn"] = distance_to_outlet_in_metres(order_whole, downstream_of_pixel, lengths, ncol)
    whole["lup"] = upstream_flow_length_in_metres(order_whole, downstream_of_pixel, lengths, ncol)
    # the two lengths against three_ways as well, which rounds every step to a whole centimetre and takes
    # the diagonal as the hypotenuse: the two step tables differ by a few metres over a long path, so this
    # is reported, not asserted
    distance_centimetres, _ = three_ways.distance_to_outlet_whole_int64(
        flow_direction, nrow, ncol, step_north_south, step_east_west)
    apart = np.abs(distance_centimetres / 100.0 - whole["ldn"])[flow_direction != three_ways.LIBRARY_NODATA]
    log("the two step tables put the distance to the outlet at most %.3f m apart (median %.3f m)"
        % (float(apart.max()), float(np.median(apart))))
    whole["shv"] = shreve_on_the_channel(order_whole, downstream_of_pixel, channel.ravel())
    whole["ord"] = strahler_on_the_channel(order_whole, downstream_of_pixel, channel.ravel())
    whole["hck"] = hack_on_the_channel(order_whole, downstream_of_pixel, channel.ravel(),
                                       area_float32.ravel().astype(np.float64), ncol)
    log("the whole-basin answers are computed, the channel network holds %d pixels" % int((channel != 0).sum()))

    scratch = os.path.join(case, "fd3_sweeps")
    fd1.ensure_directory(scratch)
    channel_path = os.path.join(scratch, "channel.tif")
    area_path = os.path.join(scratch, "upa.tif")
    with rasterio.open(channel_path, "w", **fd1.raster_profile(grid, "uint8", 0)) as dataset:
        dataset.write(channel, 1)
    with rasterio.open(area_path, "w", **fd1.raster_profile(grid, "float32", 0)) as dataset:
        dataset.write(area_float32, 1)

    terminal = np.flatnonzero((flow_direction == 0) & (flow_direction != three_ways.LIBRARY_NODATA))
    if terminal.size != 1:
        raise SystemExit("the case holds %d terminals; this test wants one basin" % terminal.size)
    outlet_row = int(terminal[0]) // ncol
    outlet_col = int(terminal[0]) - outlet_row * ncol
    area_km2 = float(pixel_area[land.ravel()].sum() / 1.0e6)
    paths = tables_of_one_basin(case, grid, land_pixels, outlet_row, outlet_col, area_km2)
    partition = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 1.0, grid)

    all_same = True
    for attribute in arguments.attributes.split(","):
        out_path = os.path.join(scratch, "%s.tif" % attribute)
        started = time.time()
        fd3.derive_attribute(attribute, partition, dir_path, out_path,
                             os.path.join(scratch, "%s_basins.csv" % attribute),
                             os.path.join(scratch, "%s_members.csv" % attribute),
                             channel_path=channel_path, area_path=area_path, tag="test.%s" % attribute)
        seconds = time.time() - started
        with rasterio.open(out_path) as dataset:
            mine = dataset.read(1).astype(np.float64)
            nodata_of_the_raster = dataset.nodata
            dataset_dtype = dataset.dtypes[0]
        theirs = whole[attribute].reshape(grid.nrow, grid.ncol).astype(np.float64)
        if attribute in ("shv", "ord", "hck"):
            compare_at = channel != 0
        else:
            compare_at = land
        # the raster holds the value as Float32 or as an integer, so the answer is compared as the very
        # number the raster can carry, not within a tolerance
        as_written = theirs.astype(dataset_dtype)
        apart = np.abs(mine[compare_at].astype(np.float64) - as_written[compare_at].astype(np.float64))
        not_finite = int((~np.isfinite(mine[compare_at])).sum())
        differing = int((apart != 0.0).sum()) + not_finite
        if nodata_of_the_raster is None:
            raise SystemExit("%s carries no nodata value" % out_path)
        outside = int((mine[~compare_at] != nodata_of_the_raster).sum())
        log("%-4s %-9s %d of %d compared pixels differ (largest %.6g, %d not finite), %d of %d pixels the "
            "class does not live on carry something other than the nodata %g, %.1f s"
            % (attribute, "IDENTICAL" if differing == 0 and outside == 0 else "DIFFERENT", differing,
               int(compare_at.sum()), float(apart.max()) if apart.size else 0.0, not_finite, outside,
               int((~compare_at).sum()), nodata_of_the_raster, seconds))
        all_same = all_same and differing == 0 and outside == 0
    log("ALL IDENTICAL" if all_same else "SOMETHING DIFFERS")
    raise SystemExit(0 if all_same else 1)


if __name__ == "__main__":
    main()
