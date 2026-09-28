"""test_tile_kernels.py -- the four kernels of tile_kernels.py (one tile at a time, the tiles the data are
distributed in) against the whole-domain answers of three_ways.py, pixel for pixel, on a real basin.

    python test_tile_kernels.py <case directory> [--tile-degrees 1]

The case directory is one that test_three_ways_on_one_basin.py prepared: input_dir.tif (the flow directions
of the basin's rectangle, MERIT convention) and case.txt.  The test computes, on that grid,

    upa   the upstream area           ldn   the distance to the outlet
    lup   the longest path upstream    ord   the Strahler order of the channel network

and compares every pixel with a whole-domain computation written out in this file, in the units of
fd1_partition (three_ways.py computes the same four over the whole domain, but in whole centimetres and
whole square metres, a different rounding; its answers are used as a second, independent check of the
distances, within the centimetre rounding).  The whole-domain answers hold the rectangle in memory, so the basin has to be a
small one -- the point here is that the answer does not depend on the cut, which a small basin shows as well
as a large one; the timing runs of Figure 7 are a different thing.

What this test does not do: the whole-domain reference here shares the step lengths and the
pixel areas of fd1_partition with the code it checks, so it is independent of the traversal and the cut but not
of the geometry -- three_ways.py, which rounds every step to a whole centimetre, is the second reference and is
independent of both; and there is no case yet for a periodic grid, for a tie between two equally long paths
across a tile edge, or for a tile with no exits.

--tile-degrees is 1 by default, not 10: on a basin of a few degrees the 10-degree tiles of HydroSHEDS would
often be a single tile, and then nothing crosses a tile edge and the test would check nothing.  The lengths
are compared within a metre (tile_kernels writes Float32 rasters, as the package's own attributes do, and a
metre in 10^7 is that format's own step) and the integers exactly.
"""
import argparse
import os
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math

from numba import njit

import fd1_partition as fd1
import three_ways
import tile_kernels
from fd1_partition import Grid


def log(message):
    print("%s [test tiles] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


@njit(cache=True)
def _step_length_here(lengths, row, drow, dcol):
    """the length of one step, written out here rather than imported from tile_kernels: a reference that
    borrows from the code it checks cannot catch a mistake in what it borrowed.
    The five lengths of a row are east-west, north, south, north-diagonal, south-diagonal."""
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
def _whole_domain_kernels(flow_direction, nrow, ncol, downstream_of_pixel, order, lengths, row_area_m2, channel_threshold_m2):
    """The four kernels over the WHOLE rectangle at once, with the step lengths and the pixel areas of
    fd1_partition -- the same numbers tile_kernels uses, so that the only difference between the two
    answers is where the grid was cut.  (three_ways.py computes the same four, but in whole centimetres
    and whole square metres, which is a different rounding and so not the reference here.)"""
    pixel_count = nrow * ncol
    area = np.zeros(pixel_count, np.float64)
    distance = np.zeros(pixel_count, np.float64)
    longest = np.zeros(pixel_count, np.float64)
    source = np.empty(pixel_count, np.int64)
    largest_order = np.zeros(pixel_count, np.uint8)
    second_order = np.zeros(pixel_count, np.uint8)
    strahler = np.zeros(pixel_count, np.uint8)
    for pixel in range(pixel_count):
        source[pixel] = pixel
        if flow_direction[pixel] != three_ways.LIBRARY_NODATA:
            area[pixel] = row_area_m2[pixel // ncol]
    # the accumulation and the longest path travel downstream: the order, upstream first
    for position in range(order.size):
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0:
            continue
        row = pixel // ncol
        step = _step_length_here(lengths, row, downstream_pixel // ncol - row, downstream_pixel % ncol - (pixel - row * ncol))
        area[downstream_pixel] += area[pixel]
        candidate = longest[pixel] + step
        if candidate > longest[downstream_pixel]:
            longest[downstream_pixel] = candidate
            source[downstream_pixel] = source[pixel]
        elif candidate == longest[downstream_pixel] and source[pixel] < source[downstream_pixel]:
            source[downstream_pixel] = source[pixel]
    # the distance to the outlet travels upstream: the same order, backwards
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0:
            continue
        row = pixel // ncol
        step = _step_length_here(lengths, row, downstream_pixel // ncol - row, downstream_pixel % ncol - (pixel - row * ncol))
        distance[pixel] = distance[downstream_pixel] + step
    # the Strahler order of the channel network, upstream first
    channel = np.zeros(pixel_count, np.uint8)
    for pixel in range(pixel_count):
        rounded = math.floor(area[pixel] + 0.5)
        if flow_direction[pixel] != three_ways.LIBRARY_NODATA and np.float32(rounded) >= channel_threshold_m2:
            channel[pixel] = 1
    for position in range(order.size):
        pixel = order[position]
        if channel[pixel] == 0:
            continue
        if largest_order[pixel] == 0:
            own = 1
        elif second_order[pixel] + 1 > largest_order[pixel]:
            own = second_order[pixel] + 1
        else:
            own = largest_order[pixel]
        strahler[pixel] = own
        downstream_pixel = downstream_of_pixel[pixel]
        if downstream_pixel < 0 or channel[downstream_pixel] == 0:
            continue
        if own >= largest_order[downstream_pixel]:
            second_order[downstream_pixel] = largest_order[downstream_pixel]
            largest_order[downstream_pixel] = own
        elif own > second_order[downstream_pixel]:
            second_order[downstream_pixel] = own
    return area, distance, longest, source, strahler, channel


def whole_domain_answers(case_directory, channel_threshold_km2=1.0):
    """the four answers over the whole rectangle at once, in the units of fd1_partition"""
    dir_path = os.path.join(case_directory, "input_dir.tif")
    grid = Grid(dir_path)
    with rasterio.open(dir_path) as dataset:
        flow_direction = dataset.read(1).ravel()
    flow_direction = three_ways.flow_directions_to_library_convention(flow_direction)
    nrow, ncol = grid.nrow, grid.ncol
    downstream_of_pixel = three_ways.build_downstream_pixel_array(flow_direction, nrow, ncol)
    order, is_forest = three_ways.topological_order_whole(flow_direction, downstream_of_pixel)
    if not is_forest:
        raise SystemExit("the flow directions of the case hold a cycle")
    lengths = grid.row_step_lengths_m(0, nrow)
    row_area_m2 = grid.row_pixel_areas_m2(0, nrow)
    area, distance, longest, source, strahler, channel = _whole_domain_kernels(
        flow_direction, nrow, ncol, downstream_of_pixel, order, lengths, row_area_m2, channel_threshold_km2 * 1e6)
    return {"nrow": nrow, "ncol": ncol, "upa": area, "ldn": distance, "lup": longest, "lup_source": source,
            "ord": strahler, "channel": channel, "flow_direction": flow_direction}


def compare(name, mine, reference, mask, tolerance=0.0, relative=0.0):
    """the tile answer against the whole-domain one on the pixels of `mask`: the same to within `tolerance`
    in absolute value, or `relative` of the value itself (the rasters are Float32)"""
    mine = np.asarray(mine, np.float64)[mask]
    reference = np.asarray(reference, np.float64)[mask]
    # a value that is not finite on a pixel of the mask is a difference; NaN - 1 is NaN, which no "> allowed"
    # catches, and the test would pass
    not_finite = ~np.isfinite(mine) | ~np.isfinite(reference)
    if not_finite.any():
        first = int(np.flatnonzero(not_finite)[0])
        log("%-12s DIFFERS: %d of %d pixels hold a value that is not finite; first: tiles %r, whole domain %r"
            % (name, int(not_finite.sum()), mine.size, mine[first], reference[first]))
        return False
    difference = np.abs(mine - reference)
    allowed = tolerance + relative * np.abs(reference)
    worst = float(difference.max()) if difference.size else 0.0
    differing = int((difference > allowed).sum())
    if differing == 0:
        log("%-12s identical to the whole-domain answer on all %d pixels (largest difference %.3g)" % (name, difference.size, worst))
        return True
    first = int(np.flatnonzero(difference > allowed)[0])
    log("%-12s DIFFERS on %d of %d pixels; first: tiles %.6f, whole domain %.6f" % (name, differing, difference.size, mine[first], reference[first]))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="a case directory of test_three_ways_on_one_basin.py (input_dir.tif, case.txt)")
    parser.add_argument("--tile-degrees", type=float, default=1.0, help="the tiles to cut into (default 1 degree, so that something crosses a tile edge)")
    parser.add_argument("--channel-threshold-km2", type=float, default=1.0)
    arguments = parser.parse_args()
    dir_path = os.path.join(arguments.case, "input_dir.tif")
    out_directory = os.path.join(arguments.case, "tile_kernels")
    os.makedirs(out_directory, exist_ok=True)
    grid = Grid(dir_path)
    tiles, pixels_per_tile = tile_kernels.tiles_of_distribution(grid, arguments.tile_degrees, tile_kernels.TILE_METRES_DEFAULT)
    log("%d tiles of %d x %d pixels over a grid of %d x %d" % (len(tiles), pixels_per_tile, pixels_per_tile, grid.nrow, grid.ncol))
    if len(tiles) < 2:
        raise SystemExit("one tile covers the whole grid: nothing would cross a tile edge; give a smaller --tile-degrees")
    log("the whole-domain answers (three_ways.py)")
    reference = whole_domain_answers(arguments.case, arguments.channel_threshold_km2)
    nrow, ncol = reference["nrow"], reference["ncol"]
    land = reference["flow_direction"] != three_ways.LIBRARY_NODATA
    channel = reference["channel"] != 0
    results = []
    # upa: the accumulation on the distribution tiles
    upa_path = os.path.join(out_directory, "upa.tif")
    str_path = os.path.join(out_directory, "str.tif")
    tile_kernels.upa_on_tiles(dir_path, upa_path, str_path, grid, tiles, channel_threshold_km2=arguments.channel_threshold_km2)
    with rasterio.open(upa_path) as dataset:
        upa_tiles = dataset.read(1).ravel().astype(np.float64)
    # both to the nearest square metre.  The two ways add the same areas in a different order, and a sum of
    # doubles is not associative: over n terms the bound is n x 2^-53 of the sum, so the tolerance is that of
    # this basin's own numbers, three times over for the two sums and the rounding, not a number measured on
    # one basin.  Measured on basin 141: 12 pixels of 6.9 million differ by 1 m2.
    land_pixels = int(land.sum())
    largest_area = float(np.max(reference["upa"]))
    area_tolerance = max(3.0 * land_pixels * 2.0 ** -53 * largest_area, 2.0)
    log("upa: %d land pixels, %.3e m2 at the outlet, so the sums may differ by %.1f m2" % (land_pixels, largest_area, area_tolerance))
    results.append(compare("upa", upa_tiles, np.floor(reference["upa"] + 0.5), land, tolerance=area_tolerance))
    with rasterio.open(str_path) as dataset:
        channel_tiles = dataset.read(1).ravel()
    # the mask is the area against a threshold, so a pixel whose area is within the summation difference of
    # the threshold may fall either side of it; those few are left out and counted
    near_the_threshold = np.abs(reference["upa"] - arguments.channel_threshold_km2 * 1e6) <= area_tolerance
    log("channel mask: %d pixels lie within %.1f m2 of the threshold and are left out" % (int((near_the_threshold & land).sum()), area_tolerance))
    results.append(compare("channel mask", channel_tiles, reference["channel"], land & ~near_the_threshold, tolerance=0.0))
    del upa_tiles
    # ldn
    ldn_path = os.path.join(out_directory, "ldn.tif")
    tile_kernels.ldn_on_tiles(dir_path, ldn_path, grid, tiles)
    with rasterio.open(ldn_path) as dataset:
        ldn_tiles = dataset.read(1).ravel().astype(np.float64)
    # The rasters are Float32, as the package's own attributes are, so the comparison is relative: one step of
    # that format is 2^-24 of the value, and the two ways add the same steps in a different order, which over
    # n steps is at most n x 2^-53 of the sum.  2^-22 covers both with room.  Measured on basin 141: the largest difference is 0.016 m in 300 km.
    results.append(compare("ldn", ldn_tiles, reference["ldn"], land, relative=2 ** -22))
    del ldn_tiles
    # lup, and the pixel every longest path starts at
    lup_path = os.path.join(out_directory, "lup.tif")
    lup_source_path = os.path.join(out_directory, "lup_source.tif")
    tile_kernels.lup_on_tiles(dir_path, lup_path, grid, tiles, source_path=lup_source_path)
    with rasterio.open(lup_path) as dataset:
        lup_tiles = dataset.read(1).ravel().astype(np.float64)
    results.append(compare("lup", lup_tiles, reference["lup"], land, relative=2 ** -22))
    with rasterio.open(lup_source_path) as dataset:
        lup_source_tiles = dataset.read(1).ravel().astype(np.int64)
    results.append(compare("lup source", lup_source_tiles, reference["lup_source"], land, tolerance=0.0))
    del lup_tiles, lup_source_tiles
    # ord, on the channel mask the tile accumulation wrote
    ord_path = os.path.join(out_directory, "ord.tif")
    report = tile_kernels.ord_on_tiles(dir_path, str_path, ord_path, grid, tiles)
    with rasterio.open(ord_path) as dataset:
        ord_tiles = dataset.read(1).ravel().astype(np.float64)
    # on EVERY land pixel, not only the channel: a pixel off the channel must be 0 in both, which is how an
    # order written where there is no channel would show
    results.append(compare("ord", ord_tiles, reference["ord"], land, tolerance=0.0))
    log("the Strahler order took %d sweeps over %d tiles" % (report["sweeps"], report["tiles"]))
    # the second reference, written by other hands: three_ways.py over the whole domain, in whole centimetres
    # and whole square metres.  Its rounding is its own: each step is rounded to a whole centimetre, up to
    # 0.5 cm a step, and a path of L metres has at most L / (the shortest step of this grid) steps, so the two
    # answers may drift by 0.005 / shortest_step of L (the number comes from the grid, not from an assumed
    # 30 m).  The orders are compared exactly.
    log("the same four against three_ways.py's whole-domain answers (whole centimetres, whole square metres)")
    flow_direction = reference["flow_direction"]
    with rasterio.open(dir_path) as dataset:
        transform = dataset.transform
    steps_north_south, steps_east_west, areas = three_ways.row_steps_and_areas(nrow, transform.c, transform.f, transform.a, transform.e)
    shortest_step = float(min(steps_north_south.min(), steps_east_west.min()))
    drift = 0.005 / shortest_step
    log("the shortest step of this grid is %.1f m, so the centimetre rounding may drift by %.1e of a length" % (shortest_step, drift))
    ldn_centimetres, _ = three_ways.distance_to_outlet_whole_int64(flow_direction, nrow, ncol, steps_north_south, steps_east_west)
    with rasterio.open(ldn_path) as dataset:
        ldn_tiles = dataset.read(1).ravel().astype(np.float64)
    results.append(compare("ldn vs three_ways", ldn_tiles, ldn_centimetres / 100.0, land, tolerance=1.0, relative=drift))
    del ldn_centimetres, ldn_tiles
    lup_centimetres, _, _ = three_ways.upstream_flow_length_whole_int64(flow_direction, nrow, ncol, steps_north_south, steps_east_west)
    with rasterio.open(lup_path) as dataset:
        lup_tiles = dataset.read(1).ravel().astype(np.float64)
    results.append(compare("lup vs three_ways", lup_tiles, lup_centimetres / 100.0, land, tolerance=1.0, relative=drift))
    del lup_centimetres, lup_tiles
    masked = flow_direction.copy()
    masked[reference["channel"] == 0] = three_ways.LIBRARY_NODATA
    strahler_whole, _ = three_ways.strahler_whole_int64(masked, nrow, ncol)
    strahler_whole[reference["channel"] == 0] = 0
    with rasterio.open(ord_path) as dataset:
        ord_tiles = dataset.read(1).ravel().astype(np.float64)
    results.append(compare("ord vs three_ways", ord_tiles, strahler_whole, land, tolerance=0.0))
    del masked, strahler_whole, ord_tiles
    print()
    if all(results):
        log("ALL PASSED: the tile answers are the whole-domain answers")
        return 0
    log("FAILED: %d of %d comparisons differ" % (len([r for r in results if not r]), len(results)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
