"""test_three_ways_on_one_basin.py -- the seven variables of three_ways.py computed three ways on one real
basin, every answer compared with the whole-domain one pixel for pixel.

    python test_three_ways_on_one_basin.py prepare <products_root> <continent> <basin_id> <160|80|40> <case_directory>
    python test_three_ways_on_one_basin.py run     <case_directory> <int64|square|cut> <upg|upa|shv|ldn|lup|ord|hck>
    python test_three_ways_on_one_basin.py compare <case_directory> <variable>
    python test_three_ways_on_one_basin.py all     <case_directory>          every way and variable, one process
                                                                             each, then the comparisons

prepare cuts the basin's rectangle out of the published products under <products_root>/<continent>/
(the flow directions of FD1.0, the basin raster of FD1.3, the region raster of the partition at the
capacity): input_dir.tif (MERIT convention, 247 outside the basin), parts_cut.tif (the region id, 0
outside the basin) and case.txt.  run writes <case_directory>/python_<way>/<variable>.tif (lup also lup_source.tif) and
<variable>.cost.txt.  compare appends to <case_directory>/compare_python.txt.

The whole rectangle is held in memory whichever way runs (see three_ways.py for the bytes per pixel):
basin 7 of South America, the Parnaiba, 749,344,500 pixels in its rectangle, takes about 22 GB for int64
and 45-62 GB for a by-part way, which a 64 GB laptop just runs; that is the basin the paper's check was
made on.
"""
import os
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import three_ways
import fd_tables
from fd1_partition import MERIT_NODATA, MERIT_SINK

VARIABLES = ["upg", "upa", "shv", "ldn", "lup", "ord", "hck"]
WAYS = ["int64", "square", "cut"]
STRIP_NROW = 1024
FLOAT_NODATA = -1.0


def log(message):
    print("%s [three_ways] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


# =============================================================================
#  [1] The case: prepare, load
# =============================================================================

def prepare_case(products_root, continent, basin_id, capacity, case_directory):
    """the basin's rectangle out of the published products, strip by strip"""
    root = os.path.join(products_root, continent)
    partition_root = os.path.join(root, "partitions", "%sdeg2" % capacity)
    # the one basin table, read with its checks
    outlet_table = fd_tables.read_basin_table(fd_tables.basin_table_path(root, continent))
    row = outlet_table[outlet_table.basin_id == basin_id]
    if len(row) != 1:
        raise SystemExit("basin %d is not in the basin table" % basin_id)
    row = row.iloc[0]
    row_min, row_max = int(row.basin_row_min), int(row.basin_row_max)
    col_min, col_max = int(row.basin_col_min), int(row.basin_col_max)
    nrow, ncol = row_max - row_min, col_max - col_min
    table_land_pixels = int(row.basin_grid_count)
    log("basin %d of %s: rows %d..%d, cols %d..%d, %d x %d = %d pixels, %d of them land" % (basin_id, continent, row_min, row_max, col_min, col_max, nrow, ncol, nrow * ncol, table_land_pixels))
    os.makedirs(case_directory, exist_ok=True)
    dir_path = os.path.join(root, "global", "fineresolution", "dir", "dir_%s_1s_merit.tif" % continent)
    bsn_path = os.path.join(root, "global", "fineresolution", "bsn", "bsn_%s_1s.tif" % continent)
    rgn_path = os.path.join(partition_root, "global", "fineresolution", "rgn", "rgn_%s_1s.tif" % continent)
    land_pixels = 0
    mouths = 0
    without_region = 0
    with rasterio.open(dir_path) as dir_ds, rasterio.open(bsn_path) as bsn_ds, rasterio.open(rgn_path) as rgn_ds:
        transform = dir_ds.window_transform(Window(col_min, row_min, ncol, nrow))
        profile = dict(driver="GTiff", count=1, width=ncol, height=nrow, crs=dir_ds.crs, transform=transform, tiled=True,
                       blockxsize=512, blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER")
        with rasterio.open(os.path.join(case_directory, "input_dir.tif"), "w", dtype="uint8", nodata=MERIT_NODATA, **profile) as dir_out, \
             rasterio.open(os.path.join(case_directory, "parts_cut.tif"), "w", dtype="uint32", nodata=0, **profile) as parts_out:
            for row_start in range(0, nrow, STRIP_NROW):
                strip_nrow = min(STRIP_NROW, nrow - row_start)
                window = Window(col_min, row_min + row_start, ncol, strip_nrow)
                inside = bsn_ds.read(1, window=window) == basin_id
                direction = dir_ds.read(1, window=window)
                direction[~inside] = MERIT_NODATA
                region = rgn_ds.read(1, window=window).astype(np.uint32)
                region[~inside] = 0
                land_pixels += int(inside.sum())
                mouths += int(((direction[inside] == 0) | (direction[inside] == MERIT_SINK)).sum())     # the mouth, or an inland sink
                without_region += int((region[inside] == 0).sum())
                out_window = Window(0, row_start, ncol, strip_nrow)
                dir_out.write(direction, 1, window=out_window)
                parts_out.write(region, 1, window=out_window)
                if (row_start // STRIP_NROW) % 8 == 0:
                    log("prepare: row %d of %d, %d land pixels so far" % (row_start, nrow, land_pixels))
    if land_pixels != table_land_pixels or mouths != 1 or without_region != 0:
        raise SystemExit("prepare: %d land pixels (table says %d), %d terminals (one expected), %d land pixels without a region" % (land_pixels, table_land_pixels, mouths, without_region))
    the_case = load_case(case_directory, manifest={"nrow": str(nrow), "ncol": str(ncol), "land_pixels": str(land_pixels)})
    part_of_pixel, tile_count, tile_columns, tile_rows = three_ways.square_tiles_of_hydrosheds(
        the_case["flow_direction"], nrow, ncol, transform.c, transform.f, transform.a, transform.e)
    region_count = count_regions(the_case, case_directory)
    with open(os.path.join(case_directory, "case.txt"), "w") as file:
        file.write("continent %s\nbasin_id %d\ncapacity_square_degrees %d\nrectangle_row_min %d\nrectangle_col_min %d\nnrow %d\nncol %d\nland_pixels %d\n"
                   % (continent, basin_id, capacity, row_min, col_min, nrow, ncol, land_pixels))
        file.write("geotransform %.12f %.15f 0.000000000000 %.12f 0.000000000000 %.15f\n" % (transform.c, transform.a, transform.f, transform.e))
        file.write("flow_direction_file input_dir.tif (MERIT convention: 247 outside the basin, 0 the mouth, 255 a pit)\n")
        file.write("cut_parts_file parts_cut.tif (the region id of the published partition, 0 outside the basin)\n")
        file.write("square_parts HydroSHEDS v2 10 x 10 degree tiles, corners at multiples of 10 degrees (hydrosheds.org: named by the south-west corner, "
                   "e.g. n40w080); the rectangle touches %d columns x %d rows of tiles; %d tiles hold land of the basin; the cut has %d regions\n"
                   % (tile_columns, tile_rows, tile_count, region_count))
        file.write("earth_model WGS84 ellipsoid (a=6378137 m, 1/f=298.257223563)\n")
    log("prepared %s: %d land pixels, %d tiles of 10 degrees, %d regions at %d square degrees" % (case_directory, land_pixels, tile_count, region_count, capacity))


def count_regions(the_case, case_directory):
    with rasterio.open(os.path.join(case_directory, "parts_cut.tif")) as dataset:
        # the parts on the grid of the flow directions before they are read by position
        if (dataset.height, dataset.width) != (the_case["nrow"], the_case["ncol"]) or dataset.transform != the_case["transform"]:
            raise SystemExit("parts_cut.tif is not on the grid of the flow directions")
        region_of_pixel = dataset.read(1).ravel().astype(np.uint32)
    part_of_pixel, part_count = three_ways.regions_of_the_cut(the_case["flow_direction"], region_of_pixel)
    if part_count < 0:
        raise SystemExit("a land pixel has no region in parts_cut.tif")
    return int(part_count)


def load_case(case_directory, manifest=None):
    """the flow directions in memory in the library's convention, the row steps and areas, the geotransform"""
    if manifest is None:
        manifest = three_ways.read_case_manifest(case_directory)
    nrow, ncol = int(manifest["nrow"]), int(manifest["ncol"])
    with rasterio.open(os.path.join(case_directory, "input_dir.tif")) as dataset:
        if dataset.height != nrow or dataset.width != ncol:
            raise SystemExit("input_dir.tif does not match case.txt")
        flow_direction = dataset.read(1).ravel()
        transform = dataset.transform
        profile = dataset.profile
    flow_direction = three_ways.flow_directions_to_library_convention(flow_direction)
    land = int((flow_direction != three_ways.LIBRARY_NODATA).sum())
    if land != int(manifest["land_pixels"]):
        raise SystemExit("%d land pixels, case.txt says %s" % (land, manifest["land_pixels"]))
    steps_north_south, steps_east_west, areas = three_ways.row_steps_and_areas(nrow, transform.c, transform.f, transform.a, transform.e)
    return {"flow_direction": flow_direction, "nrow": nrow, "ncol": ncol, "transform": transform, "profile": profile,
            "step_metres_north_south": steps_north_south, "step_metres_east_west": steps_east_west, "pixel_area_m2_by_row": areas,
            "land_pixels": land}


def parts_of(the_case, case_directory, way):
    if way == "square":
        transform = the_case["transform"]
        part_of_pixel, part_count, tile_columns, tile_rows = three_ways.square_tiles_of_hydrosheds(
            the_case["flow_direction"], the_case["nrow"], the_case["ncol"], transform.c, transform.f, transform.a, transform.e)
        return part_of_pixel, int(part_count)
    with rasterio.open(os.path.join(case_directory, "parts_cut.tif")) as dataset:
        # the parts on the grid of the flow directions before they are read by position
        if (dataset.height, dataset.width) != (the_case["nrow"], the_case["ncol"]) or dataset.transform != the_case["transform"]:
            raise SystemExit("parts_cut.tif is not on the grid of the flow directions")
        region_of_pixel = dataset.read(1).ravel().astype(np.uint32)
    part_of_pixel, part_count = three_ways.regions_of_the_cut(the_case["flow_direction"], region_of_pixel)
    if part_count < 0:
        raise SystemExit("a land pixel has no region in parts_cut.tif")
    return part_of_pixel, int(part_count)


# =============================================================================
#  [2] Rasters out, strip by strip
# =============================================================================

def write_raster(the_case, path, flat_values, dtype, nodata):
    """a flat array of the rectangle to a GeoTIFF; the pixels outside the basin set to nodata"""
    nrow, ncol = the_case["nrow"], the_case["ncol"]
    values = flat_values.reshape(nrow, ncol)
    outside = (the_case["flow_direction"] == three_ways.LIBRARY_NODATA).reshape(nrow, ncol)
    profile = dict(the_case["profile"])
    profile.update(dtype=dtype, nodata=nodata, tiled=True, blockxsize=512, blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER")
    profile.pop("predictor", None)
    with rasterio.open(path, "w", **profile) as dataset:
        for row_start in range(0, nrow, STRIP_NROW):
            strip_nrow = min(STRIP_NROW, nrow - row_start)
            strip = values[row_start:row_start + strip_nrow].astype(dtype)
            strip[outside[row_start:row_start + strip_nrow]] = nodata
            dataset.write(strip, 1, window=Window(0, row_start, ncol, strip_nrow))


def read_raster_flat(path, dtype=np.float64):
    with rasterio.open(path) as dataset:
        return dataset.read(1).ravel().astype(dtype)


# =============================================================================
#  [3] One variable one way
# =============================================================================

def run_one(case_directory, way, variable):
    started = time.time()
    the_case = load_case(case_directory)
    nrow, ncol = the_case["nrow"], the_case["ncol"]
    direction = the_case["flow_direction"]
    steps_ns, steps_ew = the_case["step_metres_north_south"], the_case["step_metres_east_west"]
    way_directory = os.path.join(case_directory, "python_" + way)
    os.makedirs(way_directory, exist_ok=True)
    part_of_pixel, part_count = (None, 0) if way == "int64" else parts_of(the_case, case_directory, way)
    if way != "int64":
        log("%s: %d parts" % (way, part_count))
    path = os.path.join(way_directory, variable + ".tif")
    started_library = time.time()
    if variable in ("upg", "upa"):
        weight = None
        if variable == "upa":
            weight = np.repeat(the_case["pixel_area_m2_by_row"], ncol)
        if way == "int64":
            values, cost = three_ways.accumulate_whole_int64(direction, nrow, ncol, weight)
        else:
            values, cost = three_ways.accumulate_by_part(direction, nrow, ncol, part_of_pixel, part_count, steps_ns, steps_ew, weight)
        del weight
        seconds = time.time() - started_library
        write_raster(the_case, path, values, "float64", FLOAT_NODATA)
    elif variable == "shv":
        if way == "int64":
            values, cost = three_ways.shreve_whole_int64(direction, nrow, ncol)
        else:
            values, cost = three_ways.shreve_by_part(direction, nrow, ncol, part_of_pixel, part_count, steps_ns, steps_ew)
        seconds = time.time() - started_library
        write_raster(the_case, path, values, "uint32", 0)
    elif variable == "ldn":
        if way == "int64":
            values, cost = three_ways.distance_to_outlet_whole_int64(direction, nrow, ncol, steps_ns, steps_ew)
        else:
            values, cost = three_ways.distance_to_outlet_by_part(direction, nrow, ncol, part_of_pixel, part_count, steps_ns, steps_ew)
        seconds = time.time() - started_library
        write_raster(the_case, path, values, "float64", FLOAT_NODATA)
    elif variable == "lup":
        if way == "int64":
            values, source, cost = three_ways.upstream_flow_length_whole_int64(direction, nrow, ncol, steps_ns, steps_ew)
        else:
            values, source, cost = three_ways.upstream_flow_length_by_part(direction, nrow, ncol, part_of_pixel, part_count, steps_ns, steps_ew)
        seconds = time.time() - started_library
        write_raster(the_case, path, values, "float64", FLOAT_NODATA)
        write_raster(the_case, os.path.join(way_directory, "lup_source.tif"), source, "int64", -1)
        del source
    elif variable == "ord":
        if way == "int64":
            values, cost = three_ways.strahler_whole_int64(direction, nrow, ncol)
        else:
            values, cost = three_ways.strahler_by_part(direction, nrow, ncol, part_of_pixel, part_count, steps_ns, steps_ew)
        seconds = time.time() - started_library
        write_raster(the_case, path, values, "uint8", 0)
    elif variable == "hck":
        upa_path = os.path.join(case_directory, "python_int64", "upa.tif")
        if not os.path.exists(upa_path):
            raise SystemExit("hck needs python_int64/upa.tif of the same case: run int64 upa first")
        upstream_area = read_raster_flat(upa_path)
        if upstream_area.size != nrow * ncol:          # an upa of another case is refused
            raise SystemExit("%s is not the grid of this case" % upa_path)
        if way == "int64":
            values, cost = three_ways.hack_whole_int64(direction, nrow, ncol, upstream_area)
        else:
            values, cost = three_ways.hack_by_part(direction, nrow, ncol, part_of_pixel, part_count, steps_ns, steps_ew, upstream_area)
        del upstream_area
        seconds = time.time() - started_library
        write_raster(the_case, path, values, "uint8", 0)
    else:
        raise SystemExit("unknown variable " + variable)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    with open(os.path.join(way_directory, variable + ".cost.txt"), "w") as file:
        file.write("way %s\nvariable %s\nseconds_of_the_library_call %.3f\nseconds_of_the_process_so_far %.3f\npeak_rss_gb %.3f\n" % (way, variable, seconds, time.time() - started, peak_gb))
        for key, value in cost.items():
            file.write("%s %s\n" % (key, value))
    log("%s %s: library %.1f s, process %.1f s, peak %.2f GB, %d parts, %d exits, acyclic %d, sweeps %d"
        % (way, variable, seconds, time.time() - started, peak_gb, cost["part_count"], cost["exit_pixel_count"], cost["part_graph_is_acyclic"], cost["strahler_sweep_count"]))


# =============================================================================
#  [4] The comparisons
# =============================================================================

def compare_rasters(reference_path, other_path):
    """(compared, differing, first differing pixel, its two values), strip by strip, exact"""
    with rasterio.open(reference_path) as reference, rasterio.open(other_path) as other:
        nrow, ncol = reference.height, reference.width
        if other.height != nrow or other.width != ncol or other.transform != reference.transform:
            raise SystemExit("%s and %s are not the same rectangle" % (reference_path, other_path))
        compared = differing = 0
        first = None
        for row_start in range(0, nrow, STRIP_NROW):
            strip_nrow = min(STRIP_NROW, nrow - row_start)
            window = Window(0, row_start, ncol, strip_nrow)
            a = reference.read(1, window=window).astype(np.float64)
            b = other.read(1, window=window).astype(np.float64)
            unequal = a != b
            compared += a.size
            count = int(unequal.sum())
            if count and first is None:
                index = int(np.flatnonzero(unequal.ravel())[0])
                first = (row_start * ncol + index, float(a.ravel()[index]), float(b.ravel()[index]))
            differing += count
    return compared, differing, first


def compare_one(case_directory, variable):
    """python square and python cut against python int64; lup also its source pixel.

    Returns the number of comparisons that did not come out identical, missing answers included, so
    that a run can be gated on it (with the differences in the log alone the process would leave with
    0 whatever they said, and "the three ways agree" could not be shown from the exit status)."""
    lines = []
    failures = 0
    for file_name in ([variable, "lup_source"] if variable == "lup" else [variable]):
        reference = os.path.join(case_directory, "python_int64", file_name + ".tif")
        if not os.path.exists(reference):
            lines.append("%s: no python int64 answer\n" % file_name)
            failures += 1
            continue
        for other_name, other_path in (("python square", os.path.join(case_directory, "python_square", file_name + ".tif")),
                                       ("python cut", os.path.join(case_directory, "python_cut", file_name + ".tif"))):
            if not os.path.exists(other_path):
                lines.append("%s %s: no answer (did not run or did not fit)\n" % (file_name, other_name))
                failures += 1
                continue
            compared, differing, first = compare_rasters(reference, other_path)
            if differing == 0:
                lines.append("%s %s: identical to python int64 on all %d pixels of the rectangle\n" % (file_name, other_name, compared))
            else:
                lines.append("%s %s: %d of %d pixels DIFFER from python int64; first at pixel %d: %.6f vs %.6f\n" % (file_name, other_name, differing, compared, first[0], first[1], first[2]))
                failures += 1
    with open(os.path.join(case_directory, "compare_python.txt"), "a") as file:
        for line in lines:
            file.write(line)
            log("compare " + line.rstrip("\n"))
    return failures


def run_all(case_directory):
    """Every way and variable, then the comparisons.  Returns the number of runs that failed and of
    comparisons that did not come out identical; the caller leaves with it."""
    if os.path.exists(os.path.join(case_directory, "compare_python.txt")):
        os.remove(os.path.join(case_directory, "compare_python.txt"))
    failures = 0
    for way in WAYS:
        for variable in VARIABLES:
            log("run %s %s" % (way, variable))
            command = [sys.executable, os.path.abspath(__file__), "run", case_directory, way, variable]
            code = subprocess.run(command).returncode
            log("done %s %s: exit %d" % (way, variable, code))
            if code != 0:
                failures += 1
    for variable in VARIABLES:
        failures += compare_one(case_directory, variable)
    log("all done; see %s" % os.path.join(case_directory, "compare_python.txt"))
    if failures:
        log("%d runs or comparisons did not come out identical" % failures)
    return failures


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 6 and arguments[0] == "prepare":
        prepare_case(arguments[1], arguments[2], int(arguments[3]), int(arguments[4]), arguments[5])
    elif len(arguments) == 4 and arguments[0] == "run" and arguments[2] in WAYS and arguments[3] in VARIABLES:
        run_one(arguments[1], arguments[2], arguments[3])
    elif len(arguments) == 3 and arguments[0] == "compare":
        sys.exit(1 if compare_one(arguments[1], arguments[2]) else 0)
    elif len(arguments) == 2 and arguments[0] == "all":
        sys.exit(1 if run_all(arguments[1]) else 0)
    else:
        print(__doc__)
        sys.exit(2)
