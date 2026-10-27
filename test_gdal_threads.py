"""GDAL's threads.  No data needed: synthetic rasters in a temporary directory, which is left there.

With GDAL 3.6 a compressed raster that is written piece by piece and read between the pieces can stand still when
GDAL decodes on its own threads: one of them, needing room in the block cache, pushes out a written tile, waits for
a free thread to compress it and holds the lock the other decoding threads wait for.
Scenarios 1 to 3 run in a process of their own with
a block cache of 16 MB, which their rasters outgrow; a process that has not ended after the time limit is stopped
and counted as failed: it stands still, or the machine is too slow for the limit (FLOWDIVIDE_TEST_LIMIT_SECONDS,
300 by default; a run takes a few seconds).

  1  two layers of squares burned by fd1.4 into a grid of 6016 x 8192, the second over half of the first, six times
     over: each run ends, every pixel holds the code of the last square burned over it, and the thread's own
     GDAL_NUM_THREADS is afterwards what it was before (not set, or the value it had).
     Six repetitions check a failure that need not appear on every run.
  2  strips of a raster of 16384 x 4096 written in turn with reads of the rows around them from another raster, a
     GeoTIFF whose name does not end in .tif, opened with fd1.open_raster, GDAL_NUM_THREADS=8 in the environment:
     the run ends and the raster holds what was written.
  3  a raster opened to update with fd1.open_raster(path, "r+"), read back and written window after window as FD3
     writes an attribute region after region, GDAL_NUM_THREADS=8 in the environment: the run ends and the raster
     holds what was written
  4  what open_raster passes to rasterio.open, seen by standing in for it: NUM_THREADS=1 and the mode asked for,
     to read and to update, for a GeoTIFF under any name; no option for another format, in either mode; for a
     raster that is not there, the error rasterio.open itself raises.  And no module of the
     package opens a raster to read or to update in another way: read as Python (ast), every call of rasterio's
     open outside open_raster creates a raster (its second argument is "w"), and rasterio is imported under no
     other name
"""
import ast
import os
import subprocess
import sys
import tempfile
import time

LIMIT_SECONDS = float(os.environ.get("FLOWDIVIDE_TEST_LIMIT_SECONDS", "300"))
BURN_NROW = 8192
BURN_NCOL = 6016
SQUARE = 128                 # the side of a square in pixels: a square holds a quarter of a tile across
PIXEL = 1.0 / 1200.0
WEST = 79.0
NORTH = 10.0
STRIPS_NROW = 4096
STRIPS_NCOL = 16384
STRIP = 512
UPDATE_NROW = 6144
UPDATE_NCOL = 8192
# windows that overlap and do not end on tiles, as the rectangles of regions do
UPDATE_WINDOWS = ((0, 0, 3000, 5000), (2500, 300, 3644, 4000), (100, 4200, 4000, 3992), (1000, 1000, 4100, 6000),
                  (0, 0, 6144, 8192), (2047, 513, 2050, 3000))
MODULES = ("fd1_partition.py", "fd2_views.py", "fd3_attributes.py", "tile_kernels.py", "flowdivide.py")
HERE = os.path.dirname(os.path.abspath(__file__))

failures = 0


def check(condition, what):
    global failures
    print("  %s  %s" % ("ok  " if condition else "FAIL", what))
    if not condition:
        failures += 1


def code_of_square(layer, square_row, square_col):
    """the code a square burns: three digits, as fd1.4 asks; the two layers give a square different codes"""
    return 100 + ((square_row * 7 + square_col * 13) % 800 if layer == 0 else (square_row * 5 + square_col * 3) % 899)


def burn_inputs(directory):
    """the grid and the two polygon files of scenario 1: every square in the first file, the squares whose row and
    column add up to an odd number in the second.  The squares' sides lie on pixel edges."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from osgeo import ogr, osr
    dir_path = os.path.join(directory, "burn_grid.tif")
    with rasterio.open(dir_path, "w", driver="GTiff", height=BURN_NROW, width=BURN_NCOL, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(WEST, NORTH, PIXEL, PIXEL), tiled=True, blockxsize=512,
                       blockysize=512, compress="DEFLATE") as out:
        out.write(np.ones((BURN_NROW, BURN_NCOL), np.uint8), 1)
    for layer_index in range(2):
        path = os.path.join(directory, "burn_layer%d.shp" % layer_index)
        system = osr.SpatialReference()
        system.ImportFromEPSG(4326)
        system.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        source = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(path)
        layer = source.CreateLayer("layer%d" % layer_index, system, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("PFAF_ID", ogr.OFTInteger))
        for square_row in range(BURN_NROW // SQUARE):
            for square_col in range(BURN_NCOL // SQUARE):
                if layer_index == 1 and (square_row + square_col) % 2 == 0:
                    continue
                west = WEST + square_col * SQUARE * PIXEL
                east = WEST + (square_col + 1) * SQUARE * PIXEL
                north = NORTH - square_row * SQUARE * PIXEL
                south = NORTH - (square_row + 1) * SQUARE * PIXEL
                ring = ogr.Geometry(ogr.wkbLinearRing)
                for x, y in ((west, north), (east, north), (east, south), (west, south), (west, north)):
                    ring.AddPoint_2D(x, y)
                polygon = ogr.Geometry(ogr.wkbPolygon)
                polygon.AddGeometry(ring)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("PFAF_ID", code_of_square(layer_index, square_row, square_col))
                feature.SetGeometry(polygon)
                layer.CreateFeature(feature)
        source = None


def burn_in_this_process(directory, label):
    """scenario 1's own process: the burn, then what the file holds and what the thread's setting is"""
    import numpy as np
    import rasterio
    from osgeo import gdal
    import fd1_partition as fd1
    dir_path = os.path.join(directory, "burn_grid.tif")
    layers = [(os.path.join(directory, "burn_layer%d.shp" % layer_index), None) for layer_index in range(2)]
    out_path = os.path.join(directory, "burned_%s.tif" % label)
    before = "4" if label.endswith("set") else None        # a setting of the thread's own must come back as it was
    gdal.SetThreadLocalConfigOption("GDAL_NUM_THREADS", before)
    codes, polygons = fd1._burn_level3_layers(dir_path, out_path, layers, "burn")
    after = gdal.GetThreadLocalConfigOption("GDAL_NUM_THREADS", None)
    with rasterio.open(out_path) as written:
        pixels = written.read(1)
    expected = np.empty((BURN_NROW, BURN_NCOL), np.uint16)
    for square_row in range(BURN_NROW // SQUARE):
        for square_col in range(BURN_NCOL // SQUARE):
            expected[square_row * SQUARE:(square_row + 1) * SQUARE, square_col * SQUARE:(square_col + 1) * SQUARE] = \
                code_of_square((square_row + square_col) % 2, square_row, square_col)
    print("RESULT pixels_as_expected=%s polygons=%d setting_given_back=%s" % (
        bool(np.array_equal(pixels, expected)), polygons, after == before))


def profile_of(fd1, nrow, ncol, dtype, nodata):
    """a raster as the steps create theirs (fd1_partition.raster_profile, without a grid)"""
    return {"driver": "GTiff", "height": nrow, "width": ncol, "count": 1, "dtype": dtype, "nodata": nodata,
            "tiled": True, "blockxsize": fd1.RASTER_BLOCK, "blockysize": fd1.RASTER_BLOCK, "compress": "DEFLATE",
            "zlevel": 6, "predictor": 2, "BIGTIFF": "YES", "NUM_THREADS": fd1.GDAL_WRITE_THREADS}


def value_strip(row0):
    """what a strip of the raster that is read holds: a low byte that does not compress away, the row above it"""
    import numpy as np
    generator = np.random.default_rng(row0)
    return generator.integers(1, 257, (STRIP, STRIPS_NCOL), dtype=np.uint32) + np.uint32(row0 * 256)


def strips_in_this_process(directory, label, how):
    """scenario 2's own process: two rasters written strip by strip, the second after the rows of the first around
    each strip were read (a ring of one row, as the steps read the flow directions around a tile).  how: "package"
    opens the first raster with fd1.open_raster; any other value uses rasterio.open(path), to check whether
    this machine's GDAL can stand still (not part of the test)."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import fd1_partition as fd1
    read_path = os.path.join(directory, "strips_read_%s.dat" % label)       # a GeoTIFF by its driver, not its name
    out_path = os.path.join(directory, "strips_written_%s.tif" % label)
    profile = profile_of(fd1, STRIPS_NROW, STRIPS_NCOL, "uint32", 0)
    with rasterio.open(read_path, "w", **profile) as out:
        for row0 in range(0, STRIPS_NROW, STRIP):
            out.write(value_strip(row0), 1, window=Window(0, row0, STRIPS_NCOL, STRIP))
    source = fd1.open_raster(read_path) if how == "package" else rasterio.open(read_path)
    with source, rasterio.open(out_path, "w", **profile) as out:
        for row0 in range(0, STRIPS_NROW, STRIP):
            read_row0 = max(row0 - 1, 0)
            read_row1 = min(row0 + STRIP + 1, STRIPS_NROW)
            around = source.read(1, window=Window(0, read_row0, STRIPS_NCOL, read_row1 - read_row0))
            out.write(around[row0 - read_row0:row0 - read_row0 + STRIP] ^ np.uint32(1), 1, window=Window(0, row0, STRIPS_NCOL, STRIP))
    as_written = True
    with fd1.open_raster(out_path) as written:
        for row0 in range(0, STRIPS_NROW, STRIP):
            if not np.array_equal(written.read(1, window=Window(0, row0, STRIPS_NCOL, STRIP)), value_strip(row0) ^ np.uint32(1)):
                as_written = False
    print("RESULT pixels_as_expected=%s threads_in_the_environment=%s" % (as_written, os.environ.get("GDAL_NUM_THREADS")))


def update_in_this_process(directory, label):
    """scenario 3's own process: a raster created, closed, opened to update, and read back and written window after
    window, another raster read beside it; what it must hold is kept in memory"""
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import fd1_partition as fd1
    read_path = os.path.join(directory, "update_read_%s.tif" % label)
    out_path = os.path.join(directory, "update_written_%s.tif" % label)
    generator = np.random.default_rng(7)
    directions = generator.integers(0, 200, (UPDATE_NROW, UPDATE_NCOL), dtype=np.uint8)
    profile = profile_of(fd1, UPDATE_NROW, UPDATE_NCOL, "float64", -9999.0)
    with rasterio.open(read_path, "w", **dict(profile, dtype="uint8", nodata=247, predictor=1)) as out:
        out.write(directions, 1)
    with rasterio.open(out_path, "w", **profile):
        pass
    expected = np.full((UPDATE_NROW, UPDATE_NCOL), -9999.0)
    with fd1.open_raster(read_path) as dir_dataset, fd1.open_raster(out_path, "r+") as out_dataset:
        for visit, (row0, col0, nrow, ncol) in enumerate(UPDATE_WINDOWS):
            window = Window(col0, row0, ncol, nrow)
            read = dir_dataset.read(1, window=window)
            values = out_dataset.read(1, window=window)
            values[values == -9999.0] = 0.0
            values += read.astype(np.float64) + visit + generator.random((nrow, ncol))
            out_dataset.write(values, 1, window=window)
            expected[row0:row0 + nrow, col0:col0 + ncol] = values
    with fd1.open_raster(out_path) as written:
        as_written = bool(np.array_equal(written.read(1), expected))
    print("RESULT pixels_as_expected=%s threads_in_the_environment=%s" % (as_written, os.environ.get("GDAL_NUM_THREADS")))


def in_a_process_of_its_own(arguments, directory, threads):
    """one scenario in a new process with a block cache of 16 MB and GDAL_NUM_THREADS as given (None: not set), its
    caches of compiled code in the test's directory.  Returns (ended, seconds, its RESULT line or the end of what it
    printed)."""
    environment = dict(os.environ, GDAL_CACHEMAX="16", PYTHONDONTWRITEBYTECODE="1",
                       NUMBA_CACHE_DIR=os.path.join(directory, "numba_cache"))
    environment.pop("GDAL_NUM_THREADS", None)
    if threads is not None:
        environment["GDAL_NUM_THREADS"] = threads
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__)] + list(arguments), env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    try:
        output, _ = process.communicate(timeout=LIMIT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return False, time.monotonic() - started, "not ended after %.0f s and stopped" % LIMIT_SECONDS
    results = [line for line in output.splitlines() if line.startswith("RESULT")]
    return process.returncode == 0, time.monotonic() - started, results[0] if results else output[-400:]


def opens_outside_the_helper(source):
    """the line numbers of the calls of rasterio's open in a module's source that are neither inside open_raster nor
    the creation of a raster (a second positional argument "w"), and the names rasterio or its open were imported
    under besides `rasterio` itself"""
    tree = ast.parse(source)
    other_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            other_names += [alias.asname for alias in node.names if alias.name.split(".")[0] == "rasterio" and alias.asname]
        elif isinstance(node, ast.ImportFrom) and node.module == "rasterio":
            other_names += [alias.asname or alias.name for alias in node.names if alias.name == "open"]
    helper_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "open_raster":
            helper_lines = set(range(node.lineno, node.end_lineno + 1))
    lines = []
    calls = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open"):
            continue
        owner = node.func.value
        is_rasterio = (isinstance(owner, ast.Name) and owner.id == "rasterio") or \
                      (isinstance(owner, ast.Attribute) and owner.attr == "rasterio")
        if not is_rasterio:
            continue
        calls += 1
        creates = len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "w"
        if node.lineno not in helper_lines and not creates:
            lines.append(node.lineno)
    return calls, lines, other_names


def main():
    directory = tempfile.mkdtemp(prefix="flowdivide_gdal_threads_")
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(directory, "numba_cache")     # the compiled kernels of this process too
    sys.dont_write_bytecode = True

    print("1  the burn of two layers")
    burn_inputs(directory)
    for label in ("first", "second", "third", "fourth", "fifth", "sixth_set"):
        ended, seconds, result = in_a_process_of_its_own(["--burn", directory, label], directory, None)
        check(ended, "run %s ended (%.1f s; %s)" % (label, seconds, result if not ended else "the limit is %.0f s" % LIMIT_SECONDS))
        check("pixels_as_expected=True" in result and "polygons=4512" in result,
              "every pixel holds the code of the last square burned over it, 4512 squares in the two layers (%s)" % result)
        check("setting_given_back=True" in result, "the thread's own GDAL_NUM_THREADS is what it was before the burn")

    print("2  strips written and read in turn, GDAL_NUM_THREADS=8 in the environment")
    for label in ("first", "second"):
        ended, seconds, result = in_a_process_of_its_own(["--strips", directory, label, "package"], directory, "8")
        check(ended, "run %s ended (%.1f s; %s)" % (label, seconds, result if not ended else "the limit is %.0f s" % LIMIT_SECONDS))
        check("pixels_as_expected=True" in result and "threads_in_the_environment=8" in result,
              "the raster holds what was written, with 8 still in the environment (%s)" % result)

    print("3  a raster updated in place, GDAL_NUM_THREADS=8 in the environment")
    ended, seconds, result = in_a_process_of_its_own(["--update", directory, "first"], directory, "8")
    check(ended, "the run ended (%.1f s; %s)" % (seconds, result if not ended else "the limit is %.0f s" % LIMIT_SECONDS))
    check("pixels_as_expected=True" in result and "threads_in_the_environment=8" in result,
          "the raster holds what was written, with 8 still in the environment (%s)" % result)

    print("4  how the package opens a raster")
    import numpy as np
    import rasterio
    import fd1_partition as fd1
    small = os.path.join(directory, "small.tif")
    other = os.path.join(directory, "small.img")
    for path, driver in ((small, "GTiff"), (other, "HFA")):
        with rasterio.open(path, "w", driver=driver, height=8, width=8, count=1, dtype="uint8", crs="EPSG:4326",
                           transform=rasterio.transform.from_origin(0.0, 8.0, 1.0, 1.0)) as out:
            out.write(np.ones((8, 8), np.uint8), 1)
    renamed = os.path.join(directory, "small_geotiff.dat")
    with rasterio.open(renamed, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=rasterio.transform.from_origin(0.0, 8.0, 1.0, 1.0)) as out:
        out.write(np.ones((8, 8), np.uint8), 1)
    asked = []
    rasterio_open = rasterio.open

    def standing_in(*arguments, **options):
        asked.append((arguments, options))
        return rasterio_open(*arguments, **options)

    fd1.rasterio.open = standing_in
    try:
        for path, what in ((small, "a GeoTIFF"), (renamed, "a GeoTIFF whose name does not end in .tif")):
            del asked[:]
            with fd1.open_raster(path) as dataset:
                read_sum = int(dataset.read(1).sum())
            check(asked[-1] == ((path, "r"), {"NUM_THREADS": 1}) and read_sum == 64,
                  "%s to read: rasterio.open(path, \"r\", NUM_THREADS=1)" % what)
            del asked[:]
            with fd1.open_raster(path, "r+") as dataset:
                dataset.write(np.full((8, 8), 2, np.uint8), 1)
            check(asked[-1] == ((path, "r+"), {"NUM_THREADS": 1}), "%s to update: rasterio.open(path, \"r+\", NUM_THREADS=1)" % what)
            with fd1.open_raster(path) as dataset:
                check(int(dataset.read(1).sum()) == 128, "and what was written into it is read back")
        del asked[:]
        with fd1.open_raster(other) as dataset:
            read_sum = int(dataset.read(1).sum())
        check(asked[-1] == ((other, "r"), {}) and read_sum == 64, "another format to read: opened without the option")
        del asked[:]
        with fd1.open_raster(other, "r+") as dataset:
            dataset.write(np.full((8, 8), 3, np.uint8), 1)
        check(asked[-1] == ((other, "r+"), {}), "another format to update: opened without the option")
        with fd1.open_raster(other) as dataset:
            check(int(dataset.read(1).sum()) == 192, "and what was written into it is read back")
        missing = os.path.join(directory, "a_raster_that_is_not_there.tif")
        errors = []
        for opener in (lambda: rasterio_open(missing, "r+"), lambda: fd1.open_raster(missing, "r+"),
                       lambda: rasterio_open(missing), lambda: fd1.open_raster(missing)):
            try:
                opener().close()
                errors.append(None)
            except Exception as error:
                errors.append(type(error))
        check(errors[0] is not None and errors[0] is errors[1] and errors[2] is not None and errors[2] is errors[3],
              "a raster that is not there: the error rasterio.open itself raises, to update and to read (%s, %s)" % (
                  getattr(errors[1], "__name__", None), getattr(errors[3], "__name__", None)))
    finally:
        fd1.rasterio.open = rasterio_open
    for module in MODULES:
        with open(os.path.join(HERE, module), encoding="utf-8") as source:
            calls, lines, other_names = opens_outside_the_helper(source.read())
        check(not lines and not other_names,
              "%s: of %d calls of rasterio.open, each creates a raster or is open_raster's own%s%s" % (
                  module, calls, "; not so at lines %s" % lines if lines else "",
                  "; rasterio or its open also imported as %s" % other_names if other_names else ""))

    print("%d failures" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--burn":
        burn_in_this_process(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "--strips":
        strips_in_this_process(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4 and sys.argv[1] == "--update":
        update_in_this_process(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
