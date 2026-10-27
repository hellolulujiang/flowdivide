"""Very fine geographic grids, and the lock of a run.  No data needed.

  1  a geographic grid whose largest pixels are below a square metre is refused when it is opened
  2  a geographic grid finer than about a sixth of an arc-second gets a default block smaller than a degree, on which
     every default capacity holds at least one block; a grid of a third of an arc-second keeps its block of a degree
  3  a run holds the lock on the output's _logs directory: a second process is refused while it is held, the lock
     is free afterwards, and no file is made under the output
  4  on a file system that gives no lock on a directory the lock is on _logs/run.lock, and a second
     taker is refused; a file system that gives no lock at all is refused in words
  5  what a stopped run left under a temporary name is moved into _tmp_/, not written over, by the
     table writer and by the text writer
  6  two tiles that share pixels are refused from their headers; tiles side by side are taken
  8  a move never replaces a file, also where no hard link can be made (the system's exclusive
     rename); GDAL's side files .aux.xml, .msk and .ovr go aside with a raster, and without it when it is gone
  7  an empty output is one that was not written whole, but for an output its step says may be empty
     (the list of the basins cut, where none had to be): a finished run is not made again for it
  9  the default views of a small grid of one's own leave out a view a quarter of whose cell is larger than
     the whole grid (all its tiles), the finest made finer if it must be, and the partition and its views are made;
     --view-factors sets the views as they are given, each lowered to a divisor of the block
 10  the MERIT recode on disk: refused when its marker is gone while the steps after it keep theirs, or when
     its marker lacks the refusal of directions into nodata (read as a whole argument); --only fd1.0 and --summary-only make or plan the
     recode again
 11  the tile kernels' command is refused while another run holds the lock of its output
"""
import errno
import os
import stat
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

import fd1_partition as fd1
import fd_tables
import flowdivide

failures = 0


def check(condition, what):
    global failures
    print("  %s  %s" % ("ok  " if condition else "FAIL", what))
    if not condition:
        failures += 1


def geographic_raster(directory, name, pixel_degrees, top_latitude=10.0):
    """a 4 x 4 raster of flow directions in degrees, its pixels of the given size"""
    path = os.path.join(directory, name)
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(20.0, top_latitude, pixel_degrees, pixel_degrees), nodata=247) as dataset:
        dataset.write(np.full((4, 4), 1, np.uint8), 1)
    return path


def main():
    directory = tempfile.mkdtemp(prefix="flowdivide_fine_grids_")

    print("1  pixels below a square metre")
    try:
        fd1.Grid(geographic_raster(directory, "millionth.tif", 1e-6))
        check(False, "a grid of 0.000001-degree pixels (0.012 m2) is refused")
    except fd1.FlowDivideError as error:
        check("less than a square metre" in str(error) or "less than\na square metre" in str(error),
              "a grid of 0.000001-degree pixels (0.012 m2) is refused, and the message says why")
    grid = fd1.Grid(geographic_raster(directory, "hundred_thousandth.tif", 1e-5))
    check(grid is not None, "a grid of 0.00001-degree pixels (1.2 m2) is opened")

    print("2  the default block")
    check(grid.block_pixels == 10000, "0.00001-degree pixels: the default block is 10,000 pixels, a tenth of a degree (%d)" % grid.block_pixels)
    check(all(flowdivide.capacity_blocks_of(name, grid.block_pixels) >= 1 for name in ("2^31", "2^30", "2^29")),
          "and the capacities 2^31, 2^30 and 2^29 each hold at least one block")
    third = fd1.Grid(geographic_raster(directory, "third_of_a_second.tif", 1.0 / 10800.0))
    check(third.block_pixels == 10800, "a third of an arc-second keeps its block of a degree (%d)" % third.block_pixels)
    second = fd1.Grid(geographic_raster(directory, "one_second.tif", 1.0 / 3600.0))
    check(second.block_pixels == 3600, "one arc-second keeps its block of a degree (%d)" % second.block_pixels)

    print("3  the lock of a run")
    root = os.path.join(directory, "output")
    os.makedirs(root)
    other = ("import sys; sys.path.insert(0, %r); import fd_tables\n"
             "try:\n"
             "    with fd_tables.the_only_run_on(%r):\n"
             "        print('held')\n"
             "except fd_tables.TableError:\n"
             "    print('refused')\n") % (os.path.dirname(os.path.abspath(__file__)), root)
    with fd_tables.the_only_run_on(root):
        said = subprocess.run([sys.executable, "-c", other], capture_output=True, text=True).stdout.strip()
        check(said == "refused", "a second process is refused while the lock is held (%s)" % said)
    said = subprocess.run([sys.executable, "-c", other], capture_output=True, text=True).stdout.strip()
    check(said == "held", "the lock is free once the run has ended (%s)" % said)
    files = [name for _, _, names in os.walk(root) for name in names]
    check(files == [], "no file is made under the output (%s)" % files)

    print("4  a file system without a lock on a directory")

    class NoLockOnADirectory:
        """the system's locks, but for a directory (as NFS: an exclusive lock needs a file open for writing)"""
        LOCK_EX, LOCK_NB, LOCK_UN = fd_tables.fcntl.LOCK_EX, fd_tables.fcntl.LOCK_NB, fd_tables.fcntl.LOCK_UN
        real = fd_tables.fcntl

        def flock(self, descriptor, operation):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EBADF, "Bad file descriptor")
            return self.real.flock(descriptor, operation)

    class NoLockAtAll(NoLockOnADirectory):
        def flock(self, descriptor, operation):
            raise OSError(errno.ENOLCK, "No locks available")

    the_systems = fd_tables.fcntl
    other_root = os.path.join(directory, "output_on_nfs")
    os.makedirs(other_root)
    try:
        fd_tables.fcntl = NoLockOnADirectory()
        with fd_tables.the_only_run_on(other_root):
            check(os.path.exists(os.path.join(other_root, "_logs", "run.lock")), "the lock is on _logs/run.lock")
            try:
                with fd_tables.the_only_run_on(other_root):
                    check(False, "a second taker is refused while the lock file is held")
            except fd_tables.TableError as error:
                check("another run is writing" in str(error), "a second taker is refused while the lock file is held")
        with fd_tables.the_only_run_on(other_root):
            check(True, "the lock file is free once the run has ended")
        fd_tables.fcntl = NoLockAtAll()
        try:
            with fd_tables.the_only_run_on(other_root):
                check(False, "a file system that gives no lock is refused")
        except fd_tables.TableError as error:
            check("gives no lock" in str(error), "a file system that gives no lock is refused (%s)" % str(error)[-60:])
    finally:
        fd_tables.fcntl = the_systems

    print("5  what a stopped run left under a temporary name")
    import pandas as pd
    table_path = os.path.join(directory, "left", "table.csv")
    os.makedirs(os.path.dirname(table_path))
    with open(table_path + ".partial", "w") as handle:
        handle.write("left by a stopped run")
    fd1.write_table(pd.DataFrame({"a": [1, 2]}), table_path)
    kept = [os.path.join(folder, name) for folder, _, names in os.walk(os.path.dirname(table_path)) for name in names
            if "_tmp_" in folder]
    check(len(kept) == 1 and os.path.basename(kept[0]) == "table.csv.partial"
          and open(kept[0]).read() == "left by a stopped run", "the table writer moved it into _tmp_/ as it was (%s)"
          % [os.path.relpath(name, directory) for name in kept])
    check(open(table_path).read().split() == ["a", "1", "2"] and not os.path.exists(table_path + ".partial"),
          "the table is the new one, and no temporary is left")
    text_path = os.path.join(directory, "left", "list.txt")
    stale = "%s.tmp.%d" % (text_path, os.getpid())
    with open(stale, "w") as handle:
        handle.write("left by a process of the same number")
    fd_tables.write_text(text_path, "new\n")
    kept = [name for folder, _, names in os.walk(os.path.dirname(text_path)) for name in names if "_tmp_" in folder]
    check(os.path.basename(stale) in kept and open(text_path).read() == "new\n",
          "the text writer moved it into _tmp_/ and wrote the new text")

    print("6  tiles that share pixels")

    def tile(name, west, north, ncol):
        tile_path = os.path.join(directory, name)
        with rasterio.open(tile_path, "w", driver="GTiff", height=4, width=ncol, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(west, north, 0.25, 0.25), nodata=247) as dataset:
            dataset.write(np.full((4, ncol), 1, np.uint8), 1)
        return tile_path

    west_tile, east_tile = tile("tile_west.tif", 20.0, 10.0, 4), tile("tile_east.tif", 21.0, 10.0, 4)
    south_tile, wide_tile = tile("tile_south.tif", 20.0, 9.0, 8), tile("tile_wide.tif", 20.75, 10.0, 4)
    try:
        fd1.refuse_tiles_that_overlap([west_tile, east_tile, south_tile])
        check(True, "three tiles side by side and below are taken")
    except fd1.FlowDivideError as error:
        check(False, "three tiles side by side and below are taken (%s)" % error)
    try:
        fd1.refuse_tiles_that_overlap([west_tile, east_tile, wide_tile])
        check(False, "a tile over the last column of another is refused")
    except fd1.FlowDivideError as error:
        check("overlap" in str(error), "a tile over the last column of another is refused")
    fd1.refuse_tiles_that_overlap([west_tile])
    check(True, "one tile alone has nothing to overlap")

    print("7  an output that may be empty")
    listing = os.path.join(directory, "cut_basins_test.txt")
    open(listing, "w").close()
    marker = os.path.join(directory, "step.ok")
    with open(marker, "w") as handle:
        handle.write("2026-10-11 00:00:00 version x\nsignature: s\ndone_marked: \n")
    chain = flowdivide.Chain.__new__(flowdivide.Chain)
    chain.marker = lambda label: marker
    plain = flowdivide.Chain.Step("cut", "arguments", None, (), outputs=[listing])
    named = flowdivide.Chain.Step("cut", "arguments", None, (), outputs=[listing], may_be_empty=[listing])
    check(chain.an_output_lost_its_marker(plain), "an empty output is taken for one that was not written whole")
    check(not chain.an_output_lost_its_marker(named), "the empty list of the basins cut is taken as it is")

    print("8  moves that never replace, and GDAL's side files")
    moves = os.path.join(directory, "moves")
    os.makedirs(moves)

    def written(name, text):
        with open(os.path.join(moves, name), "w") as handle:
            handle.write(text)
        return os.path.join(moves, name)

    first, second = written("first", "1"), written("second", "2")
    check(fd_tables.rename_exclusive(first, second) is False and open(first).read() == "1" and open(second).read() == "2",
          "the exclusive rename leaves a taken name and both files as they are")
    check(fd_tables.rename_exclusive(first, os.path.join(moves, "third")) is True and not os.path.exists(first)
          and open(os.path.join(moves, "third")).read() == "1", "the exclusive rename moves a file to a free name")
    os.makedirs(os.path.join(moves, "held_blocks"))
    check(fd_tables.move_without_replacing(os.path.join(moves, "held_blocks"), os.path.join(moves, "held_blocks_moved")) is True
          and os.path.isdir(os.path.join(moves, "held_blocks_moved")), "a directory, of which no hard link can be made, is moved")
    raster = written("ldn.tif", "raster")
    for suffix in fd_tables.GDAL_SIDE_FILES:
        written("ldn.tif" + suffix, "side " + suffix)
    fd_tables.set_aside(raster)
    kept = sorted(name for folder, _, names in os.walk(moves) for name in names if "_tmp_" in folder)
    check(kept == ["ldn.tif", "ldn.tif.aux.xml", "ldn.tif.msk", "ldn.tif.ovr"]
          and not [name for name in os.listdir(moves) if name.startswith("ldn.tif")],
          "a raster goes aside with its statistics, mask and overviews (%s)" % kept)
    written("lup.tif.ovr", "overviews of a raster that is gone")
    fd_tables.set_aside(os.path.join(moves, "lup.tif"))
    check(not os.path.exists(os.path.join(moves, "lup.tif.ovr")), "side files go aside also when their raster is gone")

    here = os.path.dirname(os.path.abspath(__file__))

    def run(*arguments):
        process = subprocess.run([sys.executable, "-B", os.path.join(here, "flowdivide.py"), "run"] + list(arguments),
                                 capture_output=True, text=True, cwd=directory)
        return process.returncode, process.stdout + process.stderr

    print("9  the views of a small grid of one's own")
    small = os.path.join(directory, "small_10m.tif")
    directions = np.full((100, 100), 1, np.uint8)        # ESRI codes: east along every row, south down the last column
    directions[:, -1] = 4
    directions[-1, -1] = 1                               # the outlet: east, off the grid
    with rasterio.open(small, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(500000.0, 4000000.0, 10.0, 10.0), nodata=255) as dataset:
        dataset.write(directions, 1)
    probe = fd1.Grid(small)
    check([factor for _, factor in flowdivide.views_by_cell_size(probe)] == [10, 50],
          "a 1 km grid of 10 m pixels: the 2.5 km view (a quarter cell of 1.6 km2) is left out (%s)" % flowdivide.views_by_cell_size(probe))
    check([factor for _, factor in flowdivide.views_by_cell_size(probe, pixels=100 * 600)] == [10, 50, 250],
          "the same tile among tiles of 100 x 600 pixels in all: the grid as a whole is judged, the 2.5 km view kept")
    check([factor for _, factor in flowdivide.views_by_cell_size(probe, pixels=16)] == [5],
          "a grid of 4 x 4 pixels: the finest view made finer, 5 pixels to a cell (a quarter of it below the 16 pixels)")
    check([factor for _, factor in flowdivide.views_by_cell_size(probe, (250,), prune=False)] == [250],
          "views given by name are kept (250 divides the block)")
    metre = os.path.join(directory, "one_metre.tif")
    with rasterio.open(metre, "w", driver="GTiff", height=124, width=126, count=1, dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(500000.0, 4000000.0, 1.0, 1.0), nodata=255) as dataset:
        dataset.write(np.full((124, 126), 1, np.uint8), 1)
    check(250 in [factor for _, factor in flowdivide.views_by_cell_size(fd1.Grid(metre))],
          "a 1 m grid of 124 x 126 pixels (0.015624 km2) keeps its 250-pixel view, whose threshold FD2 rounds to 0.0156 km2")
    stretched = os.path.join(directory, "stretched.tif")
    with rasterio.open(stretched, "w", driver="GTiff", height=19, width=821, count=1, dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(500000.0, 4000000.0, 1.0, 1.00004), nodata=255) as dataset:
        dataset.write(np.full((19, 821), 1, np.uint8), 1)
    check(250 in [factor for _, factor in flowdivide.views_by_cell_size(fd1.Grid(stretched))],
          "19 x 821 pixels of 1 x 1.00004 m (15,599.6 m2, stored as 15,600): the 250-pixel view (0.0156 km2) is kept")
    tiles = os.path.join(directory, "tiles_10m")
    os.makedirs(tiles)
    for name, column0, width in (("a.tif", 0, 100), ("b.tif", 100, 500)):
        with rasterio.open(os.path.join(tiles, name), "w", driver="GTiff", height=100, width=width, count=1, dtype="uint8",
                           crs="EPSG:32633", transform=from_origin(500000.0 + 10.0 * column0, 4000000.0, 10.0, 10.0), nodata=255) as dataset:
            dataset.write(np.full((100, width), 1, np.uint8), 1)
    check(flowdivide.footprint_pixels(tiles, fd1.Grid(os.path.join(tiles, "a.tif"))) == 60000,
          "the footprint of a directory of tiles is that of all of them (100 x 600)")
    tiny = os.path.join(directory, "tiny_10m.tif")
    with rasterio.open(tiny, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(500000.0, 4000000.0, 10.0, 10.0), nodata=255) as dataset:
        dataset.write(np.array([[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [1, 1, 1, 1]], np.uint8), 1)
    status, text = run("tiny", "--dir", tiny, "--convention", "esri", "--steps", "fd1,fd2", "--no-open",
                       "--out-root", os.path.join(directory, "tiny_out"))
    check(status == 0, "a grid of 4 x 4 pixels, one basin of 0.0016 km2: the partition and its view are made (exit %d%s)"
          % (status, "" if status == 0 else ": " + text[-300:]))
    status, text = run("small", "--dir", small, "--convention", "esri", "--steps", "fd1,fd2", "--no-open",
                       "--out-root", os.path.join(directory, "small_out"))
    check(status == 0, "the partition and the views of that grid are made (exit %d%s)" % (status, "" if status == 0 else ": " + text[-300:]))
    status, text = run("small", "--dir", small, "--convention", "esri", "--steps", "fd1,fd2", "--no-open", "--summary-only",
                       "--view-factors", "10", "--out-root", os.path.join(directory, "small_out_10"))
    check(status == 0 and "100 m = 10 pixels a cell" in text and "500 m" not in text, "--view-factors 10 gives the 100 m view alone")
    status, text = run("small", "--dir", small, "--convention", "esri", "--summary-only", "--view-factors", "ten",
                       "--out-root", os.path.join(directory, "small_out_bad"))
    check(status != 0 and "--view-factors" in text, "--view-factors that is not a number is refused in words")

    print("10  the MERIT recode on disk")
    import re
    merit = os.path.join(directory, "merit_3s.tif")
    directions = np.full((24, 24), 1, np.uint8)          # MERIT codes: east along every row, south down the last
    directions[:, -1] = 4                                # column, 0 the river mouth: one basin of 576 pixels, 4.9 km2
    directions[-1, -1] = 0
    with rasterio.open(merit, "w", driver="GTiff", height=24, width=24, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(20.0, 10.0, 1.0 / 1200.0, 1.0 / 1200.0), nodata=247) as dataset:
        dataset.write(directions, 1)
    out = os.path.join(directory, "merit_out")
    common = ["merit3s", "--dir", merit, "--convention", "merit", "--no-open", "--out-root", out]
    status, text = run(*(common + ["--steps", "fd1"]))
    check(status == 0, "a MERIT grid is partitioned (exit %d%s)" % (status, "" if status == 0 else ": " + text[-300:]))
    marker = os.path.join(out, "merit3s", "_logs", "fd1.0.ok")
    aside = os.path.join(directory, "fd1.0.ok.aside")
    os.rename(marker, aside)
    status, text = run(*(common + ["--steps", "fd3", "--attributes", "ldn"]))
    check(status != 0 and "no marker of their recode" in text, "without the recode's marker, the steps after it keep theirs: fd3 is refused")
    with open(aside) as handle:              # a marker without the refusal in its arguments, with another signature
        old = re.sub(r"(?m)^signature: .*$", "signature: 0000000000000000", handle.read().replace(" into_nodata=refused", ""))
    with open(marker, "w") as handle:
        handle.write(old)
    status, text = run(*(common + ["--steps", "fd3", "--attributes", "ldn"]))
    check(status != 0 and "recoded by an earlier version" in text, "a recode without the refusal of directions into nodata: fd3 is refused")
    for prefix in ("/data/into_nodata=refused/", "/data/case into_nodata=refused old/",
                   "/data/case into_nodata=refused periodic=False input=old/"):
        with open(marker, "w") as handle:          # the rule's text inside a path is not the rule
            handle.write(old.replace("args: recode ", "args: recode " + prefix, 1))
        status, text = run(*(common + ["--steps", "fd3", "--attributes", "ldn"]))
        check(status != 0 and "recoded by an earlier version" in text, "a path that holds the rule's text does not pass for the rule (%s)" % prefix)
    with open(marker, "w") as handle:
        handle.write(old)
    status, text = run(*(common + ["--steps", "fd1", "--summary-only"]))
    check(status == 0, "--steps fd1 --summary-only on it plans the recode again (exit %d)" % status)
    status, text = run(*(common + ["--steps", "fd1", "--only", "fd1.0"]))
    with open(marker) as handle:
        remade = "into_nodata=refused" in handle.read()
    check(status == 0 and remade, "--steps fd1 --only fd1.0 makes the recode again (exit %d)" % status)

    print("11  the lock of the tile kernels' command")
    import fd3_attributes
    recoded = os.path.join(out, "merit3s", "global", "fineresolution", "dir")
    recoded = os.path.join(recoded, sorted(name for name in os.listdir(recoded) if name.endswith(".tif"))[0])
    held = os.path.join(directory, "tiles_ldn.tif")
    with fd3_attributes._the_only_run_writing(held):
        process = subprocess.run([sys.executable, "-B", os.path.join(here, "tile_kernels.py"), recoded, "ldn", held],
                                 capture_output=True, text=True, cwd=directory)
    check(process.returncode != 0 and "another run is writing" in process.stdout + process.stderr and not os.path.exists(held),
          "a second writer of the same output is refused and writes nothing")
    process = subprocess.run([sys.executable, "-B", os.path.join(here, "tile_kernels.py"), recoded, "ldn", held],
                             capture_output=True, text=True, cwd=directory)
    check(process.returncode == 0 and os.path.exists(held), "with the lock free, the command writes its output")

    print("%d failures" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
