"""measure_fd3_memory.py -- one attribute of FD3 on a partition already on disk, run to measure its peak memory.

    python3 measure_fd3_memory.py <dataset root> <capacity directory> <attribute> <out directory>
                                  [--block <pixels>] [--periodic] [--channel <raster>] [--area <raster>]
                                  [--ldn-members <the member table of an ldn run>]

<dataset root> holds global/table/basin_table_fine_<run>.csv, global/fineresolution/dir/dir_<run>_<tag>_merit.tif and
partitions/<capacity directory>/global/table/{region_fine,piece_fine,basin_region}_<run>.csv, the layout the C chain
(CCode v10.x) and this package both write.  Nothing is written under the dataset root: the attribute's raster and its
tables go to <out directory>, which must be new or empty, and so does <attribute>_memory.txt with the peak resident set of the whole process (the
tables read, every region, the tables written), the seconds and the held blocks.  The FD3 log lines give the peak so
far after the tables and after every region.

The distance to the outlet (ldn) and the upstream flow length (lup) need only the flow directions; the Shreve
magnitude (shv) and the Strahler order (ord) need the channel mask (--channel), the Hack order (hck) the channel mask
and the upstream area (--area).  The longest flow path (lfp) reads the heads from the member table of a distance run
(--ldn-members); without it the distance is swept first in the same process.  For the paper's machine sizes: North America at 80deg2 (2^30) and 40deg2 (2^29), lup,
the attribute with the highest peak of the swept ones.  One attribute a process, so that one peak is one attribute's.
"""
import argparse
import glob
import os
import re
import resource
import platform
import sys
import time

import fd_tables
import fd1_partition as fd1
import fd3_attributes as fd3


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("capacity_directory")
    parser.add_argument("attribute", choices=["shv", "ldn", "hck", "lup", "ord", "lfp"])
    parser.add_argument("out")
    parser.add_argument("--block", type=int, default=None, help="the pixels of a block (default one degree)")
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--area", default=None)
    parser.add_argument("--min-basin-area-km2", type=float, default=1.0)
    parser.add_argument("--ldn-members", default=None, help="for lfp: the member table of a distance (ldn) run")
    arguments = parser.parse_args(argv)
    started = time.time()
    tables = glob.glob(os.path.join(arguments.root, "global", "table", "basin_table_fine_*.csv"))
    if len(tables) != 1:
        raise SystemExit("%s holds %d basin tables, not one" % (arguments.root, len(tables)))
    run = fd_tables.root_and_run_of_basin_table(tables[0])[1]
    directions = glob.glob(os.path.join(arguments.root, "global", "fineresolution", "dir", "dir_%s_*_merit.tif" % run))
    if len(directions) != 1:
        raise SystemExit("%s holds %d recoded flow-direction rasters for %s, not one" % (arguments.root, len(directions), run))
    partition_tables = os.path.join(arguments.root, "partitions", arguments.capacity_directory, "global", "table")
    # the capacity, the block and the grouping the directory is named for (flowdivide.py capacity_directory_name:
    # <N>deg2 or <N>blocks[_block<B>], hilbert_ before it for the Hilbert groups), held against the region map's own
    # key as flowdivide.py step_fd3 holds it
    named = re.fullmatch(r"(hilbert_)?(\d+)(deg2|blocks)(?:_block(\d+))?", arguments.capacity_directory)
    if named is None:
        raise SystemExit("%s is not named as a capacity directory (<N>deg2, <N>blocks_block<B>, hilbert_...)"
                         % arguments.capacity_directory)
    blocks = int(named.group(2))
    grouping = "hilbert" if named.group(1) else "l3"
    block = arguments.block
    # <N>blocks without _block<B> is a projected grid's own block of 5000 pixels (capacity_directory_name leaves it out)
    named_block = named.group(4) if named.group(4) is not None else ("5000" if named.group(3) == "blocks" else None)
    if named_block is not None:
        if block is not None and block != int(named_block):
            raise SystemExit("--block %d and the directory's block %s disagree" % (block, named_block))
        block = int(named_block)
    # the out directory and every file written there lie outside the dataset root, links followed
    root = os.path.realpath(arguments.root)
    out = os.path.realpath(arguments.out)
    if os.path.commonpath([out, root]) == root:
        raise SystemExit("the out directory %s lies inside the dataset root %s; give one outside it" % (out, root))
    # a new or empty directory: nothing in it can be a link the run would write through (the partial rasters and
    # tables, the scratch of the held blocks, the distance run lfp may make first)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise SystemExit("the out directory %s is not a new or empty directory" % out)
    os.makedirs(out, exist_ok=True)
    code = arguments.attribute
    grid = fd1.Grid(directions[0], periodic=arguments.periodic, block_pixels=block)
    covers_every_basin = arguments.attribute in ("ldn", "lup")
    partition = fd3.Partition(tables[0], os.path.join(partition_tables, "region_fine_%s.csv" % run),
                              os.path.join(partition_tables, "piece_fine_%s.csv" % run),
                              os.path.join(partition_tables, "basin_region_%s.csv" % run),
                              arguments.min_basin_area_km2, grid,
                              expect={"continent": run, "capacity_px": blocks * grid.block_pixels * grid.block_pixels,
                                      "block_px": grid.block_pixels, "groups": grouping},
                              raster_min_basin_area_km2=0.0 if covers_every_basin else None)
    seconds_tables = time.time() - started
    peak_tables = fd3.peak_memory_gb()
    report = fd3.derive_attribute(code, partition, directions[0], os.path.join(out, "%s_%s.tif" % (code, run)),
                                  os.path.join(out, "%s_basin_%s.csv" % (code, run)),
                                  os.path.join(out, "%s_member_%s.csv" % (code, run)),
                                  channel_path=arguments.channel, area_path=arguments.area,
                                  ldn_member_table=arguments.ldn_members, lines_path=None)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    lines = ["attribute %s" % code, "run %s" % run, "capacity_directory %s" % arguments.capacity_directory,
             "flowdivide %s" % fd_tables.VERSION, "held_memory_budget_bytes %d" % fd3.held_memory_bytes(),
             "regions %d" % report["regions_visited"], "seconds_tables %.1f" % seconds_tables,
             "peak_gb_after_tables %.2f" % peak_tables, "seconds %.1f" % (time.time() - started),
             "peak_bytes %d" % peak, "peak_gb %.2f" % (peak / 1e9)]
    with open(os.path.join(out, "%s_memory.txt" % code), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
