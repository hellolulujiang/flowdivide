"""fd1.6 on a made-up table: the region of every basin and its place in its
Level-01, Level-02 and Level-03 group and in its region, written into the basin table in place.  Synthetic: it checks
that the code holds; no number from it goes anywhere.

    python3 test_fd1_6_basin_table.py

The case: five basins in two Level-03 groups (611, 612) of one Level-02 unit (61), one island group (206001), and
basin 1 cut into two pieces whose outlet piece is in region 61102.  Checked: region_id is the map's value, or the
outlet piece's region for the cut basin; the places are 1 .. n in basin_id order within each unit; a map of another
key, a table without its marker and a basin without a group are refused.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import fd1_partition as fd1
import fd_tables

FAILED = []


def check(what, got, want):
    if got != want:
        FAILED.append("%s: got %r, expected %r" % (what, got, want))
        print("  FAIL  %-58s got %r, expected %r" % (what, got, want))
    else:
        print("  ok    %-58s %r" % (what, got))


def refused(what, function):
    try:
        function()
    except (fd1.FlowDivideError, fd_tables.TableError):
        check(what, "refused", "refused")
        return
    check(what, "accepted", "refused")


def made_up_run(work):
    root = os.path.join(work, "south-america")
    run = "south-america"
    partition = os.path.join(root, "partitions", "160deg2", "global", "table")
    os.makedirs(os.path.join(root, "global", "table"))
    os.makedirs(partition)
    basins = fd_tables.new_basin_table(5)
    basins["outlet_row"] = [10, 20, 30, 40, 50]
    basins["outlet_col"] = [10, 20, 30, 40, 50]
    basins["outlet_flag"] = 0
    basins["basin_grid_count"] = [900, 500, 300, 200, 100]
    basins["basin_area_km2"] = [9.0, 5.0, 3.0, 2.0, 1.0]
    basins["level3_id"] = [611, 612, 611, 206001, 612]
    basins["level2_id"] = [61, 61, 61, 206001, 61]
    basins["level1_id"] = [6, 6, 6, 206001, 6]
    basin_table = fd_tables.basin_table_path(root, run)
    fd_tables.write_basin_table(basins, basin_table, "fd1.4", "test")
    key = fd_tables.region_map_key(160 * 3600 * 3600, 3600, "l3")
    map_path = os.path.join(partition, "basin_region_%s.csv" % run)
    fd_tables.write_basin_map(map_path, np.array([0, 0, 61201, 61101, 206001, 61201], np.int64), run, "region", key)
    pieces = pd.DataFrame([{name: 0 for name in fd_tables.PIECE_TABLE_COLUMNS} for _ in range(2)])
    pieces["piece_id"] = [6, 7]
    pieces["basin_id"] = [1, 1]
    pieces["parent_piece_id"] = [0, 6]
    pieces["region_id"] = [61102, 61103]
    piece_path = os.path.join(partition, "piece_fine_%s.csv" % run)
    fd_tables.write_piece_table(pieces, piece_path, "test")
    regions = pd.DataFrame([{name: 0 for name in fd_tables.REGION_TABLE_COLUMNS} for _ in range(5)])
    regions["region_id"] = [61101, 61102, 61103, 61201, 206001]
    regions["level3_members"] = ""
    regions["nrow"] = regions["row_max"] - regions["row_min"] + 1
    regions["ncol"] = regions["col_max"] - regions["col_min"] + 1
    region_path = os.path.join(partition, "region_fine_%s.csv" % run)
    fd_tables.write_region_table(regions, region_path, "test")
    return basin_table, map_path, region_path, piece_path, key


def the_columns_are_filled():
    print("fd1.6 on five basins")
    work = tempfile.mkdtemp(prefix="flowdivide_fd1_6_")
    basin_table, map_path, region_path, piece_path, key = made_up_run(work)
    fd1.fd1_6_basin_table(basin_table, map_path, region_path, piece_path, key)
    table = fd_tables.read_basin_table(basin_table)
    check("the stage is fd1.6", fd_tables.basin_table_stage(basin_table)[1], "fd1.6")
    check("region_id: the map, and the outlet piece for the cut basin", table["region_id"].tolist(), [61102, 61201, 61101, 206001, 61201])
    check("places in Level-03 (611: 1, 3; 612: 2, 5)", table["basin_id_in_level3"].tolist(), [1, 1, 2, 1, 2])
    check("places in Level-02 (61 holds 1, 2, 3, 5)", table["basin_id_in_level2"].tolist(), [1, 2, 3, 1, 4])
    check("places in Level-01", table["basin_id_in_level1"].tolist(), [1, 2, 3, 1, 4])
    check("places in the region (61201 holds 2 and 5)", table["basin_id_in_region"].tolist(), [1, 1, 1, 1, 2])
    other_key = fd_tables.region_map_key(80 * 3600 * 3600, 3600, "l3")
    refused("a map of another partition", lambda: fd1.fd1_6_basin_table(basin_table, map_path, region_path, piece_path, other_key))
    os.remove(basin_table + ".done")
    refused("a basin table without its marker", lambda: fd1.fd1_6_basin_table(basin_table, map_path, region_path, piece_path, key))


def a_basin_without_a_group_is_refused():
    print("a basin table from before fd1.4")
    work = tempfile.mkdtemp(prefix="flowdivide_fd1_6_")
    basin_table, map_path, region_path, piece_path, key = made_up_run(work)
    table = fd_tables.read_basin_table(basin_table)
    table["level2_id"] = 0
    fd_tables.write_basin_table(table, basin_table, "fd1.3", "test")
    refused("a basin without a Level-02 group", lambda: fd1.fd1_6_basin_table(basin_table, map_path, region_path, piece_path, key))


if __name__ == "__main__":
    the_columns_are_filled()
    a_basin_without_a_group_is_refused()
    print()
    print("ALL PASSED" if not FAILED else "%d FAILED" % len(FAILED))
    sys.exit(1 if FAILED else 0)
