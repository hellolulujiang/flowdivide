"""The fold of the Level-03 groups to Level-02 and Level-01 (fd1.4), checked on made-up codes and on the tables
of a finished run.

    FLOWDIVIDE_DATA_ROOT=<dir> python3 test_level_fold.py

A Level-03 code is three digits: the first the Level-01 region, the first two the Level-02 unit.  Level-01 is the nine
regions HydroBASINS draws, the Arctic (8) its own.  On a finished run, its Level-03
table is folded here and set against the run's own Level-02 and Level-01 tables, every column of every row.  The
tables are read under FLOWDIVIDE_DATA_ROOT (HydroSHEDS_v2_30m/<continent>/ and MERIT_Hydro_90m/global/); without it
that part is skipped.
"""
import os
import sys

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
        print("  FAIL  %-50s got %r, expected %r" % (what, got, want))
    else:
        print("  ok    %-50s %r" % (what, got))


def the_codes_fold_as_the_digits_say():
    print("the code of a folded unit is the first digits of the Level-03 code")
    for code, level1, level2 in ((111, 1, 11), (622, 6, 62), (711, 7, 71), (783, 7, 78),
                                 (811, 8, 81), (863, 8, 86), (963, 9, 96), (999, 9, 99)):
        check("Level-01 of %d" % code, int(fd1.level_code_of_level3_code(code, 1)), level1)
        check("Level-02 of %d" % code, int(fd1.level_code_of_level3_code(code, 2)), level2)
    check("Level-01 of an uncoded group", int(fd1.level_code_of_level3_code(0, 1)), 0)
    check("8 is not folded into 7", int(fd1.level_code_of_level3_code(811, 1)) == int(fd1.level_code_of_level3_code(711, 1)), False)


class _Grid:
    """what fold_level3_groups needs of a grid: the block, the width, whether it wraps, and the box in degrees"""

    def __init__(self, block, ncol, periodic, west=-93.0, north=15.0, step=1.0 / 3600.0):
        self.block_pixels = block
        self.ncol = ncol
        self.periodic = periodic
        self.west = west
        self.north = north
        self.step = step

    def pixel_box_lon_lat(self, row_min, row_max, col_min, col_max):
        return (self.west + col_min * self.step, self.north - (row_max + 1) * self.step,
                self.west + (col_max + 1) * self.step, self.north - row_min * self.step)


def a_made_up_fold():
    print("a made-up Level-03 table folded")
    rows = []
    for group_id, code, kind, basins, window in ((621, 621, 2, 10, (0, 1, 0, 1)), (622, 622, 1, 1, (1, 3, 1, 2)),
                                                 (611, 611, 2, 5, (4, 4, 0, 0)), (206001, 0, 3, 2, (9, 9, 9, 9))):
        rows.append({"group_id": group_id, "group_level": 3, "group_kind": kind, "level_code": code, "level3_count": 1 if code else 0,
                     "basin_count": basins, "coded_basin_count": basins, "neighbour_basin_count": 0, "land_grid_count": basins * 100,
                     "window_row_min": window[0], "window_row_max": window[1], "window_col_min": window[2], "window_col_max": window[3]})
    grid = _Grid(10, 1000, False)
    level3 = fd1._fill_group_window_columns(pd.DataFrame(rows), grid, 10)
    level2 = fd1.fold_level3_groups(level3, 2, grid)
    check("Level-02 ids: the two units in code order, then the island", level2["group_id"].tolist(), [61, 62, 206001])
    unit_62 = level2[level2["group_id"] == 62].iloc[0]
    check("62 holds its two Level-03 groups", int(unit_62["level3_count"]), 2)
    check("its window is their union", (int(unit_62["window_row_min"]), int(unit_62["window_row_max"]),
                                        int(unit_62["window_col_min"]), int(unit_62["window_col_max"])), (0, 3, 0, 2))
    check("a folded group is of kind 2", int(unit_62["group_kind"]), 2)
    check("the island keeps its id and changes level", (int(level2.iloc[2]["group_id"]), int(level2.iloc[2]["group_level"])), (206001, 2))
    level1 = fd1.fold_level3_groups(level3, 1, grid)
    check("Level-01: 6, then the island", level1["group_id"].tolist(), [6, 206001])
    check("6 holds the basins of all three codes", int(level1.iloc[0]["basin_count"]), 16)
    # across the 180th meridian: a unit with a group at the western edge and one past the eastern edge
    rows_seam = [dict(rows[0], window_col_min=0, window_col_max=1), dict(rows[1], window_col_min=98, window_col_max=100)]
    seam_grid = _Grid(10, 1000, True)
    folded = fd1.fold_level3_groups(fd1._fill_group_window_columns(pd.DataFrame(rows_seam), seam_grid, 10), 2, seam_grid)
    check("the western part is moved one width east when that is narrower", (int(folded.iloc[0]["window_col_min"]),
                                                                            int(folded.iloc[0]["window_col_max"])), (98, 101))


def the_run_tables_fold_the_same_here(root, run):
    """the Level-03 table of a finished run, folded here, against the run's own Level-02 and Level-01 tables"""
    level3_path = fd_tables.group_table_path(root, run, "l3")
    if not os.path.exists(level3_path + ".done"):
        print("  (skipped: %s is not there yet)" % level3_path)
        return
    level3 = fd_tables.read_group_table(level3_path, 0)
    dir_path = os.path.join(root, "global", "fineresolution", "dir")
    candidates = [os.path.join(dir_path, name) for name in sorted(os.listdir(dir_path)) if name.endswith("_merit.tif")] if os.path.isdir(dir_path) else []
    if not candidates:
        print("  (skipped: no flow-direction raster under %s for the grid)" % dir_path)
        return
    grid = fd1.Grid(candidates[0], periodic=(run == "global"))
    for level, grouping in ((2, "l2"), (1, "l1")):
        run_table = fd_tables.read_group_table(fd_tables.group_table_path(root, run, grouping), 0)
        here = fd1.fold_level3_groups(level3, level, grid)
        for column in fd_tables.GROUP_TABLE_COLUMNS:
            if column in fd_tables.GROUP_TABLE_FLOAT_FORMATS:
                same = bool(np.array_equal(np.char.mod(fd_tables.GROUP_TABLE_FLOAT_FORMATS[column], here[column].to_numpy(np.float64)),
                                           np.char.mod(fd_tables.GROUP_TABLE_FLOAT_FORMATS[column], run_table[column].to_numpy(np.float64))))
            else:
                same = bool(np.array_equal(here[column].to_numpy(np.int64), run_table[column].to_numpy(np.int64)))
            if not same:
                check("%s %s: column %s as the run folded it" % (run, grouping, column), False, True)
                break
        else:
            check("%s %s: %d groups, every column as the run folded it" % (run, grouping, len(run_table)), True, True)


if __name__ == "__main__":
    the_codes_fold_as_the_digits_say()
    a_made_up_fold()
    data_root = os.environ.get("FLOWDIVIDE_DATA_ROOT", "")
    roots = {"south-america": os.path.join(data_root, "HydroSHEDS_v2_30m", "south-america"),
             "north-america": os.path.join(data_root, "HydroSHEDS_v2_30m", "north-america"),
             "global": os.path.join(data_root, "MERIT_Hydro_90m", "global")} if data_root else {}
    if not data_root:
        print("the tables of a finished run: skipped (FLOWDIVIDE_DATA_ROOT is not set)")
    for run, root in roots.items():
        print("the tables of %s" % run)
        the_run_tables_fold_the_same_here(root, run)
    print()
    print("ALL PASSED" if not FAILED else "%d FAILED" % len(FAILED))
    sys.exit(1 if FAILED else 0)
