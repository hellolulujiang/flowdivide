"""data_07_external_tools: the data of Figure 7 (one basin computed by selected packages).

Reads from the timing experiment under RUNS (FLOWDIVIDE_BENCHMARK):
    timings_multi.csv    one row per timed run (run_basins.py)
    needs_basins.csv     the memory each package would need on the five largest basins (needs/needs_model.py)
and the published basin table of South America for the rectangles of the five largest basins.

Writes into data/:
    timings_multi.csv, needs_basins.csv      copies, their modification times kept (Figure 7 checks that the needs
                                             were computed from these timings)
    basin_rectangles_south-america.csv       basin_id, the pixels of the bounding rectangle (basin_nrow x basin_ncol),
                                             of basins 1 to 5
"""
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

RUNS = paths.BENCHMARK_RUNS
BASIN_TABLE = os.path.join(paths.HYDROSHEDS30M_ROOT, "south-america", "global", "table", "basin_table_fine_south-america.csv")


def main():
    out = paths.DATA_DIRECTORY
    for name in ("timings_multi.csv", "needs_basins.csv"):
        shutil.copy2(os.path.join(RUNS, name), out)
    # the table's rectangles are left closed and right open: basin_nrow = basin_row_max - basin_row_min
    table = pd.read_csv(BASIN_TABLE, sep=" ", nrows=5, usecols=["basin_id", "basin_nrow", "basin_ncol"])
    table["rectangle_pixels"] = table["basin_nrow"].astype("int64") * table["basin_ncol"].astype("int64")
    table[["basin_id", "rectangle_pixels"]].to_csv(os.path.join(out, "basin_rectangles_south-america.csv"), index=False)
    print(table[["basin_id", "rectangle_pixels"]].to_string(index=False))


if __name__ == "__main__":
    main()
