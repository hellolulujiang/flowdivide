"""data_03_basin_groups: the data of Figure 3 (basins and basin groups of South America).

Reads from the published products under PRODUCTS/south-america/global/table:
    basin_table_fine_south-america.csv         one row per basin
    basin_group_hilbert_fine_south-america.csv the Hilbert group of every basin
    group_l3_fine_south-america.csv, group_hilbert_fine_south-america.csv
and the basin view bsn_south-america_400th_view.npy (data_03_basin_view) for the largest basin id it shows.

Writes into data/:
    basin_groups_south-america_400th.csv       basin_id, basin_area_km2, level3_id, hilbert_group of the basins 1 to N,
                                               N the largest id in the view
    basin_counts_south-america.json            the number of basins, of basins of at least 1 km2, and the basins whose
                                               rectangle taken out to whole degrees exceeds the capacity (160 deg2)
    group_l3_fine_south-america.csv, group_hilbert_fine_south-america.csv
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

CONTINENT = "south-america"
TABLE = os.path.join(paths.HYDROSHEDS30M_ROOT, CONTINENT, "global", "table")
CAPACITY_PIXELS = 160 * 3600 * 3600          # 2^31 on the 1 arc-second grid: 160 whole-degree blocks


def main():
    out = paths.DATA_DIRECTORY
    view = np.load(os.path.join(out, "bsn_%s_400th_view.npy" % CONTINENT))
    largest_id = int(view.max())
    basins = pd.read_csv(os.path.join(TABLE, "basin_table_fine_%s.csv" % CONTINENT), sep=" ",
                         usecols=["basin_id", "basin_area_km2", "level3_id",
                                  "basin_row_min", "basin_row_max", "basin_col_min", "basin_col_max"])
    if not (basins["basin_id"].to_numpy() == np.arange(1, len(basins) + 1)).all():
        raise SystemExit("the basin ids of the table do not run from 1 without gaps")
    with open(os.path.join(TABLE, "basin_group_hilbert_fine_%s.csv" % CONTINENT)) as handle:
        if not handle.readline().startswith("#basin_map"):
            raise SystemExit("the Hilbert groups are not a basin map")
        hilbert = np.array([int(handle.readline()) for _ in range(largest_id)], dtype=np.int64)
    shown = basins.iloc[:largest_id][["basin_id", "basin_area_km2", "level3_id"]].copy()
    shown["hilbert_group"] = hilbert
    shown.to_csv(os.path.join(out, "basin_groups_%s_400th.csv" % CONTINENT), index=False)

    # the rectangles are left closed and right open: the last row is basin_row_max - 1, the last column the same
    whole_degree_pixels = (((basins["basin_row_max"] - 1) // 3600 - basins["basin_row_min"] // 3600 + 1)
                           * ((basins["basin_col_max"] - 1) // 3600 - basins["basin_col_min"] // 3600 + 1) * 3600 * 3600)
    over = basins.loc[whole_degree_pixels > CAPACITY_PIXELS, "basin_id"]
    counts = {"basins": int(len(basins)),
              "basins_of_at_least_1_km2": int((basins["basin_area_km2"] >= 1.0).sum()),
              "capacity_pixels": CAPACITY_PIXELS,
              "over_the_capacity": {str(int(b)): int(whole_degree_pixels[b - 1]) for b in over}}
    with open(os.path.join(out, "basin_counts_%s.json" % CONTINENT), "w") as handle:
        json.dump(counts, handle, indent=1)
    for kind in ("l3", "hilbert"):
        shutil.copy2(os.path.join(TABLE, "group_%s_fine_%s.csv" % (kind, CONTINENT)), out)
    print("basins 1 to %d written; %s" % (largest_id, json.dumps(counts)))


if __name__ == "__main__":
    main()
