"""data_08_package_run: the data of Figure 8 (the Python package at work).

Reads the runs of the FlowDivide Python package under PACKAGE_RUNS:
    <dataset>/_logs/timings_<dataset>.csv   the wall seconds of every step
    _run_logs/<dataset>_fd1.log              the log of the FD1 run
    south-america/_logs/run_summary_south-america.txt
from the published products of North America under PRODUCTS the piece table and the preview of the Mississippi's cut
at 160 deg2 and the basin table, and from the HydroSHEDS v2 download the flow directions (DIR) and the upstream area
(ACA) of North America.

Writes into data/:
    package_stage_hours.csv                  Figure 8c: hours of FD1, FD2 and FD3 on the three grids
    grid_sizes.csv                           Figure 8c: rows and columns of the three flow-direction grids
    run_summary_south-america.txt            Figure 8a: the page the package prints at the start of a run
    pfafstetter_pieces_north-america_basin1.csv, pfafstetter_pieces_north-america_basin1_30s.tif,
    mississippi_outlet.csv, mississippi_main_stem.csv
                                             Figure 8b: the Mississippi cut into regions of 160 deg2; the main stem
                                             traced from the outlet up the channel network (1 km2), at each junction
                                             to the donor of the largest upstream area

The hours: FD1 and FD2 of the run of 2026-09-25 (package 0.6.1), FD3 of the runs of 2026-09-27 (package 0.6.4) on
South America at 2^31, North America at 2^30 and the globe at 700 deg2.  The seconds of fd1.1 are read from the FD1 log,
because the timings table holds a later rerun of that step.
"""
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

PACKAGE_RUNS = paths.PACKAGE_RUNS
sys.path.insert(0, paths.PACKAGE_DIRECTORY)
import flowdivide
PRODUCTS = paths.HYDROSHEDS30M_ROOT
RAW = paths.RAW_HYDROSHEDS
DATASETS = ["south-america", "north-america", "merit-global"]
FD3_CAPACITY = {"south-america": "2^31", "north-america": "2^30", "merit-global": "700deg2"}


def scrub_run_page(text):
    """the run page without the directories of the machine it ran on: the output directory becomes <directory>, and
    every other absolute path keeps only its last name; paths relative to the output directory stay as they are"""
    text = re.sub(r"--out-root (/\S*)", "--out-root <directory>", text)
    return re.sub(r"(?<!\S)/(?:[^\s,/]+/)+([^\s,/]*)", lambda m: "<directory>/" + m.group(1), text)


def fd1_1_seconds(dataset):
    """the wall seconds of fd1.1 in the FD1 run of 2026-09-25"""
    with open(os.path.join(PACKAGE_RUNS, "_run_logs", "%s_fd1.log" % dataset), encoding="utf8") as handle:
        found = re.findall(r"^fd1\.1 done\s+09-25 \S+\s+([0-9.]+) s", handle.read(), flags=re.M)
    if len(found) != 1:
        raise SystemExit("%s: %d fd1.1 lines of 2026-09-25 in the FD1 log, one expected" % (dataset, len(found)))
    return float(found[0])


def stage_hours():
    rows = []
    for dataset in DATASETS:
        table = pd.read_csv(os.path.join(PACKAGE_RUNS, dataset, "_logs", "timings_%s.csv" % dataset), sep=r"\s+")
        day = table["finished"].str.slice(0, 10)
        fd1 = table[table["step"].str.match(r"^fd1[._]") & (table["step"] != "fd1.1")]
        fd2 = table[table["step"].str.match(r"^fd2_")]
        fd3 = table[table["step"].str.match(r"^fd3_") & table["step"].str.endswith("_" + FD3_CAPACITY[dataset])]
        if (day[fd1.index] != "2026-09-25").any() or (day[fd2.index] != "2026-09-25").any():
            raise SystemExit("%s: an FD1 or FD2 step is not of the run of 2026-09-25" % dataset)
        if len(fd3) != 6 or (day[fd3.index] != "2026-09-27").any():
            raise SystemExit("%s: %d FD3 steps of 2026-09-27 at %s, six expected" % (dataset, len(fd3), FD3_CAPACITY[dataset]))
        for stage, seconds, package, run, capacity in (
                ("FD1", fd1["seconds"].sum() + fd1_1_seconds(dataset), "0.6.1", "2026-09-25", "all"),
                ("FD2", fd2["seconds"].sum(), "0.6.1", "2026-09-25", "all"),
                ("FD3", fd3["seconds"].sum(), "0.6.4", "2026-09-27", FD3_CAPACITY[dataset])):
            rows.append({"dataset": dataset, "stage": stage, "hours": round(seconds / 3600.0, 4),
                         "seconds": round(seconds, 1), "package": package, "run": run, "capacity": capacity})
    return pd.DataFrame(rows)


BLOCK = 2048
CHANNEL_KM2 = 1.0                    # the channel threshold of the runs
OFFSET = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}    # HydroSHEDS D8


def main_stem(outlet_row, outlet_col):
    """row, col, lon, lat of the main stem of the basin draining to the outlet, on the channel network of the package
    (upstream area of at least CHANNEL_KM2): from the outlet up, the donor of the largest upstream area at every
    junction, to the channel head"""
    directions = rasterio.open(os.path.join(RAW, "north-america_DIR_1s_v2r0.tif"))
    area = rasterio.open(os.path.join(RAW, "north-america_ACA_1s_v2r0.tif"))
    blocks = {}

    def value(row, col):
        key = (row // BLOCK, col // BLOCK)
        if key not in blocks:
            window = Window(key[1] * BLOCK, key[0] * BLOCK, BLOCK, BLOCK)
            blocks[key] = (directions.read(1, window=window, boundless=True, fill_value=0),
                           area.read(1, window=window, boundless=True, fill_value=0))
        d, a = blocks[key]
        return int(d[row % BLOCK, col % BLOCK]), float(a[row % BLOCK, col % BLOCK])

    rows, cols = [outlet_row], [outlet_col]
    row, col = outlet_row, outlet_col
    while True:
        donors = []
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                code, upstream_area = value(row + drow, col + dcol)
                step = OFFSET.get(code)
                if step and (row + drow + step[0], col + dcol + step[1]) == (row, col):
                    donors.append((upstream_area, row + drow, col + dcol))
        if not donors or max(donors)[0] < CHANNEL_KM2:
            break
        largest = max(donors)[0]
        if sum(1 for d in donors if d[0] == largest) > 1:
            raise SystemExit("two donors of equal upstream area at row %d col %d" % (row, col))
        _, row, col = max(donors)
        rows.append(row)
        cols.append(col)
    transform = directions.transform
    rows, cols = np.array(rows), np.array(cols)
    return pd.DataFrame({"row": rows, "col": cols,
                         "lon": np.round(transform.c + (cols + 0.5) * transform.a, 7),
                         "lat": np.round(transform.f + (rows + 0.5) * transform.e, 7)})


def main():
    out = paths.DATA_DIRECTORY
    hours = stage_hours()
    hours.to_csv(os.path.join(out, "package_stage_hours.csv"), index=False)
    print(hours.to_string(index=False))
    with open(os.path.join(PACKAGE_RUNS, "south-america", "_logs", "run_summary_south-america.txt"), encoding="utf8") as handle:
        page = scrub_run_page(handle.read())
    with open(os.path.join(out, "run_summary_south-america.txt"), "w", encoding="utf8") as handle:
        handle.write(page)
    cut = os.path.join(PRODUCTS, "north-america", "partitions", "160deg2", "global")
    shutil.copy2(os.path.join(cut, "table", "pfafstetter_pieces_north-america_basin1.csv"), out)
    shutil.copy2(os.path.join(cut, "preview", "pfafstetter_pieces_north-america_basin1_30s.tif"), out)
    basins = pd.read_csv(os.path.join(PRODUCTS, "north-america", "global", "table", "basin_table_fine_north-america.csv"),
                         sep=" ", nrows=1, usecols=["basin_id", "outlet_row", "outlet_col", "outlet_lon", "outlet_lat"])
    if int(basins["basin_id"].iloc[0]) != 1:
        raise SystemExit("the first row of the North America basin table is not basin 1")
    basins[["basin_id", "outlet_lon", "outlet_lat"]].to_csv(os.path.join(out, "mississippi_outlet.csv"), index=False)
    stem = main_stem(int(basins["outlet_row"].iloc[0]), int(basins["outlet_col"].iloc[0]))
    stem.to_csv(os.path.join(out, "mississippi_main_stem.csv"), sep=" ", index=False)
    print("the main stem of the Mississippi: %d pixels" % len(stem))
    sizes = []
    for dataset in DATASETS:
        with rasterio.open(flowdivide.builtin_datasets()[dataset].raw_dir) as raster:
            sizes.append({"dataset": dataset, "rows": raster.height, "columns": raster.width})
    pd.DataFrame(sizes).to_csv(os.path.join(out, "grid_sizes.csv"), index=False)
    print("written into", out)


if __name__ == "__main__":
    main()
