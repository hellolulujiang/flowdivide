"""data_a4_views_120th: the data of Figure A4 (the partition as data at the 1/120 degree view).

Reads from the published products of South America under PRODUCTS/south-america:
    global/coarseresolution/bsn/bsn_south-america_120th.tif        the basin view
    global/vector/bsn_south-america_120th.gpkg                      its table (layer basins)
    partitions/160deg2/global/coarseresolution/rgn/rgn_south-america_120th.tif    the region view at 2^31
    partitions/160deg2/global/vector/rgn_south-america_120th.gpkg   its table (layer regions)
    global/table/basin_table_fine_south-america.csv                 the pixels of each basin
and the eight colours of the views (COLOR_PALETTE of paths.py).

Writes into data/:
    bsn_south-america_120th.tif, rgn_south-america_120th_160deg2.tif
    bsn_south-america_120th_rows.csv     basin_id, basin_grid_count, basin_area_km2, outlet_lon, outlet_lat,
                                         coarse_grid_count, color_id: every row of the view
    rgn_south-america_120th_160deg2_rows.csv   region_id, coarse_grid_count, color_id
    color_id_palette.csv
"""
import os
import shutil
import sqlite3
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

CONTINENT = "south-america"
GLOBAL = os.path.join(paths.HYDROSHEDS30M_ROOT, CONTINENT, "global")
PARTITION = os.path.join(paths.HYDROSHEDS30M_ROOT, CONTINENT, "partitions", "160deg2", "global")


def layer_rows(path, layer, columns):
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query('SELECT %s FROM "%s" ORDER BY fid' % (", ".join(columns), layer), connection)


def main():
    out = paths.DATA_DIRECTORY
    shutil.copy2(os.path.join(GLOBAL, "coarseresolution", "bsn", "bsn_%s_120th.tif" % CONTINENT), out)
    shutil.copy2(os.path.join(PARTITION, "coarseresolution", "rgn", "rgn_%s_120th.tif" % CONTINENT),
                 os.path.join(out, "rgn_%s_120th_160deg2.tif" % CONTINENT))
    basins = layer_rows(os.path.join(GLOBAL, "vector", "bsn_%s_120th.gpkg" % CONTINENT), "basins",
                        ["basin_id", "basin_area_km2", "outlet_lon", "outlet_lat", "coarse_grid_count", "color_id"])
    fine = pd.read_csv(os.path.join(GLOBAL, "table", "basin_table_fine_%s.csv" % CONTINENT), sep=" ",
                       nrows=len(basins), usecols=["basin_id", "basin_grid_count"])
    if not (basins["basin_id"].to_numpy() == fine["basin_id"].to_numpy()).all():
        raise SystemExit("the rows of the basin view are not the first rows of the basin table")
    basins.insert(1, "basin_grid_count", fine["basin_grid_count"].to_numpy())
    basins.to_csv(os.path.join(out, "bsn_%s_120th_rows.csv" % CONTINENT), index=False)
    regions = layer_rows(os.path.join(PARTITION, "vector", "rgn_%s_120th.gpkg" % CONTINENT), "regions",
                         ["region_id", "coarse_grid_count", "color_id"])
    regions.to_csv(os.path.join(out, "rgn_%s_120th_160deg2_rows.csv" % CONTINENT), index=False)
    if os.path.abspath(paths.COLOR_PALETTE) != os.path.abspath(os.path.join(out, "color_id_palette.csv")):
        shutil.copy2(paths.COLOR_PALETTE, os.path.join(out, "color_id_palette.csv"))
    print("%d basins, %d of them with a cell; %d regions, %d with a cell"
          % (len(basins), int((basins["coarse_grid_count"] > 0).sum()), len(regions), int((regions["coarse_grid_count"] > 0).sum())))


if __name__ == "__main__":
    main()
