"""Fine group rasters on a small basin grid, including sparse ids and refused counts.

    python3 test_fine_group_raster.py
"""
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio

import fd1_partition as fd1
import fd_tables
import flowdivide


def made_up_run():
    work = tempfile.mkdtemp(prefix="flowdivide_fine_groups_")
    dataset = flowdivide.Dataset("test", "", "merit")
    layout = flowdivide.Layout(work, dataset)
    basin = np.tile(np.array([[0, 1, 1, 2, 3, 3, 4]], np.uint32), (1025, 1))
    profile = dict(driver="GTiff", height=basin.shape[0], width=basin.shape[1], count=1,
                   dtype="uint32", nodata=0, crs="EPSG:4326",
                   transform=rasterio.transform.from_origin(10, 20, 1 / 1200, 1 / 1200))
    path = os.path.join(work, "basin.tif")
    with rasterio.open(path, "w", **profile) as out:
        out.write(basin, 1)
    layout.set_grid(fd1.Grid(path))
    os.rename(path, layout.raster("bsn"))
    basins = fd_tables.new_basin_table(4)
    basins["outlet_row"], basins["outlet_col"], basins["outlet_flag"] = 0, [1, 3, 4, 6], 0
    basins["basin_grid_count"] = [2050, 1025, 2050, 1025]
    basins["basin_area_km2"] = [4, 3, 2, 1]
    for level, ids in ((1, [6, 6, 8, 206001]), (2, [61, 61, 81, 206001]), (3, [611, 612, 811, 206001])):
        basins["level%d_id" % level] = ids
    fd_tables.write_basin_table(basins, layout.basin_table(), "fd1.4", "test")
    for grouping in ("l1", "l2", "l3", "hilbert"):
        if grouping == "hilbert":
            fd_tables.write_basin_map(layout.basin_group_table(grouping), np.array([0, 7, 7, 19, 19]),
                                      layout.name, "group", grouping)
        lookup = fd_tables.group_of_basin(layout.root, layout.name, grouping)
        rows = []
        for identifier in np.unique(lookup[1:]):
            row = {name: 0 for name in fd_tables.GROUP_TABLE_COLUMNS}
            row.update(group_id=int(identifier), group_level=0, group_kind=2, basin_count=int((lookup[1:] == identifier).sum()),
                       land_grid_count=int(basins.loc[lookup[1:] == identifier, "basin_grid_count"].sum()),
                       window_row_max=1, window_col_max=1, window_nrow=1200, window_ncol=1200, window_grid_count=1440000)
            rows.append(row)
        fd_tables.write_group_table(pd.DataFrame(rows), layout.group_table(grouping), "test")
    chain = object.__new__(flowdivide.Chain)
    chain.layout = layout
    return chain, basin, basins


def refused(chain, grouping):
    try:
        chain.step_fd2_group_raster(grouping)
    except fd1.FlowDivideError:
        return
    raise AssertionError("the step accepted bad basin or group counts")


def main():
    chain, basin, basins = made_up_run()
    layout = chain.layout
    with patch("fd2_views.strip_rows_for", return_value=600):
        for grouping in ("l1", "l2", "l3", "hilbert"):
            chain.step_fd2_group_raster(grouping)
            lookup = fd_tables.group_of_basin(layout.root, layout.name, grouping)
            with rasterio.open(layout.group_raster(grouping)) as out, rasterio.open(layout.raster("bsn")) as source:
                assert np.array_equal(out.read(1), lookup[basin])
                assert out.dtypes == ("uint32",) and out.nodata == 0
                assert out.transform == source.transform and out.crs == source.crs
                assert out.block_shapes == [(512, 512)]
                assert out.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG"
        with fd1.open_raster(layout.raster("bsn"), "r+") as out:
            changed = basin.copy()
            changed[0, 0] = 5
            out.write(changed, 1)
        refused(chain, "l3")
        with fd1.open_raster(layout.raster("bsn"), "r+") as out:
            out.write(basin, 1)
        basins.loc[0, "basin_grid_count"] += 1
        fd_tables.write_basin_table(basins, layout.basin_table(), "fd1.4", "test")
        groups = fd_tables.read_group_table(layout.group_table("l3"), 0)
        groups.loc[groups.group_id == 611, "land_grid_count"] += 1
        fd_tables.write_group_table(groups, layout.group_table("l3"), "test")
        refused(chain, "l3")
        basins.loc[0, "basin_grid_count"] -= 1
        fd_tables.write_basin_table(basins, layout.basin_table(), "fd1.4", "test")
        refused(chain, "l3")
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
