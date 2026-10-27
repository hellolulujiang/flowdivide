# Changelog

## 1.1.4 (2026-10-28)

The published partitions are unchanged: MERIT uses `level2`, the 30 m grids `none`; neither uses `balanced`.
`--merge balanced` now ranks partners by the largest joined window.

- The default balanced merge rule is `window-max`; ties go to the smaller extra window, then the smaller region id.
- FD2 writes fine group rasters at `global/fineresolution/grp/grp_<grouping>_<name>_<tag>.tif`.
- Published outlet tables are read as space- or comma-separated CSV or Parquet, with the new global basin and outlet column names too.
- `fd1.5_regions` no longer records the merge rule in its arguments under `level2`, which does not use it. A MERIT
  run left by 1.1.3 therefore makes its regions, and the steps after them, again, with the same results.
- The user guide has a section on the limits of the package.
- The README links the published partitions.
- `fd2_views.min_basin_area_km2_until_0_7_8` is renamed
  `min_basin_area_km2_width_squared`.

## 1.1.3 (2026-10-23)

Results are unchanged. Comments, docstrings, messages and documentation describe the code's behaviour.

- The five swept attributes hold only the region's own pixels; fd1.4 writes the Level-02 groups;
  `merge_policy` takes `level2`; FD3 requires the system's file locks; Hack and Strahler orders stop at 254.
- The test descriptions give the raster comparisons and their tolerance. The guide explains how
  `--view-factors` lowers factors to divisors of the block and when a capacity directory omits its block.
- The Figure 7 script's messages refer to scripts in this repository.

## 1.1.2 (2026-10-15)

Results are unchanged for the built-in grids, whose only filter (`PFAF_ID != 353`, North America) was read correctly.

- **`--group-vectors` filters use OGR SQL.** Outside quotes, pandas' `==` and `!=` become `=` and `<>`;
  square-bracket lists are refused, and quoted strings and names are kept. In 1.1.1 only `!=` was converted,
  everywhere; OGR can accept `PFAF_ID == 353` or `PFAF_ID IN [611, 612]` while keeping every polygon.
- The guide clarifies grid and polygon inputs, default blocks, MERIT basin views, `--sink-code` and optional
  grouping vectors. `test_caps_and_counts.py` checks the OGR filter.

## 1.1.1 (2026-10-15)

The limits below are enforced. The chain's results are unchanged on the Sri Lanka sample.

- **fd1.6 writes `region_id` for the Level-03 partition of 2³¹ only.** The Python API refuses another key,
  as the command line already did.
- **fd1.5 rounds upstream counts as fd1.2 does:** read v as a double, floor(v + 0.5) when
  1 ≤ v < 9 × 10¹⁸, otherwise 0. In 1.1.0 the cut dropped the fraction; whole counts up to 2⁵² read unchanged.
- **The longest flow path is checked to 1 mm** against the distance walk's length; 1.1.0 allowed 10⁻³ of the
  length (1 m in 1 km). For up to 7,000 km in up to 700,000 steps each double-precision sum is within
  0.33 mm of the exact length; a longer path or more steps could fail a correct run.
- **Hack and Strahler orders stop at 254** (`fd1.ORDER_CAP`); 1.1.0 allowed 255. The bound applies to every
  order written or handed across a cut.
- **The tiled distance to the outlet writes nodata −1** (`tile_kernels.ldn_on_tiles`); 1.1.0 wrote −9999.
- The guide clarifies inputs, basin ids, views, attribute coverage, scratch files and markers.
  `test_caps_and_counts.py` checks the counts, capacities, order limit and tiled nodata;
  `test_fd1_6_basin_table.py` refuses other partition keys.

## 1.1.0 (2026-10-13)

Two ways a run could stand still with GDAL 3.6 are closed. A decoding thread needing room in the block cache
could push out a written tile, wait for a compression thread and hold the lock other decoding threads needed.

- **fd1.4 burns Level-03 polygons on one thread.** It creates the raster with `NUM_THREADS=1` and sets
  `GDAL_NUM_THREADS=1` for the burning thread until the output is closed; GDAL reads the output back for
  each layer after the first.
- **GeoTIFF reads and updates use the calling thread.** `fd1_partition.open_raster` sets the open option
  `NUM_THREADS=1`, whatever the environment or GDAL configuration says. Created rasters still use
  `FLOWDIVIDE_GDAL_NUM_THREADS` for compression, except fd1.4; rasters reopened to update compress on one thread.
- Not covered by that option: non-GeoTIFF inputs such as VRTs, external overviews (`.ovr`), and datasets
  a caller opens and hands to a package function.
- `earth_distance_m` holds the haversine term at 1 when rounding carries it above, so points opposite
  each other on the earth, or within rounding of it, have a finite distance.
- `test_gdal_threads.py` checks two-layer burns, reads between writes, updates and the open options.
  Scenarios that could stand still run in separate processes with time limits.

Changes before 1.1 are in [`CHANGELOG.md` at `v1.0.0`](https://github.com/hellolulujiang/flowdivide/blob/v1.0.0/CHANGELOG.md).
