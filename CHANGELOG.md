# Changelog

Renumbered on 2026-10-06: the versions first named 1.0.0 to 1.0.4 are 0.7.0 to 0.7.4, so that 1.0.0 names the version the paper describes. The version published on GitHub on 2026-09-28 as 1.0.0 is 0.7.0.

## 0.7.5 (2026-10-06)

After the review of 0.7.4 by GitHub Copilot (gpt-6.1-sol) and agy; the rasters and tables do not change.
`fd_tables.read_basin_table_columns` counts the values of every line itself: read in chunks, pandas took a line with
one value too many that opens a chunk as a line with an index and dropped the value without a word; such a table is
now refused with pandas' own message, as the whole reader refuses it, and a table with faults in several columns is
refused for the first column in the table's order, also as the whole reader does (`_column_faults`,
`merge_column_faults`). The held blocks go to `<raster>.held_blocks.<process id>`, and a run removes only the scratch
directories of runs no longer alive; on an error every dataset is closed and the scratch directory removed, a failure
there logged without hiding the error. One run writes an output at a time: `derive_attribute` holds an exclusive lock
on `<raster>.lock` (flock, released by the system however the process ends; the empty file stays, since flock locks
the file and not its name; not on Windows). The budget is 512 MiB
(`FLOWDIVIDE_HELD_MEMORY_MB`, in MiB). `measure_fd3_memory.py` holds the region map against the capacity, block and
grouping its directory is named for, as `flowdivide.py` does (a projected grid's `<N>blocks` meaning blocks of 5000
pixels), follows links before it decides that the out directory lies outside the dataset root, and writes only into a
new or empty directory. README and user guide list the eight tests that need no data.

## 0.7.4 (2026-10-06)

The memory of FD3 follows the capacity. The output blocks that two regions share were held in memory until the
last of their regions had written, and their number grows with the continent, not with the capacity (12,024 blocks,
12.6 GB, for the upstream flow length on North America at 2^30). They are now held in memory up to a budget, 512 MB by
default (`FLOWDIVIDE_HELD_MEMORY_MB`), and the rest in a scratch directory beside the output as raw `.npy` files,
read back when a region needs them; every block comes back bit for bit, so the rasters and tables do not change.
The partition reads the basin table in chunks and keeps the eleven columns it uses
(`fd_tables.read_basin_table_columns`), every check of `read_basin_table` still made; `read_basin_table` no longer
copies the whole table to check `basin_nrow` and `basin_ncol`. The Hack order keeps its main-stem donor in int32 and
reads the donor's area from the window instead of a second array (33 to 25 bytes a window pixel). Every region line of
the FD3 log, the longest flow path's included, gives the peak memory so far, and the swept attributes the held blocks
in memory and on disk. `measure_fd3_memory.py` computes one attribute on a partition already on disk (the C chain's
or this package's) into a directory of its own and reports the peak memory of the process; `test_fd3_memory.py`
checks the three changes on made-up data. A marker of a basin table that promises more rows than the file can hold
is refused as before, without allocating them.

## 0.7.3 (2026-10-03)

Track native DIR files and tile-directory membership in cache identities. Validate the native identity already stored in legacy recode arguments when a dependency is reused without FD1; changed or missing native inputs require FD1 to run again. Geometry and attribute kernels are unchanged.

## 0.7.2 (2026-10-02)

Inherits all 0.7.1 half-open rectangle and table-reader corrections from the local release tree.

* FD3 checks the accumulated affine-coordinate displacement across the whole raster, within 0.01 pixel,
  for the channel mask and upstream area. Non-finite transforms are refused. `FD3_RULES_VERSION` is 6,
  so FD3 results made under the earlier grid check are checked and made again; FD1 and FD2 markers retain
  their rules versions.
* `--only` refuses an empty selection or names outside the selected run's planned steps before any step
  executes. A private summary preflight checks recode and later steps together without writing data or
  markers or changing the live chain, so a valid later step remains selectable.
* `test_grid_and_step_selection.py` covers grid alignment and valid, unknown and empty CLI selections.

## 0.7.1 (2026-09-29)

Every rectangle is now left closed and right open, as in CCode v10.00 and in the FlowTopo materials:
`row_max`, `col_max`, `basin_row_max`, `basin_col_max`, `bbox_row_max`, `bbox_col_max`, `window_row_max`,
`window_col_max`, `cell_row_max` and `cell_col_max` are one past the last pixel (or block, or cell), so
`nrow = row_max - row_min` and a loop runs `row_min <= row < row_max`. Tables written by 0.7.0 are refused
by the readers (their `nrow` no longer matches); convert them with `convert_tables_to_half_open_bounds.py`
of CCode v10.00 or run the partition again. The rasters do not change.  `RULES_VERSION` is 2, so every marker
of a 0.7.0 run is stale and a run into an old tree makes everything again.

After the Codex review of 2026-09-30: the region rows write `nrow = row_max - row_min`; the piece rasters are
checked against a right-open window; a rectangle ending at the last column is not across the antimeridian
(`col_max > ncol` is); the centre of a rectangle is `(min + max - 1) // 2` wherever a rectangle is placed by
its centre, so no grouping or split moves; the last block of a rectangle is `(max - 1) // block`; the group
windows must have positive spans; the in-memory branch of the view check slices `[row0:row1)`.

## 0.7.0 (2026-09-28)

First public release.

* The three stages of the paper: the partition (FD1), the views and the figures (FD2), and the attributes
  region by region (FD3).
* A GeoPackage view carries a QGIS default style on `color_id`, so that the polygons are coloured the moment
  the file is opened.
* The regular-tile kernels (`tile_kernels.py`) and the three-ways check (`three_ways.py`) the paper
  compares the partition with.
* Test scripts on made-up grids and on a case cut from a real basin.
