# Changelog

Renumbered on 2026-10-06: the versions first named 1.0.0 to 1.0.4 are 0.7.0 to 0.7.4, so that 1.0.0 names the version the paper describes. The version published on GitHub on 2026-09-28 as 1.0.0 is 0.7.0.

## 0.7.8 (2026-10-07)

After the final review of 0.7.7 (Codex gpt-6-astra and agy, round 3) and the reviews of the fixes (rounds 4 and 5).
On the two HydroSHEDS grids nothing the six attributes give changes. On a periodic grid only the longest flow path of
a cut basin can change, when its members lie in regions whose windows are unrolled whole turns apart and two members'
heads are exactly equally far in the same row (the lfp tie below; the basin need not cross the antimeridian). On the
MERIT partitions on disk no region that holds a piece of a cut basin is unrolled (700deg2: 16 such regions, 350deg2:
45, all within columns 18,219 to 385,836 of 432,000; 1400deg2 cuts no basin), so their lfp does not change.

- A periodic region whose window, with a column of margin on each side, would hold a column twice (a region all the
  way round the grid, or all but one column) is refused, as in 0.7.7, with a message that says why and what to do. The
  first 0.7.8 computed a region of all but one column in a window that held the free column once; round 4 showed why
  that cannot be right: the tables do not say which column is free (FD1 may give such a region a box over every
  column, and the small basins' boxes are not kept), and with one copy of the free column the column of a child's
  outlet there, which decides a tie of the Hack order's main stem, depends on where the window starts. The user guide
  states the limit. On MERIT the widest region spans 62.3 degrees (1400deg2), so the paper's runs never meet it.
- A partition read with `raster_min_basin_area_km2=0` (as for the distance and the upstream flow length) and used for
  the longest flow path or a registered rule: a region of small basins only is in `region_order()` but holds no member
  (KeyError). Both now pass over such a region. The longest flow path also draws only the basins of the tables' area:
  such a partition keeps the cut basins below that area for the distance's raster, and lfp drew them as well, which a
  partition read for lfp does not. `flowdivide.py` and `measure_fd3_memory.py` read a partition for each attribute, so
  they never met either.
- `derive_user_attribute` takes the output lock (`_the_only_run_writing`), as the six attributes do.
- lfp, the tie between the heads of a cut basin's members (equally far from the outlet, in the same row): the column
  of each head was taken in its own region's window, and on a periodic grid two regions' windows are unrolled from
  different columns, so the head picked depended on the partition (Codex, round 5: on a 10-column grid one region
  picked column 9, three pieces column 1). Each member is now placed beside the member it flows into -- its outlet and
  its parent's inlet are neighbours, so its columns are shifted by the whole turns that bring the two within a column
  -- from the outlet member down the tree, which lays the basin out as one connected run, the order one window holding
  the whole basin gives its columns; inside one member the distance's sweep already breaks the tie in that order. The
  basin table's box is not used: FD1 may give a basin a box over every column (Codex, round 6, against a first fix that
  used it). On a grid that is not periodic nothing changes. `flowdivide.py` gives lfp's step its own rules version
  (`LFP_RULES_VERSION` 2, in lfp's signature only), so that an lfp made before reruns and the other five attributes keep
  their markers (checked: with an older lfp marker only lfp runs again).
- lfp's two path buffers start at 4 million points and double for a longer path, instead of being reserved at the
  largest member's pixel count. The user guide gives the memory a region's members take (some 50 bytes each) and
  the North America measurements at 2^31 (22.3 GB), 2^30 (14.2 GB) and 2^29 (8.8 GB), made with 0.7.7.
- `test_fd3_seam_and_reuse.py` (new, no data): a basin across the seam of a periodic grid of 10 columns whose box spans
  8 of them (its window with the margins all 10), all six attributes and their tables equal to the same basin on a
  flat grid; regions over 9 and 10 of the columns refused; on a partition read for the distance that holds a region of
  small basins only and a cut basin below the tables' area, lfp equal to lfp on a partition read for it, and a
  registered rule run; the lock of a registered rule; two heads equally far across the seam, the basin as one region
  and as three pieces in three regions: the same path, also with a basin box over every column and a region unrolled a
  whole turn away. The test grid spans the globe (10 columns of 36 degrees), so
  that a step across the seam is measured as one column. 0.7.7 fails it (KeyError twice; the tie: column 9 against 1).

## 0.7.7 (2026-10-07)

FD3 holds the region's own pixels, not its whole window, and bounds GDAL's block cache; the rasters and tables do not
change. Measured on North America, lup at 2^29, one process (`measure_fd3_memory.py`): 8.77 GB and 2,544 s with 0.7.7,
16.0 GB and 2,627 s with 0.7.6; the peak is the windows' (the tables read, 7.8 GB, came first). Of the 16.0 GB of 0.7.6, about 10 GB the arrays of the largest
windows (23 to 25 bytes for every pixel of a window, though a region takes 43 % of its window at the median and at
most 68 % on North America at 2^29), 3.2 GB GDAL's
block cache (unbounded, GDAL keeps up to 5 % of the machine's memory), 1.9 GB the partition's tables and 0.5 GB the
held output blocks.

- The six provided attributes number the region's own pixels so that each comes after the pixel it flows into
  (`fd3_attributes` section [2c], `compact_the_pixels_of_the_region`), as the C programs do. The window holds the
  flow directions (and the channel mask) and, while the pixels are numbered, the number of each (int32); the
  downstream link, the member and the values are held for the region's own pixels; the numbers are the order the
  sweeps run in, so neither the order nor the member of every window pixel is kept. Every pixel of the order is
  followed down once, row by row, to the first pixel decided, and the pixels passed are numbered on a second pass
  down the same way (nothing of the walk is kept but its length); the outlet of a child of another region and the
  pixels above it are another region's, but the walk goes on through it, so that a cycle through it is found too; a
  cycle is found on the walk (Brent), as 0.7.6 found it over the same pixels. The two ties (the farthest pixel of the distance, the main-stem donor of the Hack order) go
  to the smaller window index as before. The output window is no longer made before the sweep: the values go into
  one, in the raster's type, just before the write. `check_owned_edges` is gone: a pixel took its member from the
  pixel it flows into, so the case it looked for could not arise. On North America at 2^29, one region alone, the
  same steps with the package's functions (read, structure, lup sweep, values back): region 86301 (window 0.43
  billion pixels, 0.14 billion of its own) 11.1 GB and 4.5 s with 0.7.6, 4.3 GB and 4.5 s now; region 78207 (0.42
  billion, 0.20 billion of its own) 12.5 GB and 10.5 s, 4.9 GB and 8.5 s.
- GDAL's block cache is bounded at 512 MiB while an attribute is computed (`fd3_attributes.gdal_cache_bytes`; the
  C programs set the same with GDALSetCacheMax64): `--gdal-cache-mb` of `flowdivide.py` and of
  `measure_fd3_memory.py`, or `FLOWDIVIDE_GDAL_CACHE_MB`, set another bound in MiB. Reading 3 GB of North America's
  flow directions, the process held 3.21 GB under GDAL's own bound and 0.71 GB under 512 MiB. FD3 sets the bound
  with `rasterio.Env`, which gives the cache back when the attribute is done, so `GDAL_CACHEMAX` does not reach FD3.
  The run summary and the log print the bound.
- A registered rule (`derive_user_attribute`) keeps the window order of 0.7.6, which its kernel is written for.
- Tests: `test_fd3_compact.py` (new; 1,059 checks) compares the numbering and the six sweeps with the 0.7.6 window
  order and kernels on 300 random windows, and checks the refusals (a cycle through a child's outlet of another
  region among them). `test_fd3_memory.py`'s main-stem donor test drew
  its directions at random, which nearly always holds a cycle, so it never reached its comparison; it now draws
  windows without a cycle and compares on all 40. The Sri Lanka chain (20deg2 three levels, 6deg2 four levels, and
  held blocks 0 MiB with a GDAL cache of 1 MiB) gives the 18 FD3 products of 0.7.6 bit for bit; `test_fd3_sweeps.py`
  on basins 141 and 175 of South America: all identical.

## 0.7.6 (2026-10-06)

The memory a parse leaves behind goes back to the system. Measured with 0.7.5 on North America (the upstream flow
length, one process): 26.4 GB at 2^30 and 20.6 GB at 2^29, where the windows take at most 0.95e9 and 0.38e9 pixels;
after the basin table (11 columns, 2.65 GB) and the basin map of thirty million basins were parsed, the process held
6.6 GB with 1.9 GB of it in use, the rest freed but kept by the allocator, which the windows' large arrays cannot use.
`fd3_attributes.release_free_memory` asks the C library to hand such memory back (macOS:
`malloc_zone_pressure_relief`; Linux with glibc: `malloc_trim`; elsewhere nothing): after the partition is read and
its tables are deleted, when an attribute starts, and after every region. On North America it took the process from 6.6 to 1.9 GB. No value changes.

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
