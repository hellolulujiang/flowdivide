# Changelog

## 1.1.2 (2026-10-15)

No result changes for the built-in grids, whose only filter (`PFAF_ID != 353`, North America) was read right.

- **A filter of `--group-vectors` is read as OGR reads it** (`fd1.ogr_attribute_filter`): an SQL condition on the
  attribute table; outside quotes, pandas' `==` and `!=` are taken as `=` and `<>` and a list in square brackets is
  refused, and a quoted string or name is kept as it is. Until 1.1.1 only `!=` was converted, everywhere, and OGR reads `PFAF_ID == 353` and `PFAF_ID IN [611, 612]` without an error as conditions
  that keep every polygon, so a filter written as the user guide said (a pandas query) could group the basins on
  every polygon of the file.
- The documentation set right: a grid needs a coordinate reference system, north up, in degrees or metres; a
  polygon file of `--group-vectors` needs a coordinate reference system and a field `PFAF_ID` of three-digit codes;
  the default block on a projected or a very fine grid; on the built-in MERIT Hydro grid the basin views are rasters
  only; `--sink-code` (a direction pointing into nodata is a river mouth, not a sink); `--group-vectors` is optional
  for a grid of your own.
- Tests: `test_caps_and_counts.py` also reads a filter as OGR does.

## 1.1.1 (2026-10-15)

No result changes in the chain: on the Sri Lanka sample (a run where basins are cut and the six attributes are
computed, and a run with the default capacities and the views) every raster equals 1.1.0's pixel for pixel, every
table byte for byte, and the polygon files equal it in rows, columns and geometry (their metadata names the
version). Four limits are now the C programs': three refuse what 1.1.0 let through, and one reads a fractional
upstream count as fd1.2 already did, which a raster of whole counts (fd1.1's own, MERIT Hydro's) never holds. The
distance to the outlet on regular tiles, a helper outside the chain, now writes nodata as the chain does. The
documentation says what the code does.

- **fd1.6 writes `region_id` for the Level-03 partition of 2³¹ only**, as the C program does. The command line already
  held to it; `fd1_6_basin_table` called on its own took the key of any partition. It now refuses another key
  (`fd1.top_tier_blocks`, `fd1.level3_top_tier_key`; `capacity_blocks_of` in `flowdivide.py` uses the same rule).
- **fd1.5 reads an upstream count as fd1.2 and the C programs do**: the value v as a double, floor(v + 0.5) when
  1 ≤ v < 9 × 10¹⁸, otherwise 0 (`fd1.acc_count`, `fd1.acc_counts`). Until 1.1.0 the cut dropped the fraction, so a
  count of 1.6 read 1 there and 2 in fd1.2. A whole count up to 2⁵², which every grid's is, reads the same as before.
- **The longest flow path is checked to 1 mm**, as in the C programs: the painted path against the length the
  distance walk found. Until 1.1.0 the check allowed 10⁻³ of the length (1 m in 1 km). Both sums add the same steps
  in double precision, in other orders; each addition rounds by at most half a unit in the last place of the total,
  so on a path of up to 7,000 km in up to 700,000 steps each sum is within 0.33 mm of the exact length and a correct
  run passes. A longer path or more steps is outside that bound, and there a correct run could fail the check.
- **The Hack and Strahler orders stop at 254**, as in the C programs (`fd1.ORDER_CAP`); 1.1.0 allowed 255. Every
  order a sweep writes is held to it, one copied from below or carried in at a cut too, and so is an order handed
  across a cut to the Hack order of the region above, which was not held at all. No grid of the paper comes near it.
- **The distance to the outlet on regular tiles (`tile_kernels.ldn_on_tiles`) writes nodata −1**, as FD3 and the C
  programs do; until 1.1.0 it wrote −9999.
- The documentation set right: the inputs of a run (the provider's upstream count and area of the built-in grids,
  MERIT Hydro's table of basin ids); views given with `--view-factors` are still lowered to a divisor of the block;
  how the basins are numbered on MERIT Hydro; what the colouring guarantees; which pixels and tables the attributes
  cover and what the distance's nodata −1 means; that the longest flow path's own distance walk writes a scratch
  raster; the lines of the longest flow path come from the Python API only; the columns of the Hilbert groups; the
  ranges of ids and codes; which outputs of FD3 carry `.done` markers; what FD1.0 and FD2 take that the C programs
  refuse; what a cut that is not deep enough leaves for inspection; the scratch a running FD3 writes again.
- Tests: `test_caps_and_counts.py` (the upstream count, the top tier, the order limit, the tiled distance's nodata);
  `test_fd1_6_basin_table.py` also refuses the keys of other partitions. Unused imports and variables removed.
- Two sentences of earlier entries were not so. 1.0.0: views given with `--view-factors` are not "taken as they are";
  each is lowered to a divisor of the block and a repeat is kept once. 0.7.9: the package writes over nothing but its
  own scratch while it runs (a block of FD3 held on disk is written again in place when it goes to disk a second time).

## 1.1.0 (2026-10-13)

No result changes: on the Sri Lanka sample (a run where basins are cut and the six attributes are
computed, a run with the default capacities and the views) every raster and table equals 1.0.0 byte for byte, and
the polygon files equal it in rows, columns and geometry (their metadata names the version). On MERIT Hydro
worldwide, FD1 gives the partition published at <https://fullhydro.org> (FullDivide v1.1): the basin and region
masks pixel for pixel, and the region table, the three basin-group tables and the region and groups of every basin
cell for cell. What changes is that two ways a run could stand still with GDAL 3.6 are closed.

- **What stood still.** GDAL 3.6 can decode a read of several tiles on its own threads. One of them that needs room
  in the block cache pushes out a tile that was written and not yet compressed, waits for a free thread to compress
  it, and holds the lock the other decoding threads wait for: the run stops without an error and uses no processor.
  It takes a raster that is written piece by piece, reads between the pieces, and a raster larger than the cache.
- **fd1.4 burns the Level-03 polygons on one thread.** Its raster was created with `NUM_THREADS=ALL_CPUS`, whatever
  the environment said, and GDAL reads the raster back, chunk by chunk, for every layer after the first. Of the
  built-in grids, North America burns two layers and MERIT Hydro nine; South America burns one. With two layers a
  synthetic grid of 6016 x 8192 pixels under a cache of 16 MB stood still in three runs of six with GDAL 3.6.2;
  with one layer, in none of eight. The raster is now created with `NUM_THREADS=1`, and `GDAL_NUM_THREADS` is 1 for
  the burning thread from before the layers are opened until the output is released; twelve runs of twelve with two
  layers ended. The pixels are the same; the file's bytes are not (the tiles lie in another order), and this one
  raster is now compressed on one thread.
- **The GeoTIFFs whose pixels the package reads are decoded on the calling thread, whatever `GDAL_NUM_THREADS`
  says.** A run
  for which no setting asks GDAL for threads, which is how the package is documented and was timed, already read
  so, and nothing changes for it. With `GDAL_NUM_THREADS=8`, strips written in turn with reads of another raster
  stood still in six synthetic runs of six. Every open to read or to update, in the five modules, now goes through
  `fd1_partition.open_raster`, which opens a GeoTIFF (told by its driver, not its name) with the open option
  `NUM_THREADS=1`; GDAL takes an open option before the environment and before its configuration, so the functions
  are covered however they are called (the command line, the tile kernels, a script of one's own). What this costs
  a run that had asked for threads: its reads are no faster than without, and the rasters FD3 opens again to update
  (the longest flow path, a registered attribute) are compressed on one thread, as they are without. The rasters
  the package creates are compressed on every core as before (`FLOWDIVIDE_GDAL_NUM_THREADS`), fd1.4's excepted.
  Not covered: an input that is not a GeoTIFF (a VRT over tiles), overviews kept in a file of their own (`.ovr`),
  and a dataset a caller opens and hands to a function of the package.
- `earth_distance_m` holds the haversine term at 1 when rounding carries it above, as the C code does. It can only
  happen for two points opposite each other on the earth, or within rounding of it, and the distance was then not
  a number; every distance that was a number is the number it was.
- Tests: `test_gdal_threads.py` (the burn of two layers; strips written in turn with reads through `open_raster`
  of a GeoTIFF whose name does not say so, with eight threads asked for in the environment; a raster updated in
  place; the option `open_raster` passes for each mode; that no module opens a raster to read in another way). The
  scenarios that could stand still each run in a process of their own with a time limit. With the reads opened as
  in 1.0.0 the strips stood still in four runs of four.
- The user guide says how GDAL's threads are used.

## 1.0.0 (2026-10-12)

The version the paper describes. On every grid 0.7.9 could run it computes what 0.7.9 computes: on the Sri Lanka
sample (a run where no basin is cut, a run where five are, the views of both) every raster and table equals 0.7.9
byte for byte, the polygon files differ only in the version their metadata names, and a second identical run does
nothing.

- `paper/`: the scripts that draw the 14 figures of the paper, the figures, and where to download the data they read.
- A grid of one's own: a default view whose smallest basin (FD2's threshold, a quarter of a cell) is larger than
  any basin of the grid could be (all its tiles together) is left out, since no basin would be in it and on a small
  grid the coarsest view stopped the run with "no basin ... has the km2 of the view"; the finest is kept, made finer
  if it must be. Such a view stopped 0.7.9 as well, so on every grid 0.7.9 could run the views do not change. `--view-factors` sets the views, as pixels to a cell, taken as they are.
- The user guide's example of a rule of one's own is a whole kernel (the upstream pixel count), with what the window
  holds, how the value crosses a cut and how the partition is read; the README says the regions are visited in the
  order the attribute needs.
- The MERIT convention: a recode whose marker is gone while the steps after it keep theirs (a recode made again that
  stopped on the refusal of 0.7.9) is refused like one made before 0.7.9, whose marker is told by the rule in its own field,
  before `periodic=` and `input=` (a path that holds its text does not pass); `--only fd1.0` and `--summary-only` on such
  a tree make or plan the recode again, where both were refused.
- `tile_kernels.py` holds the lock of its outputs while it writes them, as the attribute runs do.
- `pyproject.toml` asks for Shapely 2.0 or later, whose functions the polygon writer uses.
- Tests: sections 9 to 11 of `test_fine_grids_and_run_lock.py` (the views of a small grid, the MERIT recode on disk,
  the lock of the tile kernels).
- The version is kept in one place, `fd_tables.VERSION`; `test_native_cache_provenance.py` checks that
  `pyproject.toml` says the same.
- Documentation: the README names the paper; `DATA_NOTICE.md` says that the partitions published at fullhydro.org
  are independent datasets, not official MERIT Hydro products and not part of the HydroSHEDS family of products;
  the user guide says how the package differs from the C implementation; HydroSHEDS v2.0 where the data are named.

Renumbered on 2026-10-06: the versions first named 1.0.0 to 1.0.4 are 0.7.0 to 0.7.4, so that 1.0.0 names the version the paper describes. The version published on GitHub on 2026-09-28 as 1.0.0 is 0.7.0.

## 0.7.9 (2026-10-11)

Checked against the C implementation. Nothing the six attributes or the
partition give changes on the grids of the paper: on the Sri Lanka sample (MERIT codes given as a custom coding,
capacity 4deg2, 11 regions) every raster and table equals 0.7.8 bit for bit, and the polygon files equal it in
geometry and columns (their metadata names the new version and vector rule).

- Nothing a run replaces is deleted. An output, its marker, a dropped variable, the scratch of the held blocks and
  what a stopped run left under a temporary name are
  moved into `_tmp_/<date>_<time>_<process>/` beside them, as the C programs do. Publishing no
  longer removes the old file first: a temporary that is missing leaves the old file where it is and stops the step.
- A step is written by one run at a time: a second run on the same outputs is refused (an exclusive lock on the step's
  marker), so two runs no longer share or remove each other's temporary files.
- FD1.0, MERIT convention: a direction that points into nodata or off the grid is refused, as the C verification does
  (until 0.7.8 it was made a river mouth). For a window cut out of a MERIT grid, give MERIT's codes as a custom
  coding, whole (a custom coding has no values of its own for no data, sinks and mouths):
  `--convention custom --direction-codes E=1,SE=2,S=4,SW=8,W=16,NW=32,N=64,NE=128 --nodata-code 247 --sink-code 255 --mouth-code 0`
  and those pixels become river mouths.
- FD1.0: tiles that overlap are refused (until 0.7.8 the later tile was written over the earlier one).
- FD1.2: on a grid that reaches past 85 degrees, the lowest pixel area allowed at an outlet is that of the grid's
  outermost row, so that a basin there passes.
- The grid: pixels of less than a square metre are refused (areas are kept in whole square metres); an explicit block
  must be at least one pixel; a geographic grid coarser than a degree has blocks of one pixel; on a projected grid finer
  than 10 m the default block goes down by tens until a capacity of 2^29 pixels holds four blocks (until 0.7.8 a 1 m
  grid stopped with "less than one block").
- FD2: the smallest basin of a view is a quarter of the cell's width times its height (until 0.7.8 the width squared);
  the outline of an object across the 180th meridian is traced with the object in one piece and cut at the meridian,
  as in the C implementation.
- FD3: at every cut the piece table is checked against the flow directions: the child's outlet must flow into the
  inlet pixel the table names, and that pixel must belong to the parent (the C programs' checks). A region window of
  more than 2^31 - 1 pixels with its ring is refused, as in the C programs (its pixels are named by 32-bit indices).
- Documentation: the README's command for a grid of one's own adds `--steps fd3` for the attributes and says the finest
  grid is 1 m; the user guide says when the polygon files carry a CRS, how the numbers of `three_ways.py` differ from
  the production ones, and what the C programs do that the package does not.
- A whole run, and `global-ids`, hold one lock on the output's `_logs` directory
  (the system's lock on the directory itself: no lock file, and a run refused by its own checks still leaves no file);
  without the system's file locks a run is refused. A file takes its name without replacing one that appeared
  meanwhile, and a raster's `.aux.xml` goes aside with the raster it described. The longest flow path checks, also when
  it reuses a distance table, that every segment runs to its own member's outlet without passing another member's.
  Signatures change only where results can: `FD3_RULES_VERSION` 7 (the cut checks); `outline_across_the_seam=v10.35`
  on grids periodic in longitude; `min_areas=` on views whose smallest basin changed; `into_nodata=refused` on the
  recode of the MERIT convention. A geographic grid finer than about a sixth of an arc-second gets a default block
  smaller than a degree (it stopped with "less than one block"), and one whose largest pixels are below a square
  metre is refused when the grid is opened. `--keep` and `--drop` say that a dropped variable is moved into `_tmp_/`,
  which frees no disk space until that directory is emptied by hand.
- The package now writes over nothing and removes nothing, its own temporary files
  included: what a stopped run left under a temporary name is moved into `_tmp_/` before the name is written under
  (it was opened for writing as it was); a move that must not replace a file is a hard link or, where none can be
  made, the system's exclusive rename, and is refused where there is neither (it fell back on a check and an ordinary
  rename); a raster's `.msk` and `.ovr` go aside with it like its `.aux.xml`, before it and also when it is gone; a
  step's marker, the run summary, the figures' manifest and the measuring script's report are published like the
  other files. The timings table is written once when the run ends (every step's rewrite went into `_tmp_/`).
  **A finished run is no longer made again where no basin had to be cut**: the list of the basins cut is then an
  empty file, which the plan took for one that was not written whole, so that the cut, the regions and every
  attribute ran again on each identical run (0.7.8 and before as well). On a file system that gives no lock on a
  directory (NFS) the run's lock is on `_logs/run.lock`; one that gives no lock at all is refused in words.
  Overlapping tiles are refused when a run is planned, also a run that takes up a tree assembled by 0.7.8, and a run
  without `fd1` refuses a MERIT recode made before 0.7.9. A cut basin must have one piece without a parent, ending
  at the basin's outlet (the C's check; a root piece a pixel short gave a longest flow path a pixel short).
  `docs/user-guide.md`: what goes into `_tmp_/`; a dropped variable is made again only when its step is among
  `--steps`; a direction into no data is a mouth with or without a mouth code; codes are 0 to 65535.
- Tests: `test_fd2_outline_across_the_seam.py` and `test_fine_grids_and_run_lock.py` (new); `test_fd3_seam_and_reuse.py` checks the piece table at a cut and
  the window limit. The twelve tests that need no data pass.

## 0.7.8 (2026-10-07)

On the two HydroSHEDS grids nothing the six attributes give changes. On a periodic grid only the longest flow path of
a cut basin can change, when its members lie in regions whose windows are unrolled whole turns apart and two members'
heads are exactly equally far in the same row (the lfp tie below; the basin need not cross the antimeridian). On the
MERIT partitions on disk no region that holds a piece of a cut basin is unrolled (700deg2: 16 such regions, 350deg2:
45, all within columns 18,219 to 385,836 of 432,000; 1400deg2 cuts no basin), so their lfp does not change.

- A periodic region whose window, with a column of margin on each side, would hold a column twice (a region all the
  way round the grid, or all but one column) is refused, as in 0.7.7, with a message that says why and what to do. A
  window that holds the free column once cannot serve a region of all but one column: the tables do not say which column is free (FD1 may give such a region a box over every
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
  different columns, so the head picked depended on the partition (on a 10-column grid one region picked
  column 9, three pieces column 1). Each member is now placed beside the member it flows into -- its outlet and
  its parent's inlet are neighbours, so its columns are shifted by the whole turns that bring the two within a column
  -- from the outlet member down the tree, which lays the basin out as one connected run, the order one window holding
  the whole basin gives its columns; inside one member the distance's sweep already breaks the tie in that order. The
  basin table's box is not used: FD1 may give a basin a box over every column. On a grid that is not periodic nothing changes. `flowdivide.py` gives lfp's step its own rules version
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

The rasters and tables do not change.
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

Includes the half-open rectangles and the corrections of the table readers of 0.7.1.

* FD3 checks the accumulated affine-coordinate displacement across the whole raster, within 0.01 pixel,
  for the channel mask and upstream area. Non-finite transforms are refused. `FD3_RULES_VERSION` is 6,
  so FD3 results made under the earlier grid check are checked and made again; FD1 and FD2 markers retain
  their rules versions.
* `--only` refuses an empty selection or names outside the selected run's planned steps before any step
  executes. A private summary preflight checks recode and later steps together without writing data or
  markers or changing the live chain, so a valid later step remains selectable.
* `test_grid_and_step_selection.py` covers grid alignment and valid, unknown and empty CLI selections.

## 0.7.1 (2026-09-29)

Every rectangle is now left closed and right open, as in the C implementation:
`row_max`, `col_max`, `basin_row_max`, `basin_col_max`, `bbox_row_max`, `bbox_col_max`, `window_row_max`,
`window_col_max`, `cell_row_max` and `cell_col_max` are one past the last pixel (or block, or cell), so
`nrow = row_max - row_min` and a loop runs `row_min <= row < row_max`. Tables written by 0.7.0 are refused
by the readers (their `nrow` no longer matches); run the partition again. The rasters do not change.  `RULES_VERSION` is 2, so every marker
of a 0.7.0 run is stale and a run into an old tree makes everything again.

The region rows write `nrow = row_max - row_min`; the piece rasters are
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
