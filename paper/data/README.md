# data/

The 41 files the figure scripts read. They are not kept in git: download `flowdivide-paper-data.zip` from
<https://fullhydro.org/downloads/papers/flowdivide/v1.0/> and unzip it in `paper/`, which fills this folder. `SHA256SUMS.txt` gives the SHA-256 of the zip and of each file.

All files describe South America on HydroSHEDS v2.0 at 1 arc-second (30 m), except those of Fig. 8: the run times
and grid sizes of the three grids (South America and North America on HydroSHEDS v2.0, the globe on MERIT Hydro)
and the Mississippi files of Fig. 8b (North America). They are small extracts and coarse views of the products of FlowDivide, which are published at
<https://fullhydro.org>: basin and region views (`.npy`, `.tif`), tables (`.csv`, `.txt`, `.json`), paths and
stream lines (`.geojson`, `.npz`), and the timing records of the runs reported in the paper. `README.md` one level
up lists which figure reads which file and which data script writes it.

Seven tables are copies taken on 2026-09-27, before the published tables gave every rectangle as left closed and
right open: `group_l3_fine_south-america.csv`, `group_hilbert_fine_south-america.csv`,
`region_fine_south-america_{160,80,40}deg2.csv`, `piece_fine_south-america_basin1_160deg2.csv` and
`pfafstetter_pieces_south-america_basin1_160deg2.csv`. In them a rectangle's `*_max` is its last row or column, one
less than in the tables published now (`nrow = row_max - row_min + 1`). The figures read their counts, sizes and
geographic bounds, which are the same; the data scripts run on the published products copy the current tables.

The files are derived from HydroSHEDS v2.0 (Lehner, 2026) and HydroBASINS (Lehner and Grill, 2013). They are
independent of these products and are not part of the HydroSHEDS family of products. Obtain the source data from
<https://www.hydrosheds.org> and use them under the terms given there. The files of this folder are under CC BY 4.0,
the licence of the FlowDivide products they are taken from.
