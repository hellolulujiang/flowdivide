# Figures of the FlowDivide paper

This folder draws the figures of

> Jiang, L., Chen, W. FlowDivide: a standard workflow for hydrographic attributes from flow-direction grids that
> outgrow a computer's memory. Submitted to *Environmental Modelling & Software*.

Each figure script reads only `data/` and writes one PNG into `figures/`. The PNGs in `figures/` are the figures of
the paper.

## Contents

```
paper/
├── README.md
├── requirements.txt
├── scripts/
│   ├── fig_01_*.py … fig_08_*.py, fig_a1_*.py … fig_a6_*.py   one script per figure
│   ├── data_*.py                                            build data/ again (optional, see below)
│   ├── panels_*.py, style.py                                drawing shared by the figures
│   └── paths.py                                             where the scripts read and write
├── figures/                                                 the 14 figures of the paper
└── data/                                                    the 41 files the figures read (downloaded from fullhydro.org)
```

## Set up

Python 3.11 or later (the NumPy and SciPy of `requirements.txt` need it), with the packages of `requirements.txt`. GDAL is most easily installed with conda:

    conda create -n flowdivide-paper -c conda-forge python=3.11 numpy matplotlib pandas scipy geopandas gdal rasterio pyogrio shapely fiona numba pyarrow
    conda activate flowdivide-paper

Figure 8 also imports the flowdivide package from the root of this repository.

## Data

The figures read 41 files (288 MB; 87 MB zipped). They are not kept in git; they are published as
`flowdivide-paper-data.zip` at <https://fullhydro.org/downloads/papers/flowdivide/v1.0/>. Download it into `paper/`
and unzip it there, so that the files land in `paper/data/`:

    cd paper
    curl -O https://data.fullhydro.org/papers/flowdivide/v1.0/flowdivide-paper-data.zip
    unzip flowdivide-paper-data.zip
    shasum -a 256 -c data/SHA256SUMS.txt    # checks the zip and the 41 files; every line should end in OK

`data/README.md` describes the files and their sources.

## Draw the figures

From `paper/`:

    cd scripts
    for s in fig_*.py; do python "$s"; done

Each script writes `figures/<script name>.png`, replacing the file there. With the package versions of
`requirements.txt` the figures are identical, byte for byte, to those in `figures/`; other versions of Matplotlib
or other fonts may move the text slightly.

## Figures, scripts and data

| Figure | Script | Data script | Files read from `data/` |
|---|---|---|---|
| 1 | `fig_01_test_grid_upg.py`, with `panels_test_grid.py` and `panels_upg_tiles_vs_pieces.py` | `data_01_test_grid_10x10.py` (imported) | none |
| 2 | `fig_02_workflow.py` | none | none |
| 3 | `fig_03_regions_south_america.py` | `data_03_basin_view.py`, then `data_03_basin_groups.py` | `bsn_south-america_400th_view.npy`, `basin_groups_south-america_400th.csv`, `basin_counts_south-america.json`, `group_l3_fine_south-america.csv`, `group_hilbert_fine_south-america.csv` |
| 4 | `fig_04_amazon_pfafstetter_regions.py` | `data_04_amazon_pieces.py`, `data_04_amazon_longest_flow_path.py` | `pfafstetter_pieces_south-america_basin1_160deg2.csv`, `pfafstetter_pieces_south-america_basin1_30s_160deg2.tif`, `piece_fine_south-america_basin1_160deg2.csv`, `lfp_amazon_400th.geojson` |
| 5 | `fig_05_regions_three_capacities.py` | `data_05_region_views.py`; the basin view of `data_03_basin_view.py` | `rgn_south-america_400th_view_{160,80,40}deg2.npy`, `region_fine_south-america_{160,80,40}deg2.csv`, `bsn_south-america_400th_view.npy` |
| 6 | `fig_06_partition_as_data.py` | `data_06_basin_table_extracts.py`; files of `data_03`, `data_04`, `data_05` | `basin_table_fine_south-america_head.txt`, `example_basin_10827_row.txt`, `bsn_south-america_400th_view.npy`, `rgn_south-america_400th_view_160deg2.npy`, `region_fine_south-america_160deg2.csv`, `piece_fine_south-america_basin1_160deg2.csv` |
| 7 | `fig_07_external_tools.py` | `data_07_external_tools.py` | `timings_multi.csv`, `needs_basins.csv`, `basin_rectangles_south-america.csv` |
| 8 | `fig_08_package_run.py` (imports the flowdivide package) | `data_08_package_run.py` | `package_stage_hours.csv`, `grid_sizes.csv`, `run_summary_south-america.txt`, `pfafstetter_pieces_north-america_basin1.csv`, `pfafstetter_pieces_north-america_basin1_30s.tif`, `mississippi_outlet.csv`, `mississippi_main_stem.csv` |
| A.1, A.2, A.3 | `fig_a1_basin_tiles_vs_pieces.py`, `fig_a2_upstream_length_tiles_vs_pieces.py`, `fig_a3_strahler_tiles_vs_pieces.py` | `data_01_test_grid_10x10.py` (imported) | none |
| A.4 | `fig_a4_views_120th.py` | `data_a4_views_120th.py`; files of `data_05`, `data_06` | `bsn_south-america_120th.tif`, `rgn_south-america_120th_160deg2.tif`, `bsn_south-america_120th_rows.csv`, `rgn_south-america_120th_160deg2_rows.csv`, `color_id_palette.csv`, `region_fine_south-america_160deg2.csv`, `example_basin_10827_row.txt` |
| A.5 | `fig_a5_length_attributes.py` | `data_a5_length_views.py`, `data_a5_example_basin.py`; the basin view of `data_03_basin_view.py` | `ldn_south-america_400th_view.npy`, `lup_south-america_20th_view.npy`, `lfp_south-america_400th_ge100km2.geojson`, `example_basin_10827_{outlet.json, boundary.geojson, ldn_lup.npz, lfp.npz, lines.npz}`, `bsn_south-america_400th_view.npy` |
| A.6 | `fig_a6_stream_orders.py` | `data_a6_network_lines.py`; files of `data_a5_example_basin.py` | `net_south-america_400th_ord_ge4.npz`, `example_basin_10827_lines.npz`, `example_basin_10827_outlet.json`, `example_basin_10827_boundary.geojson` |

## Build data/ again (optional)

The `data_*.py` scripts write `data/` from inputs that are not in this repository. Give their locations in
environment variables (`paths.py` lists them):

| Variable | What it points to |
|---|---|
| `FLOWDIVIDE_PRODUCTS` | the products of FlowDivide on HydroSHEDS v2.0, one folder per continent, as published at <https://fullhydro.org> |
| `HYDROSHEDS_V2` | HydroSHEDS v2.0 as distributed at <https://www.hydrosheds.org> |
| `FLOWDIVIDE_BENCHMARK` | the timing records of the benchmark of Fig. 7 |
| `FLOWDIVIDE_RUNS` | the logs of the package runs of Fig. 8 |
| `FLOWDIVIDE_HYDROSHEDS`, `FLOWDIVIDE_MERIT` | read by the package itself for the sizes of the three grids (`data_08_package_run.py`): the directory of HydroSHEDS v2.0's DIR, ACC and ACA mosaics (the same as `HYDROSHEDS_V2`), and MERIT Hydro's flow directions, one file |

The timing records and run logs are summarised in `data/` (`timings_multi.csv`, `needs_basins.csv`,
`package_stage_hours.csv`, `run_summary_south-america.txt`). From `paper/scripts`, run the data scripts in this order,
one thread each:

    export OMP_NUM_THREADS=1
    for s in data_03_basin_view data_03_basin_groups data_04_amazon_pieces data_04_amazon_longest_flow_path \
             data_05_region_views data_06_basin_table_extracts data_07_external_tools data_08_package_run \
             data_a4_views_120th data_a5_length_views data_a5_example_basin data_a6_network_lines; do python "$s.py"; done

## Terms

The code in this folder is under the MIT License of this repository. The figures and the files of `data/` are
derived from HydroSHEDS v2.0 (Lehner, 2026) and HydroBASINS (Lehner and Grill, 2013); Figure 8 also reports run
times on MERIT Hydro (Yamazaki et al., 2019). They are independent of these products: they are not part of the
HydroSHEDS family of products and are not official MERIT Hydro products. Obtain the source data from their
producers and use them under their terms ([`DATA_NOTICE.md`](../DATA_NOTICE.md)).
