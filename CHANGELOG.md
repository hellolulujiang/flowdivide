# Changelog

## 1.0.0 (2026-09-28)

First public release.

* The three stages of the paper: the partition (FD1), the views and the figures (FD2), and the attributes
  region by region (FD3).
* A GeoPackage view carries a QGIS default style on `color_id`, so that the polygons are coloured the moment
  the file is opened.
* The regular-tile kernels (`tile_kernels.py`) and the three-ways check (`three_ways.py`) the paper
  compares the partition with.
* Test scripts on made-up grids and on a case cut from a real basin.
