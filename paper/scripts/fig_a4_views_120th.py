"""Figure A4: the partition as data at the 1/120 degree view, the four panels of Figure 6a-d.
    (a) the basin view, every basin filled by its color_id, the basins of (b) marked;
    (b) the table of the view: the first N rows of the basin table (the basins of at least a quarter of a cell),
        with the cells each holds and its color_id; a basin with no cell keeps its row, color_id -9999;
    (c) the region view at 2^31, the windows of the regions of (d) in red;
    (d) the region table with the same two columns; under it the window of region 62211 in cells.
Reads from data/ the views and their rows (data_a4_views_120th), the region table (data_05_region_views), the row of basin 10827
(data_06_basin_table_extracts) and the colour palette (data_a4_views_120th); writes fig_a4_views_120th.png into ../figures.

    python3 fig_a4_views_120th.py
"""
import csv
import os
import paths
import sys
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
CONTINENT = "south-america"
PALETTE_PATH = os.path.join(DATA_DIRECTORY, "color_id_palette.csv")

CELLS_PER_DEGREE = 120
PIXELS_PER_DEGREE = 3600
FACTOR = 30
OCEAN = "#f4f2ee"
BOUNDARY_LINE = (0.16, 0.18, 0.21, 1.0)
TEXT_COLOUR = "#2b3038"
HIGHLIGHT = "#fff3b0"
INDEX_COLUMN = "#e8e8e8"
RED = "#8c1d0b"
EXAMPLE_BASIN = 10827
LAST_BASIN = None            # the last row of the view, basin N, found from the file (7,383,055 basins in the table; 93,661 in the 1/120 view)
MARKER_HALF_WIDTH = 0.9
SHOWN_REGIONS = ["61101", "61501", "62210", "62211", "206001", "206002"]     # the same regions as Figure 6d: 62210, 62211 are the two regions at the mouth of the Amazon (10 and 11 of Figure 4)
TABLE_FONT = 7.6

style.apply_style()


def read_palette():
    palette = {}
    with open(PALETTE_PATH) as handle:
        for row in csv.DictReader(handle):
            palette[int(row["color_id"])] = row["hex"]
    return palette


def read_rows(path):
    with open(path) as handle:
        header = handle.readline().split()
        return [dict(zip(header, line.split())) for line in handle if line.strip()]


def read_raster(path):
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray().astype(np.int64)
    transform = dataset.GetGeoTransform()
    extent = (transform[0], transform[0] + dataset.RasterXSize * transform[1],
              transform[3] + dataset.RasterYSize * transform[5], transform[3])
    dataset = None
    return array, extent


def colour_ids(rows_path, id_field, wanted):
    """object id -> color_id for every row with a cell; the rows of the wanted objects (None in wanted: the last
    row); the number of rows without a cell (coarse_grid_count 0, color_id -9999) and their ids when few"""
    table = pd.read_csv(rows_path)
    with_cell = table[table["coarse_grid_count"] > 0]
    colours = dict(zip(with_cell[id_field].astype(int), with_cell["color_id"].astype(int)))
    missing_ids_all = [int(i) for i in table.loc[table["coarse_grid_count"] == 0, id_field]]
    last_id = int(table[id_field].iloc[-1])
    wanted = [last_id if w is None else w for w in wanted]
    rows = {int(r[id_field]): r for r in table[table[id_field].isin(wanted)].to_dict("records")}
    return colours, rows, len(missing_ids_all), (missing_ids_all if len(missing_ids_all) < 1000 else []), last_id


def paint_by_colour_id(axis, raster, extent, colours, palette):
    """the raster's ids replaced by their color_id and drawn with the palette; 0 stays ocean"""
    top = int(raster.max()) + 1
    lookup = np.zeros(top, dtype=np.int16)
    ids = np.fromiter(colours.keys(), dtype=np.int64)
    lookup[ids] = np.fromiter(colours.values(), dtype=np.int16)
    painted = np.where(raster > 0, lookup[np.clip(raster, 0, top - 1)], 0)
    colour_list = [OCEAN] + [palette[k] for k in range(1, max(palette) + 1)]
    axis.imshow(painted, cmap=ListedColormap(colour_list), vmin=0, vmax=len(colour_list) - 1,
                extent=extent, interpolation="nearest", origin="upper")


def draw_boundaries(axis, raster, extent):
    boundary = np.zeros(raster.shape, dtype=bool)
    boundary[:, :-1] |= (raster[:, :-1] != raster[:, 1:]) & (raster[:, :-1] > 0) & (raster[:, 1:] > 0)
    boundary[:-1, :] |= (raster[:-1, :] != raster[1:, :]) & (raster[:-1, :] > 0) & (raster[1:, :] > 0)
    layer = np.zeros(raster.shape + (4,), dtype=float)
    layer[boundary] = BOUNDARY_LINE
    axis.imshow(layer, extent=extent, interpolation="nearest", origin="upper")


def basin_panel(axis, raster, extent, colours, palette, example_row, last_row):
    paint_by_colour_id(axis, raster, extent, colours, palette)
    lon_min, lon_max, lat_min, lat_max = extent
    rows, cols = raster.shape
    for basin_id in (1, 2, 3, 4):          # the four largest basins, circled ids
        where = np.nonzero(raster == basin_id)
        lat = lat_max - (np.median(where[0]) + 0.5) * (lat_max - lat_min) / rows
        lon = lon_min + (np.median(where[1]) + 0.5) * (lon_max - lon_min) / cols
        axis.text(lon, lat, str(basin_id), ha="center", va="center", fontsize=8, fontweight="bold",
                  color=TEXT_COLOUR, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=TEXT_COLOUR, lw=0.6))
    centre_lon = 0.5 * (float(example_row["bbox_minlon"]) + float(example_row["bbox_maxlon"]))
    centre_lat = 0.5 * (float(example_row["bbox_minlat"]) + float(example_row["bbox_maxlat"]))
    for colour, width, order in (("#ffffff", 1.7, 7), (RED, 0.8, 8)):
        axis.add_patch(Rectangle((centre_lon - MARKER_HALF_WIDTH, centre_lat - MARKER_HALF_WIDTH),
                                 2 * MARKER_HALF_WIDTH, 2 * MARKER_HALF_WIDTH, fill=False, edgecolor=colour,
                                 linewidth=width, zorder=order))
    axis.text(centre_lon + MARKER_HALF_WIDTH + 0.6, centre_lat, str(EXAMPLE_BASIN), fontsize=7, color=RED,
              fontweight="bold", ha="left", va="center", zorder=9,
              bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
    last_lon, last_lat = float(last_row["outlet_lon"]), float(last_row["outlet_lat"])
    axis.plot([last_lon], [last_lat], marker="o", markersize=2.6, markerfacecolor=RED, markeredgecolor="white",
              markeredgewidth=0.5, linestyle="none", zorder=9)
    axis.text(last_lon + 1.4, last_lat + 1.4, f"{LAST_BASIN:,}", fontsize=7, color=RED,
              fontweight="bold", ha="left", va="center", zorder=9,
              bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))


def region_panel(axis, raster, extent, colours, palette, shown_regions):
    paint_by_colour_id(axis, raster, extent, colours, palette)
    draw_boundaries(axis, raster, extent)
    lon_min, lon_max, lat_min, lat_max = extent
    label_corner = {"62211": "top-right", "62210": "bottom-right", "61501": "top-right",
                    "61101": "top-left", "206001": "left", "206002": "right"}
    for row in shown_regions:
        west = lon_min + int(row["col_min"]) / PIXELS_PER_DEGREE
        north = lat_max - int(row["row_min"]) / PIXELS_PER_DEGREE
        width = int(row["ncol"]) / PIXELS_PER_DEGREE
        height = int(row["nrow"]) / PIXELS_PER_DEGREE
        axis.add_patch(Rectangle((west, north - height), width, height, fill=False, edgecolor=RED, linewidth=0.9))
        corner = label_corner.get(row["region_id"], "top-left")
        if corner in ("left", "right"):
            x = west - 0.5 if corner == "left" else west + width + 0.5
            axis.text(x, north - height / 2, row["region_id"], fontsize=7, color=RED, fontweight="bold",
                      ha="right" if corner == "left" else "left", va="center",
                      bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
            continue
        x = west + 0.4 if "left" in corner else west + width - 0.4
        y = north - 0.4 if "top" in corner else north - height + 0.4
        axis.text(x, y, row["region_id"], fontsize=7, color=RED, fontweight="bold",
                  ha="left" if "left" in corner else "right", va="top" if "top" in corner else "bottom",
                  bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))


def style_map(axis, extent, letter, title_text, subtitle_text):
    lon_min, lon_max, lat_min, lat_max = extent
    axis.set_xlim(-90.0, lon_max)
    axis.set_ylim(lat_min, lat_max)
    axis.set_xticks([-75, -60, -45])
    axis.set_yticks([-45, -30, -15, 0])
    axis.set_xticklabels(["75°W", "60°W", "45°W"])
    axis.set_yticklabels(["45°S", "30°S", "15°S", "0°"])
    axis.tick_params(length=2, pad=1.5, labelsize=style.SIZE_TICK, colors=TEXT_COLOUR)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#9a9a9a")
    style.panel_heading(axis, letter, title_text, subtitle_text)


def draw_table(axis, header, align, body_rows, ellipsis_after, column_widths, shaded, letter, title_text, subtitle_text,
               row_height=0.115, font_size=None):
    """the hand-drawn table of Figure 6, unchanged"""
    if isinstance(ellipsis_after, int):
        ellipsis_after = [ellipsis_after]
    font_size = font_size or TABLE_FONT
    axis.set_axis_off()
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    header_height = row_height * (1.65 if any("\n" in name for name in header) else 1.0)
    top = 0.90
    x_edges = np.concatenate([[0.0], np.cumsum(column_widths)]) / sum(column_widths)
    bottom = top - header_height - (len(body_rows) + len(ellipsis_after)) * row_height
    for column_index, colour in shaded.items():
        axis.add_patch(Rectangle((x_edges[column_index], bottom), x_edges[column_index + 1] - x_edges[column_index],
                                 top - bottom, facecolor=colour, edgecolor="none", zorder=0))
    axis.plot([0, 1], [top, top], color=TEXT_COLOUR, lw=0.7)
    axis.plot([0, 1], [top - header_height, top - header_height], color=TEXT_COLOUR, lw=0.5)
    axis.plot([0, 1], [bottom, bottom], color=TEXT_COLOUR, lw=0.7)
    pad = 0.012

    def x_of(column_index):
        if header[column_index] == "…":
            return (x_edges[column_index] + x_edges[column_index + 1]) / 2
        if align[column_index] == "right":
            return x_edges[column_index + 1] - pad
        if align[column_index] == "left":
            return x_edges[column_index] + pad
        return (x_edges[column_index] + x_edges[column_index + 1]) / 2

    for column_index, name in enumerate(header):
        axis.text(x_of(column_index), top - header_height / 2, name, ha="center" if name == "…" else align[column_index],
                  va="center", fontsize=font_size, color=TEXT_COLOUR, fontweight="bold", linespacing=1.15)
    lines = [[("…" if name == "…" else value) for name, value in zip(header, cells)] for cells in body_rows]
    for position in sorted(ellipsis_after, reverse=True):
        lines.insert(position, ["…"] * len(header))
    for row_index, cells in enumerate(lines):
        y = top - header_height - (row_index + 0.5) * row_height
        for column_index, value in enumerate(cells):
            axis.text(x_of(column_index), y, value, ha="center" if header[column_index] == "…" else align[column_index],
                      va="center", fontsize=font_size, color=TEXT_COLOUR)
    style.panel_heading(axis, letter, title_text, subtitle_text)
    return top, row_height, bottom, x_edges


def main():
    palette = read_palette()
    basin_raster, extent = read_raster(os.path.join(DATA_DIRECTORY, "bsn_%s_%dth.tif" % (CONTINENT, CELLS_PER_DEGREE)))
    region_raster, region_extent = read_raster(os.path.join(DATA_DIRECTORY, "rgn_%s_%dth_160deg2.tif" % (CONTINENT, CELLS_PER_DEGREE)))
    assert extent == region_extent, (extent, region_extent)
    global LAST_BASIN
    basin_colours, basin_rows, basins_missing, _, LAST_BASIN = colour_ids(
        os.path.join(DATA_DIRECTORY, "bsn_%s_%dth_rows.csv" % (CONTINENT, CELLS_PER_DEGREE)), "basin_id",
        (1, 2, 3, 4, EXAMPLE_BASIN, None))
    region_colours, region_view_rows, regions_missing, regions_missing_ids, _ = colour_ids(
        os.path.join(DATA_DIRECTORY, "rgn_%s_%dth_160deg2_rows.csv" % (CONTINENT, CELLS_PER_DEGREE)), "region_id",
        [int(r) for r in SHOWN_REGIONS])
    region_rows = read_rows(os.path.join(DATA_DIRECTORY, "region_fine_%s_160deg2.csv" % CONTINENT))
    example_row = read_rows(os.path.join(DATA_DIRECTORY, "example_basin_%d_row.txt" % EXAMPLE_BASIN))[0]
    pixels_of = {basin_id: int(row["basin_grid_count"]) for basin_id, row in basin_rows.items()}
    last_row = basin_rows[LAST_BASIN]
    last_has_cell = int(last_row["coarse_grid_count"]) > 0
    assert last_has_cell or int(last_row["color_id"]) == -9999, last_row
    rows_in_the_view = len(basin_colours) + basins_missing
    print("the 1/%d view: %s basins (ids 1..%s), %s with a cell, %s without; regions with a cell: %d, without: %d %s" % (
        CELLS_PER_DEGREE, format(rows_in_the_view, ","), format(LAST_BASIN, ","), format(len(basin_colours), ","), format(basins_missing, ","),
        len(region_colours), regions_missing, regions_missing_ids))

    figure = plt.figure(figsize=(7.48, 6.0))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 2.3], height_ratios=[1, 1.12],
                               left=0.05, right=0.99, top=0.95, bottom=0.04, wspace=0.08, hspace=0.34)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    # (a) the basin view, filled by color_id, the basins of (b) marked
    basin_panel(axis_a, basin_raster, extent, basin_colours, palette, example_row, {"outlet_lon": last_row["outlet_lon"], "outlet_lat": last_row["outlet_lat"]})
    style_map(axis_a, extent, "(a)", "Basin view at 1/120°", "%s of the view's %s basins hold a cell" % (format(len(basin_colours), ","), format(rows_in_the_view, ",")))

    # (b) the basin view's table: the same rows as Figure 6b, with the two columns of the view
    header_b = ["basin_id", "…", "pixels", "area\n(km²)", "…", "cells at\n1/120°", "color_id"]
    align_b = ["right"] * 7
    body_b = []
    for basin_id in (1, 2, 3, 4, EXAMPLE_BASIN, LAST_BASIN):
        row = basin_rows[basin_id]
        area = float(row["basin_area_km2"])
        body_b.append([f"{basin_id:,}", "", f"{pixels_of[basin_id]:,}",
                       f"{area:,.0f}" if area >= 1 else f"{area:.4f}", "",
                       f"{int(row['coarse_grid_count']):,}", str(int(row["color_id"]))])
    widths_b = [1.1, 0.3, 1.5, 1.05, 0.3, 1.0, 0.9]
    draw_table(axis_b, header_b, align_b, body_b, [4, 5], widths_b, {5: HIGHLIGHT, 6: HIGHLIGHT}, "(b)", "Basin table of the view",
               "the first %s rows of the basin table, cell or no cell" % format(rows_in_the_view, ","), row_height=0.09, font_size=7.0)
    top, row_height, bottom, _ = 0.9, 0.09, 0.9 - 0.09 * 1.65 - 8 * 0.09, None      # the header, six rows and two ellipsis lines
    last_note = ("basin %s: the last row of the view, the smallest basin of at least a quarter of a cell (%.4f km²);\n%s"
                 % (format(LAST_BASIN, ","), float(last_row["basin_area_km2"]),
                    "no cell at this view, colour −9999; its row stays" if not last_has_cell else "%d cell%s at this view" % (int(last_row["coarse_grid_count"]), "" if int(last_row["coarse_grid_count"]) == 1 else "s")))
    axis_b.text(0.0, bottom - 0.06, last_note, fontsize=style.SIZE_TITLE, color=RED, ha="left", va="top", fontweight="bold", linespacing=1.35)

    # (c) the region view, filled by color_id, the windows of the regions of (d)
    row_of = {r["region_id"]: r for r in region_rows}
    shown_regions = [row_of[region_id] for region_id in SHOWN_REGIONS]
    region_panel(axis_c, region_raster, extent, region_colours, palette, shown_regions)
    style_map(axis_c, extent, "(c)", "Region view at 1/120°", "%d of %d regions with a cell" % (len(region_colours), len(region_rows)))

    # (d) the region view's table
    header_d = ["region_id", "…", "nrow", "ncol", "pixels", "basins", "…", "level", "cells at\n1/120°", "color_id"]
    align_d = ["right"] * 10
    body_d = []
    for row in shown_regions:
        region_id = int(row["region_id"])
        view = region_view_rows[region_id]
        body_d.append([row["region_id"], "", f"{int(row['nrow']):,}", f"{int(row['ncol']):,}",
                       f"{int(row['region_grid_count']):,}", f"{int(row['basin_count']):,}", "", row["topological_level"],
                       f"{int(view['coarse_grid_count']):,}", str(int(view["color_id"]))])
    widths_d = [1.0, 0.3, 0.85, 0.85, 1.4, 0.8, 0.3, 0.6, 1.0, 0.9]
    top, row_height, bottom, x_edges = draw_table(axis_d, header_d, align_d, body_d, [1, 2, 4], widths_d,
                                                  {8: HIGHLIGHT, 9: HIGHLIGHT, 7: INDEX_COLUMN},
                                                  "(d)", "Region table of the view", "one row per region", row_height=0.085, font_size=7.4)
    first = row_of["62211"]
    degrees = [int(first[key]) // PIXELS_PER_DEGREE for key in ("nrow", "ncol")]
    cells = [int(first[key]) // FACTOR for key in ("nrow", "ncol")]
    axis_d.text(0.0, bottom - 0.06,
                f"{first['region_id']}: {int(first['nrow']):,} × {int(first['ncol']):,} pixels = {degrees[0]}° × {degrees[1]}° "
                f"= {cells[0]:,} × {cells[1]:,} cells of 30 × 30 pixels;\nthe same window, the same ids, one row per object at every view",
                fontsize=style.SIZE_TITLE, color=RED, ha="left", va="top", fontweight="bold", linespacing=1.35)

    output = style.figure_path(FIGURE_DIRECTORY, "fig_a4_views_120th")
    figure.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.04)
    print(output)


if __name__ == "__main__":
    main()
