"""Figure 6: the partition as data, the two rasters and three tables a user reads.
    (a) the basin raster and (b) the basin table: the ids run from 1 without a gap, so a raster value is a row number;
    (c) the region raster and (d) the region table: every region a rectangle on whole degrees (row_min, col_min, nrow,
        ncol multiples of 3,600 pixels), read as one window of the continental file; the level is counted upstream from the
        region that holds the cut basin's outlet, which is 0;
    (e) the piece table: the first four pieces of the Amazon and the last.
The maps mark exactly the rows of their tables: basins 1 to 4, basin 10827 (the example basin of Figures A5 and A6) and
the last basin in (a); the windows of the listed regions in (c).
Reads from data/ the 1/400 degree views (data_03_basin_view, data_05_region_views), the region and piece tables (data_05_region_views, data_04_amazon_pieces),
the head of the basin table and the row of basin 10827 (data_06_basin_table_extracts); writes
fig_06_partition_as_data.png into ../figures.

    python3 fig_06_partition_as_data.py
"""
import collections
import os
import paths
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY

MOSAIC = (-93.0, -32.0, -57.0, 15.0)          # lon min, lon max, lat min, lat max
PIXELS_PER_DEGREE = 3600
PALETTE = ["#9fbdd4", "#d8c79c", "#aecfa6", "#e0b7a4", "#bcb3d4",
           "#c9d894", "#e6c4d1", "#9dc9c2", "#d6cbae", "#b2c2da",
           "#cfb9c9", "#a7d3dc"]
OCEAN = "#f4f2ee"
BOUNDARY_LINE = (0.16, 0.18, 0.21, 1.0)
TEXT_COLOUR = "#2b3038"
HIGHLIGHT = "#fff3b0"
INDEX_COLUMN = "#e8e8e8"
RED = "#8c1d0b"
AMAZON_BASIN = 1
EXAMPLE_BASIN = 10827             # the example basin of Figures A5 and A6: its row in (b), its place in (a)
MARKER_HALF_WIDTH = 0.9           # degrees: the red box around it in (a); the basin itself is 0.036 by 0.049 degrees
WHOLE_BASIN_REGION = "61501"      # the Orinoco's HydroBASINS Level-03 unit: 19 whole basins, no cut
TABLE_FONT = 7.6                  # few rows and words, large enough to read
VIEW_BASIN_COLOUR = {18: "#b2c2da", 252: "#c9d894", 343: "#e0b7a4"}     # the three basins of the window of (f)
VIEW_OTHER_BASIN = "#d6cbae"
VIEW_CELL_EDGE = "white"
PIECE_LABEL_OFFSET = {7383056: (3.5, 3.2), 7383058: (0.0, -4.5), 7383059: (2.5, 4.5), 7383080: (0.0, -4.0)}   # degrees, label away from the piece

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
style.apply_style()


def read_rows(path):
    with open(path) as handle:
        header = handle.readline().split()
        return [dict(zip(header, line.split())) for line in handle if line.strip()]


def greedy_region_colours(region_raster, region_list):
    neighbours = collections.defaultdict(set)
    for one_side, other_side in ((region_raster[:, :-1], region_raster[:, 1:]),
                                 (region_raster[:-1, :], region_raster[1:, :])):
        differ = (one_side != other_side) & (one_side > 0) & (other_side > 0)
        for one, other in zip(one_side[differ], other_side[differ]):
            neighbours[int(one)].add(int(other))
            neighbours[int(other)].add(int(one))
    colour_of_region = {}
    for region_id in sorted(region_list, key=lambda r: -len(neighbours[r])):
        taken = {colour_of_region[other] for other in neighbours[region_id] if other in colour_of_region}
        for colour_index in range(len(PALETTE)):
            if colour_index not in taken:
                colour_of_region[region_id] = colour_index
                break
        else:
            colour_of_region[region_id] = 0
    return colour_of_region


def draw_boundaries(axis, raster, extent):
    boundary = np.zeros(raster.shape, dtype=bool)
    boundary[:, :-1] |= (raster[:, :-1] != raster[:, 1:]) & (raster[:, :-1] > 0) & (raster[:, 1:] > 0)
    boundary[:-1, :] |= (raster[:-1, :] != raster[1:, :]) & (raster[:-1, :] > 0) & (raster[1:, :] > 0)
    layer = np.zeros(raster.shape + (4,), dtype=float)
    layer[boundary] = BOUNDARY_LINE
    axis.imshow(layer, extent=extent, interpolation="nearest", origin="upper")


def basin_raster_panel(axis, basin_raster, extent, example_row, last_row):
    """The basins coloured by their id spread by the golden ratio, so that neighbours rarely share a colour; the
    basins of table (b) marked: 1 to 4 numbered, the example basin 10827 in a red box, the last basin (one pixel at
    the southern tip) with a red dot"""
    painted = np.zeros(basin_raster.shape, dtype=np.int16)
    land = basin_raster > 0
    painted[land] = (np.floor((basin_raster[land].astype(np.float64) * 0.6180339887) % 1.0 * len(PALETTE))).astype(np.int16) + 1
    axis.imshow(painted, cmap=ListedColormap([OCEAN] + PALETTE), vmin=0, vmax=len(PALETTE),
                extent=extent, interpolation="nearest", origin="upper")
    lon_min, lon_max, lat_min, lat_max = extent
    rows, cols = basin_raster.shape
    for basin_id in (1, 2, 3, 4):          # the four largest basins, circled ids
        where = np.nonzero(basin_raster == basin_id)
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
    axis.text(last_lon + 1.4, last_lat + 1.4, f"{int(last_row['basin_id']):,}", fontsize=7, color=RED,
              fontweight="bold", ha="left", va="center", zorder=9,
              bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))


def region_raster_panel(axis, region_raster, extent, shown_regions):
    """The 59 regions; the whole-degree rectangles of the regions listed in panel (d), in red, labels on white."""
    region_list = sorted(int(r) for r in np.unique(region_raster) if r > 0)
    colour_of_region = greedy_region_colours(region_raster, region_list)
    painted = np.zeros(region_raster.shape, dtype=np.int16)
    for region_id in region_list:
        painted[region_raster == region_id] = colour_of_region[region_id] + 1
    axis.imshow(painted, cmap=ListedColormap([OCEAN] + PALETTE), vmin=0, vmax=len(PALETTE),
                extent=extent, interpolation="nearest", origin="upper")
    draw_boundaries(axis, region_raster, extent)
    lon_min, lon_max, lat_min, lat_max = extent
    label_corner = {"62211": "top-right", "62210": "bottom-right", "62209": "top-left",
                    "62208": "bottom-right", "62202": "top-left", "61501": "top-right",
                    "61101": "top-left", "61201": "bottom-left", "206001": "left", "206002": "right"}
    for row in shown_regions:
        west = lon_min + int(row["col_min"]) / PIXELS_PER_DEGREE
        north = lat_max - int(row["row_min"]) / PIXELS_PER_DEGREE
        width = int(row["ncol"]) / PIXELS_PER_DEGREE
        height = int(row["nrow"]) / PIXELS_PER_DEGREE
        axis.add_patch(Rectangle((west, north - height), width, height, fill=False, edgecolor=RED, linewidth=0.9))
        corner = label_corner.get(row["region_id"], "top-left")
        if corner in ("left", "right"):           # a small rectangle (an island group): the label beside it
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
    """A table drawn by hand: header, body rows,
    an ellipsis row after each position of ellipsis_after (one position or a list of them), an ellipsis
    column wherever the header is "…" (columns of the released table left out); some columns shaded
    (colour by column index)."""
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
        if header[column_index] == "\u2026":
            return (x_edges[column_index] + x_edges[column_index + 1]) / 2
        if align[column_index] == "right":
            return x_edges[column_index + 1] - pad
        if align[column_index] == "left":
            return x_edges[column_index] + pad
        return (x_edges[column_index] + x_edges[column_index + 1]) / 2

    for column_index, name in enumerate(header):
        axis.text(x_of(column_index), top - header_height / 2, name, ha="center" if name == "\u2026" else align[column_index], va="center",
                  fontsize=font_size, color=TEXT_COLOUR, fontweight="bold", linespacing=1.15)
    lines = [[("\u2026" if name == "\u2026" else value) for name, value in zip(header, cells)] for cells in body_rows]
    for position in sorted(ellipsis_after, reverse=True):
        lines.insert(position, ["\u2026"] * len(header))
    for row_index, cells in enumerate(lines):
        y = top - header_height - (row_index + 0.5) * row_height
        for column_index, value in enumerate(cells):
            axis.text(x_of(column_index), y, value, ha="center" if header[column_index] == "\u2026" else align[column_index], va="center",
                      fontsize=font_size, color=TEXT_COLOUR)
    style.panel_heading(axis, letter, title_text, subtitle_text)
    return top, row_height, bottom, x_edges


def main():
    basin_raster = np.load(os.path.join(DATA_DIRECTORY, "bsn_south-america_400th_view.npy"))
    region_raster = np.load(os.path.join(DATA_DIRECTORY, "rgn_south-america_400th_view_160deg2.npy"))
    region_rows = read_rows(os.path.join(DATA_DIRECTORY, "region_fine_south-america_160deg2.csv"))
    basin_rows = read_rows(os.path.join(DATA_DIRECTORY, "basin_table_fine_south-america_head.txt"))
    example_row = read_rows(os.path.join(DATA_DIRECTORY, "example_basin_%d_row.txt" % EXAMPLE_BASIN))[0]
    piece_rows = read_rows(os.path.join(DATA_DIRECTORY, "piece_fine_south-america_basin1_160deg2.csv"))
    extent = MOSAIC

    # three rows, a map on the left and its table on the right, the piece table alone in the third
    # the map marks exactly the rows of its table
    figure = plt.figure(figsize=(7.48, 9.2))
    grid = figure.add_gridspec(3, 2, width_ratios=[1.0, 2.3], height_ratios=[1, 1.12, 0.9],
                               left=0.05, right=0.99, top=0.96, bottom=0.03, wspace=0.08, hspace=0.42)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])
    axis_e = figure.add_subplot(grid[2, 1])

    # (a) basin mask, marking exactly the six basins of (b)
    basin_raster_panel(axis_a, basin_raster, extent, example_row, basin_rows[-1])
    style_map(axis_a, extent, "(a)", "Basin raster", "7,383,055 basins")

    # (b) basin table: the four largest, the example basin of the appendix, the last
    header_b = ["basin_id", "outlet\nrow", "outlet\ncol", "pixels", "area\n(km\u00b2)", "\u2026",
                "row_min", "row_max", "col_min", "col_max", "\u2026"]
    align_b = ["right"] * 11
    body_b = []
    for row in basin_rows[:4] + [example_row] + basin_rows[-1:]:
        area = float(row["basin_area_km2"])
        body_b.append([f"{int(row['basin_id']):,}", f"{int(row['outlet_row']):,}", f"{int(row['outlet_col']):,}",
                       f"{int(row['basin_grid_count']):,}", f"{area:,.0f}" if area >= 1 else f"{area:.4f}", "",
                       f"{int(row['basin_row_min']):,}", f"{int(row['basin_row_max']):,}",
                       f"{int(row['basin_col_min']):,}", f"{int(row['basin_col_max']):,}", ""])
    widths_b = [1.15, 0.95, 0.9, 1.5, 1.05, 0.3, 0.95, 0.95, 0.95, 0.95, 0.3]
    draw_table(axis_b, header_b, align_b, body_b, [4, 5], widths_b, {0: HIGHLIGHT}, "(b)", "Basin table", "one row per basin",
               row_height=0.09, font_size=7.0)

    # (c) region mask with the rectangles of the regions of (d)
    by_id = sorted(region_rows, key=lambda r: int(r["region_id"]))
    row_of = {r["region_id"]: r for r in by_id}
    # the first two regions, the Orinoco's unit of whole basins, the two Amazon regions of the crossing shown
    # in (e), and the last two, island groups; an ellipsis row where ids are skipped
    shown_ids = [by_id[0]["region_id"], WHOLE_BASIN_REGION, "62210", "62211", by_id[-2]["region_id"], by_id[-1]["region_id"]]
    shown_regions = [row_of[region_id] for region_id in shown_ids]
    region_raster_panel(axis_c, region_raster, extent, shown_regions)
    style_map(axis_c, extent, "(c)", "Region raster", "59 regions")

    # (d) region table: the Amazon's outlet region and the next, its last, and a region of whole basins;
    # under it the rectangle of the outlet region in cells of the three views, whole numbers all
    header_d = ["region_id", "\u2026", "row_min", "col_min", "nrow", "ncol", "pixels", "basins", "\u2026", "cut\nbasin", "\u2026", "level"]
    align_d = ["right"] * 12
    body_d = []
    for row in shown_regions:
        cut = row["cut_basin_id"] if int(row["cut_basin_id"]) > 0 else "\u2013"
        body_d.append([row["region_id"], "", f"{int(row['row_min']):,}", f"{int(row['col_min']):,}", f"{int(row['nrow']):,}",
                       f"{int(row['ncol']):,}", f"{int(row['region_grid_count']):,}", f"{int(row['basin_count']):,}", "", cut,
                       "", row["topological_level"]])
    widths_d = [1.0, 0.3, 0.9, 0.9, 0.85, 0.85, 1.4, 0.8, 0.3, 0.65, 0.3, 0.6]
    top, row_height, bottom, x_edges = draw_table(axis_d, header_d, align_d, body_d, [1, 2, 4], widths_d,
                                                  {2: HIGHLIGHT, 3: HIGHLIGHT, 4: HIGHLIGHT, 5: HIGHLIGHT, 11: INDEX_COLUMN},
                                                  "(d)", "Region table", "one row per region", row_height=0.085, font_size=7.4)
    # the window of 62211 in cells of the views is given under Figure A4

    # (e) piece table: the piece at the outlet, the two pieces of the crossing from region 62210 into 62211,
    # and the last piece
    amazon_pieces = [r for r in piece_rows if int(r["basin_id"]) == AMAZON_BASIN]
    shown_pieces = amazon_pieces[:4] + amazon_pieces[-1:]
    header_e = ["piece_id", "basin_id", "\u2026", "drains into\npiece", "region_id", "outlet\nrow", "outlet\ncol", "\u2026",
                "pixels", "upstream\ncount at\noutlet", "\u2026"]
    align_e = ["right"] * 11
    body_e = []
    for row in shown_pieces:
        parent = int(row["parent_piece_id"])
        body_e.append([f"{int(row['piece_id']):,}", row["basin_id"], "", f"{parent:,}" if parent > 0 else "\u2013 (sea)",
                       row["region_id"], f"{int(row['outlet_row']):,}", f"{int(row['outlet_col']):,}", "", f"{int(row['piece_grid_count']):,}",
                       f"{int(row['acc_at_outlet']):,}", ""])
    widths_e = [1.05, 0.9, 0.3, 1.1, 0.95, 0.85, 0.85, 0.3, 1.25, 1.3, 0.3]
    draw_table(axis_e, header_e, align_e, body_e, 4, widths_e, {3: HIGHLIGHT, 4: HIGHLIGHT, 9: HIGHLIGHT},
               "(e)", "Piece table", "one row per piece of a cut basin", row_height=0.1, font_size=7.0)

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    output = style.figure_path(FIGURE_DIRECTORY, "fig_06_partition_as_data")
    figure.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.04)
    print(output)

if __name__ == "__main__":
    main()
