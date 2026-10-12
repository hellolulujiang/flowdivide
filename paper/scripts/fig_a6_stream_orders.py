"""Figure A6: three orders of the channel network (1 km2 threshold) as lines, one line per stream segment.
    above, South America, the segments of Strahler order 4 and above, the same segments in the three panels;
    below, basin 10827 (10.3 km2), the example basin of Figure 6, every segment at 1 arc-second;
    (a, d) the Strahler order; (b, e) the Shreve magnitude, the number of channel heads above the segment; (c, f) the
    Hack order, 1 on the main stem.
The line width follows the Strahler order in every panel, so the three panels of a row differ in colour only; each row
has its own colour scales below it.
Reads from data/ the continent's lines (data_a6_network_lines) and the files of basin 10827 (data_a5_example_basin); writes
fig_a6_stream_orders.png into ../figures.

    python3 fig_a6_stream_orders.py
"""
import json
import os
import paths
import sys
import time
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
EXAMPLE_BASIN = 10827                              # 10.3 km2, the example basin of Figure 6
MOSAIC = (-93.0, -32.0, -57.0, 15.0)              # lon min, lon max, lat min, lat max
BASIN_FRAME = (-48.636, -48.587, -27.907, -27.848)   # around basin 10827's bbox (-48.630, -27.902, -48.594, -27.853)
BASIN_XTICKS = ([-48.63, -48.61, -48.59], ["48.63°W", "48.61°W", "48.59°W"])
BASIN_YTICKS = ([-27.90, -27.88, -27.86], ["27.90°S", "27.88°S", "27.86°S"])
MARKER_HALF_WIDTH = 0.9                           # degrees: the red box on the continent around the example basin
OCEAN = "#f4f2ee"
LAND = "#ffffff"
TEXT_COLOUR = "#2b3038"
CONTINENT_MIN_ORDER = 4
HIGHEST_STRAHLER = 11                             # the Amazon's outlet order on South America
HACK_CLASSES = 8                                  # 1 .. 7 and "8 and above" (the Amazon reaches 12)

# line width in points by Strahler order, continent and basin; heavier where the order is higher
WIDTH_STRAHLER_CONTINENT = {4: 0.12, 5: 0.20, 6: 0.32, 7: 0.50, 8: 0.70, 9: 0.95, 10: 1.20, 11: 1.50}
WIDTH_STRAHLER_BASIN = {1: 0.7, 2: 1.1, 3: 1.6, 4: 2.2, 5: 2.9, 6: 3.6, 7: 4.4}   # heavier: a 10 km2 basin fills the panel


def load_lines(npz_name):
    with np.load(os.path.join(DATA_DIRECTORY, npz_name)) as stored:
        coordinates = stored["coordinates"]
        offsets = stored["offsets"]
        attributes = {key: stored[key] for key in ("strahler_order", "shreve_magnitude", "hack_order")}
    segments = np.split(coordinates, offsets[1:-1])
    return segments, attributes


def draw_lines(axis, segments, colours, widths, order_key):
    """one LineCollection, the lines sorted so that the heaviest are drawn last and lie on top"""
    order = np.argsort(order_key, kind="stable")
    collection = LineCollection([segments[index] for index in order], colors=colours[order], linewidths=widths[order],
                                capstyle="round", joinstyle="round", antialiased=True, zorder=3)
    axis.add_collection(collection)


def width_of(values, table):
    keys = np.array(sorted(table))
    lookup = np.array([table[key] for key in keys])
    clipped = np.clip(values, keys[0], keys[-1])
    return lookup[np.searchsorted(keys, clipped)]


def continent_frame(axis):
    axis.set_xlim(MOSAIC[0], MOSAIC[1])
    axis.set_ylim(MOSAIC[2], MOSAIC[3])
    axis.set_aspect("equal")
    axis.set_xticks([-90, -75, -60, -45])
    axis.set_yticks([-45, -30, -15, 0])
    axis.set_xticklabels(["90°W", "75°W", "60°W", "45°W"])
    axis.set_yticklabels(["45°S", "30°S", "15°S", "0°"])
    axis.tick_params(labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#8a8f98")
    axis.set_facecolor(LAND)


def basin_frame(axis):
    axis.set_xlim(BASIN_FRAME[0], BASIN_FRAME[1])
    axis.set_ylim(BASIN_FRAME[2], BASIN_FRAME[3])
    axis.set_aspect("equal")
    axis.set_xticks(BASIN_XTICKS[0])
    axis.set_yticks(BASIN_YTICKS[0])
    axis.set_xticklabels(BASIN_XTICKS[1])
    axis.set_yticklabels(BASIN_YTICKS[1])
    axis.tick_params(labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#8a8f98")
    axis.set_facecolor(LAND)


def colour_bar(axis, mappable, label_text, ticks, tick_labels):
    # under the panel, below the longitude labels, so that nothing sits on the map
    bar_axis = axis.inset_axes([0.22, -0.21, 0.56, 0.03])
    bar = axis.figure.colorbar(mappable, cax=bar_axis, orientation="horizontal", ticks=ticks)
    bar.ax.set_xticklabels(tick_labels)
    bar.ax.tick_params(labelsize=style.SIZE_LABEL, length=1.8, pad=1.2, width=0.5)
    bar.ax.minorticks_off()
    bar.outline.set_linewidth(0.5)
    bar.ax.set_title(label_text, fontsize=style.SIZE_LABEL, pad=2.0, color=TEXT_COLOUR)


def main():
    style.apply_style()
    started = time.time()
    continent_segments, continent = load_lines("net_south-america_400th_ord_ge%d.npz" % CONTINENT_MIN_ORDER)
    basin_segments, basin = load_lines("example_basin_%d_lines.npz" % EXAMPLE_BASIN)
    boundary = gpd.read_file(os.path.join(DATA_DIRECTORY, "example_basin_%d_boundary.geojson" % EXAMPLE_BASIN))
    with open(os.path.join(DATA_DIRECTORY, "example_basin_%d_outlet.json" % EXAMPLE_BASIN)) as handle:
        outlet = json.load(handle)
    print("%d continent segments (order >= %d), %d basin segments, read in %.0f s" % (
        len(continent_segments), CONTINENT_MIN_ORDER, len(basin_segments), time.time() - started))
    print("basin %d: Strahler 1..%d, magnitude 1..%d, Hack 1..%d" % (
        EXAMPLE_BASIN, basin["strahler_order"].max(), basin["shreve_magnitude"].max(), basin["hack_order"].max()))

    # the three colour scales, shared by the two rows
    strahler_map = ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.22, 1.0, HIGHEST_STRAHLER)))
    strahler_norm = BoundaryNorm(np.arange(1, HIGHEST_STRAHLER + 2) - 0.5, strahler_map.N)
    magnitude_map = plt.get_cmap("viridis")
    magnitude_norm = LogNorm(vmin=1.0, vmax=2.0e6)
    hack_map = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(1.0, 0.3, HACK_CLASSES)))    # 1 darkest
    hack_norm = BoundaryNorm(np.arange(1, HACK_CLASSES + 2) - 0.5, hack_map.N)

    figure = plt.figure(figsize=(7.48, 8.15))
    grid = figure.add_gridspec(2, 3, left=0.062, right=0.988, top=0.965, bottom=0.098, wspace=0.06, hspace=0.42,
                               height_ratios=[1.0, 1.24])
    axes = np.array([[figure.add_subplot(grid[row, column]) for column in range(3)] for row in range(2)])

    # ---- the continent: the same segments, three colourings ----
    magnitude_continent = continent["shreve_magnitude"].astype(float)
    hack_continent = np.minimum(continent["hack_order"], HACK_CLASSES)
    continent_widths = width_of(continent["strahler_order"], WIDTH_STRAHLER_CONTINENT)
    draw_lines(axes[0, 0], continent_segments, strahler_map(strahler_norm(continent["strahler_order"])),
               continent_widths, continent["strahler_order"])
    draw_lines(axes[0, 1], continent_segments, magnitude_map(magnitude_norm(magnitude_continent)),
               continent_widths, magnitude_continent)
    draw_lines(axes[0, 2], continent_segments, hack_map(hack_norm(hack_continent)),
               continent_widths, -hack_continent)
    for axis in axes[0]:
        continent_frame(axis)
        # where the basin of the lower row lies: a box of fixed size, the basin itself being 0.036 by 0.049 degrees
        centre_lon = 0.5 * (outlet["bbox"][0] + outlet["bbox"][2])
        centre_lat = 0.5 * (outlet["bbox"][1] + outlet["bbox"][3])
        for colour, width, order in (("#ffffff", 1.7, 7), ("#d7301f", 0.7, 8)):
            axis.add_patch(Rectangle((centre_lon - MARKER_HALF_WIDTH, centre_lat - MARKER_HALF_WIDTH),
                                     2 * MARKER_HALF_WIDTH, 2 * MARKER_HALF_WIDTH,
                                     fill=False, edgecolor=colour, linewidth=width, zorder=order))
    style.panel_heading(axes[0, 0], "(a)", "Strahler order")
    style.panel_heading(axes[0, 1], "(b)", "Shreve magnitude")
    style.panel_heading(axes[0, 2], "(c)", "Hack order")
    style.panel_foot(axes[0, 0], "Strahler order %d and above: %s segments" % (
        CONTINENT_MIN_ORDER, format(len(continent_segments), ",")), y=-0.075)
    style.panel_foot(axes[0, 1], "the same segments", y=-0.075)
    style.panel_foot(axes[0, 2], "the same segments; 1 = the main stem", y=-0.075)

    # the continent's colour bars under its own row: the two rows have their own scales
    # the bar shows the orders drawn, 4 and above, in the colours the lines carry
    drawn_map = ListedColormap(strahler_map.colors[CONTINENT_MIN_ORDER - 1:])
    drawn_norm = BoundaryNorm(np.arange(CONTINENT_MIN_ORDER, HIGHEST_STRAHLER + 2) - 0.5, drawn_map.N)
    strahler_ticks = list(range(CONTINENT_MIN_ORDER, HIGHEST_STRAHLER + 1))
    colour_bar(axes[0, 0], ScalarMappable(norm=drawn_norm, cmap=drawn_map), "Strahler order",
               strahler_ticks, [str(t) for t in strahler_ticks])
    magnitude_ticks = [1, 10, 100, 1000, 10000, 100000, 1000000]
    colour_bar(axes[0, 1], ScalarMappable(norm=magnitude_norm, cmap=magnitude_map), "Shreve magnitude (channel heads)",
               magnitude_ticks, ["1", "10", "$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$", "$10^{6}$"])
    hack_ticks = list(range(1, HACK_CLASSES + 1))
    colour_bar(axes[0, 2], ScalarMappable(norm=hack_norm, cmap=hack_map), "Hack order",
               hack_ticks, [str(t) for t in hack_ticks[:-1]] + ["\u2265%d" % HACK_CLASSES])

    # ---- the basin: every segment at 1 arc-second, on its own scales (orders 1..3 on a 10 km2 basin) ----
    magnitude_basin = basin["shreve_magnitude"].astype(float)
    hack_basin = basin["hack_order"]
    top_strahler = int(basin["strahler_order"].max()); top_magnitude = int(magnitude_basin.max()); top_hack = int(hack_basin.max())
    basin_strahler_map = ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.30, 1.0, top_strahler)))
    basin_strahler_norm = BoundaryNorm(np.arange(1, top_strahler + 2) - 0.5, basin_strahler_map.N)
    basin_magnitude_map = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 1.0, top_magnitude)))
    basin_magnitude_norm = BoundaryNorm(np.arange(1, top_magnitude + 2) - 0.5, basin_magnitude_map.N)
    basin_hack_map = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(1.0, 0.35, top_hack)))
    basin_hack_norm = BoundaryNorm(np.arange(1, top_hack + 2) - 0.5, basin_hack_map.N)
    basin_widths = width_of(basin["strahler_order"], WIDTH_STRAHLER_BASIN)
    draw_lines(axes[1, 0], basin_segments, basin_strahler_map(basin_strahler_norm(basin["strahler_order"])),
               basin_widths, basin["strahler_order"])
    draw_lines(axes[1, 1], basin_segments, basin_magnitude_map(basin_magnitude_norm(magnitude_basin)),
               basin_widths, magnitude_basin)
    draw_lines(axes[1, 2], basin_segments, basin_hack_map(basin_hack_norm(hack_basin)),
               basin_widths, -hack_basin)
    for axis in axes[1]:
        basin_frame(axis)
        boundary.boundary.plot(ax=axis, color="#5a6068", linewidth=0.5, zorder=2)
        axis.plot(outlet["outlet_lon"], outlet["outlet_lat"], zorder=9, **style.OUTLET_MARKER)
    basin_title = "Basin %d, " % EXAMPLE_BASIN
    style.panel_heading(axes[1, 0], "(d)", basin_title + "Strahler order")
    style.panel_heading(axes[1, 1], "(e)", basin_title + "Shreve magnitude")
    style.panel_heading(axes[1, 2], "(f)", basin_title + "Hack order")
    style.panel_foot(axes[1, 0], "%.1f km²; all %s segments at 1 arc-second; outlet order %d" % (
        outlet["basin_area_km2"], format(len(basin_segments), ","),
        int(basin["strahler_order"].max())), y=-0.065)
    style.panel_foot(axes[1, 1], "%s channel heads" % format(int(basin["shreve_magnitude"].max()), ","), y=-0.065)
    style.panel_foot(axes[1, 2], "1 = the main stem", y=-0.065)

    colour_bar(axes[1, 0], ScalarMappable(norm=basin_strahler_norm, cmap=basin_strahler_map), "Strahler order",
               list(range(1, top_strahler + 1)), [str(t) for t in range(1, top_strahler + 1)])
    colour_bar(axes[1, 1], ScalarMappable(norm=basin_magnitude_norm, cmap=basin_magnitude_map), "Shreve magnitude (channel heads)",
               list(range(1, top_magnitude + 1)), [str(t) for t in range(1, top_magnitude + 1)])
    colour_bar(axes[1, 2], ScalarMappable(norm=basin_hack_norm, cmap=basin_hack_map), "Hack order",
               list(range(1, top_hack + 1)), [str(t) for t in range(1, top_hack + 1)])

    for row in range(2):
        axes[row, 1].set_yticklabels([])
        axes[row, 2].set_yticklabels([])

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_a6_stream_orders")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path, "%.0f s" % (time.time() - started))


if __name__ == "__main__":
    main()
