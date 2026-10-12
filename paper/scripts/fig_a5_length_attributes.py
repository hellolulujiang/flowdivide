"""Figure A5: three length attributes of South America, in km, and the same three on basin 10827 (10.3 km2), the
example basin of Figure 6, at 1 arc-second.
    (a) the distance to the outlet along the flow path, on the 1/400 degree view;
    (b) the upstream flow length, read down to 20 cells a degree by the largest value in each cell, so that every river
        cell is a printed pixel;
    (c) the longest flow path of every basin of at least 100 km2, thicker for larger basins, over the basin raster;
    (d, e) the distance to the outlet and the upstream flow length over basin 10827, every pixel at 1 arc-second;
    (f) the longest flow path of basin 10827 over its channel network (1 km2), with its boundary and outlet.
Reads from data/ the views (data_03_basin_view, data_a5_length_views), the longest flow paths (data_a5_length_views) and the files of basin 10827
(data_a5_example_basin); writes fig_a5_length_attributes.png into ../figures.

    python3 fig_a5_length_attributes.py
"""
import json
import os
import paths
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
MOSAIC = (-93.0, -32.0, -57.0, 15.0)              # lon min, lon max, lat min, lat max
OCEAN = "#f4f2ee"
LAND = "#e3e1dc"
LFP_COLOUR = "#1b4a70"
NETWORK_COLOUR = "#9aa3ad"
EXAMPLE_BASIN = 10827                                  # 10.3 km2, the example basin of Figure 6
BASIN_FRAME = (-48.636, -48.587, -27.907, -27.848)   # as Figure A5: around basin 10827's bbox (-48.630, -27.902, -48.594, -27.853)
BASIN_XTICKS = ([-48.63, -48.61, -48.59], ["48.63°W", "48.61°W", "48.59°W"])
BASIN_YTICKS = ([-27.90, -27.88, -27.86], ["27.90°S", "27.88°S", "27.86°S"])
MARKER_HALF_WIDTH = 0.9                           # degrees: the red box on the continent around the example basin
# the basin colours of Figure 6a (fig_06_partition_as_data.PALETTE); the paths are dark enough to stand out
BASIN_PALETTE = ["#9fbdd4", "#d8c79c", "#aecfa6", "#e0b7a4", "#bcb3d4", "#c9d894", "#e6c4d1", "#9dc9c2", "#d6cbae",
                 "#b2c2da", "#cfb9c9", "#a7d3dc"]
TEXT_COLOUR = "#2b3038"


def frame(axis):
    axis.set_xlim(MOSAIC[0], MOSAIC[1])
    axis.set_ylim(MOSAIC[2], MOSAIC[3])
    axis.set_xticks([-90, -75, -60, -45])
    axis.set_yticks([-45, -30, -15, 0])
    axis.set_xticklabels(["90°W", "75°W", "60°W", "45°W"])
    axis.set_yticklabels(["45°S", "30°S", "15°S", "0°"])
    axis.tick_params(labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#8a8f98")
    axis.set_facecolor(OCEAN)


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
    axis.set_facecolor("#ffffff")


def example_basin_marker(axis, outlet):
    """where the basin of the lower row lies: a box of fixed size, the basin itself being 0.036 by 0.049 degrees"""
    centre_lon = 0.5 * (outlet["bbox"][0] + outlet["bbox"][2])
    centre_lat = 0.5 * (outlet["bbox"][1] + outlet["bbox"][3])
    for colour, width, order in (("#ffffff", 1.7, 7), ("#d7301f", 0.7, 8)):
        axis.add_patch(Rectangle((centre_lon - MARKER_HALF_WIDTH, centre_lat - MARKER_HALF_WIDTH),
                                 2 * MARKER_HALF_WIDTH, 2 * MARKER_HALF_WIDTH, fill=False, edgecolor=colour,
                                 linewidth=width, zorder=order))


def colour_bar(axis, image, label_text, ticks, tick_labels):
    # under the panel, below the longitude labels, so that nothing sits on the map
    bar_axis = axis.inset_axes([0.24, -0.20, 0.52, 0.03])
    bar = axis.figure.colorbar(image, cax=bar_axis, orientation="horizontal", ticks=ticks)
    bar.ax.set_xticklabels(tick_labels)
    bar.ax.tick_params(labelsize=style.SIZE_LABEL, length=1.8, pad=1.2, width=0.5)
    bar.ax.minorticks_off()
    bar.outline.set_linewidth(0.5)
    bar.ax.set_title(label_text, fontsize=style.SIZE_LABEL, pad=2.0, color=TEXT_COLOUR)


def main():
    style.apply_style()
    ldn = np.load(os.path.join(DATA_DIRECTORY, "ldn_south-america_400th_view.npy"))
    lup = np.load(os.path.join(DATA_DIRECTORY, "lup_south-america_20th_view.npy"))
    basin = np.load(os.path.join(DATA_DIRECTORY, "bsn_south-america_400th_view.npy"))
    with open(os.path.join(DATA_DIRECTORY, "lfp_south-america_400th_ge100km2.geojson")) as handle:
        paths = json.load(handle)["features"]
    ldn_km = np.where(ldn >= 0, ldn / 1000.0, np.nan)
    lup_km = np.where(lup >= 0, lup / 1000.0, np.nan)
    low, high = 0.0, np.nanpercentile(lup_km, 95)
    print("LDN km max %.0f  LUP km max %.0f  paths %d" % (np.nanmax(ldn_km), np.nanmax(lup_km), len(paths)))

    with np.load(os.path.join(DATA_DIRECTORY, "example_basin_%d_ldn_lup.npz" % EXAMPLE_BASIN)) as stored:
        basin_ldn_km = stored["ldn_m"] / 1000.0
        basin_lup_km = stored["lup_m"] / 1000.0
        basin_extent = tuple(stored["extent"])
    with np.load(os.path.join(DATA_DIRECTORY, "example_basin_%d_lfp.npz" % EXAMPLE_BASIN)) as stored:
        basin_path = np.column_stack([stored["lon"], stored["lat"]])
    with np.load(os.path.join(DATA_DIRECTORY, "example_basin_%d_lines.npz" % EXAMPLE_BASIN)) as stored:
        basin_segments = np.split(stored["coordinates"], stored["offsets"][1:-1])
    with open(os.path.join(DATA_DIRECTORY, "example_basin_%d_boundary.geojson" % EXAMPLE_BASIN)) as handle:
        basin_boundary = np.asarray(json.load(handle)["features"][0]["geometry"]["coordinates"][0])
    with open(os.path.join(DATA_DIRECTORY, "example_basin_%d_outlet.json" % EXAMPLE_BASIN)) as handle:
        outlet = json.load(handle)

    figure = plt.figure(figsize=(7.48, 8.1))
    grid = figure.add_gridspec(2, 3, left=0.062, right=0.988, top=0.962, bottom=0.10, wspace=0.06, hspace=0.55,
                               height_ratios=[1.0, 1.24])
    axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    lower = [figure.add_subplot(grid[1, column]) for column in range(3)]
    colour_map = plt.get_cmap("YlGnBu").copy()
    colour_map.set_bad(OCEAN)

    image = axes[0].imshow(ldn_km, cmap=colour_map, vmin=0, vmax=6000, extent=MOSAIC, interpolation="nearest",
                           origin="upper")
    colour_bar(axes[0], image, "distance to the outlet (km)", [0, 3000, 6000], ["0", "3000", "6000"])
    style.panel_heading(axes[0], "(a)", "Distance to the outlet")
    longest_km = max(path["properties"]["longest_flow_path_m"] for path in paths) / 1000.0
    style.panel_foot(axes[0], "along the flow path, up to {:,.0f} km".format(round(longest_km, -2)), y=-0.31)

    terrain = plt.get_cmap("terrain").copy()
    terrain.set_bad(OCEAN)
    image = axes[1].imshow(lup_km, cmap=terrain, vmin=low, vmax=high, extent=MOSAIC, interpolation="nearest",
                           origin="upper")
    colour_bar(axes[1], image, "upstream flow length (km)", [0, 200, 400], ["0", "200", "400"])
    style.panel_heading(axes[1], "(b)", "Upstream flow length")
    style.panel_foot(axes[1], "longest path from a divide to the pixel", y=-0.31)
    print("LUP colour range %.1f to %.1f km" % (low, high))

    # the basin raster under the paths, in pale colours, so that each longest flow path is seen inside the basin
    # it belongs to
    painted = np.zeros(basin.shape, dtype=np.int16)
    on_land = basin > 0
    painted[on_land] = (np.floor((basin[on_land].astype(np.float64) * 0.6180339887) % 1.0 * len(BASIN_PALETTE))).astype(np.int16) + 1
    axes[2].imshow(painted, cmap=ListedColormap([OCEAN] + BASIN_PALETTE), vmin=0, vmax=len(BASIN_PALETTE), extent=MOSAIC,
                   interpolation="nearest", origin="upper")
    for feature in sorted(paths, key=lambda f: f["properties"]["basin_area_km2"]):
        geometry = feature["geometry"]
        lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        area = feature["properties"]["basin_area_km2"]
        width = 0.25 + 0.9 * min(max(np.log10(area / 100.0), 0.0) / 4.8, 1.0)
        for line in lines:
            xy = np.asarray(line)
            axes[2].plot(xy[:, 0], xy[:, 1], color=LFP_COLOUR, linewidth=width, solid_capstyle="round", zorder=3)
    # the legend of (c), in the slot the colour bars take in (a) and (b): two entries only;
    # the caption says that a line is drawn thicker for a larger basin
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[2].legend([Line2D([], [], color=LFP_COLOUR, linewidth=0.9, solid_capstyle="round"),
                    Patch(facecolor=BASIN_PALETTE[0], edgecolor=BASIN_PALETTE[4], linewidth=0.6)],
                   ["longest flow path", "basin raster, coloured by basin id"], loc="upper center", bbox_to_anchor=(0.5, -0.13),
                   ncol=1, frameon=False, fontsize=style.SIZE_LABEL, handlelength=1.8, handletextpad=0.5, borderaxespad=0.0,
                   labelspacing=0.5)
    style.panel_heading(axes[2], "(c)", "Longest flow path")
    # no footer under (c)

    for axis in axes:
        frame(axis)
        example_basin_marker(axis, outlet)
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    # ---- the example basin at 1 arc-second ----
    basin_map = plt.get_cmap("YlGnBu").copy()
    basin_map.set_bad("#ffffff")
    # one range for (d) and (e): both run from 0 to the longest flow path (8.9 km for basin 10827), the distance to the
    # outlet at its head and the upstream flow length at the outlet
    top_km = float(np.ceil(np.nanmax(basin_ldn_km) / 5.0) * 5.0) if np.nanmax(basin_ldn_km) > 20 else float(np.ceil(np.nanmax(basin_ldn_km)))
    image = lower[0].imshow(basin_ldn_km, cmap=basin_map, vmin=0, vmax=top_km, extent=basin_extent,
                            interpolation="nearest", origin="upper")
    ldn_ticks = [0, top_km / 2.0, top_km]
    colour_bar(lower[0], image, "distance to the outlet (km)", ldn_ticks, ["%g" % t for t in ldn_ticks])
    style.panel_heading(lower[0], "(d)", "Basin %d, distance to the outlet" % EXAMPLE_BASIN)
    style.panel_foot(lower[0], "%.1f km²; every pixel at 1 arc-second; up to %.1f km" % (
        outlet["basin_area_km2"], np.nanmax(basin_ldn_km)), y=-0.065)

    basin_terrain = plt.get_cmap("terrain").copy()
    basin_terrain.set_bad("#ffffff")
    # most pixels lie below 1 km and the outlet holds the whole longest flow path, so the scale is logarithmic: the
    # hillslope gradient and the channel network both show
    image = lower[1].imshow(np.maximum(basin_lup_km, 0.03), cmap=basin_terrain, norm=LogNorm(vmin=0.03, vmax=top_km),
                            extent=basin_extent, interpolation="nearest", origin="upper")
    lup_ticks = [t for t in (0.1, 1, 10, 60) if t < top_km] + [top_km]
    colour_bar(lower[1], image, "upstream flow length (km), log scale", lup_ticks, ["%g" % t for t in lup_ticks])
    style.panel_heading(lower[1], "(e)", "Basin %d, upstream flow length" % EXAMPLE_BASIN)
    style.panel_foot(lower[1], "up to %.1f km, at the outlet" % np.nanmax(basin_lup_km), y=-0.065)
    print("basin LDN drawn 0 to %.0f km, LUP 0.03 to %.0f km log; LUP 95th percentile %.2f km" % (
        top_km, top_km, np.nanpercentile(basin_lup_km, 95)))

    lower[2].add_collection(LineCollection(basin_segments, colors=NETWORK_COLOUR, linewidths=0.45, capstyle="round",
                                           joinstyle="round", zorder=2))
    lower[2].plot(basin_path[:, 0], basin_path[:, 1], color=LFP_COLOUR, linewidth=1.6, solid_capstyle="round", zorder=4)
    lower[2].legend([Line2D([], [], color=LFP_COLOUR, linewidth=1.6, solid_capstyle="round"),
                     Line2D([], [], color=NETWORK_COLOUR, linewidth=0.7)],
                    ["longest flow path, %.1f km" % (outlet["longest_flow_path_m"] / 1000.0), "channel network, 1 km² threshold"],
                    loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=1, frameon=False, fontsize=style.SIZE_LABEL,
                    handlelength=1.8, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.5)
    style.panel_heading(lower[2], "(f)", "Basin %d, longest flow path" % EXAMPLE_BASIN)
    for axis in lower:
        basin_frame(axis)
        axis.plot(basin_boundary[:, 0], basin_boundary[:, 1], color="#5a6068", linewidth=0.5, zorder=3)
        # the outlet in every panel of the basin row
        axis.plot(outlet["outlet_lon"], outlet["outlet_lat"], zorder=9, **style.OUTLET_MARKER)
    lower[1].set_yticklabels([])
    lower[2].set_yticklabels([])

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_a5_length_attributes")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
