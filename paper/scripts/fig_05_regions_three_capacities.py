"""The same continent partitioned at three capacities: 160, 80 and 40 square degrees.

All three partitions start from the HydroBASINS Level-03 grouping of Figure 3b.  A basin whose whole-degree rectangle
exceeds the capacity on its own is cut at tributary outlets (Figure 4) and its pieces become regions; a group of whole
basins that exceeds it is divided along whole-degree lines into groups that fit.  Halving the cap roughly doubles the regions and cuts a
few more basins, and every region's rectangle stays under the cap.

    python3 fig_05_regions_three_capacities.py [south-america]

Reads rgn_<continent>_400th_view_<cap>deg2.npy, bsn_<continent>_400th_view.npy and
region_fine_<continent>_<cap>deg2.csv from data/ (made by
data_03_basin_view.py and data_05_region_views.py) and writes
fig_05_regions_three_capacities.png into ../figures.
"""
import os
import paths
import sys
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_fill_holes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY

CAPACITIES = (160, 80, 40)
PANEL_LETTERS = ("(a)", "(b)", "(c)")

CONTINENT_SETTINGS = {
    "south-america": dict(
        mosaic=(-93.0, -32.0, -57.0, 15.0),      # lon min, lon max, lat min, lat max
        figure_size=(7.48, 5.30),
        longitude_ticks=[-90, -75, -60, -45],
        latitude_ticks=[-45, -30, -15, 0],
    ),
}

PALETTE = ["#9fbdd4", "#d8c79c", "#aecfa6", "#e0b7a4", "#bcb3d4",
           "#c9d894", "#e6c4d1", "#9dc9c2", "#d6cbae", "#b2c2da",
           "#cfb9c9", "#a7d3dc"]
OCEAN = "#f4f2ee"
CUT_BASIN_OUTLINE = "#8c1d0b"
BOUNDARY_LINE = (0.16, 0.18, 0.21, 1.0)
TEXT_COLOUR = "#2b3038"


def read_region_table(path):
    """region_id -> dict of the columns this figure uses."""
    table = {}
    with open(path) as handle:
        header = handle.readline().split()
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            parts = line.split()
            table[int(parts[index["region_id"]])] = dict(
                land_pixels=int(parts[index["region_grid_count"]]),
                window_pixels=int(parts[index["window_grid_count"]]),
                cut_basin_id=int(parts[index["cut_basin_id"]]),
                piece_count=int(parts[index["piece_count"]]),
                minlon=float(parts[index["minlon"]]),
                maxlon=float(parts[index["maxlon"]]),
                minlat=float(parts[index["minlat"]]),
                maxlat=float(parts[index["maxlat"]]),
            )
    return table


def greedy_region_colours(region_raster, region_list):
    """Give touching regions different colours, so the shapes read as shapes."""
    neighbours = collections.defaultdict(set)
    left, right = region_raster[:, :-1], region_raster[:, 1:]
    differ = (left != right) & (left > 0) & (right > 0)
    for one, other in zip(left[differ], right[differ]):
        neighbours[int(one)].add(int(other))
        neighbours[int(other)].add(int(one))
    upper, lower = region_raster[:-1, :], region_raster[1:, :]
    differ = (upper != lower) & (upper > 0) & (lower > 0)
    for one, other in zip(upper[differ], lower[differ]):
        neighbours[int(one)].add(int(other))
        neighbours[int(other)].add(int(one))
    colour_of_region = {}
    for region_id in sorted(region_list, key=lambda r: -len(neighbours[r])):
        taken = {colour_of_region[other] for other in neighbours[region_id]
                 if other in colour_of_region}
        for colour_index in range(len(PALETTE)):
            if colour_index not in taken:
                colour_of_region[region_id] = colour_index
                break
        else:
            colour_of_region[region_id] = 0
    return colour_of_region


def draw_panel(axis, region_raster, region_table, basin_raster, extent, title_text, label_letter):
    lon_min, lon_max, lat_min, lat_max = extent
    region_list = sorted(int(r) for r in np.unique(region_raster) if r > 0)
    painted = np.zeros(region_raster.shape, dtype=np.int16)
    colour_of_region = greedy_region_colours(region_raster, region_list)
    for region_id in region_list:
        painted[region_raster == region_id] = colour_of_region[region_id] + 1
    colour_map = ListedColormap([OCEAN] + PALETTE)
    axis.imshow(painted, cmap=colour_map, vmin=0, vmax=len(PALETTE),
                extent=(lon_min, lon_max, lat_min, lat_max), interpolation="nearest",
                origin="upper")

    # the line between two regions, drawn from the raster itself
    boundary = np.zeros(region_raster.shape, dtype=bool)
    boundary[:, :-1] |= (region_raster[:, :-1] != region_raster[:, 1:]) & \
                        (region_raster[:, :-1] > 0) & (region_raster[:, 1:] > 0)
    boundary[:-1, :] |= (region_raster[:-1, :] != region_raster[1:, :]) & \
                        (region_raster[:-1, :] > 0) & (region_raster[1:, :] > 0)
    boundary_layer = np.zeros(region_raster.shape + (4,), dtype=float)
    boundary_layer[boundary] = BOUNDARY_LINE
    axis.imshow(boundary_layer, extent=(lon_min, lon_max, lat_min, lat_max),
                interpolation="nearest", origin="upper")

    # the basins that had to be cut, outlined in dark red as in Figure 3a
    rows, cols = region_raster.shape
    longitude = lon_min + (np.arange(cols) + 0.5) * (lon_max - lon_min) / cols
    latitude = lat_max - (np.arange(rows) + 0.5) * (lat_max - lat_min) / rows
    cut_basins = sorted({values["cut_basin_id"] for values in region_table.values()
                         if values["cut_basin_id"] > 0})
    for basin_id in cut_basins:
        one_basin = basin_raster == basin_id
        if one_basin.any():
            # the outer boundary only: a closed basin sitting inside a big one would otherwise
            # come out as a small ring, and the Andes hold hundreds of them
            outer = binary_fill_holes(one_basin).astype(float)
            axis.contour(longitude, latitude, outer, levels=[0.5],
                         colors=[CUT_BASIN_OUTLINE], linewidths=0.7)

    # a region too small to survive the read-down: a dot where the table puts it
    for region_id, values in region_table.items():
        if region_id in colour_of_region:
            continue
        axis.plot([0.5 * (values["minlon"] + values["maxlon"])],
                  [0.5 * (values["minlat"] + values["maxlat"])],
                  marker="o", markersize=2.4, markerfacecolor="#12161c",
                  markeredgecolor="white", markeredgewidth=0.3, linestyle="none", clip_on=True, zorder=7)

    style.panel_heading(axis, label_letter, title_text)
    regions_from_cut_basins = sum(1 for values in region_table.values()
                                  if values["cut_basin_id"] > 0)
    return cut_basins, regions_from_cut_basins


def main():
    continent = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in CONTINENT_SETTINGS \
                else "south-america"
    settings = CONTINENT_SETTINGS[continent]
    extent = settings["mosaic"]
    lon_min, lon_max, lat_min, lat_max = extent
    basin_raster = np.load(os.path.join(DATA_DIRECTORY, "bsn_%s_400th_view.npy" % continent))

    style.apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(settings["figure_size"][0],
                                               settings["figure_size"][1] * 0.80))
    plt.subplots_adjust(left=0.058, right=0.988, top=0.945, bottom=0.205, wspace=0.06)
    stats_offset = -0.085

    for axis, square_degrees, letter in zip(axes, CAPACITIES, PANEL_LETTERS):
        region_raster = np.load(os.path.join(
            DATA_DIRECTORY, "rgn_%s_400th_view_%ddeg2.npy" % (continent, square_degrees)))
        region_table = read_region_table(os.path.join(
            DATA_DIRECTORY, "region_fine_%s_%ddeg2.csv" % (continent, square_degrees)))
        if region_raster.shape != basin_raster.shape:
            sys.exit("region view %s and basin view %s are not on the same grid"
                     % (region_raster.shape, basin_raster.shape))
        cap_pixels = square_degrees * 3600 * 3600
        over_the_cap = [r for r, values in region_table.items() if values["window_pixels"] > cap_pixels]
        if over_the_cap:
            sys.exit("%d deg2: regions over the cap in the table: %s" % (square_degrees, over_the_cap))
        # the capacity is named by the power of two and realised as the round number of whole degrees below it:
        # 2^31 -> 160, 2^30 -> 80, 2^29 -> 40; the same words as in the text and in Table 2
        power_of_two = {160: 31, 80: 30, 40: 29}[square_degrees]       # mathtext: Arial has no \u2070 \u2074\u2013\u2079 superscripts
        cut_basins, regions_from_cut_basins = draw_panel(
            axis, region_raster, region_table, basin_raster, extent,
            "Capacity $2^{%d}$ (%d deg\u00b2)" % (power_of_two, square_degrees), letter)
        pieces = sum(v["piece_count"] for v in region_table.values())
        style.panel_foot(axis, "%d regions; %d basins (dark red outline) cut into\n%d pieces, packed into %d regions"
                         % (len(region_table), len(cut_basins), pieces, regions_from_cut_basins), y=stats_offset)
        print("%3d deg2: %d regions, %d cut basins %s into %d regions, largest rectangle %d = cap %s"
              % (square_degrees, len(region_table), len(cut_basins), cut_basins,
                 regions_from_cut_basins, max(v["window_pixels"] for v in region_table.values()),
                 max(v["window_pixels"] for v in region_table.values()) == cap_pixels))

    for axis in axes:
        axis.set_xlim(lon_min, lon_max)
        axis.set_ylim(lat_min, lat_max)
        axis.set_xticks(settings["longitude_ticks"])
        axis.set_yticks(settings["latitude_ticks"])
        axis.tick_params(labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
        axis.set_xticklabels(["%d°%s" % (abs(v), "W" if v < 0 else "E")
                              for v in settings["longitude_ticks"]])
        axis.set_yticklabels(["0°" if v == 0 else "%d°%s" % (abs(v), "S" if v < 0 else "N")
                              for v in settings["latitude_ticks"]])
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#8a8f98")
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return                            # the overlap lint runs the script and only inspects the figure
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_05_regions_three_capacities")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
