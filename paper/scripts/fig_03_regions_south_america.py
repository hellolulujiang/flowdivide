"""Figure 3: basins and basin groups of South America (HydroSHEDS v2, 1 arc-second, capacity 2^31).
    (a) the basins, shaded by drainage area, the four whose whole-degree rectangle exceeds the capacity outlined;
    (b) the groups along the HydroBASINS Level-03 units; (c) the groups along a Hilbert curve.
Reads from data/ the 1/400 degree basin view (data_03_basin_view) and the basin areas, groups and counts (data_03_basin_groups);
writes fig_03_regions_south_america.png into ../figures.  Drawn at print size for a two-column figure (7.48 inch).

    python3 fig_03_regions_south_america.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
CAP_CELLS_FOR_BASINS = 160 * 3600 * 3600

CONTINENT_SETTINGS = {
    "south-america": dict(
        decimated_basin_file="bsn_south-america_400th_view.npy",
        mosaic=(-93.0, -32.0, -57.0, 15.0),      # lon min, lon max, lat min, lat max
        view=(-93.0, -32.0, -57.0, 15.0),
        figure_size=(7.48, 5.30),
        longitude_ticks=[-90, -75, -60, -45],
        latitude_ticks=[-45, -30, -15, 0],
        panels_side_by_side=True,
    ),
}

CAP_PIXELS = 160 * 3600 * 3600
OVER_CAP_HATCH_COLOUR = "#96301a"

# the eight basins that drain at least 250,000 km2, identified by area and outlet
GREAT_RIVER_NAME = {
    1: "Amazon", 2: "Paraná", 3: "Orinoco", 4: "Tocantins", 5: "São Francisco",
    6: "Uruguay", 7: "Parnaíba", 8: "Magdalena",
}
GREAT_RIVER_LABEL_OFFSET = {          # degrees, to keep the crowded names apart
    8: (-2.6, 2.2),                   # Magdalena, out to the north-west
    3: (2.4, -0.4),                   # Orinoco, out to the east
    4: (-2.6, 1.2),                   # Tocantins, up and left, off the Sao Francisco
    5: (1.8, -4.6),                   # Sao Francisco, down its own basin
    7: (1.2, 1.0),                    # Parnaiba, up and right
    6: (-1.0, -1.4),                  # Uruguay, clear of the Parana
}

OVER_CAP_NAME_OFFSET = {              # degrees from the middle of the basin's land
    1: (0.0, 1.5),                    # Amazon, inside
    2: (0.0, -1.0),                   # Parana, inside
    3: (1.0, 2.2),                    # Orinoco, up and to the east, clear of the Amazon
    4: (8.5, 6.0),                    # Tocantins-Araguaia, out in the Atlantic to the north-east
    5: (10.8, -10.5),                 # Sao Francisco, off the coast to the south-east, right-aligned
}
OVER_CAP_NAME_LEADER = {4, 5}         # these two get a hairline from the name to the basin
OVER_CAP_NAME_ALIGN = {5: "right"}    # the name ends at the frame instead of being centred
OVER_CAP_NAMES_DRAWN = {1, 2, 3, 4, 5}   # the five largest basins, named and numbered (those of Figure 7c)

PALETTE = ["#9fbdd4", "#d8c79c", "#aecfa6", "#e0b7a4", "#bcb3d4",
           "#c9d894", "#e6c4d1", "#9dc9c2", "#d6cbae", "#b2c2da",
           "#cfb9c9", "#a7d3dc"]
OCEAN = "#f4f2ee"
NAME_THE_RIVERS = False        # False: every region its own colour, nothing labelled


def one_colour_a_region(count):
    """A colour a region, spread by the golden angle in hue and stepped in brightness so that two
    regions next to each other in the list never come out close."""
    import colorsys
    colours = []
    for index in range(count):
        hue = (index * 0.6180339887498949) % 1.0
        saturation = (0.42, 0.62, 0.52)[index % 3]
        value = (0.94, 0.80, 0.88)[(index // 3) % 3]
        colours.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return colours


def read_basin_groups(continent):
    """basin_id -> (Level-03 group, Hilbert group, area in km2) as arrays indexed by basin id, for the basins of the view"""
    table = pd.read_csv(os.path.join(DATA_DIRECTORY, "basin_groups_%s_400th.csv" % continent))
    size = int(table["basin_id"].max()) + 1
    level3, hilbert, area = np.zeros(size, np.int64), np.zeros(size, np.int64), np.zeros(size, np.float64)
    level3[table["basin_id"]] = table["level3_id"]
    hilbert[table["basin_id"]] = table["hilbert_group"]
    area[table["basin_id"]] = table["basin_area_km2"]
    return level3, hilbert, area


def read_region_table(path):
    """group_id -> (land_grid_count, window_grid_count, basin_count)."""
    table = {}
    with open(path) as handle:
        header = handle.readline().split()
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            parts = line.split()
            table[int(parts[index["group_id"]])] = (
                int(parts[index["land_grid_count"]]),
                int(parts[index["window_grid_count"]]),
                int(parts[index["basin_count"]]),
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


def basin_counts(continent):
    """the number of basins, of basins of at least 1 km2, and {basin: whole-degree pixels} of those over the capacity"""
    with open(os.path.join(DATA_DIRECTORY, "basin_counts_%s.json" % continent)) as handle:
        counts = json.load(handle)
    return counts["basins"], counts["basins_of_at_least_1_km2"], {int(k): v for k, v in counts["over_the_capacity"].items()}


def draw_basin_panel(axis, basin_raster, title_text, label_letter, over_the_cap_basins,
                     area_of_basin):
    """The basin raster.  This panel is the input, not a third partition, so it is drawn with
    one ramp on the logarithm of basin area: the great rivers come out dark, the fringe of small
    coastal basins light.  That is what there is to see here."""
    painted = np.full(basin_raster.shape, np.nan)
    land = basin_raster > 0
    lookup = np.zeros(int(basin_raster.max()) + 1, dtype=np.float64)
    for basin_id, area in area_of_basin.items():
        if basin_id < lookup.size:
            lookup[basin_id] = area
    painted[land] = np.log10(np.maximum(lookup[basin_raster[land]], 0.01))
    colour_map = plt.get_cmap("YlGnBu").copy()
    colour_map.set_bad(OCEAN)
    image = axis.imshow(painted, cmap=colour_map, vmin=-1.0, vmax=7.0,
                        extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), interpolation="nearest",
                        origin="upper")
    rows, cols = basin_raster.shape
    longitude = LON_MIN + (np.arange(cols) + 0.5) * (LON_MAX - LON_MIN) / cols
    latitude = LAT_MAX - (np.arange(rows) + 0.5) * (LAT_MAX - LAT_MIN) / rows
    for basin_id in over_the_cap_basins:
        one_basin = (basin_raster == basin_id).astype(float)
        if one_basin.any():
            axis.contour(longitude, latitude, one_basin, levels=[0.5],
                         colors=["#8c1d0b"], linewidths=1.1)
    # the names of the basins that exceed the capacity on their own: Amazon and Parana inside
    # their outline, the two narrow ones out in the Atlantic with a hairline to their land
    for basin_id in sorted(set(over_the_cap_basins) | OVER_CAP_NAMES_DRAWN):
        if basin_id not in GREAT_RIVER_NAME or basin_id not in OVER_CAP_NAMES_DRAWN:
            continue
        row_index, col_index = np.nonzero(basin_raster == basin_id)
        if row_index.size == 0:
            continue
        centre = (float(longitude[col_index].mean()), float(latitude[row_index].mean()))
        offset_lon, offset_lat = OVER_CAP_NAME_OFFSET.get(basin_id, (0.0, 0.0))
        at = (centre[0] + offset_lon, centre[1] + offset_lat)
        # the circled basin id, as Figures 6 and A4 write it
        text_style = dict(fontsize=style.SIZE_LABEL, fontweight="bold", ha="center", va="center", color="#12161c", zorder=6,
                          bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor="#12161c", linewidth=0.6))
        if basin_id in OVER_CAP_NAME_LEADER:
            axis.annotate(str(basin_id), xy=centre, xytext=at,
                          arrowprops=dict(arrowstyle="-", linewidth=0.5, color="#12161c", shrinkA=0, shrinkB=4),
                          **text_style)
        else:
            axis.text(at[0], at[1], str(basin_id), **text_style)
        # the river's name under the circled id, so that a reader of Figure 7 can find the basin here
        axis.text(at[0], at[1] - 1.9, GREAT_RIVER_NAME[basin_id], fontsize=style.SIZE_LABEL,
                  ha=OVER_CAP_NAME_ALIGN.get(basin_id, "center"), va="top", color="#12161c", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75))
    # colour bar in the empty ocean at the lower left
    bar_axis = axis.inset_axes([0.07, 0.075, 0.34, 0.028])
    colour_bar = axis.figure.colorbar(image, cax=bar_axis, orientation="horizontal", ticks=[-1, 1, 3, 5, 7])
    colour_bar.ax.set_xticklabels(["\u22640.1", "10", "10\u00b3", "10\u2075", "10\u2077"],
                                  fontfamily="DejaVu Sans")
    colour_bar.ax.tick_params(labelsize=style.SIZE_LABEL, length=1.8, pad=1.2, width=0.5)
    colour_bar.outline.set_linewidth(0.5)
    colour_bar.ax.set_title("drainage area (km\u00b2)", fontsize=style.SIZE_LABEL, pad=2.0, color="#2b3038")
    style.panel_heading(axis, label_letter, title_text)


def separate_the_labels(figure, axis, label_records, rounds=300):
    """Push apart region ids that came out on top of one another.

    A region id starts at the middle of its own land, and two small neighbours can put their
    middles within a few pixels of each other.  This pass works on the drawn boxes, in display
    units, so it needs to know nothing about the continent: it separates whatever overlaps and
    keeps every box inside the frame.  A region too small to see gets a hairline back to its
    mark once it has been pushed away from it."""
    if not label_records:
        return
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer=renderer)
    margin = 1.5
    gap = 2.5
    for _ in range(rounds):
        boxes = [record["text"].get_window_extent(renderer=renderer) for record in label_records]
        shift_x = [0.0] * len(label_records)
        shift_y = [0.0] * len(label_records)
        something_moved = False
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                box_first = boxes[first]
                box_second = boxes[second]
                overlap_x = min(box_first.x1, box_second.x1) - max(box_first.x0, box_second.x0)
                overlap_y = min(box_first.y1, box_second.y1) - max(box_first.y0, box_second.y0)
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue
                something_moved = True
                if overlap_y <= overlap_x:
                    step = 0.5 * (overlap_y + gap)
                    if box_first.y0 + box_first.y1 <= box_second.y0 + box_second.y1:
                        shift_y[first] -= step
                        shift_y[second] += step
                    else:
                        shift_y[first] += step
                        shift_y[second] -= step
                else:
                    step = 0.5 * (overlap_x + gap)
                    if box_first.x0 + box_first.x1 <= box_second.x0 + box_second.x1:
                        shift_x[first] -= step
                        shift_x[second] += step
                    else:
                        shift_x[first] += step
                        shift_x[second] -= step
        if not something_moved:
            break
        for index, record in enumerate(label_records):
            if shift_x[index] == 0.0 and shift_y[index] == 0.0:
                continue
            box = boxes[index]
            wanted_x = shift_x[index]
            wanted_y = shift_y[index]
            if box.x0 + wanted_x < frame.x0 + margin:
                wanted_x = frame.x0 + margin - box.x0
            if box.x1 + wanted_x > frame.x1 - margin:
                wanted_x = frame.x1 - margin - box.x1
            if box.y0 + wanted_y < frame.y0 + margin:
                wanted_y = frame.y0 + margin - box.y0
            if box.y1 + wanted_y > frame.y1 - margin:
                wanted_y = frame.y1 - margin - box.y1
            position_in_display = axis.transData.transform(record["text"].get_position())
            moved_in_display = (position_in_display[0] + wanted_x,
                                position_in_display[1] + wanted_y)
            moved_in_data = axis.transData.inverted().transform(moved_in_display)
            record["text"].set_position((float(moved_in_data[0]), float(moved_in_data[1])))
    for record in label_records:
        if not record["too_small_to_see"]:
            continue
        anchor_lon, anchor_lat = record["anchor"]
        label_lon, label_lat = record["text"].get_position()
        anchor_in_display = axis.transData.transform((anchor_lon, anchor_lat))
        label_in_display = axis.transData.transform((label_lon, label_lat))
        distance = ((anchor_in_display[0] - label_in_display[0]) ** 2 +
                    (anchor_in_display[1] - label_in_display[1]) ** 2) ** 0.5
        if distance < 6.0:
            continue
        # the hairline stops 1.5 points short of the label's box instead of running under it to the label's middle
        patch = record["text"].get_bbox_patch()
        box = (patch.get_window_extent(renderer=renderer) if patch is not None
               else record["text"].get_window_extent(renderer=renderer)).expanded(1.0, 1.0)
        box = box.padded(1.5)
        end_in_display = label_in_display
        for step in range(1, 401):
            fraction = step / 400.0
            point = (anchor_in_display[0] + fraction * (label_in_display[0] - anchor_in_display[0]),
                     anchor_in_display[1] + fraction * (label_in_display[1] - anchor_in_display[1]))
            if box.contains(point[0], point[1]):
                break
            end_in_display = point
        end_lon, end_lat = axis.transData.inverted().transform(end_in_display)
        axis.plot([anchor_lon, float(end_lon)], [anchor_lat, float(end_lat)], linestyle="-",
                  linewidth=0.35, color="#12161c", zorder=4, clip_on=True)


def draw_panel(axis, region_raster, region_table, basin_raster, title_text, label_letter):
    region_list = sorted(int(r) for r in np.unique(region_raster) if r > 0)
    painted = np.zeros(region_raster.shape, dtype=np.int16)
    colour_of_region = greedy_region_colours(region_raster, region_list)
    for region_id in region_list:
        painted[region_raster == region_id] = colour_of_region[region_id] + 1
    colour_map = ListedColormap([OCEAN] + PALETTE)
    top = len(PALETTE)
    axis.imshow(painted, cmap=colour_map, vmin=0, vmax=top,
                extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), interpolation="nearest",
                origin="upper")

    # the line between two regions, drawn from the raster itself
    boundary = np.zeros(region_raster.shape, dtype=bool)
    boundary[:, :-1] |= (region_raster[:, :-1] != region_raster[:, 1:]) & \
                        (region_raster[:, :-1] > 0) & (region_raster[:, 1:] > 0)
    boundary[:-1, :] |= (region_raster[:-1, :] != region_raster[1:, :]) & \
                        (region_raster[:-1, :] > 0) & (region_raster[1:, :] > 0)
    boundary_layer = np.zeros(region_raster.shape + (4,), dtype=float)
    boundary_layer[boundary] = (0.16, 0.18, 0.21, 1.0)
    axis.imshow(boundary_layer, extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
                interpolation="nearest", origin="upper")

    # the region id at the middle of its own land, so a reader can look a region up in the table
    rows, cols = region_raster.shape
    longitude = LON_MIN + (np.arange(cols) + 0.5) * (LON_MAX - LON_MIN) / cols
    latitude = LAT_MAX - (np.arange(rows) + 0.5) * (LAT_MAX - LAT_MIN) / rows
    label_records = []
    for region_id in region_list:
        row_index, col_index = np.nonzero(region_raster == region_id)
        if row_index.size == 0:
            continue
        centre_lon = float(longitude[col_index].mean())
        centre_lat = float(latitude[row_index].mean())
        anchor_lon = min(max(centre_lon, VIEW_LON_MIN + 0.6), VIEW_LON_MAX - 0.6)
        anchor_lat = min(max(centre_lat, VIEW_LAT_MIN + 0.6), VIEW_LAT_MAX - 0.6)
        centre_lon = min(max(centre_lon, VIEW_LON_MIN + 3.4), VIEW_LON_MAX - 3.4)
        centre_lat = min(max(centre_lat, VIEW_LAT_MIN + 2.2), VIEW_LAT_MAX - 2.2)
        if (centre_lon < VIEW_LON_MIN + 9.0) and (centre_lat > VIEW_LAT_MAX - 6.0):
            centre_lat = VIEW_LAT_MAX - 8.5
        a_region_too_small_to_see = row_index.size < 40
        if a_region_too_small_to_see:
            # too small to see at this resolution: mark where it is, and put the id beside the mark
            axis.plot([anchor_lon], [anchor_lat], marker="o", markersize=2.4,
                      markerfacecolor="#12161c", markeredgecolor="white", markeredgewidth=0.3,
                      linestyle="none", clip_on=True, zorder=7)
            centre_lat = centre_lat - 2.6
        text_object = axis.text(centre_lon, centre_lat, str(region_id), fontsize=style.SIZE_LABEL,
                                ha="center", va="center", color="#12161c", clip_on=True, zorder=6,
                                bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                                          edgecolor="none", alpha=0.70))
        label_records.append({"text": text_object, "anchor": (anchor_lon, anchor_lat),
                              "too_small_to_see": a_region_too_small_to_see})
    over_the_cap = [r for r in region_list if region_table[r][1] > CAP_PIXELS]
    over_mask = (np.isin(region_raster, over_the_cap) & (region_raster > 0)).astype(float)
    rows, cols = region_raster.shape
    longitude = LON_MIN + (np.arange(cols) + 0.5) * (LON_MAX - LON_MIN) / cols
    latitude = LAT_MAX - (np.arange(rows) + 0.5) * (LAT_MAX - LAT_MIN) / rows
    axis.contourf(longitude, latitude, over_mask, levels=[0.5, 1.5],
                  colors="none", hatches=["/////"])
    style.panel_heading(axis, label_letter, title_text)
    return region_list, over_the_cap, label_records


def label_great_rivers(axis, basin_raster, region_raster, region_table):
    """Name every basin of at least 250,000 km2 that holds a region on its own."""
    rows, cols = basin_raster.shape
    longitude = LON_MIN + (np.arange(cols) + 0.5) * (LON_MAX - LON_MIN) / cols
    latitude = LAT_MAX - (np.arange(rows) + 0.5) * (LAT_MAX - LAT_MIN) / rows
    for basin_id, river_name in GREAT_RIVER_NAME.items():
        mask = basin_raster == basin_id
        if not mask.any():
            continue
        region_id = int(region_raster[mask][0])
        land_cells, _, _ = region_table[region_id]
        row_index, col_index = np.nonzero(mask)
        offset_lon, offset_lat = GREAT_RIVER_LABEL_OFFSET.get(basin_id, (0.0, 0.0))
        centre_lon = float(longitude[col_index].mean()) + offset_lon
        centre_lat = float(latitude[row_index].mean()) + offset_lat
        share_of_region = (basin_raster[region_raster == region_id] == basin_id).mean()
        alone = share_of_region > 0.90
        axis.text(centre_lon, centre_lat, river_name, fontsize=7.0, ha="center", va="center",
                  color="#12161c" if alone else "#5c636d",
                  fontweight="bold" if alone else "normal",
                  fontstyle="normal" if alone else "italic",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.80 if alone else 0.62))


def main():
    continent = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in CONTINENT_SETTINGS \
                else "south-america"
    settings = CONTINENT_SETTINGS[continent]
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, VIEW_LON_MIN, VIEW_LON_MAX, VIEW_LAT_MIN, VIEW_LAT_MAX
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = settings["mosaic"]
    VIEW_LON_MIN, VIEW_LON_MAX, VIEW_LAT_MIN, VIEW_LAT_MAX = settings["view"]
    basin_raster = np.load(os.path.join(DATA_DIRECTORY, settings["decimated_basin_file"]))
    style.apply_style()
    matplotlib.rcParams["hatch.linewidth"] = 0.30
    matplotlib.rcParams["hatch.color"] = OVER_CAP_HATCH_COLOUR
    if settings["panels_side_by_side"]:
        figure, axes = plt.subplots(1, 3, figsize=(settings["figure_size"][0],
                                                   settings["figure_size"][1] * 0.80))
        plt.subplots_adjust(left=0.058, right=0.988, top=0.945, bottom=0.205, wspace=0.06)
        stats_offset = -0.085
    else:
        figure, axes = plt.subplots(3, 1, figsize=(settings["figure_size"][0],
                                                   settings["figure_size"][1] * 1.45))
        plt.subplots_adjust(left=0.062, right=0.988, top=0.975, bottom=0.062, hspace=0.20)
        stats_offset = -0.095

    basin_count_in_table, basins_at_least_one_square_kilometre, over_the_cap_basins = basin_counts(continent)
    level3_of_basin, hilbert_of_basin, area_lookup = read_basin_groups(continent)
    area_of_basin = {basin_id: area_lookup[basin_id] for basin_id in range(1, area_lookup.size)}
    draw_basin_panel(axes[0], basin_raster, "Watershed delineation", "(a)", over_the_cap_basins,
                     area_of_basin)
    axes[0].text(0.965, 0.965,
                 "%s complete basins\n%s of at least 1 km\u00b2"
                 % (format(basin_count_in_table, ","),
                    format(basins_at_least_one_square_kilometre, ",")),
                 transform=axes[0].transAxes, fontsize=style.SIZE_LABEL, ha="right", va="top",
                 color="#2b3038", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75))
    # the criterion is the basin's rectangle taken out to whole degrees, as a region's window is; the capacity
    # 2^31 is 160 whole-degree cells, 2.07 x 10^9 pixels
    style.panel_foot(axes[0], "dark red: the %d basins whose whole-degree rectangle\nexceeds the capacity, 2.07 \u00d7 10$^{9}$ pixels"
                     % len(over_the_cap_basins), y=stats_offset)

    label_records_of_axis = {}
    panels = [
        (level3_of_basin, "group_l3_fine_%s.csv" % continent, "Basin groups by Level-03 unit", "(b)"),
        (hilbert_of_basin, "group_hilbert_fine_%s.csv" % continent, "Basin groups by Hilbert curve", "(c)"),
    ]
    for axis, (lookup, region_table_name, title_text, letter) in zip(axes[1:], panels):
        region_table = read_region_table(os.path.join(DATA_DIRECTORY, region_table_name))
        region_raster = np.where(basin_raster > 0, lookup[np.clip(basin_raster, 0, len(lookup) - 1)], 0)
        region_list, over_the_cap, label_records = draw_panel(axis, region_raster, region_table,
                                                              basin_raster, title_text, letter)
        label_records_of_axis[id(axis)] = label_records
        if NAME_THE_RIVERS:
            label_great_rivers(axis, basin_raster, region_raster, region_table)
        over_the_cap_in_table = sum(1 for values in region_table.values() if values[1] > CAP_PIXELS)
        # the groups that hold a basin over the capacity on its own
        regions_of_an_oversized_basin = set(int(lookup[basin_id]) for basin_id in over_the_cap_basins
                                            if basin_id < len(lookup))
        forced = sum(1 for region_id, values in region_table.items()
                     if values[1] > CAP_PIXELS and region_id in regions_of_an_oversized_basin)
        style.panel_foot(axis, "%d basin groups, %d over the capacity (hatched);\n%d of them by a basin over on its own"
                         % (len(region_table), over_the_cap_in_table, forced), y=stats_offset)
        axis.set_xlim(VIEW_LON_MIN, VIEW_LON_MAX)
        axis.set_ylim(VIEW_LAT_MIN, VIEW_LAT_MAX)
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
    for axis in axes:
        axis.set_xlim(VIEW_LON_MIN, VIEW_LON_MAX)
        axis.set_ylim(VIEW_LAT_MIN, VIEW_LAT_MAX)
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
    if settings["panels_side_by_side"]:
        axes[1].set_yticklabels([])
        axes[2].set_yticklabels([])

    for axis in axes[1:]:
        separate_the_labels(figure, axis, label_records_of_axis.get(id(axis), []))

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return                            # the overlap lint runs the script and only inspects the figure
    stem = "fig_03_regions_%s" % continent.replace("-", "_")
    out_path = style.figure_path(FIGURE_DIRECTORY, stem)
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
