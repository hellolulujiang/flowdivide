"""How a basin that does not fit an int32 rectangle becomes regions: the Amazon, step by step.

(a) The nine Pfafstetter pieces of level 1.  From the outlet the mainstem is walked upstream;
    the four tributaries with the largest upstream cell count are the even pieces 2, 4, 6, 8,
    the mainstem stretches between them the odd pieces 1, 3, 5, 7, 9.  A piece whose
    whole-degree rectangle exceeds the capacity (6 and 9) is hatched, with its rectangle drawn.
(b) A piece over the capacity is replaced by its own nine pieces (61-69, 91-99); every one of
    the 25 pieces now fits.
(c) Consecutive pieces in Pfafstetter code order are merged while the merged rectangle stays
    under the capacity: 11 regions, numbered so that a region's number is larger than that of
    every region upstream of it.
(d) The region graph drawn as a row: every arrow points to a larger number, so a value that travels
    downstream is computed region by region in the order of their numbers; one that travels upstream
    takes a region after those its outlets drain into.
The dark blue line on (a)-(c) is the Amazon's longest flow path at 1/400 degree (data_04_amazon_longest_flow_path).

    python3 fig_04_amazon_pfafstetter_regions.py

Reads pfafstetter_pieces_south-america_basin1_160deg2.csv, the 30 arc-second piece preview and
the Amazon rows of piece_fine_south-america from data/ (data_04_amazon_pieces.py)
and writes fig_04_amazon_pfafstetter_regions.png into ../figures.
"""
import json
import os
import paths
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse
from osgeo import gdal
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY

CAP_SQUARE_DEGREES = 160
CAP_PIXELS = CAP_SQUARE_DEGREES * 3600 * 3600
VIEW = (-81.0, -48.0, -21.5, 6.5)                 # lon min, lon max, lat min, lat max

PALETTE = ["#9fbdd4", "#d8c79c", "#aecfa6", "#e0b7a4", "#bcb3d4",
           "#c9d894", "#e6c4d1", "#9dc9c2", "#d6cbae", "#b2c2da",
           "#cfb9c9", "#a7d3dc"]
OCEAN = "#f4f2ee"
OVER_CAP_HATCH_COLOUR = "#96301a"
BOUNDARY_LINE = (0.16, 0.18, 0.21, 1.0)
TEXT_COLOUR = "#2b3038"
ARROW_COLOUR = "#12161c"
RIVER_COLOUR = (0.05, 0.16, 0.45, 1.0)
OUTLET_FACE_COLOUR = "#d7301f"          # a basin outlet is always a red inverted triangle with a black edge
OUTLET_EDGE_COLOUR = "#12161c"
OUTLET_MARKER_SIZE = 6.0


def read_pieces_table(path):
    """piece index -> the columns used here; the index is the row number of the piece table, and the
    30 arc-second preview stores that index for the deepest level."""
    table = {}
    with open(path) as handle:
        header = handle.readline().split()
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            parts = line.split()
            table[int(parts[index["piece"]])] = dict(
                outlet_row=int(parts[index["outlet_row"]]),
                outlet_col=int(parts[index["outlet_col"]]),
                level=int(parts[index["level"]]),
                sub_of=int(parts[index["sub_of"]]),
                flows_into=int(parts[index["flows_into"]]),
                used=int(parts[index["used"]]),
                region=int(parts[index["region"]]),
                pixels=int(parts[index["labelled_grid_count"]]),
                window_pixels=int(parts[index["window_grid_count"]]),
                minlon=float(parts[index["minlon"]]), maxlon=float(parts[index["maxlon"]]),
                minlat=float(parts[index["minlat"]]), maxlat=float(parts[index["maxlat"]]),
            )
    return table


def pfafstetter_code_of(table, piece_index):
    """The piece index back to its Pfafstetter code: the digit is the position among the
    siblings of the same parent, in the order of the piece table (1, 2, 3, .. 9)."""
    parent = table[piece_index]["sub_of"]
    siblings = sorted(p for p, values in table.items() if values["sub_of"] == parent
                      and values["level"] == table[piece_index]["level"])
    digit = siblings.index(piece_index) + 1
    if parent == 0:
        return digit
    return pfafstetter_code_of(table, parent) * 10 + digit


def read_pieces(path):
    """piece_id -> (region_id, parent_piece_id, outlet_lon, outlet_lat, pixels) for the Amazon's used pieces."""
    pieces_used = {}
    with open(path) as handle:
        header = handle.readline().split()
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            parts = line.split()
            pieces_used[int(parts[index["piece_id"]])] = dict(
                region=int(parts[index["region_id"]]),
                parent=int(parts[index["parent_piece_id"]]),
                outlet_lon=float(parts[index["outlet_lon"]]),
                outlet_lat=float(parts[index["outlet_lat"]]),
                pixels=int(parts[index["piece_grid_count"]]),
            )
    return pieces_used


def greedy_colours(label_raster, label_list):
    """Give touching labels different colours, so the shapes read as shapes."""
    neighbours = collections.defaultdict(set)
    for one, other in ((label_raster[:, :-1], label_raster[:, 1:]),
                       (label_raster[:-1, :], label_raster[1:, :])):
        differ = (one != other) & (one > 0) & (other > 0)
        for a, b in zip(one[differ], other[differ]):
            neighbours[int(a)].add(int(b))
            neighbours[int(b)].add(int(a))
    colour_of = {}
    for label in sorted(label_list, key=lambda r: -len(neighbours[r])):
        taken = {colour_of[other] for other in neighbours[label] if other in colour_of}
        for colour_index in range(len(PALETTE)):
            if colour_index not in taken:
                colour_of[label] = colour_index
                break
        else:
            colour_of[label] = 0
    return colour_of


def draw_labelled_raster(axis, label_raster, extent, label_text_of, hatched_labels=(), offset_of=None):
    """Colour every label, draw the lines between labels from the raster, hatch some, and put
    each label's text at the middle of its own pixels.  Returns the text objects."""
    lon_min, lon_max, lat_min, lat_max = extent
    label_list = sorted(int(v) for v in np.unique(label_raster) if v > 0)
    colour_of = greedy_colours(label_raster, label_list)
    painted = np.zeros(label_raster.shape, dtype=np.int16)
    for label in label_list:
        painted[label_raster == label] = colour_of[label] + 1
    axis.imshow(painted, cmap=ListedColormap([OCEAN] + PALETTE), vmin=0, vmax=len(PALETTE),
                extent=(lon_min, lon_max, lat_min, lat_max), interpolation="nearest", origin="upper")
    boundary = np.zeros(label_raster.shape, dtype=bool)
    boundary[:, :-1] |= (label_raster[:, :-1] != label_raster[:, 1:]) & \
                        (label_raster[:, :-1] > 0) & (label_raster[:, 1:] > 0)
    boundary[:-1, :] |= (label_raster[:-1, :] != label_raster[1:, :]) & \
                        (label_raster[:-1, :] > 0) & (label_raster[1:, :] > 0)
    layer = np.zeros(label_raster.shape + (4,), dtype=float)
    layer[boundary] = BOUNDARY_LINE
    axis.imshow(layer, extent=(lon_min, lon_max, lat_min, lat_max), interpolation="nearest",
                origin="upper")
    rows, cols = label_raster.shape
    longitude = lon_min + (np.arange(cols) + 0.5) * (lon_max - lon_min) / cols
    latitude = lat_max - (np.arange(rows) + 0.5) * (lat_max - lat_min) / rows
    if hatched_labels:
        mask = np.isin(label_raster, list(hatched_labels)).astype(float)
        axis.contourf(longitude, latitude, mask, levels=[0.5, 1.5], colors="none", hatches=["/////"])
    centres = {}
    texts = []
    for label in label_list:
        row_index, col_index = np.nonzero(label_raster == label)
        centre = (float(longitude[col_index].mean()), float(latitude[row_index].mean()))
        centres[label] = centre
        shift = (offset_of or {}).get(label_text_of[label], (0.0, 0.0))
        texts.append(axis.text(centre[0] + shift[0], centre[1] + shift[1], label_text_of[label], fontsize=7.0,
                               ha="center", va="center", color="#12161c", zorder=6,
                               bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                                         edgecolor="none", alpha=0.72)))
    return centres, texts


def separate_the_labels(figure, axis, texts, rounds=300):
    """Push apart labels that came out on top of one another, in display units, keeping every
    box inside the frame (the same pass as in fig_03_regions_south_america.py)."""
    if not texts:
        return
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer=renderer)
    margin = 1.5
    gap = 1.0
    for _ in range(rounds):
        boxes = [text.get_window_extent(renderer=renderer) for text in texts]
        shift_x = [0.0] * len(texts)
        shift_y = [0.0] * len(texts)
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
                    sign = -1.0 if box_first.y0 + box_first.y1 <= box_second.y0 + box_second.y1 else 1.0
                    shift_y[first] += sign * step
                    shift_y[second] -= sign * step
                else:
                    step = 0.5 * (overlap_x + gap)
                    sign = -1.0 if box_first.x0 + box_first.x1 <= box_second.x0 + box_second.x1 else 1.0
                    shift_x[first] += sign * step
                    shift_x[second] -= sign * step
        if not something_moved:
            break
        for index, text in enumerate(texts):
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
            position = axis.transData.transform(text.get_position())
            moved = axis.transData.inverted().transform((position[0] + wanted_x, position[1] + wanted_y))
            text.set_position((float(moved[0]), float(moved[1])))


def draw_longest_flow_path(axis, path_lon, path_lat):
    """The Amazon's longest flow path as a dark blue line, one vertex per 1/400-degree cell the path enters."""
    axis.plot(path_lon, path_lat, color=RIVER_COLOUR, linewidth=0.9, solid_capstyle="round",
              solid_joinstyle="round", zorder=3)


def move_labels_off_the_river(figure, axis, texts, path_lon, path_lat):
    """A mainstem stretch has its label at its middle, which is on the river.  For every label
    whose box holds a vertex of the river, try the four directions in turn, each by one box
    height plus a gap, and keep the first position with no river vertex inside the box."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    river_in_display = axis.transData.transform(np.column_stack([path_lon, path_lat]))
    gap = 2.0
    for text in texts:
        original_position = text.get_position()
        box = text.get_window_extent(renderer=renderer)
        height = box.y1 - box.y0
        width = box.x1 - box.x0
        for step_count in range(1, 6):
            box = text.get_window_extent(renderer=renderer)
            inside = ((river_in_display[:, 0] >= box.x0) & (river_in_display[:, 0] <= box.x1) &
                      (river_in_display[:, 1] >= box.y0) & (river_in_display[:, 1] <= box.y1))
            if not inside.any():
                break
            moved = False
            for shift_x, shift_y in ((0.0, height + gap), (0.0, -(height + gap)),
                                     (-(width + gap), 0.0), (width + gap, 0.0)):
                position_in_display = axis.transData.transform(original_position)
                candidate = (position_in_display[0] + step_count * shift_x,
                             position_in_display[1] + step_count * shift_y)
                candidate_in_data = axis.transData.inverted().transform(candidate)
                text.set_position((float(candidate_in_data[0]), float(candidate_in_data[1])))
                figure.canvas.draw()
                box = text.get_window_extent(renderer=renderer)
                inside = ((river_in_display[:, 0] >= box.x0) & (river_in_display[:, 0] <= box.x1) &
                          (river_in_display[:, 1] >= box.y0) & (river_in_display[:, 1] <= box.y1))
                if not inside.any():
                    moved = True
                    break
            if moved:
                break


def draw_outlet(axis, outlet):
    """The basin outlet: a red inverted triangle with a black edge, the one symbol for an outlet in
    every figure of ours."""
    axis.plot([outlet["outlet_lon"]], [outlet["outlet_lat"]], marker="v", markersize=OUTLET_MARKER_SIZE,
              markerfacecolor=OUTLET_FACE_COLOUR, markeredgecolor=OUTLET_EDGE_COLOUR, markeredgewidth=0.6,
              linestyle="none", zorder=7)


def style_axis(axis, title_text, letter):
    axis.set_xlim(VIEW[0], VIEW[1])
    axis.set_ylim(VIEW[2], VIEW[3])
    axis.set_xticks([-80, -70, -60, -50])
    axis.set_yticks([-20, -10, 0])
    axis.set_xticklabels(["80°W", "70°W", "60°W", "50°W"])
    axis.set_yticklabels(["20°S", "10°S", "0°"])
    axis.tick_params(labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#8a8f98")
    style.panel_heading(axis, letter, title_text)


def main():
    table = read_pieces_table(os.path.join(DATA_DIRECTORY,
                                           "pfafstetter_pieces_south-america_basin1_160deg2.csv"))
    pieces_used = read_pieces(os.path.join(DATA_DIRECTORY, "piece_fine_south-america_basin1_160deg2.csv"))
    dataset = gdal.Open(os.path.join(DATA_DIRECTORY,
                                     "pfafstetter_pieces_south-america_basin1_30s_160deg2.tif"))
    deepest = dataset.ReadAsArray().astype(np.int32)
    transform = dataset.GetGeoTransform()
    extent = (transform[0], transform[0] + dataset.RasterXSize * transform[1],
              transform[3] + dataset.RasterYSize * transform[5], transform[3])

    # the deepest-level index of every cell up to its level-2 and level-1 piece
    top_index = max(table) + 1
    level2_of = np.zeros(top_index, dtype=np.int32)
    level1_of = np.zeros(top_index, dtype=np.int32)
    for piece_index, values in table.items():
        walk = piece_index
        while table[walk]["level"] > 2:
            walk = table[walk]["sub_of"]
        level2_of[piece_index] = walk if table[walk]["level"] == 2 else 0
        while table[walk]["level"] > 1:
            walk = table[walk]["sub_of"]
        level1_of[piece_index] = walk
    level1_raster = np.where(deepest > 0, level1_of[deepest], 0)
    with open(os.path.join(DATA_DIRECTORY, "lfp_amazon_400th.geojson")) as handle:
        path_feature = json.load(handle)["features"][0]
    path_lon = [point[0] for point in path_feature["geometry"]["coordinates"]]
    path_lat = [point[1] for point in path_feature["geometry"]["coordinates"]]
    print("vertices on the Amazon's longest flow path at 1/400 degree:", len(path_lon))
    level2_raster = np.where(deepest > 0, level2_of[deepest], 0)

    # (b): the used pieces; (c): the region of every used piece
    region_of_used = {}
    for piece_index, values in table.items():
        if values["used"]:
            region_of_used[piece_index] = values["region"]
    used_raster = np.zeros_like(deepest)
    for piece_index in region_of_used:
        if table[piece_index]["level"] == 1:
            used_raster[level1_raster == piece_index] = piece_index
        else:
            used_raster[level2_raster == piece_index] = piece_index
    region_raster = np.zeros_like(deepest)
    for piece_index, region_id in region_of_used.items():
        region_raster[used_raster == piece_index] = region_id
    code_of = {piece_index: pfafstetter_code_of(table, piece_index) for piece_index in table
               if table[piece_index]["level"] <= 2}
    over_the_cap_level1 = [p for p, v in table.items() if v["level"] == 1 and v["window_pixels"] > CAP_PIXELS]
    region_ids = sorted(set(region_of_used.values()))
    print("level-1 pieces over the cap:", [code_of[p] for p in over_the_cap_level1],
          "used pieces:", len(region_of_used), "regions:", len(region_ids))
    for p in over_the_cap_level1:
        print("   piece %d rectangle %d pixels = %.2f x 2^31" % (code_of[p], table[p]["window_pixels"],
                                                                table[p]["window_pixels"] / 2.0 ** 31))

    style.apply_style()
    figure = plt.figure(figsize=(7.48, 3.7))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.30], left=0.058, right=0.988,
                               top=0.935, bottom=0.03, wspace=0.06, hspace=0.62)
    axes = [figure.add_subplot(grid[0, k]) for k in range(3)]
    graph_axis = figure.add_subplot(grid[1, :])
    stats_offset = -0.105                            # the footers a little lower
    outlet = pieces_used[min(pieces_used)]                      # the first unit row is piece 1, at the mouth

    # (a) nine pieces, the two over the cap hatched with their rectangles
    label_text = {p: str(code_of[p]) for p in table if table[p]["level"] == 1}
    _, texts_a = draw_labelled_raster(axes[0], level1_raster, extent, label_text,
                                      hatched_labels=over_the_cap_level1)
    for p in over_the_cap_level1:
        v = table[p]
        axes[0].add_patch(Rectangle((v["minlon"], v["minlat"]), v["maxlon"] - v["minlon"],
                                    v["maxlat"] - v["minlat"], fill=False, linestyle=(0, (3, 2)),
                                    linewidth=0.7, edgecolor=OVER_CAP_HATCH_COLOUR, zorder=5))
    used_piece_of_piece = {}
    for piece_id, values in pieces_used.items():
        for piece_index in region_of_used:
            if table[piece_index]["pixels"] == values["pixels"]:
                used_piece_of_piece[piece_index] = piece_id
    for p in table:
        if table[p]["level"] == 1 and code_of[p] % 2 == 0:
            first_sub = p if p in used_piece_of_piece else next(q for q in table if table[q]["sub_of"] == p)
            mouth = pieces_used[used_piece_of_piece[first_sub]]
            axes[0].plot([mouth["outlet_lon"]], [mouth["outlet_lat"]], marker="o", markersize=2.6,
                         markerfacecolor="#12161c", markeredgecolor="white", markeredgewidth=0.4,
                         linestyle="none", zorder=7)
    draw_longest_flow_path(axes[0], path_lon, path_lat)
    draw_outlet(axes[0], outlet)
    style_axis(axes[0], "Nine Pfafstetter pieces", "(a)")
    style.panel_foot(axes[0], "pieces 6 and 9 exceed the capacity, 160 deg\u00b2 (hatched)", y=stats_offset)

    # (b) the used pieces
    label_text = {p: str(code_of[p]) for p in region_of_used}
    _, texts_b = draw_labelled_raster(axes[1], used_raster, extent, label_text)
    draw_longest_flow_path(axes[1], path_lon, path_lat)
    draw_outlet(axes[1], outlet)
    style_axis(axes[1], "Pieces over the capacity cut again", "(b)")
    style.panel_foot(axes[1], "%d pieces, all within the capacity" % len(region_of_used), y=stats_offset)

    # (c) regions, numbered
    label_text = {r: "%02d" % (r % 100) for r in region_ids}
    centres, texts_c = draw_labelled_raster(axes[2], region_raster, extent, label_text)
    draw_longest_flow_path(axes[2], path_lon, path_lat)
    links = set()
    for piece_id, values in pieces_used.items():
        if values["parent"] in pieces_used and pieces_used[values["parent"]]["region"] != values["region"]:
            links.add((values["region"], pieces_used[values["parent"]]["region"]))
    draw_outlet(axes[2], outlet)
    style_axis(axes[2], "Merged in code order into regions", "(c)")
    style.panel_foot(axes[2], "%d regions, numbered from upstream" % len(region_ids), y=stats_offset)
    print("region links:", sorted(links))
    move_labels_off_the_river(figure, axes[0], texts_a, path_lon, path_lat)
    separate_the_labels(figure, axes[0], texts_a)
    move_labels_off_the_river(figure, axes[1], texts_b, path_lon, path_lat)
    separate_the_labels(figure, axes[1], texts_b)
    move_labels_off_the_river(figure, axes[2], texts_c, path_lon, path_lat)
    separate_the_labels(figure, axes[2], texts_c)

    # (d) the region graph as a row: every arrow points to a larger number
    x_of = {r: float(k + 1) for k, r in enumerate(region_ids)}
    graph_axis.set_xlim(0.3, len(region_ids) + 0.7)
    graph_axis.set_ylim(-1.05, 1.55)
    figure.canvas.draw()
    box = graph_axis.get_window_extent()
    NODE_ASPECT = (2.8 / box.height) / ((len(region_ids) + 0.4) / box.width)   # a circle on the page
    graph_axis.set_xticks([])
    graph_axis.set_yticks([])
    graph_axis.axis("off")
    for r in region_ids:
        graph_axis.add_patch(Ellipse((x_of[r], 0.0), 0.62, 0.62 * NODE_ASPECT, facecolor="white",
                                     edgecolor="#12161c", linewidth=0.6, zorder=5))
        graph_axis.text(x_of[r], 0.0, "%02d" % (r % 100), fontsize=style.SIZE_LABEL, ha="center", va="center",
                        color="#12161c", zorder=6)
    for upstream, downstream in sorted(links):
        span = x_of[downstream] - x_of[upstream]
        graph_axis.add_patch(FancyArrowPatch((x_of[upstream], 0.0), (x_of[downstream], 0.0),
                                             arrowstyle="-|>", mutation_scale=6, linewidth=0.7,
                                             color=ARROW_COLOUR, shrinkA=15, shrinkB=15,
                                             connectionstyle="arc3,rad=%.3f" % (-0.55 / max(span, 1.0) ** 0.5
                                                                                 if span > 1.0 else -0.9),
                                             zorder=4))
    style.panel_heading(graph_axis, "(d)", "Region graph: every arrow points downstream, to a larger number")
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_04_amazon_pfafstetter_regions")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
