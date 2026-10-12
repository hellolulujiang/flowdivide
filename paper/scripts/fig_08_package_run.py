"""Figure 8: the Python package at work.
    (a) one run as a reader types it, from the page the package prints at the start of a run;
    (b) the Mississippi (North America, HydroSHEDS v2.0 at 1 arc-second) cut into regions of 160 deg2;
    (c) the hours of the three stages on the three grids.
Reads from data/ the files written by data_08_package_run.py, the grid sizes of (c) included;
writes fig_08_package_run.png into ../figures.

    python3 fig_08_package_run.py
"""
import argparse
import os
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import offset_copy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
import style

# the package as released: its map helpers draw (b)
sys.path.insert(0, paths.PACKAGE_DIRECTORY)
import fd2_views as fd2                     # the package's own map helpers draw (b)
from fd1_partition import read_table

FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
DATA = paths.DATA_DIRECTORY
DATASETS = [("south-america", "South America\nHydroSHEDS v2.0, 30 m"), ("north-america", "North America\nHydroSHEDS v2.0, 30 m"), ("merit-global", "The globe\nMERIT Hydro, 90 m")]


def grid_size_label(dataset):
    """'259,200 × 219,600' (rows × columns) of the dataset's flow-direction grid (data_08_package_run)"""
    sizes = pd.read_csv(os.path.join(DATA, "grid_sizes.csv")).set_index("dataset")
    return "{:,} × {:,}".format(int(sizes.loc[dataset, "rows"]), int(sizes.loc[dataset, "columns"]))
BASIN_NAME = {"south-america": "Amazon", "north-america": "Mississippi"}
STAGES = [("FD1", "FD1  the partition"), ("FD2", "FD2  the views"), ("FD3", "FD3  the six attributes")]
STAGE_COLOURS = ["#e6e6e4", "#b7b7b3", "#7f7f7c"]      # three greys: in (c) the colour only tells the stages apart
CAPACITY_BLOCKS = 160                       # the capacity of (b), 2^31 = 160 square degrees on the 1 arc-second grid

# the strip: monospace across the page, the notes from column 44, 108 characters wide at most (7.3 inch)
STRIP_FONT = style.SIZE_LABEL
STRIP_NOTE_COLUMN = 44
STRIP_WIDTH_CHARACTERS = 108
HEADING_COLOUR = "#8c1d0b"
NAME_COLOUR = "#0b4f9c"
DIM = "#6b7078"
BOX_FACE = "#fbfaf7"
BOX_EDGE = "#8a8f98"


# ---------------------------------------------------------------------------------------------------------
# (a) one run as a reader types it
# ---------------------------------------------------------------------------------------------------------

def read_page(dataset):
    """the page the package printed when the run started"""
    path = os.path.join(DATA, "run_summary_%s.txt" % dataset)
    with open(path, encoding="utf8") as handle:
        return handle.read().split("\n")


def value_of(lines, label):
    """the text after a label of the page ('  flow directions      ...'), and its continuation lines"""
    for k, line in enumerate(lines):
        match = re.match(r"^  (%s) {2,}(.*)$" % re.escape(label), line)
        if match:
            rest = [match.group(2)]
            for follow in lines[k + 1:]:
                if re.match(r"^ {23}\S", follow):
                    rest.append(follow.strip())
                else:
                    break
            return rest
    return []


def strip_lines(dataset):
    """The run as a READER would type it on a grid of their own: the options are the package's own, the values
    that belong to our run are placeholders (<name>, <flow directions>, <polygons>, <directory>), and the grey
    note to the right says what this run used or what the option accepts."""
    lines = read_page(dataset)
    out = []
    flow = value_of(lines, "flow directions")
    # every value shown comes off the page; a missing one stops the script
    match = re.search(r"convention, (.*?), (\d+) rows x (\d+) columns", flow[1]) if len(flow) > 1 else None
    if match is None:
        raise SystemExit("the run page of %s does not say the grid: %r" % (dataset, flow[1:2]))
    fine_text = match.group(1)                                    # "1 arc-second (30 m)", or "10 m pixels"
    rows_columns = (match.group(2), match.group(3))
    conventions = [name for name in ("hydrosheds", "merit", "esri", "taudem", "grass", "degrees", "custom") if "%s convention" % name in flow[1]]
    if len(conventions) != 1:
        raise SystemExit("the run page of %s does not say one direction convention: %r" % (dataset, flow[1]))
    convention = conventions[0]
    # The run as one command: the built-in grid south-america, which fixes the convention, the groups, the
    # capacities with their depths and the channel threshold (the land limit is set too, and left out here); the run read the upstream pixel
    # count (--acc) and the grid's own area.  Every value shown still comes off the run page.
    groups_given = value_of(lines, "basin groups")
    if not (groups_given and groups_given[0].startswith("predefined")):
        raise SystemExit("the run page of %s does not name predefined units" % dataset)
    layers = [os.path.basename(text.replace("predefined units: ", "")) for text in groups_given]
    names, degrees, depths = [], [], []
    for text in value_of(lines, "capacities"):
        match = re.match(r"(\S+) = (\d+) blocks( \((\d+) square degrees\))?.*?the cut (\S+)", text)
        if match:
            name, blocks, _, square, levels = match.groups()
            names.append(name); degrees.append(square or blocks); depths.append(levels if levels.isdigit() else "auto")
    if not names:
        raise SystemExit("the run page of %s does not say the capacities" % dataset)
    threshold = value_of(lines, "channel threshold")
    if not threshold:
        raise SystemExit("the run page of %s does not say the channel threshold" % dataset)
    km2 = threshold[0].split(";")[0].replace(" km2", "").strip()
    command = " ".join(value_of(lines, "command"))
    if " run %s " % dataset not in command or "--acc" not in command:
        raise SystemExit("the run page of %s does not show a built-in run that reads --acc: %r" % (dataset, command[:120]))
    attributes = value_of(lines, "attributes")
    if attributes and " on " in attributes[0]:
        codes = attributes[0].partition(" on ")[0].replace(",", "").split()
    else:
        import flowdivide
        codes = list(flowdivide.ALL_ATTRIBUTES)
    if len(codes) != 6:
        raise SystemExit("the attributes of %s are not the six: %s" % (dataset, codes))
    out.append(("command", "$ flowdivide run %s \\" % dataset, "the built-in grid: {}, {:,} × {:,} pixels".format(fine_text, int(rows_columns[0]), int(rows_columns[1]))))
    out.append(("option", "    --acc <upstream pixel count> \\", "read, not computed; the area is the grid's own"))
    out.append(("option", "    --steps fd1,fd2,fd3 \\", "the partition, the views, the attributes"))
    out.append(("option", "    --attributes %s \\" % ",".join(codes), "the six, on the largest capacity, %s" % names[0]))
    out.append(("option", "    --keep bsn,rgn,pieces \\", "the partition kept; the channel mask is not"))
    out.append(("option", "    --out-root <directory>", "the products go under <directory>/%s/" % dataset))
    out.append(("option", "  # set by the built-in grid:", ""))
    out.append(("option", "  #   convention %s; groups" % convention, ("HydroBASINS Level-03 units, of whole basins" if all(re.match(r"hybas_.*lev03", layer) for layer in layers)
                                                                          else "units of whole basins: %s" % layers[0])))
    out.append(("option", "  #   capacities %s" % ",".join("%s:%s" % (n, d) for n, d in zip(names, depths)), "%s deg²; the cut %s deep" % (", ".join(degrees), ", ".join(depths))))
    out.append(("option", "  #   channel threshold", "%s km²" % km2))
    widest_text = max(len(text) for kind, text, note in out)
    widest_note = max(len(note) for kind, text, note in out)
    if widest_text > STRIP_NOTE_COLUMN - 2 or STRIP_NOTE_COLUMN + widest_note > STRIP_WIDTH_CHARACTERS:
        raise SystemExit("a strip line is too wide: the command %d (column %d), the note %d, %d allowed" % (widest_text, STRIP_NOTE_COLUMN, widest_note, STRIP_WIDTH_CHARACTERS))
    return out


def draw_strip(axis, rows):
    """the command in a box: the prompt in grey, the command bold, the options in ink, the notes in grey at one column"""
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.02", facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=0.6,
                                  transform=axis.transAxes, clip_on=False))
    figure = axis.figure
    height_inches = axis.get_position().height * figure.get_figheight()
    line = min(STRIP_FONT * 1.7 / 72.0, (height_inches - 0.14) / (len(rows) + 0.5)) / height_inches      # the lines spread over the box, up to 1.7 of the size
    y = 1.0 - 0.08 / height_inches - 0.85 * line
    x0 = 0.02
    # a column is placed by the advance of one monospace character, so that no text is padded with spaces
    probe = figure.text(0.0, 0.0, "M" * 20, fontsize=STRIP_FONT, family="monospace")
    advance = probe.get_window_extent(renderer=figure.canvas.get_renderer()).width * 72.0 / figure.dpi / 20.0
    probe.remove()
    at_column = lambda column: offset_copy(axis.transAxes, figure, x=column * advance, y=0.0, units="points")
    for kind, text, note in rows:
        if kind == "command":
            axis.text(x0, y, text[:1], fontsize=STRIP_FONT, family="monospace", ha="left", va="top", color=DIM, transform=axis.transAxes)
            axis.text(x0, y, text[1:], fontsize=STRIP_FONT, family="monospace", ha="left", va="top", color=style.INK, fontweight="bold", transform=at_column(1))
        else:
            axis.text(x0, y, text, fontsize=STRIP_FONT, family="monospace", ha="left", va="top", color=style.INK, transform=axis.transAxes)
        if note:
            axis.text(x0, y, note, fontsize=STRIP_FONT, family="monospace", ha="left", va="top", color=DIM, transform=at_column(STRIP_NOTE_COLUMN))
        y -= line


def check_strip_fits(axis):
    """the character counts of strip_lines only estimate the width; measure the drawn text instead (bold text and
    font fallback are wider than the count says)"""
    figure = axis.figure
    figure.canvas.draw()
    box = axis.get_window_extent()
    page = figure.get_window_extent()
    for text in axis.texts:
        extent = text.get_window_extent()
        if extent.x1 > box.x1 or extent.x0 < box.x0 or extent.y0 < box.y0 or extent.y1 > box.y1:
            raise SystemExit("a strip line does not fit the box: %r is %.1f points outside it"
                             % (text.get_text().strip()[:40], max(extent.x1 - box.x1, box.x0 - extent.x0, box.y0 - extent.y0, extent.y1 - box.y1)))
        if extent.x1 > page.x1 or extent.x0 < page.x0:
            raise SystemExit("a strip line runs off the page: %r" % text.get_text().strip()[:40])


# ---------------------------------------------------------------------------------------------------------
# (b) the largest basin of a continent cut into regions, as the package's Figure 4c draws it
# ---------------------------------------------------------------------------------------------------------

def regions_of_basin(dataset, basin_id=1):
    """the region raster of the basin on the 30 arc-second preview, its extent, the main stem, the outlet and the
    pieces used and the pieces of level 1 cut once more, from the package's tables of the cut at 160 deg2"""
    table = read_table(os.path.join(DATA, "pfafstetter_pieces_%s_basin%d.csv" % (dataset, basin_id)))
    stem = read_table(os.path.join(DATA, "mississippi_main_stem.csv"))
    outlet = pd.read_csv(os.path.join(DATA, "mississippi_outlet.csv"))
    deepest, extent = fd2._read_view(os.path.join(DATA, "pfafstetter_pieces_%s_basin%d_30s.tif" % (dataset, basin_id)))
    by_id = {int(row.piece): row for row in table.itertuples(index=False)}
    used_ids = set(int(c) for c in table.loc[table["used"] == 1, "piece"])
    cut_ids = set(int(c) for c in table["sub_of"])
    region_of = np.zeros(int(table["piece"].max()) + 1, np.int32)
    for piece_id in table["piece"].astype(int):
        if piece_id in cut_ids:
            continue
        walk = piece_id
        while walk not in used_ids:
            walk = int(by_id[walk].sub_of)
        region_of[piece_id] = int(by_id[walk].region) % 100
    region_raster = np.where(deepest > 0, region_of[deepest], 0)
    # a piece of level 1 that is not used was over the capacity and is replaced by its nine pieces of level 2
    if table.loc[table["used"] == 1, "level"].max() > 2:
        raise SystemExit("a piece of level 2 was cut again; the note under (b) names level-1 pieces only")
    cut_again = sorted(int(c) for c in table.loc[(table["level"] == 1) & (table["used"] == 0), "piece"])
    return (region_raster, extent, stem, (float(outlet["outlet_lon"].iloc[0]), float(outlet["outlet_lat"].iloc[0])),
            len(used_ids), cut_again)


def keep_labels_off_the_stem(axis, texts, region_raster, extent, stem):
    """a region number drawn on the main stem moves to the pixel of its region farthest from the stem and from the
    region's edge"""
    rows_count, columns_count = region_raster.shape
    cell_x = (extent[1] - extent[0]) / columns_count
    cell_y = (extent[3] - extent[2]) / rows_count
    stem_rows = np.clip(((extent[3] - stem["lat"].to_numpy()) / cell_y).astype(int), 0, rows_count - 1)
    stem_columns = np.clip(((stem["lon"].to_numpy() - extent[0]) / cell_x).astype(int), 0, columns_count - 1)
    off_the_stem = np.ones(region_raster.shape, dtype=bool)
    off_the_stem[stem_rows, stem_columns] = False
    distance_to_stem = ndimage.distance_transform_edt(off_the_stem)
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    stem_points = axis.transData.transform(np.column_stack([stem["lon"].to_numpy(), stem["lat"].to_numpy()]))
    for text in texts:
        if text.get_window_extent(renderer=renderer).count_contains(stem_points) == 0:
            continue
        inside = region_raster == int(text.get_text())
        room = np.minimum(ndimage.distance_transform_edt(inside), distance_to_stem)
        room[~inside] = 0.0
        row, column = np.unravel_index(int(np.argmax(room)), room.shape)
        text.set_position((extent[0] + (column + 0.5) * cell_x, extent[3] - (row + 0.5) * cell_y))


def draw_regions(axis, dataset):
    region_raster, extent, stem, outlet, used, cut_again = regions_of_basin(dataset)
    numbers = sorted(int(v) for v in np.unique(region_raster) if v > 0)
    texts = fd2._draw_labelled_raster(axis, region_raster, extent, {number: "%02d" % number for number in numbers})
    if stem is not None:
        axis.plot(stem["lon"], stem["lat"], color=fd2.RIVER_COLOUR, linewidth=0.9, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    axis.plot([outlet[0]], [outlet[1]], zorder=7, **style.OUTLET_MARKER)
    view = (extent[0], extent[1], extent[2], extent[3])
    fd2._map_axis(axis, view, fd2._tick_values(view[0], view[1]), fd2._tick_values(view[2], view[3]))
    axis.set_aspect(1.0 / np.cos(np.radians(0.5 * (view[2] + view[3]))), adjustable="box", anchor="C")
    if stem is not None:
        keep_labels_off_the_stem(axis, texts, region_raster, extent, stem)
    fd2._separate_labels(axis.figure, axis, texts)
    return len(numbers), used, cut_again, stem is not None


# ---------------------------------------------------------------------------------------------------------
# (c) the hours per stage
# ---------------------------------------------------------------------------------------------------------

def hours_by_stage():
    """{dataset: {stage: hours}} from package_stage_hours.csv; every dataset must have the three stages"""
    table = pd.read_csv(os.path.join(DATA, "package_stage_hours.csv"))
    hours = {}
    for dataset, _ in DATASETS:
        rows = table[table["dataset"] == dataset]
        if sorted(rows["stage"]) != [stage for stage, _ in STAGES] or not np.isfinite(rows["hours"]).all():
            raise SystemExit("%s: the stages in package_stage_hours.csv are %s" % (dataset, list(rows["stage"])))
        hours[dataset] = dict(zip(rows["stage"], rows["hours"]))
    return hours


def draw_hours(axis, hours):
    width = 0.26
    for s, (stage, label) in enumerate(STAGES):
        for d, (dataset, _) in enumerate(DATASETS):
            value = hours.get(dataset, {}).get(stage, 0.0)
            x = d + (s - 1) * width
            axis.bar(x, value, width * 0.92, color=STAGE_COLOURS[s], edgecolor="#4d4d4d", linewidth=0.4, label=label if d == 0 else None)
            if value > 0:
                axis.text(x, value, "%.1f" % value if value < 10 else "%.0f" % value, ha="center", va="bottom", fontsize=style.SIZE_LABEL, color=style.TEXT_COLOUR)
    axis.set_xticks(range(len(DATASETS)))
    # four short lines: the continent, the dataset, the resolution, the grid as rows x columns
    axis.set_xticklabels(["%s\n%s" % (label.replace(", ", "\n"), grid_size_label(dataset)) for dataset, label in DATASETS], fontsize=style.SIZE_TICK)
    axis.set_ylabel("hours", fontsize=style.SIZE_TICK)
    axis.tick_params(labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    axis.legend(fontsize=style.SIZE_LABEL, frameon=False, loc="upper right", handlelength=1.2, handletextpad=0.5, borderaxespad=0.2, ncol=1)
    axis.set_xlim(-0.6, len(DATASETS) - 0.4)
    top = max([v for by in hours.values() for v in by.values()] + [1.0])
    axis.set_ylim(0, top * 1.45)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------------------------------------

def main(strip_dataset, map_dataset):
    style.apply_style()
    fd2._apply_style()
    width, height = 7.48, 5.10
    figure = plt.figure(figsize=(width, height))

    def axes_inches(x0, y0, w, h):
        return figure.add_axes([x0 / width, y0 / height, w / width, h / height])

    # the command across the top; the map and the hours below
    rows = strip_lines(strip_dataset)
    strip = axes_inches(0.08, 3.35, 7.32, 1.38)
    draw_strip(strip, rows)
    check_strip_fits(strip)
    style.panel_heading(strip, "(a)", "The run, as one command", "notes in grey: %s" % dict(DATASETS)[strip_dataset].replace("\n", ", "))

    map_axis = axes_inches(0.30, 0.70, 3.40, 2.12)
    regions, used, cut_again, with_stem = draw_regions(map_axis, map_dataset)
    style.panel_heading(map_axis, "(b)", "The %s cut into regions of 160 deg²" % BASIN_NAME.get(map_dataset, map_dataset),
                        dict(DATASETS)[map_dataset].replace("\n", ", "))
    style.panel_foot(map_axis, "%d Pfafstetter pieces (piece %s cut once more),\npacked into %d regions (numbered);\n"
                     "line: the main stem; triangle: the outlet" % (used, ", ".join(map(str, cut_again)), regions), y=-0.13)

    hours = hours_by_stage()
    bars = axes_inches(4.20, 0.92, 3.20, 1.90)
    draw_hours(bars, hours)
    style.panel_heading(bars, "(c)", "Hours per stage", "one laptop, each stage its own resumed command")

    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
    out = style.figure_path(FIGURE_DIRECTORY, "fig_08_package_run")
    figure.savefig(out, dpi=400, facecolor="white")
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strip", default="south-america", help="the dataset whose run page makes the strip (a)")
    parser.add_argument("--map", default="north-america", help="the dataset whose largest basin is mapped in (b)")
    arguments = parser.parse_args()
    out = main(arguments.strip, arguments.map)
    if sys.platform == "darwin":
        subprocess.call(["open", out])
