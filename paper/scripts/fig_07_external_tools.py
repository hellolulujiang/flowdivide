"""Figure 7: FlowDivide against the tools a reader already has, on one basin of South America.

Table 3's rows are how a tool holds the grid; this figure measures what that costs:
    everything resident, the whole grid in one process           pyflwdir, WhiteboxTools, Cho's add-ons
    everything resident, regular tiles one per process           TauDEM (here one process)
    the whole grid on disk, read in pages                        GRASS r.stream.*
    one part at a time, regular tiles                            FlowDivide on the 10 x 10 degree tiles
    regions cut at tributary outlets within a capacity           FlowDivide on the published partition
Table 3's remaining row, regions drawn beforehand to hold whole basins, has no run: those are published
products, not programs to run on a flow-direction grid.

(a) The wall time of ONE PROCESS that turns the flow-direction GeoTIFF into ONE variable and writes it,
    on basin 5, the São Francisco (a rectangle of 2.01e9 pixels of HydroSHEDS v2 at 1 arc-second,
    690 million of them land).  One variable per attribute
    class: the upstream area (accumulation), the distance to the outlet (labelling), the upstream flow
    length (maximum), the Strahler order (stream order).  Five repetitions, the fastest and the slowest left
    out and the middle three averaged, one run at a time, one thread everywhere: OMP_NUM_THREADS and its kin
    set to 1, TauDEM through mpiexec -n 1, Cho's modules with nprocs=1, WhiteboxTools with --max_procs=1, so
    that what is compared is the work and not the number of threads.
    A bar therefore carries everything that tool must do before its kernel can run -- parsing the grid
    and building its traversal order, importing a map into GRASS, reading the tables of the partition --
    and the building of any file it is handed that another tool works out inside its own run: the
    cell-area raster TauDEM's aread8 and Cho's accumulation need as a weight, the outlet raster TauDEM's
    d8hdisttostrm needs, the flow directions in TauDEM's own codes, and the channel network cut into
    links at its junctions that GRASS r.stream.order needs instead of a plain channel mask.  What the
    data are distributed as -- the flow directions, the channel mask, the upstream area, the partition
    and its tables -- is charged to nobody.
    What cannot be made equal, and is said here instead: TauDEM's gridnet writes the longest upslope
    length, the total length and the Strahler order in one command, so those two bars each carry the
    work of three variables, which is how the tool is built; pyflwdir is its development version 0.5.13.dev0,
    whose distance computes the step lengths once per row, where the released 0.5.11 re-evaluates the
    degree-to-metre row length at every pixel; GRASS
    r.stream.* is given a 4000 MB segment cache, which is what makes it page; the tools do not end in
    the same place (GRASS a map in its mapset, the others a GeoTIFF; pyflwdir, a library that returns an
    array, writes it as one inside the timer); the partition line is the only one that reads continental rasters and writes on the
    continental grid; its accumulation is the first stage's kernel, because the third stage has none;
    and each tool uses its own cell-area formula, which agreed to within a thousandth on basin 4.
(b) The peak memory of those runs, in GiB (the resident set read by ps, KiB / 2^20), against the 64 GiB of the machine: the largest peak footprint of the
    runs drawn in (a), read the same way for every series.  GRASS r.stream.* stays low because it pages
    its own segment files to disk, and how much disk that took is written beside its bar.
(c) What each tool would need on the five largest basins of South America, computed and not run: its
    bytes per pixel, from the run on basin 5, times each basin's rectangle.  An extrapolation, so it is
    labelled as one; the two FlowDivide lines are flat because the part sets the peak, not the basin.
Reads from data/ the timing table and the needs of panel (c) (copied by data_07_external_tools from the runs of the
experiment, run_basins.py and needs_model.py) and the rectangles of the five largest basins; nothing is typed in by
hand, and the figure refuses to draw when the needs were not computed from these timings.  Writes
fig_07_external_tools.png into ../figures.

    python3 fig_07_external_tools.py
"""
import hashlib
import os
import paths
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style

FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
DATA = paths.DATA_DIRECTORY
TIMINGS = os.environ.get("FD_TIMINGS") or os.path.join(DATA, "timings_multi.csv")
NEEDS = os.environ.get("FD_NEEDS") or os.path.join(DATA, "needs_basins.csv")
BODY_BASIN = int(os.environ.get("FIG7_BASIN", "5"))   # the São Francisco: the smallest of the four basins the
                                    # capacity cuts, so the one every tool finishes
MEMORY_GB = 64.0                    # the machine
# the five largest basins of South America, by basin number; 1, 2, 4 and 5 are the four the partition cuts,
# because their rectangle taken out to whole degrees passes the capacity
NAMES = [(5, "São Francisco"), (4, "Tocantins"), (3, "Orinoco"), (2, "Paraná"), (1, "Amazon")]


_rectangles = {}


def rectangle_pixels(basin):
    """the pixels of a basin's bounding rectangle (data_07_external_tools)"""
    if not _rectangles:
        table = pd.read_csv(os.path.join(DATA, "basin_rectangles_south-america.csv"))
        _rectangles.update(zip(table["basin_id"].astype(int), table["rectangle_pixels"].astype(int)))
    return _rectangles.get(basin)


BASINS = [(basin, name, rectangle_pixels(basin)) for basin, name in NAMES if rectangle_pixels(basin)]
if not BASINS:
    raise SystemExit("no basin rectangles in %s" % DATA)
PIXELS_OF = {basin: pixels for basin, _, pixels in BASINS}
NAME_OF = {basin: name for basin, name, _ in BASINS}
# the two FlowDivide lines read one part at a time, so their peak is set by the part, not by the basin
BOUNDED_BY = {"fd_tiles": "bounded by the tile", "fd_pub": "bounded by the capacity"}

# the four groups: key, the row of Table 3 as the legend reads it, the colour.  Four colours, and they
# stay apart in greyscale as well (0.78, 0.62, 0.30, 0.44 of white)
# The groups are what bounds the memory, which is what this figure measures: everything resident (in one
# process, or in stripes across processes -- TauDEM's stripes are all resident at once, so its memory grows
# with the grid like the others'), the grid paged from disk, or one part at a time.  Table 3 groups by where
# the grid is cut, so TauDEM and our tiles share a row there and not a colour here.
WAYS = [("memory",    "Everything resident: the whole grid in one process", "#d2d5d8", ""),
        ("resident_tiles", "Everything resident: regular tiles, one per process, exchanging at their edges", "#d2d5d8", "////"),
        ("disk",      "The whole grid on disk, read in pages", "#e0a03a", ""),
        ("tiles",     "One part at a time: regular tiles", "#2f6f9f", ""),
        ("partition", "One part at a time: regions cut at tributary outlets within a capacity", "#c0392b", "")]
COLOUR_OF_WAY = {key: colour for key, _, colour, _ in WAYS}
HATCH_OF_WAY = {key: hatch for key, _, _, hatch in WAYS}

# key, the name as the reader sees it, the short name for (b) and (c), the group of Table 3
SERIES = [
    ("pyflwdir", "pyflwdir", "pyflwdir", "memory"),
    ("whitebox", "WhiteboxTools", "WhiteboxTools", "memory"),
    ("cho", "GRASS add-ons by Cho", "Cho's add-ons", "memory"),
    ("taudem", "TauDEM", "TauDEM", "resident_tiles"),
    ("grass", "GRASS r.stream.*", "GRASS r.stream.*", "disk"),
    ("fd_tiles", "FlowDivide, the 10° × 10° tiles", "FlowDivide, tiles", "tiles"),
    ("fd_pub", "FlowDivide, the published partition", "FlowDivide, partition", "partition"),
]
WAY_OF = {key: way for key, _, _, way in SERIES}
KERNELS = [("upa", "Upstream area", "the accumulation class"),
           ("ldn", "Distance to the outlet", "the labelling class"),
           ("lup", "Upstream flow length", "the maximum class"),
           ("ord", "Strahler order", "the stream-order class")]
# the step of each series for each kernel ("exact" or "prefix"); a pair that is not here is one the tool
# does not provide from flow directions
# A bar is one process that turned the flow directions into one variable, timed from outside: the
# orchestration writes one such row per (series, variable, repetition), with the step named
# "one process: <series> <variable>".  A pair that is not here is one the series cannot give
# from flow directions in metres or square metres; WHY_NOT says why.
KERNEL_STEP = {(series, kernel): ("exact", "one process: %s %s" % (series, kernel))
               for series, kernel in (("pyflwdir", "upa"), ("pyflwdir", "ldn"), ("pyflwdir", "ord"),
                                      ("whitebox", "ord"),
                                      ("taudem", "upa"), ("taudem", "ldn"), ("taudem", "lup"), ("taudem", "ord"),
                                      ("cho", "upa"),
                                      ("grass", "ldn"), ("grass", "ord"),
                                      ("fd_tiles", "upa"), ("fd_tiles", "ldn"), ("fd_tiles", "lup"), ("fd_tiles", "ord"),
                                      ("fd_pub", "upa"), ("fd_pub", "ldn"), ("fd_pub", "lup"), ("fd_pub", "ord"))}
# the series key the orchestration writes for our two lines is shorter than the figure's
STEP_SERIES = {"fd_tiles": "tiles", "fd_pub": "published"}
KERNEL_STEP = {(series, kernel): ("exact", "one process: %s %s" % (STEP_SERIES.get(series, series), kernel))
               for (series, kernel) in KERNEL_STEP}
CHO_PREFIXES = ("r.flowaccumulation", "r.accumulate", "r.upflowlength", "r.lfp")
# One variable, computed on its own, from the flow directions on disk: that is what a bar is.  So a bar carries whatever the tool must do first before its kernel
# can run -- parsing the grid and building its traversal order, importing a map into GRASS, reading the
# tables of the partition.  TauDEM and WhiteboxTools read the GeoTIFF themselves and have nothing to add.
# A step named here is added to that kernel's bar, once; the steps are those of the timings table.
# nothing: every bar is one process that does its own reading, parsing, importing and table reading
PREPARATION = {}
# the same rule for the files a tool is handed that another tool works out inside its own run: the cell
# area (pyflwdir and FlowDivide compute it from the transform as they go; TauDEM and Cho need a raster of
# it), the outlet pixel as a raster (TauDEM's d8hdisttostrm; GRASS makes its own with r.mapcalc, which is
# timed above), and the channel network cut into links at its junctions (GRASS r.stream.order cannot take
# a plain channel mask).  The flow directions, the channel mask and the upstream area are what the data
# are distributed as, so they are charged to nobody.
PREPARED_INPUT = {
    ("taudem", "upa"): ["dir_taudem", "cellarea"],
    ("taudem", "ldn"): ["dir_taudem", "outlet"],
    ("taudem", "lup"): ["dir_taudem"],
    ("taudem", "ord"): ["dir_taudem"],
    ("cho", "upa"): ["cellarea"],
    ("grass", "ord"): ["links"],
}
WHY_NOT = {
    ("whitebox", "upa"): "the pixel count only, no area in square metres",
    ("whitebox", "ldn"): "map units (degrees) only, no metres",
    ("cho", "lup"): "map units (degrees) only, no metres",
    ("pyflwdir", "lup"): "no upstream flow length in the library",
    ("whitebox", "lup"): "its only one takes elevation, not flow directions",
    ("cho", "ldn"): "no distance to the outlet among the add-ons",
    ("cho", "ord"): "no stream order among the add-ons",
    ("grass", "upa"): "r.stream.* has no accumulation; r.watershed takes elevation",
    ("grass", "lup"): "no upstream flow length in r.stream.*",
    ("fd_tiles", "upa"): "",
}


def rows_of(table, basin, key):
    """the rows of one series on one basin: the runs the orchestration timed from outside, one process
    and one variable each.  The rows a tool script writes about its own steps stay in the table for
    diagnosis and are not read here."""
    if len(table) == 0:
        return table
    selected = table[table["basin"] == basin]
    step = selected["step"].astype(str)
    return selected[step.str.startswith("one process: %s " % STEP_SERIES.get(key, key))]


def mean_of_step(timings, basin, tool, name):
    """the mean wall time of one step of one tool on one basin, or 0 when it is not in the table"""
    if len(timings) == 0:
        return 0.0
    rows = timings[(timings["basin"] == basin) & (timings["tool"] == tool)
                   & (timings["step"].astype(str) == name) & timings["wall_s"].notna()]
    return float(rows["wall_s"].mean()) if len(rows) else 0.0


def prepared_input_seconds(timings, basin, key, kernel):
    """the time it took to build the files this series is handed and another series works out itself.
    A missing row is an error, not a zero: a bar that silently loses part of its work looks valid."""
    names = PREPARED_INPUT.get((key, kernel))
    if not names:
        return 0.0
    total = 0.0
    for name in names:
        seconds = mean_of_step(timings, basin, "inputs", name)
        if seconds <= 0.0:
            raise SystemExit("the timings table holds no row for the prepared input '%s' of basin %d, which "
                             "%s's %s is charged for; run the experiment with FD_REBUILD_INPUTS=1 so that "
                             "the inputs are timed" % (name, basin, key, kernel))
        total += seconds
    return total


def preparation_seconds(timings, basin, key, kernel):
    """what the tool must do before this kernel can run, from the steps PREPARATION names.  The table
    given here must be the whole one: a step a basin pays once (importing a map into GRASS) is recorded
    with rep 0, and the repetitions are filtered out of the table the kernels are read from."""
    names = PREPARATION.get((key, kernel))
    if not names:
        return 0.0
    tool = {"cho": "grass", "grass": "grass", "fd_pub": "flowdivide", "fd_tiles": "flowdivide"}.get(key, key)
    return sum(mean_of_step(timings, basin, tool, name) for name in names)


def wall_of_the_repetitions(seconds):
    """the wall time a bar shows, from the repetitions of one pair: the largest and the smallest are
    dropped and the rest averaged whenever there are three or more, so three repetitions leave the middle
    one.  A single slow repetition, one the machine slowed and not one that did more work, would otherwise
    move the mean, and the figure is about what a method costs."""
    values = sorted(float(v) for v in seconds)
    if len(values) >= 3:
        values = values[1:-1]
    return sum(values) / len(values)


def kernel_seconds(timings, basin, key, kernel, every_row=None):
    """the wall time of one variable of one series on one basin, and how many runs it is, or None.
    The kernel plus its preparation: one variable, computed on its own, from the flow directions.
    `every_row`: the table before the repetitions were filtered, where the once-per-basin steps live."""
    spec = KERNEL_STEP.get((key, kernel))
    if spec is None:
        return None
    rows = rows_of(timings, basin, key)
    if len(rows) == 0:
        return None
    step = rows["step"].astype(str)
    picked = step == spec[1] if spec[0] == "exact" else step.str.startswith(spec[1])
    note_text = rows["note"].fillna("")
    mine = rows[picked & rows["wall_s"].notna() & ~note_text.str.contains("stopped") & note_text.str.contains("exit 0")]
    if len(mine) == 0:
        return None
    repetitions = list(mine["rep"])
    if len(set(repetitions)) != len(repetitions):
        raise SystemExit("the timings table holds two rows of the same repetition for %s's %s on basin %d "
                         "(repetitions %s): the table has been appended to twice, so the mean would mix runs"
                         % (key, kernel, basin, repetitions))
    whole_table = every_row if every_row is not None else timings
    return (wall_of_the_repetitions(mine["wall_s"])
            + preparation_seconds(whole_table, basin, key, kernel)
            + prepared_input_seconds(whole_table, basin, key, kernel)), len(mine)


def prepared_input_peak_gb(timings, basin, key):
    """the largest peak memory among the files this series is handed and another series works out inside
    its own run.  A reader of this series must run those passes too, so their memory counts against the
    machine just as their time counts in (a)."""
    names = set()
    for (series, kernel), charged in PREPARED_INPUT.items():
        if series == key:
            names.update(charged)
    if not names or len(timings) == 0:
        return 0.0
    rows = timings[(timings["basin"] == basin) & (timings["tool"] == "inputs")
                   & timings["step"].astype(str).isin(names)]
    peaks = rows["peak_rss_gb"].dropna() if len(rows) else []
    return float(max(peaks)) if len(peaks) else 0.0


def memory_gb(timings, basin, key, every_row=None):
    """The peak memory of a series on a basin, one definition for every series: the largest peak of its
    rows, and of the passes that build the files it is handed.  A run that was stopped for memory is not
    a measurement and is left out."""
    rows = rows_of(timings, basin, key)
    if len(rows) == 0:
        return None
    # the same predicate as kernel_seconds: a run that was stopped, or whose process did not exit 0, is
    # not a measurement of memory either
    note_text = rows["note"].fillna("")
    rows = rows[~note_text.str.contains("stopped") & note_text.str.contains("exit 0")]
    if len(rows) == 0:
        return None
    peaks = rows["peak_rss_gb"]
    peaks = peaks.dropna()
    if len(peaks) == 0:
        return None
    whole_table = every_row if every_row is not None else timings
    return max(float(peaks.max()), prepared_input_peak_gb(whole_table, basin, key))


def bytes_a_pixel(timings, key, every_row=None):
    """the memory the tool used for every pixel of the drawn basin's rectangle, from its own peak there.
    A tool that holds the whole grid needs this much for any grid, so the same number says whether a
    larger basin fits in the machine."""
    peak = memory_gb(timings, BODY_BASIN, key, every_row)
    if peak is None:
        return None
    return peak * 2 ** 30 / PIXELS_OF[BODY_BASIN]


def disk_bytes_a_pixel(timings, key):
    """the temporary files the tool wrote for every pixel of the drawn basin's rectangle, its largest step"""
    rows = rows_of(timings, BODY_BASIN, key)
    if len(rows) == 0 or "disk_gb" not in rows:
        return None
    disks = rows["disk_gb"].dropna()
    if len(disks) == 0 or float(disks.max()) <= 0:
        return None
    return float(disks.max()) * 2 ** 30 / PIXELS_OF[BODY_BASIN]


def how_far(timings, basin, key):
    """"computed", "out of memory", "partly", or "" when the tool was not run on this basin"""
    rows = rows_of(timings, basin, key)
    if len(rows) == 0:
        return ""
    note = rows["note"].fillna("")
    if len(rows[note.str.contains("stopped: memory")]):
        return "out of memory"
    kernels = [kernel for kernel, _, _ in KERNELS if (key, kernel) in KERNEL_STEP]
    done = [kernel for kernel in kernels if kernel_seconds(timings, basin, key, kernel)]
    if len(done) == len(kernels):
        return "computed"
    if not done:
        return ""
    return "partly"


def timings_fingerprint(path):
    """the same short name for the table's content that needs/needs_model.py writes into needs_basins.csv"""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def repetitions_of(timings, basin, key, kernel):
    """which repetitions of this (series, variable) pair exited 0 on this basin, as a set"""
    spec = KERNEL_STEP.get((key, kernel))
    if spec is None:
        return set()
    rows = rows_of(timings, basin, key)
    if len(rows) == 0:
        return set()
    step = rows["step"].astype(str)
    picked = step == spec[1] if spec[0] == "exact" else step.str.startswith(spec[1])
    note_text = rows["note"].fillna("")
    mine = rows[picked & rows["wall_s"].notna() & ~note_text.str.contains("stopped") & note_text.str.contains("exit 0")]
    return set(int(rep) for rep in mine["rep"])


def check_every_bar_is_there(timings, every_row):
    """Before anything is drawn: the table must hold, for the basin of the body, one run that exited 0 for
    every (series, variable) pair the figure draws, and the same number of repetitions for all of them.  A
    pair that is missing would otherwise be drawn as "not measured yet" beside bars from a finished run,
    and a pair measured fewer times would be a mean of fewer runs than its neighbours.
    FIG7_ALLOW_PARTIAL=1 draws what there is, for looking at a run while it is still going."""
    runs_of_pair = {}
    repetitions_of_pair = {}
    for key, _, _, _ in SERIES:
        for kernel, _, _ in KERNELS:
            if (key, kernel) not in KERNEL_STEP:
                continue
            measured = kernel_seconds(timings, BODY_BASIN, key, kernel, every_row)
            runs_of_pair[(key, kernel)] = 0 if measured is None else measured[1]
            repetitions_of_pair[(key, kernel)] = repetitions_of(timings, BODY_BASIN, key, kernel)
    missing = [pair for pair, runs in runs_of_pair.items() if runs == 0]
    counts = sorted({runs for runs in runs_of_pair.values() if runs > 0})
    sets = {pair: repetitions for pair, repetitions in repetitions_of_pair.items() if repetitions}
    trouble = []
    if missing:
        trouble.append("no run that exited 0 for: " + ", ".join("%s %s" % pair for pair in sorted(missing)))
    if len(counts) > 1:
        uneven = ["%s %s: %d" % (pair[0], pair[1], runs) for pair, runs in sorted(runs_of_pair.items()) if runs > 0]
        trouble.append("the pairs were measured a different number of times (%s)" % "; ".join(uneven))
    if len({tuple(sorted(repetitions)) for repetitions in sets.values()}) > 1:
        # the same number of runs can still be different runs: pair A measured in repetitions 1,2,3 and
        # pair B in 1,2,4 is not one experiment
        listed = ["%s %s: %s" % (pair[0], pair[1], ",".join(str(rep) for rep in sorted(repetitions)))
                  for pair, repetitions in sorted(sets.items())]
        trouble.append("the pairs were not measured in the same repetitions (%s)" % "; ".join(listed))
    if not trouble:
        print("every one of the %d pairs of basin %d was measured %d times" % (len(runs_of_pair), BODY_BASIN, counts[0]))
        return
    message = ("the timings table is not a whole run of basin %d: %s. Run runs/multi/run_basins.py to the end, "
               "or set FIG7_ALLOW_PARTIAL=1 to draw what there is" % (BODY_BASIN, "; ".join(trouble)))
    if os.environ.get("FIG7_ALLOW_PARTIAL", "0") == "1":
        print("WARNING: " + message)
        return
    raise SystemExit(message)


def draw(out):
    timings = pd.read_csv(TIMINGS) if os.path.exists(TIMINGS) else pd.DataFrame()
    if len(timings) == 0:
        raise SystemExit("no timings at %s" % TIMINGS)
    every_row = timings                  # the steps a basin pays once are recorded with rep 0
    timings = timings[timings["rep"] > 0]
    check_every_bar_is_there(timings, every_row)
    style.apply_style()
    figure = plt.figure(figsize=(7.48, 7.10))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.80], width_ratios=[1.20, 1.0], left=0.243,
                               right=0.940, top=0.806, bottom=0.128, hspace=0.34, wspace=0.60)
    axis_a = figure.add_subplot(grid[:, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 1])

    # ---- the legend of the four groups, across the top: the rows of Table 3
    handles = [Patch(facecolor=colour, hatch=hatch, edgecolor="#3d3a34", linewidth=0.5, label=name)
               for _, name, colour, hatch in WAYS]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.998), ncol=1, frameon=False,
                  fontsize=style.SIZE_LEGEND, handlelength=1.15, handleheight=0.95, handletextpad=0.5,
                  labelspacing=0.38, title="What bounds the memory", title_fontsize=style.SIZE_TITLE,
                  alignment="left")

    # ---- (a) one kernel of each class: the same seven rows in every group, a reason where there is no bar
    slots, bars, headings, missing = 0.0, [], [], []
    for kernel, heading, kind in KERNELS:
        headings.append((slots, heading, kind))
        slots += 1.0
        for key, name, _, way in SERIES:
            measured = kernel_seconds(timings, BODY_BASIN, key, kernel, every_row)
            if measured is None:
                missing.append((slots, name, WHY_NOT.get((key, kernel), "not measured yet")))
            else:
                bars.append((slots, name, measured[0], COLOUR_OF_WAY[way], HATCH_OF_WAY[way]))
            slots += 1.0
        slots += 0.55
    for slot, _, value, colour, hatch in bars:
        axis_a.barh([slot], [value], height=0.74, color=colour, hatch=hatch, edgecolor="#3d3a34", linewidth=0.5)
    for slot, name, value, _, _ in bars:
        axis_a.text(value + 7, slot, "%.0f" % value, va="center", ha="left", fontsize=style.SIZE_FOOT,
                    color=style.TEXT_COLOUR)
    for slot, name, why in missing:
        axis_a.text(7, slot, why, va="center", ha="left", fontsize=style.SIZE_LABEL, color="#8d9298")
    ordered = sorted([(slot, name) for slot, name, _, _, _ in bars]
                     + [(slot, name) for slot, name, _ in missing], key=lambda row: row[0])
    axis_a.set_yticks([slot for slot, _ in ordered])
    axis_a.set_yticklabels([name for _, name in ordered], fontsize=style.SIZE_FOOT)
    left_of_axes = blended_transform_factory(axis_a.transAxes, axis_a.transData)
    for slot, heading, kind in headings:
        axis_a.text(-0.335, slot + 0.12, "%s, %s" % (heading, kind), transform=left_of_axes, ha="left",
                    va="center", fontsize=style.SIZE_TITLE, fontweight="bold", color=style.TEXT_COLOUR)
    axis_a.set_ylim(slots - 0.35, -0.75)
    longest = max([value for _, _, value, _, _ in bars] + [1.0])
    axis_a.set_xlim(0, longest * 1.18)          # room for the number at the end of the longest bar
    axis_a.set_xlabel("wall time on basin %d, the %s (s)" % (BODY_BASIN, NAME_OF[BODY_BASIN]),
                      fontsize=style.SIZE_TICK)
    axis_a.tick_params(axis="y", length=0, pad=2)
    axis_a.tick_params(axis="x", labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    axis_a.grid(axis="x", color="#e4e6e8", linewidth=0.5)
    axis_a.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis_a.spines[spine].set_visible(False)
    style.panel_heading(axis_a, "(a)", "One kernel of each attribute class",
                        pad_points=4.0)

    # ---- (b) the peak memory of the same runs; the paged way pays in temporary files, so that is said too
    labels, peaks, colours, hatches, notes = [], [], [], [], []
    for key, _, short, way in SERIES:
        peak = memory_gb(timings, BODY_BASIN, key, every_row)
        if peak is None:
            continue
        on_disk = disk_bytes_a_pixel(timings, key)
        labels.append(short)
        peaks.append(peak)
        colours.append(COLOUR_OF_WAY[way])
        hatches.append(HATCH_OF_WAY[way])
        notes.append(" + %.0f on disk" % (on_disk * PIXELS_OF[BODY_BASIN] / 2 ** 30)
                     if way == "disk" and on_disk else "")
    places = np.arange(len(peaks))
    for place, peak, colour, hatch in zip(places, peaks, colours, hatches):
        axis_b.barh([place], [peak], color=colour, hatch=hatch, edgecolor="#3d3a34", linewidth=0.5, height=0.68)
    for place, peak, note in zip(places, peaks, notes):
        # a bar that ends just short of the 64 GB line would put its number on the line, so that number
        # goes inside the bar instead
        on_the_line = MEMORY_GB - 6.0 <= peak <= MEMORY_GB + 2.0
        axis_b.text(peak - 1.6 if on_the_line else peak + 1.6, place, "%.0f%s" % (peak, note), va="center",
                    ha="right" if on_the_line else "left", fontsize=style.SIZE_FOOT,
                    color="white" if on_the_line else style.TEXT_COLOUR)
    axis_b.axvline(MEMORY_GB, color="#3d3a34", linewidth=0.9, linestyle=(0, (4, 2)))
    axis_b.text(MEMORY_GB + 1.2, -0.62, "64 GiB", fontsize=style.SIZE_FOOT, color="#3d3a34", ha="left",
                va="center")
    axis_b.set_yticks(places)
    axis_b.set_yticklabels(labels, fontsize=style.SIZE_FOOT)
    axis_b.set_ylim(len(peaks) - 0.4, -0.75)
    axis_b.set_xlim(0, max(max(peaks), MEMORY_GB) * 1.30)
    axis_b.set_xlabel("peak memory (GiB)", fontsize=style.SIZE_TICK)
    axis_b.tick_params(axis="y", length=0, pad=2)
    axis_b.tick_params(axis="x", labelsize=style.SIZE_TICK, length=2.2, pad=1.6)
    axis_b.grid(axis="x", color="#e4e6e8", linewidth=0.5)
    axis_b.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis_b.spines[spine].set_visible(False)
    style.panel_heading(axis_b, "(b)", "Memory needed on basin %d" % BODY_BASIN,
                        "the peak of the whole run", pad_points=4.0)

    # ---- (c) what each tool needs on each basin, computed by runs/needs/needs_model.py and read here:
    # the memory of the largest of its variables, and for the paged tool the temporary files as well
    # panel (c) is computed by needs_model.py from the same table; if that file is older than the timings,
    # the panel would show numbers from an earlier run while (a) and (b) show this one
    if os.path.exists(NEEDS) and os.path.getmtime(NEEDS) < os.path.getmtime(TIMINGS):
        raise SystemExit("%s is older than %s: run needs/needs_model.py again before drawing panel (c)"
                         % (NEEDS, TIMINGS))
    needs = pd.read_csv(NEEDS) if os.path.exists(NEEDS) else pd.DataFrame()
    # the two files must be the same measurements: a time is not a strong enough tie, since the model can be
    # built from one table and the timings appended to afterwards within the same second
    if len(needs):
        if "timings_fingerprint" not in needs.columns:
            raise SystemExit("%s does not say which timing table it was built from (no timings_fingerprint "
                             "column): run needs/needs_model.py again" % NEEDS)
        written_from = sorted(set(needs["timings_fingerprint"].astype(str)))
        mine = timings_fingerprint(TIMINGS)
        if written_from != [mine]:
            raise SystemExit("%s was built from the timing table %s, and this figure reads %s (%s): run "
                             "needs/needs_model.py on this table before drawing panel (c)"
                             % (NEEDS, ", ".join(written_from), mine, TIMINGS))
        # and on the same rectangles: (c) is each basin's pixels times a measured bytes a pixel, so a
        # different rectangle there would be a different number here
        if "rectangle_pixels" in needs.columns:
            for basin, pixels in needs.groupby("basin")["rectangle_pixels"].max().items():
                here = rectangle_pixels(int(basin))
                if here is not None and int(pixels) != int(here):
                    raise SystemExit("%s says basin %d has %d pixels in its rectangle and this figure reads "
                                     "%d: the two are not the same partition, so panel (c) is not drawn"
                                     % (NEEDS, int(basin), int(pixels), int(here)))
    axis_c.set_xlim(-0.75, len(BASINS) - 0.25)
    axis_c.set_ylim(len(SERIES) - 0.4, -0.85)
    labels_c = []
    for row, (key, _, short, _) in enumerate(SERIES):
        mine = needs[needs["tool"] == key] if len(needs) else needs
        if len(mine) == 0:
            labels_c.append(short)
            continue
        labels_c.append(short)
        for column, (basin, _, _) in enumerate(BASINS):
            here = mine[mine["basin"] == basin]
            if len(here) == 0:
                continue
            memory = float(here["memory_gb"].max())
            on_disk = float(here["disk_gb"].max())
            fits = bool(here["fits"].min())
            if key == "grass":
                text = "%.0f + %.0f" % (memory, on_disk)
            else:
                text = "%.0f" % memory
            axis_c.text(column, row, text, ha="center", va="center", fontsize=style.SIZE_LABEL,
                        color=style.TEXT_COLOUR, fontweight=("normal" if fits else "bold"),
                        bbox=(None if fits else dict(boxstyle="round,pad=0.22", facecolor="white",
                                                     edgecolor="#2f4858", linewidth=0.7)))
    # the basin ids of Figure 3a head the columns; the names, too wide for a column at 7 pt, go in the note below
    axis_c.set_xticks([])
    for column, (basin, _, _) in enumerate(BASINS):
        axis_c.text(column, -0.62, "%d" % basin, ha="center", va="center", fontsize=style.SIZE_LABEL,
                    color=style.TEXT_COLOUR)
    axis_c.text(-0.80, -0.62, "basin", ha="right", va="center", fontsize=style.SIZE_FOOT, color=style.TEXT_COLOUR)
    axis_c.set_yticks(range(len(SERIES)))
    axis_c.set_yticklabels(labels_c, fontsize=style.SIZE_FOOT)
    axis_c.tick_params(length=0, pad=2)
    for spine in ("top", "right", "left", "bottom"):
        axis_c.spines[spine].set_visible(False)
    names = ["%d %s" % (basin, name) for basin, name, _ in BASINS]
    axis_c.text(-0.10, -0.03, "basins: %s,\n%s\nboxed: more memory than the machine's 64 GiB\n"
                "GRASS r.stream.*: memory, then files on disk\nscaled by each basin's bounding rectangle"
                % (", ".join(names[:3]), ", ".join(names[3:])),
                transform=axis_c.transAxes, fontsize=style.SIZE_FOOT, color=style.TEXT_COLOUR, ha="left",
                va="top", linespacing=1.5)
    style.panel_heading(axis_c, "(c)", "Memory it would need, in GiB",
                        "estimated from basin %d, not run" % BODY_BASIN,
                        pad_points=4.0)

    os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
    figure.savefig(out, dpi=400, facecolor="white")
    plt.close(figure)
    return out


if __name__ == "__main__":
    out = style.figure_path(FIGURE_DIRECTORY, "fig_07_external_tools")
    print(draw(out))
