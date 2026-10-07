"""fd_tables.py -- the tables of FlowDivide, one layout for both grids.

Every FlowDivide table is written by one writer here and read by one reader here, with its header, its columns and
its number formats, so that a table written and read back gives the same text.  A table of another layout is refused, not read:
the FlowDivide results are read by FlowDivide only, and a run is made again rather than an old layout kept.

THE WORDS: a count of grid cells is <what>_grid_count in every column name, whatever the resolution.  A BASIN is the whole drainage area of one outlet; a GROUP a set of whole basins read together (a
HydroBASINS Level-01, Level-02 or Level-03 unit, or a run of the Hilbert curve); a REGION what is computed at once
(whole basins, or the PIECES of one cut basin).  A WINDOW is a rectangle on whole blocks.

THE FILES OF ONE RUN (<run> the continent, or "global" for MERIT; <root> the run's root):
    <root>global/table/basin_table_fine_<run>.csv       one row per basin (BASIN_TABLE_COLUMNS)
    <root>global/table/group_<grouping>_fine_<run>.csv  one row per group (GROUP_TABLE_COLUMNS), l1, l2, l3, hilbert[_<cap>]
    <root>global/table/basin_group_<grouping>_fine_<run>.csv   the group of every basin, a basin map (Hilbert groupings)
    <partition root>global/table/region_fine_<run>.csv  one row per region (REGION_TABLE_COLUMNS)
    <partition root>global/table/piece_fine_<run>.csv   one row per piece of a cut basin (PIECE_TABLE_COLUMNS)
    <partition root>global/table/basin_region_<run>.csv the region of every basin, a basin map

THE BASIN TABLE, AND WHICH STEP FILLS WHICH COLUMN (the table is rewritten in place by each step):
    basin_id                               fd1.2  1 .. N by drainage area, largest first; the row number
    global_basin_id                        fd1.2 on MERIT (= basin_id), fd1.7 on the 30 m grid (0 until then)
    level1_id .. level3_id                 fd1.4  the group ids (a HydroBASINS code, or an island or seam group's id)
    basin_id_in_level1 .. _in_region       fd1.6  the place of the basin in its group, 1 .. n in basin_id order
    region_id                              fd1.6  the region of the 2^31 partition that holds the outlet
    outlet_row outlet_col outlet_flag      fd1.2
    basin_grid_count basin_area_km2        fd1.2
    outlet_lon outlet_lat, the boxes       fd1.3
A column not filled yet holds 0 (an id or a place), -1 (a row or a column; then basin_nrow and basin_ncol are 0) or
-9999 (a longitude or latitude).  Every table is written under a temporary name, renamed when whole, and given a .done
marker; the basin table's marker says "basins=<n> stage=<step>", which its reader checks.

Function index:
    [1] Markers and publication: write_done_marker, read_marker_checks, file_is_complete, publish_with_marker
    [2] The basin table: new_basin_table, basin_table_path, write_basin_table, read_basin_table, basin_table_stage,
        write_global_basin_table
    [3] The group table: group_table_path, group_basin_map_path, write_group_table, read_group_table, group_block_pixels,
        group_of_basin
    [4] The region and piece tables: write_region_table, read_region_table, write_piece_table, read_piece_table
    [5] Basin maps: region_map_key, write_basin_map, read_basin_map, region_of_basin
"""
import os
import platform
import time

import numpy as np
import pandas as pd

VERSION = "0.7.4"

UNSET_ID = 0
UNSET_PIXEL = -1
UNSET_DEGREE = -9999.0
NO_AREA = -1.0

GROUP_KIND_ONE_BASIN = 1          # one basin voted for the Level-03 code: the great rivers
GROUP_KIND_MANY = 2               # several basins voted for the code; every Level-01/02 group
GROUP_KIND_ISLAND = 3             # no code, collected around a seed
GROUP_KIND_ASTRIDE_THE_SEAM = 4   # no code, a group across the 180th meridian

SEAM_GROUP_ID_BASE = 100000
ISLAND_GROUP_ID_BASE = 200000

BASIN_TABLE_COLUMNS = [
    "basin_id", "global_basin_id", "level1_id", "basin_id_in_level1", "level2_id", "basin_id_in_level2", "level3_id",
    "basin_id_in_level3", "region_id", "basin_id_in_region", "outlet_row", "outlet_col", "outlet_lon", "outlet_lat",
    "outlet_flag", "basin_grid_count", "basin_area_km2", "basin_row_min", "basin_row_max", "basin_col_min",
    "basin_col_max", "basin_nrow", "basin_ncol", "bbox_minlon", "bbox_minlat", "bbox_maxlon", "bbox_maxlat"]
# the number formats, column by column: seven decimals hold one arc-second, six decimals of km2 one
# square metre
BASIN_TABLE_FLOAT_FORMATS = {"outlet_lon": "%.7f", "outlet_lat": "%.7f", "basin_area_km2": "%.6f",
                             "bbox_minlon": "%.7f", "bbox_minlat": "%.7f", "bbox_maxlon": "%.7f", "bbox_maxlat": "%.7f"}

GROUP_TABLE_COLUMNS = [
    "group_id", "group_level", "group_kind", "level_code", "level3_count", "basin_count", "coded_basin_count",
    "neighbour_basin_count", "land_grid_count", "window_row_min", "window_row_max", "window_col_min", "window_col_max",
    "window_nrow", "window_ncol", "window_grid_count", "fill_percent", "window_minlon", "window_minlat",
    "window_maxlon", "window_maxlat"]
GROUP_TABLE_FLOAT_FORMATS = {"fill_percent": "%.3f", "window_minlon": "%.7f", "window_minlat": "%.7f",
                             "window_maxlon": "%.7f", "window_maxlat": "%.7f"}

REGION_TABLE_COLUMNS = [
    "region_id", "level1_code", "level2_code", "level3_code", "row_min", "row_max", "col_min", "col_max", "nrow", "ncol",
    "region_grid_count", "basin_count", "piece_count", "window_grid_count", "fill_percent", "minlon", "minlat", "maxlon",
    "maxlat", "cut_basin_id", "bbox_row_min", "bbox_row_max", "bbox_col_min", "bbox_col_max", "topological_level",
    "level3_members"]
REGION_TABLE_FLOAT_FORMATS = {"fill_percent": "%.3f", "minlon": "%.1f", "minlat": "%.1f", "maxlon": "%.1f",
                              "maxlat": "%.1f"}

PIECE_TABLE_COLUMNS = [
    "piece_id", "basin_id", "level3_code", "piece_kind", "parent_piece_id", "region_id", "outlet_row", "outlet_col",
    "outlet_lon", "outlet_lat", "parent_inlet_row", "parent_inlet_col", "piece_grid_count", "acc_at_outlet",
    "aca_at_outlet_km2", "piece_area_km2", "row_min", "row_max", "col_min", "col_max", "bbox_grid_count", "minlon",
    "minlat", "maxlon", "maxlat", "downstream_depth", "crossing_length_m"]
PIECE_TABLE_FLOAT_FORMATS = {"outlet_lon": "%.7f", "outlet_lat": "%.7f", "aca_at_outlet_km2": "%.6f",
                             "piece_area_km2": "%.6f", "minlon": "%.7f", "minlat": "%.7f", "maxlon": "%.7f",
                             "maxlat": "%.7f", "crossing_length_m": "%.6f"}

BASIN_MAP_MAGIC = "#basin_map 3"


class TableError(Exception):
    pass


# =============================================================================
#  [1] Markers and publication
# =============================================================================

def write_done_marker(path, checks_summary):
    """<path>.done: the file, the time, the version, the machine and
    the checks, under a temporary name and renamed"""
    marker = path + ".done"
    temporary = "%s.tmp.%d" % (marker, os.getpid())
    with open(temporary, "w") as handle:
        handle.write("file %s\ncompleted %s\nversion flowdivide %s\nmachine %s\nchecks %s\n"
                     % (path, time.strftime("%Y-%m-%dT%H:%M:%S"), VERSION, platform.node() or "?", checks_summary or "-"))
    os.replace(temporary, marker)


def read_marker_checks(path):
    """the checks line of <path>.done, or None when there is no marker"""
    marker = path + ".done"
    if not os.path.exists(marker):
        return None
    with open(marker) as handle:
        for line in handle:
            if line.startswith("checks "):
                return line[len("checks "):].strip()
    return ""


def file_is_complete(path):
    """was the file published whole: its .done marker is there"""
    if not os.path.exists(path + ".done"):
        raise TableError("%s has no completion marker %s.done: the step that writes it did not finish" % (path, path))


def publish_with_marker(temporary, path, checks_summary):
    """the old marker removed first (it describes the old file), the file renamed, the new marker written"""
    if os.path.exists(path + ".done"):
        os.remove(path + ".done")
    os.replace(temporary, path)
    write_done_marker(path, checks_summary)


# the longitudes and latitudes of the four tables: one that prints as a negative zero ("-0.0000000", "-0.0") is printed
# as 0, the test made on the printed text
DEGREE_COLUMNS = {"outlet_lon", "outlet_lat", "bbox_minlon", "bbox_minlat", "bbox_maxlon", "bbox_maxlat", "window_minlon",
                  "window_minlat", "window_maxlon", "window_maxlat", "minlon", "minlat", "maxlon", "maxlat"}


def _without_negative_zero(text):
    """the printed degrees, a negative zero without its sign"""
    unsigned = np.char.lstrip(text, "-")
    negative_zero = np.char.startswith(text, "-") & (np.char.strip(unsigned, "0.") == "")
    return np.where(negative_zero, unsigned, text)


def _write_frame(frame, columns, float_formats, path, checks_summary, text_columns=()):
    """a table in its formats: integer columns as integers, the float columns each with its own format"""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise TableError("the table for %s lacks the columns %s" % (path, ", ".join(missing)))
    out = pd.DataFrame(index=frame.index)
    for name in columns:
        values = frame[name]
        if name in float_formats:
            printed = np.char.mod(float_formats[name], values.to_numpy(np.float64))
            out[name] = _without_negative_zero(printed) if name in DEGREE_COLUMNS else printed
        elif name in text_columns:
            out[name] = values.astype(str)
        else:
            out[name] = values.to_numpy(np.int64)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = "%s.tmp.%d" % (path, os.getpid())
    out.to_csv(temporary, sep=" ", index=False, header=True, lineterminator="\n")
    publish_with_marker(temporary, path, checks_summary)


def _check_column_types(frame, path, float_columns, text_columns):
    """what the reader refuses: an integer column holding a fraction or a word, a number
    column holding inf or nan, and an id or a code outside 0 .. 2^32 - 1 (it would wrap when taken as unsigned)"""
    if len(frame) == 0:
        return
    for name in frame.columns:
        if name in text_columns:
            continue
        kind = frame[name].dtype.kind
        if name in float_columns:
            if kind not in "iuf" or not np.isfinite(frame[name].to_numpy(np.float64)).all():
                raise TableError("column %s of %s holds a value that is not a finite number" % (name, path))
            continue
        if kind not in "iu":
            raise TableError("column %s of %s holds a value that is not a whole number" % (name, path))
        # an id is unsigned 32-bit and a code signed 32-bit
        if name.endswith("_id") or name.endswith("_code"):
            values = frame[name].to_numpy(np.int64)
            largest = 2 ** 32 - 1 if name.endswith("_id") else 2 ** 31 - 1
            if (values < 0).any() or (values > largest).any():
                raise TableError("column %s of %s holds a value outside 0 .. %d" % (name, path, largest))


def _check_header(path, columns, what):
    """the marker there and the header exactly the columns (the first column that differs is named)"""
    file_is_complete(path)
    with open(path) as handle:
        header = handle.readline().split()
    if header != columns:
        for index, (found, wanted) in enumerate(zip(header, columns)):
            if found != wanted:
                raise TableError("%s is not a %s of this version: column %d is '%s', it should be '%s'"
                                 % (path, what, index + 1, found, wanted))
        raise TableError("%s is not a %s of this version: it has %d columns, this version writes %d"
                         % (path, what, len(header), len(columns)))


def _read_frame(path, columns, what, float_columns=(), text_columns=()):
    """a whole table, its header demanded exactly (the first column that differs is named), its values of the types
    the reader takes"""
    _check_header(path, columns, what)
    frame = pd.read_csv(path, sep=" ", float_precision="round_trip", keep_default_na=False)
    _check_column_types(frame, path, float_columns, text_columns)
    return frame


# =============================================================================
#  [2] The basin table
# =============================================================================

def basin_table_path(root, run):
    return os.path.join(root, "global", "table", "basin_table_fine_%s.csv" % run)


def root_and_run_of_basin_table(path):
    """(<root>, <run>) of <root>global/table/basin_table_fine_<run>.csv"""
    name = os.path.basename(path)
    if not (name.startswith("basin_table_fine_") and name.endswith(".csv")):
        raise TableError("%s is not named as a basin table" % path)
    return os.path.dirname(os.path.dirname(os.path.dirname(path))), name[len("basin_table_fine_"):-len(".csv")]


def new_basin_table(basin_count):
    """a table of basin_count rows with every column unset, basin_id 1 .. N"""
    table = pd.DataFrame({name: np.zeros(basin_count, np.int64) for name in BASIN_TABLE_COLUMNS})
    table["basin_id"] = np.arange(1, basin_count + 1, dtype=np.int64)
    for name in ("outlet_row", "outlet_col", "basin_row_min", "basin_row_max", "basin_col_min", "basin_col_max"):
        table[name] = UNSET_PIXEL
    for name in ("outlet_lon", "outlet_lat", "bbox_minlon", "bbox_minlat", "bbox_maxlon", "bbox_maxlat"):
        table[name] = UNSET_DEGREE
    table["basin_area_km2"] = NO_AREA
    return table


def _check_basin_rows(table, source):
    """the consistency of every row and the order of the rows, on the whole table at once"""
    count = len(table)
    if count == 0:
        raise TableError("%s holds no basin" % source)
    basin_id = table["basin_id"].to_numpy(np.int64)
    if not np.array_equal(basin_id, np.arange(1, count + 1)):
        first = int(np.nonzero(basin_id != np.arange(1, count + 1))[0][0])
        raise TableError("row %d of %s carries basin_id %d; the id must be the row number" % (first + 1, source, basin_id[first]))
    flag = table["outlet_flag"].to_numpy(np.int64)
    grid_count = table["basin_grid_count"].to_numpy(np.int64)
    area = table["basin_area_km2"].to_numpy(np.float64)
    if (table["outlet_row"].to_numpy(np.int64) < 0).any() or (table["outlet_col"].to_numpy(np.int64) < 0).any() \
            or ((flag != 0) & (flag != 1)).any() or (grid_count < 1).any() or not ((area > 0) | (area == NO_AREA)).all():
        raise TableError("%s holds a row whose outlet, flag, pixel count or area is not a basin's" % source)
    has_area = area != NO_AREA
    if has_area.any() and not has_area.all():
        raise TableError("%s gives an area for some basins and none for others" % source)
    if has_area.all():
        # the order of the areas as the table writes them, six decimals: MERIT BasinFull v1.0
        # numbered its basins by that value, and its ids rise in the raw Float32 upa at 4,031,320 rows and at none
        # once rounded.  Only the pairs that rise as read are formatted again
        rising = np.flatnonzero(np.diff(area) > 0)
        if any(float("%.6f" % area[index + 1]) > float("%.6f" % area[index]) for index in rising):
            raise TableError("%s is not sorted by area" % source)
    elif (np.diff(grid_count) > 0).any():
        raise TableError("%s has no areas and is not sorted by pixel count" % source)
    row_min = table["basin_row_min"].to_numpy(np.int64)
    row_max = table["basin_row_max"].to_numpy(np.int64)
    col_min = table["basin_col_min"].to_numpy(np.int64)
    col_max = table["basin_col_max"].to_numpy(np.int64)
    unset = row_min == UNSET_PIXEL
    unset_right = (row_max == UNSET_PIXEL) & (col_min == UNSET_PIXEL) & (col_max == UNSET_PIXEL)
    outlet_row = table["outlet_row"].to_numpy(np.int64)
    set_right = (row_min >= 0) & (col_min >= 0) & (row_max > row_min) & (col_max > col_min) \
        & (outlet_row >= row_min) & (outlet_row < row_max)
    if not np.where(unset, unset_right, set_right).all():
        raise TableError("%s holds a row whose box is neither unset nor a box around its outlet" % source)


def _box_sizes(table):
    """(basin_nrow, basin_ncol) as the box gives them, 0 for a box not set"""
    box_is_set = table["basin_row_min"].to_numpy(np.int64) >= 0
    nrow = np.where(box_is_set, table["basin_row_max"].to_numpy(np.int64) - table["basin_row_min"].to_numpy(np.int64), 0)
    ncol = np.where(box_is_set, table["basin_col_max"].to_numpy(np.int64) - table["basin_col_min"].to_numpy(np.int64), 0)
    return nrow, ncol


def _with_box_sizes(table):
    out = table.copy()
    out["basin_nrow"], out["basin_ncol"] = _box_sizes(out)
    return out


def _check_box_sizes(table, path):
    """basin_nrow and basin_ncol as the box gives them, checked on the columns themselves: the copy of the whole
    table that _with_box_sizes makes would double the 6.5 GB of North America's thirty million rows"""
    nrow, ncol = _box_sizes(table)
    if not (np.array_equal(nrow, table["basin_nrow"].to_numpy(np.int64)) and np.array_equal(ncol, table["basin_ncol"].to_numpy(np.int64))):
        raise TableError("%s holds a row whose basin_nrow or basin_ncol is not its box" % path)


def write_basin_table(table, path, stage, checks_summary=""):
    """the basin table, checked before a byte is written; basin_nrow and basin_ncol are made from the box"""
    if not stage or " " in stage or ";" in stage:
        raise TableError("the stage of a basin table is one word, got '%s'" % stage)
    table = _with_box_sizes(table)
    _check_basin_rows(table, path)
    _write_frame(table, BASIN_TABLE_COLUMNS, BASIN_TABLE_FLOAT_FORMATS, path,
                 "basins=%d stage=%s; %s" % (len(table), stage, checks_summary))


def basin_table_stage(path):
    """(the number of rows, the step) the marker of a basin table promises"""
    checks = read_marker_checks(path)
    if checks is None:
        raise TableError("%s has no completion marker %s.done: its writing did not finish" % (path, path))
    basins = None
    stage = ""
    for word in checks.replace(";", " ").split():
        if word.startswith("basins="):
            basins = int(word[len("basins="):])
        elif word.startswith("stage="):
            stage = word[len("stage="):]
    if basins is None or basins <= 0 or basins > 2 ** 32 - 1:
        raise TableError("the completion marker of %s does not say how many basins the table holds (basins=)" % path)
    return basins, stage


def read_basin_table(path):
    """the whole basin table, every check: the marker, the exact header, basin_id the row number, the
    area order, the box sizes, the number of rows the marker promises"""
    promised, stage = basin_table_stage(path)
    table = _read_frame(path, BASIN_TABLE_COLUMNS, "basin table", BASIN_TABLE_FLOAT_FORMATS)
    if len(table) != promised:
        raise TableError("%s holds %d rows and its completion marker promises %d: the table was cut short" % (path, len(table), promised))
    _check_basin_rows(table, path)
    _check_box_sizes(table, path)
    return table


# the columns the checks of a basin table read (_check_basin_rows, _check_box_sizes)
BASIN_TABLE_CHECKED_COLUMNS = ("basin_id", "outlet_row", "outlet_col", "outlet_flag", "basin_grid_count", "basin_area_km2",
                               "basin_row_min", "basin_row_max", "basin_col_min", "basin_col_max", "basin_nrow", "basin_ncol")
# rows parsed at once: the parser's buffers grow with the chunk (on 3 million made-up rows the peak above the
# kept columns was 2.6 GB for chunks of 2 million rows and 0.6 GB for chunks of 250,000)
BASIN_TABLE_CHUNK_ROWS = 250000


def read_basin_table_columns(path, columns, chunk_rows=BASIN_TABLE_CHUNK_ROWS):
    """the basin table with every check of read_basin_table, but held in memory only in the columns asked for.  The
    rows are read in chunks; every chunk passes the column checks of the whole table (_check_column_types, column by
    column, so a chunk is checked as the table would be), and only the columns asked for and those the row checks read
    are kept, in arrays of the length the marker promises, filled chunk by chunk.  The row checks then run on the whole
    table as in read_basin_table.  North America's thirty million rows take 6.5 GB in all 27 columns, and read whole
    their parse took about five times as much; the attributes need eleven columns."""
    promised, stage = basin_table_stage(path)
    _check_header(path, BASIN_TABLE_COLUMNS, "basin table")
    unknown = [name for name in columns if name not in BASIN_TABLE_COLUMNS]
    if unknown:
        raise TableError("the basin table has no columns %s" % ", ".join(unknown))
    kept = [name for name in BASIN_TABLE_COLUMNS if name in columns or name in BASIN_TABLE_CHECKED_COLUMNS]
    # the dtypes read_basin_table gives a table of this writer: Float64 the float columns, Int64 the others (a
    # chunk is converted to them only after _check_column_types has passed it).  A row takes at least 54 bytes (27
    # values, 26 blanks, the newline): a marker that promises more rows than the file can hold is not believed, the
    # rows are only counted, and the table is refused as cut short, as read_basin_table refuses it
    fits = promised <= os.path.getsize(path) // (2 * len(BASIN_TABLE_COLUMNS))
    arrays = {name: np.empty(promised if fits else 0, np.float64 if name in BASIN_TABLE_FLOAT_FORMATS else np.int64) for name in kept}
    rows = 0
    for chunk in pd.read_csv(path, sep=" ", float_precision="round_trip", keep_default_na=False, chunksize=chunk_rows):
        _check_column_types(chunk, path, BASIN_TABLE_FLOAT_FORMATS, ())
        count = len(chunk)
        if fits and rows + count <= promised:
            for name in kept:
                arrays[name][rows:rows + count] = chunk[name].to_numpy(arrays[name].dtype)
        rows += count
        del chunk
    if rows != promised:
        raise TableError("%s holds %d rows and its completion marker promises %d: the table was cut short" % (path, rows, promised))
    table = pd.DataFrame(index=pd.RangeIndex(rows))
    for name in kept:
        table[name] = arrays.pop(name)
    _check_basin_rows(table, path)
    _check_box_sizes(table, path)
    for name in kept:                     # the columns only the checks read go, without copying the others
        if name not in columns:
            del table[name]
    return table


def write_global_basin_table(tables_by_run, path):
    """fd1.7: every run's rows in global_basin_id order after a first column "run"; the ids must be 1 .. N"""
    parts = []
    for run, table in tables_by_run:
        part = _with_box_sizes(table)
        part.insert(0, "run", run)
        parts.append(part)
    whole = pd.concat(parts, ignore_index=True).sort_values("global_basin_id", kind="stable").reset_index(drop=True)
    global_id = whole["global_basin_id"].to_numpy(np.int64)
    if not np.array_equal(global_id, np.arange(1, len(whole) + 1)):
        raise TableError("the global ids of the runs are not 1 .. %d without a gap or a repeat" % len(whole))
    if (np.diff(whole["basin_area_km2"].to_numpy(np.float64)) > 0).any():
        raise TableError("the global ids do not follow the area")
    _write_frame(whole, ["run"] + BASIN_TABLE_COLUMNS, BASIN_TABLE_FLOAT_FORMATS, path,
                 "basins=%d stage=fd1.7; runs %s" % (len(whole), " ".join(run for run, _ in tables_by_run)), text_columns=("run",))


# =============================================================================
#  [3] The group table
# =============================================================================

def group_table_path(root, run, grouping):
    return os.path.join(root, "global", "table", "group_%s_fine_%s.csv" % (grouping, run))


def group_basin_map_path(root, run, grouping):
    return os.path.join(root, "global", "table", "basin_group_%s_fine_%s.csv" % (grouping, run))


def group_block_pixels(groups):
    """the block the windows of a group table are on: window_nrow over the block rows, the same for every row"""
    rows = groups["window_row_max"].to_numpy(np.int64) - groups["window_row_min"].to_numpy(np.int64)
    nrow = groups["window_nrow"].to_numpy(np.int64)
    if (rows < 1).any() or (nrow % rows != 0).any():
        raise TableError("a window of the group table is not on whole blocks")
    blocks = np.unique(nrow // rows)
    if blocks.size != 1:
        raise TableError("the windows of the group table lie on blocks of %s pixels, not of one size" % blocks.tolist())
    return int(blocks[0])


def _check_group_rows(groups, source):
    """the checks on every row of a group table"""
    good = ((groups["group_id"] > 0) & groups["group_level"].between(0, 3)
            & groups["group_kind"].between(GROUP_KIND_ONE_BASIN, GROUP_KIND_ASTRIDE_THE_SEAM) & (groups["level_code"] >= 0)
            & (groups["basin_count"] > 0) & (groups["land_grid_count"] > 0) & (groups["window_row_min"] >= 0)
            & (groups["window_col_min"] >= 0) & (groups["window_row_max"] > groups["window_row_min"])
            & (groups["window_col_max"] > groups["window_col_min"])
            & (groups["window_grid_count"] == groups["window_nrow"] * groups["window_ncol"]))
    if not good.all():
        raise TableError("row %d of %s is not a group" % (int(np.nonzero(~good.to_numpy())[0][0]) + 1, source))


def write_group_table(groups, path, checks_summary=""):
    _check_group_rows(groups, path)
    ids = groups["group_id"].to_numpy(np.int64)
    if (ids <= 0).any() or np.unique(ids).size != ids.size:
        raise TableError("the group ids for %s are not positive and distinct" % path)
    _write_frame(groups, GROUP_TABLE_COLUMNS, GROUP_TABLE_FLOAT_FORMATS, path, checks_summary)


def read_group_table(path, pixels_per_block=0):
    """pixels_per_block 0: the block the table was written on (a Hilbert grouping may run on blocks of its own)"""
    groups = _read_frame(path, GROUP_TABLE_COLUMNS, "group table", GROUP_TABLE_FLOAT_FORMATS)
    block = pixels_per_block or group_block_pixels(groups)
    rows = groups["window_row_max"] - groups["window_row_min"]
    cols = groups["window_col_max"] - groups["window_col_min"]
    if not ((groups["window_nrow"] == rows * block) & (groups["window_ncol"] == cols * block)).all():
        raise TableError("a row of %s has window columns that disagree with blocks of %d pixels" % (path, block))
    if groups["group_id"].duplicated().any():
        raise TableError("a group appears twice in %s" % path)
    _check_group_rows(groups, path)
    return groups


def group_of_basin(root, run, grouping, basins=None):
    """the group id of every basin, indexed by basin_id (element 0 unused): l1, l2, l3 from the basin table, any other
    grouping from its basin map"""
    if grouping in ("l1", "l2", "l3"):
        if basins is None:
            basins = read_basin_table(basin_table_path(root, run))
        values = basins["level%s_id" % grouping[1]].to_numpy(np.int64)
        if (values == UNSET_ID).any():
            raise TableError("a basin has no Level-0%s group (run fd1.4 first)" % grouping[1])
        return np.concatenate([[0], values]).astype(np.uint32)
    map_path = group_basin_map_path(root, run, grouping)
    values, in_the_map = read_basin_map(map_path, run, "group", grouping)
    # the map covers the basin table, basin for basin
    basin_count = len(basins) if basins is not None else basin_table_stage(basin_table_path(root, run))[0]
    if in_the_map != basin_count:
        raise TableError("%s holds %d basins, the basin table %d" % (map_path, in_the_map, basin_count))
    if (values[1:] == 0).any():
        raise TableError("a basin has no group of %s" % grouping)
    return values


# =============================================================================
#  [4] The region and piece tables
# =============================================================================

def _check_region_rows(regions, source):
    """the checks on every row of a region table"""
    good = ((regions["region_id"] > 0) & (regions["row_max"] > regions["row_min"]) & (regions["col_max"] > regions["col_min"])
            & (regions["nrow"] == regions["row_max"] - regions["row_min"])
            & (regions["ncol"] == regions["col_max"] - regions["col_min"]))
    if not good.all():
        raise TableError("row %d of %s is not a region (its window and its nrow, ncol disagree)"
                         % (int(np.nonzero(~good.to_numpy())[0][0]) + 1, source))


def _check_piece_rows(pieces, source):
    """the checks on every row of a piece table"""
    good = ((pieces["piece_id"] > 0) & (pieces["basin_id"] > 0) & (pieces["row_max"] > pieces["row_min"])
            & (pieces["col_max"] > pieces["col_min"])
            & (pieces["bbox_grid_count"] == (pieces["row_max"] - pieces["row_min"]) * (pieces["col_max"] - pieces["col_min"])))
    if not good.all():
        raise TableError("row %d of %s is not a piece" % (int(np.nonzero(~good.to_numpy())[0][0]) + 1, source))


def write_region_table(regions, path, checks_summary=""):
    _check_region_rows(regions, path)
    out = regions.copy()
    out["level3_members"] = [("-" if not text else ",".join(text.split())) for text in out["level3_members"]]
    _write_frame(out, REGION_TABLE_COLUMNS, REGION_TABLE_FLOAT_FORMATS, path, checks_summary, text_columns=("level3_members",))


def read_region_table(path):
    regions = _read_frame(path, REGION_TABLE_COLUMNS, "region table", REGION_TABLE_FLOAT_FORMATS, ("level3_members",))
    _check_region_rows(regions, path)
    regions["level3_members"] = ["" if text == "-" else " ".join(str(text).split(",")) for text in regions["level3_members"]]
    return regions


def write_piece_table(pieces, path, checks_summary=""):
    _check_piece_rows(pieces, path)
    _write_frame(pieces, PIECE_TABLE_COLUMNS, PIECE_TABLE_FLOAT_FORMATS, path, checks_summary)


def read_piece_table(path):
    pieces = _read_frame(path, PIECE_TABLE_COLUMNS, "piece table", PIECE_TABLE_FLOAT_FORMATS)
    _check_piece_rows(pieces, path)
    return pieces


# =============================================================================
#  [5] Basin maps
# =============================================================================

def region_map_key(capacity_pixels, block_pixels, grouping):
    """the key of a region map: its capacity, its block and the grouping it was made from"""
    if " " in grouping or not grouping:
        raise TableError("a grouping name has no blanks, got '%s'" % grouping)
    return "capacity_px=%d,block_px=%d,groups=%s" % (capacity_pixels, block_pixels, grouping)


def write_basin_map(path, values, run, kind, key):
    """values[1 .. N], one line each, after a header naming the run, the kind, the key, the count and the byte count"""
    body = "".join("%d\n" % int(value) for value in values[1:])
    header = "%s run=%s kind=%s key=%s basins=%d payload_bytes=%d\n" % (BASIN_MAP_MAGIC, run, kind, key, len(values) - 1,
                                                                       len(body.encode("ascii")))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = "%s.tmp.%d" % (path, os.getpid())
    with open(temporary, "w", newline="\n") as handle:
        handle.write(header)
        handle.write(body)
    publish_with_marker(temporary, path, "%s map of %s, %d basins, key %s" % (kind, run, len(values) - 1, key))


def read_basin_map(path, run, kind, key, wanted_count=None):
    """(values[0 .. N] with element 0 unused, N) of a map of this run, kind and key; the byte count checked"""
    file_is_complete(path)
    with open(path, "rb") as handle:
        header = handle.readline().decode("ascii").split()
        body = handle.read()
    if len(header) < 2 or " ".join(header[:2]) != BASIN_MAP_MAGIC:
        raise TableError("%s is not a basin map of this version" % path)
    fields = dict(word.split("=", 1) for word in header[2:] if "=" in word)
    if fields.get("run") != run or fields.get("kind") != kind or fields.get("key") != key:
        raise TableError("%s is the %s map of %s with %s, not the %s map of %s with %s"
                         % (path, fields.get("kind"), fields.get("run"), fields.get("key"), kind, run, key))
    count = int(fields["basins"])
    if len(body) != int(fields["payload_bytes"]):
        raise TableError("%s holds %d bytes of values and its header says %s: the file was cut short"
                         % (path, len(body), fields["payload_bytes"]))
    words = body.split()
    # digits only: int() would also read "1_2" or "+3"
    if any(not word.isdigit() or not word.isascii() for word in words):
        raise TableError("%s holds a value that is not a whole number" % path)
    values = np.array([int(word) for word in words], dtype=np.int64) if words else np.zeros(0, np.int64)
    # a value is an unsigned 32-bit id; -1 or 2^32 + 1 would wrap
    if values.size and ((values < 0).any() or (values > 2 ** 32 - 1).any()):
        raise TableError("%s holds a value outside 0 .. 4294967295" % path)
    if values.size != count:
        raise TableError("%s holds %d values and its header says %d" % (path, values.size, count))
    if wanted_count is not None:
        if wanted_count > count:
            raise TableError("%s holds %d basins, fewer than the %d asked for" % (path, count, wanted_count))
        values = values[:wanted_count]
    return np.concatenate([[0], values]).astype(np.uint32), count


def region_of_basin(map_path, piece_table_path, run, key, basin_count):
    """the region of every basin at one partition: the map's value, or for a cut basin (0 in the map) the region of
    its outlet piece (the piece with no parent), exactly one"""
    region, in_the_map = read_basin_map(map_path, run, "region", key, basin_count)
    if in_the_map != basin_count:
        raise TableError("%s holds %d basins, the basin table %d" % (map_path, in_the_map, basin_count))
    region = region.astype(np.int64)
    pieces = read_piece_table(piece_table_path)
    outlets = pieces[pieces["parent_piece_id"] == 0]
    for basin_id, region_id in zip(outlets["basin_id"].to_numpy(np.int64), outlets["region_id"].to_numpy(np.int64)):
        if basin_id > basin_count or region[basin_id] != 0 or region_id == 0:
            raise TableError("%s names an outlet piece of basin %d, which the map does not mark as cut (or it has two, "
                             "or no region)" % (piece_table_path, basin_id))
        region[basin_id] = region_id
    if (region[1:] == 0).any():
        raise TableError("basin %d carries no region" % (int(np.nonzero(region[1:] == 0)[0][0]) + 1))
    return region.astype(np.uint32)
