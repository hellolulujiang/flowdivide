"""test_fd3_seam_and_reuse.py -- FD3 across the seam of a periodic grid, and a partition used for more than one attribute.

    python3 test_fd3_seam_and_reuse.py

1. A basin on a periodic grid of 10 columns that crosses the seam and leaves two columns free (its box spans 8 of
   the 10, its window with the margins all 10): the six attributes, lfp included, equal pixel for pixel and value for
   value those of the same basin laid out on a flat grid (the columns rolled so that it does not cross).
2. A region that spans 9 or 10 of the 10 columns: refused, with a message that says why (its window would hold a
   column twice; such a region is not computed).
3. A partition read with raster_min_basin_area_km2=0, as for the distance and the upstream flow length, that holds a
   region of small basins only and a basin below the tables' area that is cut: the longest flow path on it equals the
   one on a partition read for lfp (up to 0.7.7: KeyError; in the first 0.7.8: the cut basin's path drawn too), and a
   registered rule runs on it.
4. A registered rule takes the output lock: a second run on the same output is refused.
5. Two pixels equally far from the outlet of a basin across the seam: lfp draws the same path whether the basin is one
   region or three pieces in three regions, also when the basin table gives the basin a box over every column and a
   region's window is unrolled a whole turn away (the members' heads placed by the links between the members).

Synthetic: it checks that the code holds; no number from it goes anywhere.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd1_partition as fd1
import fd3_attributes as fd3
import fd_tables

FAILURES = []
NROW, NCOL = 6, 10           # the whole globe: 10 columns of 36 degrees, 6 rows of 30, so that the periodic grid is one


def check(what, condition):
    print(("ok    " if condition else "FAIL  ") + what)
    if not condition:
        FAILURES.append(what)


def write_raster(path, values, nodata):
    profile = {"driver": "GTiff", "width": values.shape[1], "height": values.shape[0], "count": 1, "dtype": str(values.dtype),
               "crs": "EPSG:4326", "transform": from_origin(-180.0, 90.0, 360.0 / NCOL, 180.0 / NROW), "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)


def write_partition(directory, run, basins, regions, region_of_basin):
    """the four tables of a partition in the formats of fd_tables: basins and regions as dictionaries, the map from a
    basin to its region (0 for a cut basin; there is none here)"""
    os.makedirs(directory, exist_ok=True)
    table = fd_tables.new_basin_table(len(basins))
    for column in ("outlet_row", "outlet_col", "basin_grid_count", "basin_area_km2", "basin_row_min", "basin_row_max",
                   "basin_col_min", "basin_col_max", "region_id"):
        table[column] = [basin[column] for basin in basins]
    table["outlet_lon"] = 0.0
    table["outlet_lat"] = 0.0
    region_table = pd.DataFrame([{name: 0 for name in fd_tables.REGION_TABLE_COLUMNS} for _ in regions])
    for index, region in enumerate(regions):
        for name, value in (("region_id", region["region_id"]), ("row_min", region["row_min"]), ("row_max", region["row_max"]),
                            ("col_min", region["col_min"]), ("col_max", region["col_max"]),
                            ("nrow", region["row_max"] - region["row_min"]), ("ncol", region["col_max"] - region["col_min"]),
                            ("region_grid_count", region["pixels"]), ("basin_count", region["basins"]),
                            ("window_grid_count", (region["row_max"] - region["row_min"]) * (region["col_max"] - region["col_min"])),
                            ("bbox_row_min", region["row_min"]), ("bbox_row_max", region["row_max"]),
                            ("bbox_col_min", region["col_min"]), ("bbox_col_max", region["col_max"])):
            region_table.loc[index, name] = value
    region_table["fill_percent"] = 50.0
    region_table["level3_members"] = ""
    piece = pd.DataFrame({name: pd.Series(dtype=np.int64) for name in fd_tables.PIECE_TABLE_COLUMNS})
    paths = {"basin": os.path.join(directory, "basin_table_fine_%s.csv" % run),
             "region": os.path.join(directory, "region_fine_%s.csv" % run),
             "piece": os.path.join(directory, "piece_fine_%s.csv" % run),
             "basin_region": os.path.join(directory, "basin_region_%s.csv" % run)}
    fd_tables.write_basin_table(table, paths["basin"], "fd1.6", "test")
    fd_tables.write_region_table(region_table, paths["region"], "test")
    fd_tables.write_piece_table(piece, paths["piece"], "test")
    fd_tables.write_basin_map(paths["basin_region"], np.asarray([0] + list(region_of_basin), np.int64), run, "region",
                              fd_tables.region_map_key(NROW * NCOL, 1200, "l3"))
    return paths


def the_basin(columns, outlet_last):
    """flow directions of one basin over `columns` (in the order the water goes east), every other column no data:
    row 2 runs east and ends in a mouth on the last column, the rows above flow south and the rows below north, with
    some diagonals; returns the directions and the upstream count as an area"""
    directions = np.full((NROW, NCOL), fd1.MERIT_NODATA, np.uint8)
    for k, column in enumerate(columns):
        last = k == len(columns) - 1
        for row in range(NROW):
            if row == 2:
                directions[row, column] = 0 if (last and outlet_last) else 1
            elif row < 2:
                directions[row, column] = 2 if (row == 0 and k % 2 == 1 and not last) else 4
            else:
                directions[row, column] = 128 if (row == NROW - 1 and k % 2 == 0 and not last) else 64
    return directions


def upstream_count(directions, periodic):
    count = np.zeros((NROW, NCOL), np.float64)
    land = np.argwhere(fd1.IS_LAND[directions] != 0)
    for row, column in land:
        r, c = int(row), int(column)
        for _ in range(NROW * NCOL):
            count[r, c] += 1
            code = directions[r, c]
            if fd1.IS_TERMINAL[code]:
                break
            r, c = r + int(fd1.DROW[code]), c + int(fd1.DCOL[code])
            if periodic:
                c %= NCOL
            if not (0 <= r < NROW and 0 <= c < NCOL) or fd1.IS_LAND[directions[r, c]] == 0:
                break
    return count.astype(np.float32)


def run_six(directory, directions, periodic, basins, regions, partition_min=0.0):
    """the six attributes on one grid; returns the rasters and the basin tables"""
    dir_path = os.path.join(directory, "dir.tif")
    write_raster(dir_path, directions, fd1.MERIT_NODATA)
    area = upstream_count(directions, periodic)
    write_raster(os.path.join(directory, "upa.tif"), area, 0.0)
    write_raster(os.path.join(directory, "str.tif"), (area > 0).astype(np.uint8), 0)
    grid = fd1.Grid(dir_path, periodic=periodic, block_pixels=1200)
    paths = write_partition(os.path.join(directory, "tables"), "seam", basins, regions, [1])
    rasters, tables = {}, {}
    for code in ("shv", "ldn", "hck", "lup", "ord", "lfp"):
        partition = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 0.0, grid,
                                  raster_min_basin_area_km2=partition_min if code in ("ldn", "lup") else None)
        out = os.path.join(directory, "%s.tif" % code)
        fd3.derive_attribute(code, partition, dir_path, out, os.path.join(directory, "%s_basin.csv" % code),
                             os.path.join(directory, "%s_member.csv" % code), channel_path=os.path.join(directory, "str.tif"),
                             area_path=os.path.join(directory, "upa.tif"), tag="test." + code)
        with rasterio.open(out) as dataset:
            rasters[code] = dataset.read(1)
        tables[code] = pd.read_csv(os.path.join(directory, "%s_basin.csv" % code), sep=" ")
    return rasters, tables


def test_seam(scratch):
    # the basin on the periodic grid: columns 6..9 then across the seam 0..3, columns 4 and 5 free; its box, unrolled,
    # is [6, 14): 8 of the 10 columns, and its window with the margins [5, 15) all 10
    seam_columns = [6, 7, 8, 9, 0, 1, 2, 3]
    flat_columns = list(range(8))
    pixels = NROW * len(seam_columns)
    basin = {"outlet_row": 2, "basin_grid_count": pixels, "basin_area_km2": 5.0, "basin_row_min": 0, "basin_row_max": NROW,
             "region_id": 1}
    seam = os.path.join(scratch, "seam")
    flat = os.path.join(scratch, "flat")
    os.makedirs(seam)
    os.makedirs(flat)
    on_the_seam, seam_tables = run_six(seam, the_basin(seam_columns, True), True,
                                       [dict(basin, outlet_col=3, basin_col_min=6, basin_col_max=14)],
                                       [{"region_id": 1, "row_min": 0, "row_max": NROW, "col_min": 6, "col_max": 14, "pixels": pixels, "basins": 1}])
    laid_flat, flat_tables = run_six(flat, the_basin(flat_columns, True), False,
                                     [dict(basin, outlet_col=7, basin_col_min=0, basin_col_max=8)],
                                     [{"region_id": 1, "row_min": 0, "row_max": NROW, "col_min": 0, "col_max": 8, "pixels": pixels, "basins": 1}])
    rolled = [(column - 6) % NCOL for column in range(NCOL)]       # the flat column of every seam column
    for code in ("shv", "ldn", "hck", "lup", "ord", "lfp"):
        seam_values = on_the_seam[code]
        flat_values = np.ascontiguousarray(laid_flat[code][:, rolled])
        check("across the seam, a box of 8 of 10 columns: %s the same on every pixel as on the flat grid" % code,
              seam_values.dtype == flat_values.dtype and np.array_equal(seam_values.view(np.uint8), flat_values.view(np.uint8))
              and bool((seam_values != (0 if code not in ("ldn", "lup") else (-1 if code == "ldn" else -9999))).any()))
    for code, column in (("shv", "magnitude_at_outlet"), ("ldn", "farthest_metres"), ("hck", "largest_order"),
                         ("lup", "lup_at_outlet_m"), ("ord", "order_at_outlet"), ("lfp", "longest_flow_path_m")):
        check("across the seam: the %s table's %s as on the flat grid (%s)" % (code, column, seam_tables[code][column].tolist()),
              seam_tables[code][column].tolist() == flat_tables[code][column].tolist())


def test_all_the_way_round(scratch):
    for own, columns in ((9, [5, 6, 7, 8, 9, 0, 1, 2, 3]), (10, list(range(NCOL)))):
        directory = os.path.join(scratch, "ring%d" % own)
        os.makedirs(directory)
        dir_path = os.path.join(directory, "dir.tif")
        write_raster(dir_path, the_basin(columns, True), fd1.MERIT_NODATA)
        grid = fd1.Grid(dir_path, periodic=True, block_pixels=1200)
        pixels = NROW * own
        first = columns[0]
        paths = write_partition(os.path.join(directory, "tables"), "ring",
                                [{"outlet_row": 2, "outlet_col": columns[-1], "basin_grid_count": pixels, "basin_area_km2": 5.0,
                                  "basin_row_min": 0, "basin_row_max": NROW, "basin_col_min": first, "basin_col_max": first + own, "region_id": 1}],
                                [{"region_id": 1, "row_min": 0, "row_max": NROW, "col_min": first, "col_max": first + own, "pixels": pixels, "basins": 1}], [1])
        partition = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 0.0, grid)
        try:
            partition.region_rectangle(1)
            check("a region over %d of the 10 columns of a periodic grid: refused" % own, False)
        except fd3.FlowDivideError as error:
            check("a region over %d of the 10 columns of a periodic grid: refused (%s)" % (own, str(error)[:70]),
                  "is not computed" in str(error) and "%d of the 10 columns" % own in str(error))


@fd3.njit(cache=False)
def mark_the_members(order, downstream, member_of_pixel, dir_window, out_window, channel_window, area_window, lengths, ncol,
                     visited_of_member):
    for position in range(order.size):
        pixel = order[position]
        if member_of_pixel[pixel] >= 0:
            out_window[pixel // ncol, pixel - (pixel // ncol) * ncol] = 1
            visited_of_member[member_of_pixel[pixel]] += 1
    return 0


def write_pieces(path, pieces):
    table = pd.DataFrame([{name: 0 for name in fd_tables.PIECE_TABLE_COLUMNS} for _ in pieces])
    for index, piece in enumerate(pieces):
        for name, value in piece.items():
            table.loc[index, name] = value
    for name in ("outlet_lon", "outlet_lat", "minlon", "minlat", "maxlon", "maxlat", "crossing_length_m", "aca_at_outlet_km2"):
        table[name] = table[name].astype(np.float64)
    fd_tables.write_piece_table(table, path, "test")


def test_reuse_and_lock(scratch):
    # basin 1 (5 km2, region 1) over columns 0..2; column 3 sea; basin 2 (0.5 km2) over columns 6..9, cut into piece 4
    # (columns 8..9, region 3, the outlet) and piece 5 (columns 6..7, region 4, flowing into piece 4); basin 3 (0.3 km2,
    # whole, region 2) over columns 4..5.  At a table area of 1 km2 and a raster area of 0, region 2 holds the small
    # basin 3 and no member, and the cut basin 2 is kept for the distance's raster
    directory = os.path.join(scratch, "reuse")
    os.makedirs(directory)
    directions = np.full((NROW, NCOL), fd1.MERIT_NODATA, np.uint8)
    for first, last in ((0, 2), (4, 5), (6, 9)):
        directions[0:2, first:last + 1] = 4
        directions[2, first:last + 1] = 1
        directions[2, last] = 0
        directions[3:, first:last + 1] = 64
    dir_path = os.path.join(directory, "dir.tif")
    write_raster(dir_path, directions, fd1.MERIT_NODATA)
    grid = fd1.Grid(dir_path, periodic=False, block_pixels=1200)
    basins = [{"outlet_row": 2, "outlet_col": 2, "basin_grid_count": NROW * 3, "basin_area_km2": 5.0, "basin_row_min": 0,
               "basin_row_max": NROW, "basin_col_min": 0, "basin_col_max": 3, "region_id": 1},
              {"outlet_row": 2, "outlet_col": 9, "basin_grid_count": NROW * 4, "basin_area_km2": 0.5, "basin_row_min": 0,
               "basin_row_max": NROW, "basin_col_min": 6, "basin_col_max": 10, "region_id": 3},
              {"outlet_row": 2, "outlet_col": 5, "basin_grid_count": NROW * 2, "basin_area_km2": 0.3, "basin_row_min": 0,
               "basin_row_max": NROW, "basin_col_min": 4, "basin_col_max": 6, "region_id": 2}]
    regions = [{"region_id": 1, "row_min": 0, "row_max": NROW, "col_min": 0, "col_max": 3, "pixels": NROW * 3, "basins": 1},
               {"region_id": 2, "row_min": 0, "row_max": NROW, "col_min": 4, "col_max": 6, "pixels": NROW * 2, "basins": 1},
               {"region_id": 3, "row_min": 0, "row_max": NROW, "col_min": 8, "col_max": 10, "pixels": NROW * 2, "basins": 0},
               {"region_id": 4, "row_min": 0, "row_max": NROW, "col_min": 6, "col_max": 8, "pixels": NROW * 2, "basins": 0}]
    paths = write_partition(os.path.join(directory, "tables"), "reuse", basins, regions, [1, 0, 2])
    write_pieces(paths["piece"], [
        {"piece_id": 4, "basin_id": 2, "piece_kind": 1, "parent_piece_id": 0, "region_id": 3, "outlet_row": 2, "outlet_col": 9,
         "parent_inlet_row": -1, "parent_inlet_col": -1, "piece_grid_count": NROW * 2, "acc_at_outlet": NROW * 4,
         "aca_at_outlet_km2": 0.5, "piece_area_km2": 0.25, "row_min": 0, "row_max": NROW, "col_min": 8, "col_max": 10,
         "bbox_grid_count": NROW * 2, "downstream_depth": 0},
        {"piece_id": 5, "basin_id": 2, "piece_kind": 1, "parent_piece_id": 4, "region_id": 4, "outlet_row": 2, "outlet_col": 7,
         "parent_inlet_row": 2, "parent_inlet_col": 8, "piece_grid_count": NROW * 2, "acc_at_outlet": NROW * 2,
         "aca_at_outlet_km2": 0.25, "piece_area_km2": 0.25, "row_min": 0, "row_max": NROW, "col_min": 6, "col_max": 8,
         "bbox_grid_count": NROW * 2, "downstream_depth": 1}])
    reused = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 1.0, grid,
                           raster_min_basin_area_km2=0.0)
    fresh = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 1.0, grid)
    check("the partition read for the distance holds region 2 with small basins only and the cut basin 2",
          2 in reused.small_by_region and 2 not in reused.members_by_region and 2 in reused.basins and 2 not in fresh.basins)
    results = {}
    for name, partition in (("reused", reused), ("fresh", fresh)):
        try:
            fd3.derive_attribute("lfp", partition, dir_path, os.path.join(directory, "lfp_%s.tif" % name),
                                 os.path.join(directory, "lfp_basin_%s.csv" % name), os.path.join(directory, "lfp_member_%s.csv" % name),
                                 tag="test.lfp." + name)
            with rasterio.open(os.path.join(directory, "lfp_%s.tif" % name)) as dataset:
                results[name] = (dataset.read(1), open(os.path.join(directory, "lfp_basin_%s.csv" % name)).read())
        except Exception as error:
            check("lfp on the %s partition: runs (%s: %s)" % (name, type(error).__name__, error), False)
    if len(results) == 2:
        check("lfp on the partition read for the distance: the raster and the table of a partition read for lfp "
              "(basin 1 only; %d path pixels)" % int((results["fresh"][0] != 0).sum()),
              np.array_equal(results["reused"][0], results["fresh"][0]) and results["reused"][1] == results["fresh"][1]
              and set(np.unique(results["fresh"][0]).tolist()) == {0, 1})
    fd3.register_attribute("tmk", "the members marked", "uint8", 0, downstream_first=False, channel=False, area=False,
                           kernel=mark_the_members)
    out = os.path.join(directory, "tmk.tif")
    members_pixels = sum(member.pixel_count for member in reused.members.values())
    try:
        fd3.derive_user_attribute("tmk", reused, dir_path, out, os.path.join(directory, "tmk.csv"))
        with rasterio.open(out) as dataset:
            marked = dataset.read(1)
        check("a registered rule on that partition: runs, the %d pixels of its members marked, none of region 2 (%d)"
              % (members_pixels, int(marked.sum())), int(marked.sum()) == members_pixels and int(marked[:, 4:6].sum()) == 0)
    except Exception as error:
        check("a registered rule on that partition: runs (%s: %s)" % (type(error).__name__, error), False)
    if fd3.fcntl is None:
        check("the lock of a registered rule: not on this system (no fcntl), skipped", True)
        return
    with fd3._the_only_run_writing(out):
        try:
            fd3.derive_user_attribute("tmk", reused, dir_path, out, os.path.join(directory, "tmk.csv"))
            check("a registered rule while another run writes its output: refused", False)
        except fd3.FlowDivideError as error:
            check("a registered rule while another run writes its output: refused", "another run is writing" in str(error))


def test_lfp_tie_across_the_seam(scratch):
    """a basin over columns 9, 0 and 1 of a periodic grid, its outlet at (2, 0), and two pixels equally far from it:
    (0, 9) and (0, 1) (each a step south, a step east or west, a step south).  Computed as one region, the head is the
    one further west in the basin, column 9; cut into three pieces in three regions (column 0, the outlet piece; 9 and
    1 flowing into it), the members' heads are compared across regions, and up to 0.7.8's round 5 the column of each
    was taken in its own region's window, which made it column 1 (Codex, round 5).  Both must give the same lfp"""
    directions = np.full((NROW, NCOL), fd1.MERIT_NODATA, np.uint8)
    directions[0, 0], directions[1, 0], directions[2, 0] = 4, 4, 0            # column 0: south to the mouth (2, 0)
    directions[0, 9], directions[1, 9], directions[2, 9] = 4, 1, 64           # column 9: to (1, 9), then east across the seam
    directions[0, 1], directions[1, 1], directions[2, 1] = 4, 16, 64          # column 1: to (1, 1), then west
    results = {}
    for layout in ("one region", "three pieces"):
        directory = os.path.join(scratch, "tie_" + layout.replace(" ", "_"))
        os.makedirs(directory)
        dir_path = os.path.join(directory, "dir.tif")
        write_raster(dir_path, directions, fd1.MERIT_NODATA)
        grid = fd1.Grid(dir_path, periodic=True, block_pixels=1200)
        basin = {"outlet_row": 2, "outlet_col": 0, "basin_grid_count": 9, "basin_area_km2": 5.0, "basin_row_min": 0,
                 "basin_row_max": 3, "basin_col_min": 9, "basin_col_max": 12, "region_id": 1}
        if layout == "one region":
            paths = write_partition(os.path.join(directory, "tables"), "tie", [basin],
                                    [{"region_id": 1, "row_min": 0, "row_max": 3, "col_min": 9, "col_max": 12, "pixels": 9, "basins": 1}], [1])
        else:
            paths = write_partition(os.path.join(directory, "tables"), "tie", [basin],
                                    [{"region_id": 1, "row_min": 0, "row_max": 3, "col_min": 0, "col_max": 1, "pixels": 3, "basins": 0},
                                     {"region_id": 2, "row_min": 0, "row_max": 3, "col_min": 9, "col_max": 10, "pixels": 3, "basins": 0},
                                     {"region_id": 3, "row_min": 0, "row_max": 3, "col_min": 1, "col_max": 2, "pixels": 3, "basins": 0}], [0])
            common = {"basin_id": 1, "piece_kind": 1, "piece_grid_count": 3, "aca_at_outlet_km2": 1.0, "piece_area_km2": 5.0 / 3,
                      "row_min": 0, "row_max": 3, "bbox_grid_count": 3}
            write_pieces(paths["piece"], [
                dict(common, piece_id=2, parent_piece_id=0, region_id=1, outlet_row=2, outlet_col=0, parent_inlet_row=-1,
                     parent_inlet_col=-1, acc_at_outlet=9, col_min=0, col_max=1, downstream_depth=0),
                dict(common, piece_id=3, parent_piece_id=2, region_id=2, outlet_row=1, outlet_col=9, parent_inlet_row=1,
                     parent_inlet_col=0, acc_at_outlet=3, col_min=9, col_max=10, downstream_depth=1),
                dict(common, piece_id=4, parent_piece_id=2, region_id=3, outlet_row=1, outlet_col=1, parent_inlet_row=1,
                     parent_inlet_col=0, acc_at_outlet=3, col_min=1, col_max=2, downstream_depth=1)])
        partition = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 1.0, grid)
        fd3.derive_attribute("lfp", partition, dir_path, os.path.join(directory, "lfp.tif"), os.path.join(directory, "lfp_basin.csv"),
                             os.path.join(directory, "lfp_member.csv"), tag="test.tie")
        with rasterio.open(os.path.join(directory, "lfp.tif")) as dataset:
            painted = dataset.read(1)
        table = pd.read_csv(os.path.join(directory, "lfp_basin.csv"), sep=" ")
        results[layout] = (painted, int(table.head_row[0]), int(table.head_col[0]))
    one, three = results["one region"], results["three pieces"]
    check("two heads equally far across the seam: one region and three pieces draw the same path, from column %d / %d"
          % (one[2], three[2]), np.array_equal(one[0], three[0]) and one[1:] == three[1:] and one[2] == 9)


def test_lfp_tie_with_a_loose_box(scratch):
    """Codex, round 6: a basin whose box in the basin table spans every column ([5, 15)) although it holds columns 3..7
    only, as FD1 may unroll it, and whose regions' windows are unrolled from different columns (region 2 at [14, 15),
    the same column as 4).  Its two farthest pixels, (0, 4) and (0, 6), are exactly equally far.  As one region the
    head is the one further west, column 4; cut into three pieces it must be the same: the members' heads are placed by
    the links between them (a child's outlet beside its parent's inlet), not by the basin's box"""
    directions = np.full((NROW, NCOL), fd1.MERIT_NODATA, np.uint8)
    directions[0, 5], directions[1, 5], directions[2, 5], directions[3, 5] = 4, 4, 4, 0   # the trunk down column 5
    directions[0, 4], directions[0, 6] = 1, 16                                            # the two heads, into (0, 5)
    directions[3, 3], directions[3, 4], directions[3, 6], directions[3, 7] = 1, 1, 16, 16  # the near pixels of row 3
    results = {}
    for layout in ("one region", "three pieces"):
        directory = os.path.join(scratch, "loose_" + layout.replace(" ", "_"))
        os.makedirs(directory)
        dir_path = os.path.join(directory, "dir.tif")
        write_raster(dir_path, directions, fd1.MERIT_NODATA)
        grid = fd1.Grid(dir_path, periodic=True, block_pixels=1200)
        basin = {"outlet_row": 3, "outlet_col": 5, "basin_grid_count": 10, "basin_area_km2": 5.0, "basin_row_min": 0,
                 "basin_row_max": 4, "basin_col_min": 5, "basin_col_max": 15, "region_id": 1}
        if layout == "one region":
            paths = write_partition(os.path.join(directory, "tables"), "loose", [basin],
                                    [{"region_id": 1, "row_min": 0, "row_max": 4, "col_min": 3, "col_max": 8, "pixels": 10, "basins": 1}], [1])
        else:
            paths = write_partition(os.path.join(directory, "tables"), "loose", [basin],
                                    [{"region_id": 1, "row_min": 0, "row_max": 4, "col_min": 3, "col_max": 8, "pixels": 8, "basins": 0},
                                     {"region_id": 2, "row_min": 0, "row_max": 1, "col_min": 14, "col_max": 15, "pixels": 1, "basins": 0},
                                     {"region_id": 3, "row_min": 0, "row_max": 1, "col_min": 6, "col_max": 7, "pixels": 1, "basins": 0}], [0])
            common = {"basin_id": 1, "piece_kind": 1, "aca_at_outlet_km2": 1.0, "row_min": 0}
            write_pieces(paths["piece"], [
                dict(common, piece_id=2, parent_piece_id=0, region_id=1, outlet_row=3, outlet_col=5, parent_inlet_row=-1,
                     parent_inlet_col=-1, piece_grid_count=8, acc_at_outlet=10, piece_area_km2=4.0, row_max=4, col_min=3, col_max=8,
                     bbox_grid_count=20, downstream_depth=0),
                dict(common, piece_id=3, parent_piece_id=2, region_id=2, outlet_row=0, outlet_col=4, parent_inlet_row=0,
                     parent_inlet_col=5, piece_grid_count=1, acc_at_outlet=1, piece_area_km2=0.5, row_max=1, col_min=14, col_max=15,
                     bbox_grid_count=1, downstream_depth=1),
                dict(common, piece_id=4, parent_piece_id=2, region_id=3, outlet_row=0, outlet_col=6, parent_inlet_row=0,
                     parent_inlet_col=5, piece_grid_count=1, acc_at_outlet=1, piece_area_km2=0.5, row_max=1, col_min=6, col_max=7,
                     bbox_grid_count=1, downstream_depth=1)])
        partition = fd3.Partition(paths["basin"], paths["region"], paths["piece"], paths["basin_region"], 1.0, grid)
        fd3.derive_attribute("lfp", partition, dir_path, os.path.join(directory, "lfp.tif"), os.path.join(directory, "lfp_basin.csv"),
                             os.path.join(directory, "lfp_member.csv"), tag="test.loose")
        with rasterio.open(os.path.join(directory, "lfp.tif")) as dataset:
            painted = dataset.read(1)
        table = pd.read_csv(os.path.join(directory, "lfp_basin.csv"), sep=" ")
        results[layout] = (painted, int(table.head_row[0]), int(table.head_col[0]))
    one, three = results["one region"], results["three pieces"]
    check("two heads equally far, the basin's box over every column, a region unrolled a turn away: one region and three "
          "pieces draw the same path, from column %d / %d" % (one[2], three[2]),
          np.array_equal(one[0], three[0]) and one[1:] == three[1:] and one[2] == 4)


if __name__ == "__main__":
    scratch = tempfile.mkdtemp(prefix="fd3_seam_")          # left in the system's temporary directory, nothing removed
    test_seam(scratch)
    test_all_the_way_round(scratch)
    test_lfp_tie_across_the_seam(scratch)
    test_lfp_tie_with_a_loose_box(scratch)
    test_reuse_and_lock(scratch)
    print("the test's files are in %s" % scratch)
    print("%d failures" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
