"""test_fd2_outline_across_the_seam.py -- the boundary lines of a basin astride the 180th meridian.

    python3 test_fd2_outline_across_the_seam.py

On a grid round the earth a basin with cells in the first and the last column is outlined in one piece and its lines
are cut at the meridian (fd2_views.outline_across_the_seam).  Checked on made-up masks of
10 columns of 36 degrees and 6 rows of 30:

1. a block of 4 x 3 cells across the meridian with a hole of two cells across it: 14 cell sides in two lines, every
   point within -180 .. 180, no line along the meridian; outlining the two halves apart gives 24 cell sides,
   4 along the meridian and 6 round the hole;
2. the same lines, moved, as the block gives when it lies away from the meridian (the columns rolled);
3. a block with its east side exactly on the meridian and a second part beyond it: the block is one closed ring;
   the part beyond has its three sides at -180 .. -144 and its side on the meridian at +180 (a point on the meridian
   is taken on the east side: the same meridian, written with the other sign);
4. random objects across the meridian: the length of the lines is the length of the outer rings of the object laid
   out flat, and no side lies along the meridian unless the flat outline has it;
5. an object with a cell in every column is refused;
6. vectorise_mask on a periodic view writes those lines for the basin across the meridian and the flat outline for a
   basin away from it.

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
import fd2_views as fd2
from fd1_partition import FlowDivideError

FAILURES = []
NROW, NCOL = 6, 10
WIDTH, HEIGHT = 360.0 / NCOL, 180.0 / NROW
TRANSFORM = from_origin(-180.0, 90.0, WIDTH, HEIGHT)


def check(what, condition):
    print(("ok    " if condition else "FAIL  ") + what)
    if not condition:
        FAILURES.append(what)


def sides_of(lines):
    """the lines as cell sides: (how many along a row, how many along a column, how many of those on the meridian)"""
    along_row = 0.0
    along_column = 0.0
    on_the_meridian = 0.0
    for line in lines:
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            along_row += abs(x1 - x0) / WIDTH
            along_column += abs(y1 - y0) / HEIGHT
            if abs(abs(x0) - 180.0) < 1e-9 and abs(abs(x1) - 180.0) < 1e-9:
                on_the_meridian += abs(y1 - y0) / HEIGHT
    return round(along_row, 9), round(along_column, 9), round(on_the_meridian, 9)


def flat_outline(mask):
    """the outer rings of the 8-connected polygons of a mask that does not reach the grid's sides, as lines"""
    from rasterio import features
    return [[(float(x), float(y)) for x, y in ring["coordinates"][0]]
            for ring, value in features.shapes(mask, mask=mask.astype(bool), connectivity=8, transform=TRANSFORM) if value == 1]


def block_with_a_hole(first_col):
    mask = np.zeros((NROW, NCOL), np.uint8)
    for k in range(4):
        mask[1:4, (first_col + k) % NCOL] = 1
    mask[2, (first_col + 1) % NCOL] = 0
    mask[2, (first_col + 2) % NCOL] = 0
    return mask


def test_the_block_with_a_hole():
    astride = block_with_a_hole(8)                      # columns 8, 9, 0, 1; the hole at columns 9 and 0
    lines = fd2.outline_across_the_seam(astride, TRANSFORM, 0, 1)
    check("a 4 x 3 block with a hole across the meridian: 14 cell sides (%s) in two lines (%d)" % (sides_of(lines), len(lines)),
          sides_of(lines) == (8.0, 6.0, 0.0) and len(lines) == 2)
    check("every point within -180 .. 180", all(-180.0 <= x <= 180.0 for line in lines for x, _ in line))
    check("each line starts and ends on the meridian, once on each side",
          all(abs(abs(line[0][0]) - 180.0) < 1e-9 and abs(abs(line[-1][0]) - 180.0) < 1e-9 for line in lines)
          and sorted(line[0][0] for line in lines) == [-180.0, 180.0])
    # the grid's own frame, the two halves apart
    halves = flat_outline(astride)
    check("in the grid's own frame the two halves give 24 cell sides, 4 of them along the meridian (%s)" % (sides_of(halves),),
          sides_of(halves) == (12.0, 12.0, 4.0))
    away = block_with_a_hole(3)                         # columns 3 .. 6: the same block away from the meridian
    flat = flat_outline(away)
    check("the same block away from the meridian: 14 cell sides in one ring (%s)" % (sides_of(flat),),
          sides_of(flat) == (8.0, 6.0, 0.0) and len(flat) == 1)
    # the same corners, the columns rolled by 5: every point of the lines is a point of the flat ring moved 180 degrees
    moved = {(round(((x + 5 * WIDTH + 180.0) % 360.0) - 180.0, 9), round(y, 9)) for x, y in flat[0]}
    moved |= {(180.0, y) for x, y in moved if x == -180.0}
    check("the lines pass through the corners of the flat ring, moved, and through the meridian",
          all((round(x, 9), round(y, 9)) in moved or abs(abs(x) - 180.0) < 1e-9 for line in lines for x, y in line))


def test_a_side_on_the_meridian():
    mask = np.zeros((NROW, NCOL), np.uint8)
    mask[1:3, 8:10] = 1                                 # its east side is the meridian
    mask[4, 0] = 1                                      # a second part beyond it, not touching the first
    lines = fd2.outline_across_the_seam(mask, TRANSFORM, 0, 1)
    closed = [line for line in lines if line[0] == line[-1]]
    check("a part with its east side on the meridian and a part beyond it: every side once (%d lines, %s)"
          % (len(lines), sides_of(lines)), sides_of(lines) == (6.0, 6.0, 3.0))
    check("the part with its east side on the meridian is one closed ring, 108 .. 180",
          len(closed) == 1 and min(x for x, _ in closed[0]) == 108.0 and max(x for x, _ in closed[0]) == 180.0)
    opened = [line for line in lines if line[0] != line[-1]]
    check("the part beyond the meridian: three sides at -180 .. -144, and its side on the meridian written at +180",
          len(opened) == 2
          and any(min(x for x, _ in line) == -180.0 and max(x for x, _ in line) == -144.0 and len(line) == 4 for line in opened)
          and any(all(x == 180.0 for x, _ in line) and len(line) == 2 for line in opened))


def test_random_objects():
    rng = np.random.default_rng(20261011)
    same = 0
    tried = 0
    for _ in range(300):
        flat = np.zeros((NROW, NCOL), np.uint8)
        width = int(rng.integers(2, 7))
        flat[:, 2:2 + width] = (rng.random((NROW, width)) < 0.6).astype(np.uint8)
        if not flat[:, 2].any() or not flat[:, 2 + width - 1].any():
            continue
        shift = int(rng.integers(NCOL - 2 - width + 1, NCOL - 2))    # rolled so that it lies across the meridian
        astride = np.roll(flat, shift, axis=1)
        if not astride[:, 0].any() or not astride[:, NCOL - 1].any():
            continue
        tried += 1
        lines = fd2.outline_across_the_seam(astride, TRANSFORM, 0, 1)
        expected = flat_outline(flat)
        # a side of the flat outline that the roll puts on the meridian stays a side there
        on_the_meridian = 0.0
        for ring in expected:
            for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
                if x0 == x1 and abs(((x0 + shift * WIDTH + 180.0) % 360.0)) < 1e-9:
                    on_the_meridian += abs(y1 - y0) / HEIGHT
        along_row, along_column, _ = sides_of(expected)
        if sides_of(lines) == (along_row, along_column, round(on_the_meridian, 9)) and \
                all(-180.0 <= x <= 180.0 for line in lines for x, _ in line):
            same += 1
    check("%d random objects across the meridian: the lines measure what the outer rings of the object laid out flat "
          "measure, side for side (%d)" % (tried, same), tried >= 100 and same == tried)


def test_every_column():
    mask = np.zeros((NROW, NCOL), np.uint8)
    mask[2, :] = 1
    try:
        fd2.outline_across_the_seam(mask, TRANSFORM, 0, 7)
        check("an object with a cell in every column: refused", False)
    except FlowDivideError as error:
        check("an object with a cell in every column: refused (%s)" % error, "object 7 holds every column" in str(error))


def test_vectorise_mask(scratch):
    import geopandas
    view = np.zeros((NROW, NCOL), np.uint32)
    view[block_with_a_hole(8) == 1] = 1                 # basin 1 across the meridian, with the hole
    view[1:4, 4:6] = 2                                  # basin 2 away from it
    view[2, 9] = 3                                      # basin 3 fills the hole
    view[2, 0] = 3
    coarse_path = os.path.join(scratch, "bsn_view.tif")
    profile = {"driver": "GTiff", "width": NCOL, "height": NROW, "count": 1, "dtype": "uint32", "crs": "EPSG:4326",
               "transform": TRANSFORM, "nodata": 0}
    with rasterio.open(coarse_path, "w", **profile) as dataset:
        dataset.write(view, 1)
    rows = []
    for object_id in (1, 2, 3):
        cells = np.argwhere(view == object_id)
        rows.append({"object_id": object_id, "coarse_grid_count": len(cells), "cell_row_min": int(cells[:, 0].min()),
                     "cell_row_max": int(cells[:, 0].max()) + 1, "cell_col_min": int(cells[:, 1].min()),
                     "cell_col_max": int(cells[:, 1].max()) + 1})
    fd2.write_table(pd.DataFrame(rows), coarse_path + ".objects.csv")
    table = pd.DataFrame({"basin_id": [1, 2, 3], "basin_area_km2": [30.0, 20.0, 10.0], "outlet_lon": [0.0, 0.0, 0.0],
                          "outlet_lat": [0.0, 0.0, 0.0]})
    results = {}
    for periodic in (True, False):
        stem = os.path.join(scratch, "bsn_%s" % ("round" if periodic else "flat"))
        lines_stem = os.path.join(scratch, "bnd_%s" % ("round" if periodic else "flat"))
        fd2.vectorise_mask(coarse_path, table, "basin_id", stem, "basins", 3, 1, formats=("gpkg",), boundaries_stem=lines_stem,
                           periodic=periodic)
        frame = geopandas.read_file(lines_stem + ".gpkg", engine="pyogrio", layer="boundaries").sort_values("basin_id")
        results[periodic] = {int(row.basin_id): [list(part.coords) for part in row.geometry.geoms] for row in frame.itertuples()}
    check("vectorise_mask on a grid round the earth: basin 1, across the meridian, 14 cell sides in two lines (%s)"
          % (sides_of(results[True][1]),), sides_of(results[True][1]) == (8.0, 6.0, 0.0) and len(results[True][1]) == 2)
    check("basin 3, the hole, across the meridian too: its 2 x 1 outline, 6 cell sides, none along the meridian (%s)"
          % (sides_of(results[True][3]),), sides_of(results[True][3]) == (4.0, 2.0, 0.0))
    check("basin 2, away from the meridian: the same lines on both grids", results[True][2] == results[False][2]
          and sides_of(results[True][2]) == (4.0, 6.0, 0.0))
    check("on a grid that does not go round, basin 1 keeps the outline of its two halves (%s)" % (sides_of(results[False][1]),),
          sides_of(results[False][1]) == (12.0, 12.0, 4.0))
    polygons = geopandas.read_file(os.path.join(scratch, "bsn_round.gpkg"), engine="pyogrio", layer="basins").sort_values("basin_id")
    check("the polygons of basin 1 are unchanged by the outline: two parts, 10 cells' area",
          len(polygons.geometry.iloc[0].geoms) == 2 and abs(polygons.geometry.iloc[0].area - 10 * WIDTH * HEIGHT) < 1e-6)


if __name__ == "__main__":
    scratch = tempfile.mkdtemp(prefix="fd2_seam_")          # left in the system's temporary directory, nothing removed
    test_the_block_with_a_hole()
    test_a_side_on_the_meridian()
    test_random_objects()
    test_every_column()
    test_vectorise_mask(scratch)
    print("the test's files are in %s" % scratch)
    print("%d failures" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
