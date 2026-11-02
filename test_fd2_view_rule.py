"""Which basin one coarse cell of the basin view goes to (fd2), the one rule of 1.2.0 (most_land):

    python3 test_fd2_view_rule.py

The cell goes to the candidate holding the most of its land, a tie to the smallest id (the C code's v10.50).  A cell
is land when at least a quarter of it is pixels of any basin (those beyond the view included); a candidate is a
basin of the view holding at least a quarter of that land.  The eight blocks are those of the C test
(tests/test_fd2_view_rule.c), then a whole fold on a small raster: the tag the views carry, a group view following
the basin view, and the refusal of v9.63's rule, which is gone.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rasterio
import fd2_views as fd2

FAILED = []
FACTOR = 10
IN_THE_VIEW = 8          # ids 1..8 are of the view; 9 and above are land and never take a cell
ID_LIMIT = 20


def check(what, got, want):
    if got != want:
        FAILED.append("%s: got %r, expected %r" % (what, got, want))
        print("  FAIL  %-62s got %r, expected %r" % (what, got, want))
    else:
        print("  ok    %-62s %r" % (what, got))


def one_cell(ids, counts):
    block = np.zeros(FACTOR * FACTOR, np.uint32)
    at = 0
    for basin_id, count in zip(ids, counts):
        block[at:at + count] = basin_id
        at += count
    block = block.reshape(FACTOR, FACTOR)
    out_row = np.zeros(1, np.uint32)
    count_of_basin = np.zeros(ID_LIMIT, np.int64)
    touched = np.zeros(FACTOR * FACTOR + 1, np.uint32)
    cell_count = np.zeros(ID_LIMIT, np.int64)
    row_min = np.full(ID_LIMIT, np.iinfo(np.int32).max, np.int32)
    row_max = np.full(ID_LIMIT, -1, np.int32)
    col_min = np.full(ID_LIMIT, np.iinfo(np.int32).max, np.int32)
    col_max = np.full(ID_LIMIT, -1, np.int32)
    flags = np.zeros(2, np.int64)
    fd2._fold_strip_basin(block, FACTOR, IN_THE_VIEW, out_row, fd2.min_land_pixels_of(FACTOR), count_of_basin, touched,
                          cell_count, row_min, row_max, col_min, col_max, 0, flags)
    if count_of_basin.any() or flags[0]:
        FAILED.append("the counts were not reset or an id was outside the table")
    return int(out_row[0])


def the_eight_blocks():
    print("one cell of 10 x 10 pixels")
    cases = [
        ("1. a small basin holding most of the land", (3, 1), (60, 30), 1, 3),
        ("2. a tie", (5, 2), (45, 45), 2, 2),
        ("3. land of a basin beyond the view counts", (4, 1, 9), (54, 26, 20), 1, 4),
        ("4. the most land beyond the view", (9, 2, 3), (50, 30, 20), 2, 2),
        ("5. less than a quarter land", (1,), (24,), 0, 0),
        ("6. a quarter land exactly", (7,), (25,), 7, 7),
        ("7. three candidates", (8, 1, 6), (30, 30, 40), 1, 6),
        ("8. no candidate", (10, 2), (76, 24), 0, 0),
    ]
    for name, ids, counts, _largest_under_v9_63, most in cases:
        check(name, one_cell(ids, counts), most)


def write_mask(path, values):
    profile = {"driver": "GTiff", "dtype": "uint32", "count": 1, "width": values.shape[1], "height": values.shape[0],
               "crs": "EPSG:4326", "transform": rasterio.Affine(0.1, 0, 0, 0, -0.1, 2.0), "nodata": 0}
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)


def a_whole_fold():
    print("a fold of a 20 x 40 mask by 10 and the tags it carries")
    fine = np.zeros((20, 40), np.uint32)
    fine[:, :] = 1                      # basin 1, the largest
    fine[0:10, 10:16] = 2               # cell (0, 1): basin 2 holds 60 of 100, basin 1 40
    fine[0:10, 20:30] = 0
    fine[0:4, 20:30] = 3                # cell (0, 2): 40 land, all of basin 3
    fine[10:20, 30:33] = 1              # cell (1, 3): 1 holds 30, 4 holds 70
    fine[10:20, 33:40] = 4
    with tempfile.TemporaryDirectory() as folder:
        fine_path = os.path.join(folder, "bsn.tif")
        write_mask(fine_path, fine)
        out = os.path.join(folder, "bsn_most_land.tif")
        fd2.fold_mask_at_factors(fine_path, [(out, FACTOR, 4, None, 0.25)], object_count_limit=5, tag="test")
        with rasterio.open(out) as dataset:
            check("the cells (v9.63 gave [[1, 1, 3, 1], [1, 1, 1, 1]])", dataset.read(1).tolist(), [[1, 2, 3, 1], [1, 1, 1, 4]])
            check("the tag", dataset.tags().get("FLOWDIVIDE_VIEW_RULE"), fd2.VIEW_RULE_TAG)
        check("the tag is the C code's, word for word", fd2.VIEW_RULE_TAG,
              "v10.49: the view holds the basins of at least its minimum area, the ids 1..N; a cell of a quarter land goes to the "
              "one of them holding the most of that land, of those holding a quarter of it, a tie to the smallest id; the region "
              "view follows the basin view")
        # a group view following the most_land view: basins 1, 2 in group 1, 3, 4 in group 2
        lookup = np.array([0, 1, 1, 2, 2], np.uint32)
        out = os.path.join(folder, "grp_most_land.tif")
        fd2.fold_mask_at_factors(fine_path, [(out, FACTOR, None, os.path.join(folder, "bsn_most_land.tif"), None)], lookup=lookup,
                                 object_count_limit=3, tag="test", rule="region")
        with rasterio.open(out) as dataset:
            check("a group view follows the most_land view", dataset.read(1).tolist(), [[1, 1, 2, 1], [1, 1, 1, 2]])
            check("and carries its rule", dataset.tags().get("FLOWDIVIDE_VIEW_RULE"), fd2.VIEW_RULE_TAG)
        for refused in ("largest_basin", "most-land"):
            try:
                fd2.fold_mask_at_factors(fine_path, [(os.path.join(folder, "x.tif"), FACTOR, 4, None, 0.25)], object_count_limit=5,
                                         tag="test", rule=refused)
                check("the rule '%s' is refused" % refused, False, True)
            except fd2.FlowDivideError:
                check("the rule '%s' is refused" % refused, True, True)


if __name__ == "__main__":
    the_eight_blocks()
    a_whole_fold()
    print("%s: %d failure(s)" % ("FAILED" if FAILED else "PASSED", len(FAILED)))
    sys.exit(1 if FAILED else 0)
