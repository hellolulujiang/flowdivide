"""Checked without any product data:

  the pixel area under each earth model   the four reference values of MERIT's own formula (CaMa-Flood's
                                          rgetara), and the ordering of the three models
  the basin ids carried over               the rule that gives every outlet the id a published table gives
                                          its pixel, and each of its refusals

    python3 test_pixel_area_and_basin_ids.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd1_partition as fd1

FAILED = []


def check(what, got, want):
    if got != want:
        FAILED.append("%s: got %r, expected %r" % (what, got, want))
        print("  FAIL  %-58s got %r, expected %r" % (what, got, want))
    else:
        print("  ok    %-58s %r" % (what, got))


def the_pixel_area_of_every_model():
    print("the pixel area of each earth model at 3 arc-seconds")
    step = 1.0 / 1200.0
    # the values of rgetara
    for latitude, expected in ((0.0, 8547.963867), (30.0, 7421.337891), (55.0, 4935.946777), (79.0, 1646.810303)):
        got = fd1.pixel_area_m2(latitude, step, step, fd1.EARTH_MODEL_MERIT)
        # the reference values are given to six decimals: half a unit in the last of them, not a rounded equality
        check("MERIT's own formula at %.0f degrees (%.6f m2)" % (latitude, got), abs(got - expected) < 5e-6, True)
    merit = fd1.pixel_area_m2(55.0, step, step, fd1.EARTH_MODEL_MERIT)
    zone = fd1.pixel_area_m2(55.0, step, step, fd1.EARTH_MODEL_WGS84_ZONE)
    check("MERIT's is smaller than the exact ellipsoid at 55 degrees", merit < zone, True)
    check("and the two differ by about two parts in a thousand", round((zone - merit) / merit, 5), 0.00229)
    try:
        fd1.pixel_area_m2(0.0, step, step, "what")
        check("an unknown model is refused", "no error", "FlowDivideError")
    except fd1.FlowDivideError:
        check("an unknown model is refused", "FlowDivideError", "FlowDivideError")


class _Grid:
    """what _basin_ids_from_a_published_table needs of a grid"""

    def __init__(self, ncol):
        self.ncol = ncol


def the_basin_ids_carried_over_from_a_table():
    """the rule that gives every outlet the id a published table gives its pixel, and its refusals"""
    print("the basin ids carried over from a published table")
    import tempfile
    import numpy as np
    work = tempfile.mkdtemp(prefix="flowdivide_carried_ids_")
    ncol = 1000

    def a_table(name, rows):
        path = os.path.join(work, name)
        with open(path, "w") as handle:
            handle.write("basin_number idxs_outlet_row idxs_outlet_col\n")
            for number, row, col in rows:
                handle.write("%d %d %d\n" % (number, row, col))
        return path

    # three outlets: the published ids are 1, 2, 3 at three pixels
    published = a_table("published.csv", [(2, 5, 7), (1, 9, 3), (3, 0, 1)])
    ours = np.array([0 * ncol + 1, 5 * ncol + 7, 9 * ncol + 3], np.int64)      # our outlets, by pixel
    order = fd1._basin_ids_from_a_published_table(published, ours, _Grid(ncol), "test")
    check("the outlet that carries id 1 comes first", int(ours[order][0]), 9 * ncol + 3)
    check("then id 2, then id 3", [int(v) for v in ours[order][1:]], [5 * ncol + 7, 0 * ncol + 1])

    def refused(what, path, outlets):
        try:
            fd1._basin_ids_from_a_published_table(path, np.asarray(outlets, np.int64), _Grid(ncol), "test")
            check(what, "no error", "FlowDivideError")
        except fd1.FlowDivideError:
            check(what, "FlowDivideError", "FlowDivideError")

    refused("an outlet that is not in the table is refused", published,
            [0 * ncol + 1, 5 * ncol + 7, 9 * ncol + 4])
    refused("a table of another size is refused", published, [0 * ncol + 1, 5 * ncol + 7])
    refused("ids that are not 1 .. N are refused",
            a_table("gap.csv", [(2, 5, 7), (1, 9, 3), (4, 0, 1)]),
            [0 * ncol + 1, 5 * ncol + 7, 9 * ncol + 3])
    refused("a repeated id is refused",
            a_table("repeat.csv", [(2, 5, 7), (1, 9, 3), (2, 0, 1)]),
            [0 * ncol + 1, 5 * ncol + 7, 9 * ncol + 3])
    # a table whose header this package does not know
    no_columns = os.path.join(work, "no_columns.csv")
    with open(no_columns, "w") as handle:
        handle.write("a b c\n1 2 3\n")
    refused("a table without the columns I know is refused", no_columns, [0 * ncol + 1])


def main():
    the_pixel_area_of_every_model()
    the_basin_ids_carried_over_from_a_table()
    print("")
    if FAILED:
        print("FAILED %d:" % len(FAILED))
        for line in FAILED:
            print("  " + line)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
