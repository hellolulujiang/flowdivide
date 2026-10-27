"""Checked without any product data:

  the pixel area under each earth model   the four reference values of MERIT's own formula (CaMa-Flood's
                                          rgetara), and the ordering of the two models
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
    import pandas as pd
    work = tempfile.mkdtemp(prefix="flowdivide_carried_ids_")
    ncol = 1000

    def a_table(name, rows, form):
        path = os.path.join(work, name + (".parquet" if form == "parquet" else ".csv"))
        columns = (["basin_number", "idxs_outlet_row", "idxs_outlet_col"] if form == "space" else
                   ["global_basin_id", "global_outlet_row", "global_outlet_col"])
        table = pd.DataFrame(rows, columns=columns)
        if form != "space":
            table["region_id"], table["basin_cell_count"], table["basin_area_km2"] = 1, 1, 0.01
            table["outlet_lon"], table["outlet_lat"] = 0.0, 0.0
            table = table[["region_id", "global_basin_id", "basin_cell_count", "basin_area_km2",
                           "global_outlet_row", "global_outlet_col", "outlet_lon", "outlet_lat"]]
        if form == "parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, sep=" " if form == "space" else ",", index=False)
        return path

    def refused(what, path, outlets):
        try:
            fd1._basin_ids_from_a_published_table(path, np.asarray(outlets, np.int64), _Grid(ncol), "test")
            check(what, "no error", "FlowDivideError")
        except fd1.FlowDivideError:
            check(what, "FlowDivideError", "FlowDivideError")

    for form in ("space", "comma", "parquet"):
        print("  %s table" % form)
        published = a_table("published_" + form, [(2, 5, 7), (1, 9, 3), (3, 0, 1)], form)
        ours = np.array([0 * ncol + 1, 5 * ncol + 7, 9 * ncol + 3], np.int64)
        order = fd1._basin_ids_from_a_published_table(published, ours, _Grid(ncol), "test")
        check("the outlet that carries id 1 comes first", int(ours[order][0]), 9 * ncol + 3)
        check("then id 2, then id 3", [int(v) for v in ours[order][1:]], [5 * ncol + 7, 0 * ncol + 1])
        refused("an outlet that is not in the table is refused", published, [1, 5007, 9004])
        refused("a table of another size is refused", published, [1, 5007])
        refused("ids that are not 1 .. N are refused",
                a_table("gap_" + form, [(2, 5, 7), (1, 9, 3), (4, 0, 1)], form), ours)
        refused("a repeated id is refused",
                a_table("repeat_" + form, [(2, 5, 7), (1, 9, 3), (2, 0, 1)], form), ours)
        refused("two outlets taking the same id are refused", published, [1, 5007, 5007])
        no_columns = os.path.join(work, "no_columns_" + form + (".parquet" if form == "parquet" else ".csv"))
        table = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        if form == "parquet":
            table.to_parquet(no_columns, index=False)
        else:
            table.to_csv(no_columns, sep=" " if form == "space" else ",", index=False)
        refused("a table without the columns I know is refused", no_columns, [1])


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
