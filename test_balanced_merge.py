"""The balanced default chooses the largest joined window, with extra window and id breaking ties.

    python3 test_balanced_merge.py
"""
import os
import sys

import numpy as np

import fd1_partition as fd1


class Grid:
    block_pixels = 1

    def window_of_rectangle(self, *box):
        return box

    def window_pixels(self, row0, row1, col0, col1):
        return (row1 - row0) * (col1 - col0)


def first_partner(rule, partners):
    regions = []
    for identifier, box, pixels in [(11101, (0, 2, 0, 2), 2)] + partners:
        region = fd1.RegionBuild(identifier, 111, 1)
        region.rectangle = box
        region.pixels = pixels
        region.basins = np.array([identifier], np.int64)
        regions.append(region)
    saved = fd1.MERGE_RULE
    fd1.MERGE_RULE = rule
    try:
        merges = fd1._merge_regions_under_the_capacity(Grid(), regions, 20, "balanced", 100, "test")
    finally:
        fd1.MERGE_RULE = saved
    return merges[0]["absorbed_region_id"]


def main():
    default = fd1.MERGE_RULE
    assert default == os.environ.get("FLOWDIVIDE_MERGE_RULE", "window-max")
    partners = [(11102, (0, 2, 2, 5), 6), (11103, (0, 4, 0, 4), 3)]
    assert first_partner("land-max", partners) == 11102
    assert first_partner("window-max", partners) == 11103
    if "FLOWDIVIDE_MERGE_RULE" not in os.environ:
        assert first_partner(default, partners) == 11103
    assert first_partner("window-max", [(11102, (0, 2, 2, 5), 2), (11103, (0, 2, 1, 5), 2)]) == 11103
    assert first_partner("window-max", [(11103, (0, 2, 2, 5), 2), (11102, (0, 2, 2, 5), 2)]) == 11102
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
