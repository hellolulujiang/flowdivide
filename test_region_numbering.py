"""The numbering of the published regions, checked on made-up regions and on the global 90 m partition at
1400 square degrees (a part of a divided unit takes four digits that join the run of codes it holds, e.g. 9112;
the islands read like the 30 m products, with the continent in the id).

    python3 test_region_numbering.py

A region that is a whole Level-02 unit takes the two digits of the unit; a part of a divided unit takes
four digits -- the unit's two, then the last digit of the smallest and of the largest Level-03 code in it,
so that the id names the run of units it holds (9112 is {911, 912}; a part holding one unit repeats its
digit, 5699); the groups astride the antimeridian become 10001, 10002; an island group becomes
20000 + its Level-01 region * 100 + its number within that region, the largest first.  A partition with a
divided Level-03 unit, a part whose codes are not a run, or a region of pieces keeps the build ids
(level3 * 100 + n).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fd1_partition import (RegionBuild, _renumber_the_regions_for_publication, region_row_order_key,
                           ISLAND_GROUP_CONTINENTS)

# The regions of the global 90 m partition at 1400 square degrees as fd1.5 builds them: the build id, the
# land it holds, and the whole-degree window of the groups that carry no code.  Frozen here rather than read
# back from the published table, which carries the published numbering (11 in that file is a
# Level-02 unit, not a Level-03 code).
CODED_BUILD_IDS = [
    11101, 12101, 13101, 14101, 15101, 16101, 17101, 18101, 21101, 22101, 23101, 24101, 25101, 26101, 27101,
    28101, 29101, 31101, 32101, 33101, 34101, 36101, 41101, 42101, 43101, 44101, 45101, 46101,
    47101, 48101, 49101, 51101, 52101, 53101, 54101, 55101, 57101, 61101, 62101, 63101, 64101,
    65101, 66101, 67101, 71101, 72101, 73101, 74101, 75101, 76101, 78101, 81101, 82101, 83101,
    84101, 85101, 86101,
]
# The four Level-02 units the 1400 square degree capacity cannot hold, each divided into runs of consecutive
# Level-03 codes (measured on the real group table: 35 into 351..353 of 1071 blocks and
# 354..356 of 1035, 56 into 561..568 of 1364 and 569 of 30, 77 into 771 of 1036 and 772..774 of 460, 91 into
# 911..912 of 989 and 913..914 of 1116).  The keeper of a part is its smallest code's region, so the build
# id of a part is that code * 100 + 1.
DIVIDED_UNITS = {
    35101: [351, 352, 353], 35401: [354, 355, 356],
    56101: [561, 562, 563, 564, 565, 566, 567, 568], 56901: [569],
    77101: [771], 77201: [772, 773, 774],
    91101: [911, 912], 91301: [913, 914],
}
# build id: (land pixels, the basins it holds, minlon, minlat, maxlon, maxlat)
UNCODED_BUILD = {
    100001: (19168845, 0, -185.0, 64.0, -177.0, 72.0),
    100002: (9650, 0, -181.0, -17.0, -179.0, -16.0),
    200001: (2159235, 29529, -174.0, 1.0, -154.0, 27.0),
    200002: (166162, 6709, -179.0, -45.0, -175.0, -29.0),
    200003: (1449375, 52827, 51.0, -54.0, 78.0, -37.0),
    200004: (607082, 17349, 55.0, -22.0, 73.0, -3.0),
    200005: (499754, 14668, -26.0, 14.0, -22.0, 18.0),
    200006: (800259, 26581, -39.0, -60.0, -26.0, -53.0),
    200007: (575928, 40769, -180.0, -23.0, -154.0, 1.0),
    200008: (545958, 101265, -154.0, -28.0, -134.0, -7.0),
    200009: (208512, 9028, 76.0, 74.0, 83.0, 80.0),
    200010: (351105, 12444, -32.0, 36.0, -24.0, 40.0),
    200011: (6901, 1004, -131.0, -26.0, -124.0, -23.0),
    200012: (26752, 5168, 97.0, -1.0, 117.0, 10.0),
    200013: (71122, 2908, 37.0, -47.0, 51.0, -45.0),
    200014: (19374, 2153, 96.0, -13.0, 106.0, -10.0),
    200015: (22107, 1065, -110.0, -28.0, -105.0, -26.0),
    200016: (28621, 1733, -30.0, -21.0, -5.0, -7.0),
    200017: (10431, 1964, -80.0, 15.0, -64.0, 33.0),
    200018: (37900, 2326, -13.0, -55.0, 4.0, -37.0),
    200019: (4380, 719, 162.0, -19.0, 165.0, -8.0),
    200020: (41057, 24090, 71.0, -1.0, 74.0, 13.0),
    200021: (1146, 433, -179.0, 27.0, -175.0, 29.0),
    200022: (8181, 869, 156.0, 76.0, 159.0, 78.0),
    200023: (1176, 246, -16.0, 79.0, -15.0, 80.0),
}

FAILED = []


class DegreeGrid:
    """a stand-in for the run's grid: in these checks a region's rectangle already holds degrees, in the
    shape of a pixel rectangle (row_min, row_max, col_min, col_max) = (minlat, maxlat, minlon, maxlon)"""

    def window_of_rectangle(self, row_min, row_max, col_min, col_max):
        return (row_min, row_max, col_min, col_max)

    def pixel_box_lon_lat(self, row_min, row_max, col_min, col_max):
        return (col_min, row_min, col_max, row_max)


GRID = DegreeGrid()


def check(what, got, want):
    if got != want:
        FAILED.append("%s: got %r, expected %r" % (what, got, want))
        print("  FAIL  %-52s got %r, expected %r" % (what, got, want))
    else:
        print("  ok    %-52s %r" % (what, got))


def region_of(build_id, level3, kind=1, cut_basin_id=0, pixels=0, box=None, basins=0, members=None):
    region = RegionBuild(build_id, level3, kind)
    if members is not None:
        region.level3_members = sorted(members)
    region.cut_basin_id = cut_basin_id
    region.pixels = pixels
    region.basins = np.zeros(basins, np.int64)
    if box is not None:
        minlon, minlat, maxlon, maxlat = box
        region.rectangle = (minlat, maxlat, minlon, maxlon)
    return region


def island_of(build_id, shift=0.0, basins=None):
    pixels, held, minlon, minlat, maxlon, maxlat = UNCODED_BUILD[build_id]
    kind = 4 if build_id < 200000 else 3
    return region_of(build_id, 0, kind=kind, pixels=pixels, basins=held if basins is None else basins,
                     box=(minlon + shift, minlat + shift, maxlon + shift, maxlat + shift))


def a_unit_that_is_one_region_takes_its_two_digits():
    print("a Level-02 unit that is one region")
    regions = [region_of(11101, 111), region_of(12101, 121)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids changed", changed, True)
    check("the two units", [r.region_id for r in regions], [11, 12])


def a_divided_unit_gives_its_parts_four_digits():
    print("a Level-02 unit divided into parts")
    regions = [region_of(91101, 911, members=[911, 912]), region_of(91301, 913, members=[913, 914]),
               region_of(36101, 361)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids changed", changed, True)
    check("the two parts and the whole unit", [r.region_id for r in regions], [9112, 9134, 36])


def a_part_holding_one_unit_repeats_its_digit():
    print("a part that holds one Level-03 unit")
    regions = [region_of(56101, 561, members=[561, 562, 563, 564, 565, 566, 567, 568]),
               region_of(56901, 569, members=[569])]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids changed", changed, True)
    check("the eight-unit part and the one-unit part", [r.region_id for r in regions], [5618, 5699])


def a_part_whose_codes_are_not_a_run_keeps_the_build_ids():
    print("a part whose Level-03 codes are not a run")
    regions = [region_of(35101, 351, members=[351, 353]), region_of(35201, 352, members=[352, 354, 355, 356])]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids were left alone", changed, False)
    check("the build ids", [r.region_id for r in regions], [35101, 35201])


def an_island_carries_its_level1_and_its_size():
    print("the island groups of one Level-01 region")
    regions = [region_of(11101, 111), island_of(200001), island_of(200021), island_of(200007)]
    _renumber_the_regions_for_publication(regions, [], GRID, "test")
    # Hawaii 2,159,235 pixels, Samoa 575,928, Midway 1,146: the largest first
    check("the unit, Hawaii, Midway, Samoa", [r.region_id for r in regions], [11, 20501, 20503, 20502])


def a_seam_group_keeps_the_plain_numbers():
    print("the groups astride the antimeridian")
    regions = [region_of(11101, 111), island_of(100002), island_of(100001)]
    _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the unit and the two seam groups", [r.region_id for r in regions], [11, 10002, 10001])


def an_island_that_moved_keeps_the_build_ids():
    print("an island group whose window has moved")
    regions = [region_of(11101, 111), island_of(200001, shift=5.0)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids were left alone", changed, False)
    check("the build ids", [r.region_id for r in regions], [11101, 200001])


def an_island_not_in_the_table_keeps_the_build_ids():
    print("an island group that is not in the table")
    regions = [region_of(11101, 111), region_of(200099, 0, kind=3, pixels=10, box=(0.0, 0.0, 1.0, 1.0), basins=3)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids were left alone", changed, False)


def an_island_that_holds_other_basins_keeps_the_build_ids():
    print("an island group that holds other basins than the table was made on")
    regions = [region_of(11101, 111), island_of(200001, basins=29528)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("one basin fewer: the ids were left alone", changed, False)
    check("the build ids", [r.region_id for r in regions], [11101, 200001])
    regions = [region_of(11101, 111), island_of(200001)]
    regions[1].pixels -= 1                                  # the same basins over less land
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("one land pixel fewer: the ids were left alone", changed, False)


def a_failure_after_the_numbering_leaves_every_field_alone():
    """the only refusal left after the ids are worked out is a Level-01 region with more than 99 island groups;
    with the real table it cannot happen, so the table is stood in for here"""
    print("a refusal after the numbering has started")
    import fd1_partition
    kept = fd1_partition.ISLAND_GROUP_CONTINENTS
    pixels, basins, minlon, minlat, maxlon, maxlat = (100, 1, 0.0, 0.0, 1.0, 1.0)
    made_up = {200000 + n: (5, basins, pixels, minlon, minlat, maxlon, maxlat, "made up") for n in range(1, 101)}
    fd1_partition.ISLAND_GROUP_CONTINENTS = made_up
    try:
        regions = [region_of(11101, 111)]
        regions += [region_of(200000 + n, 0, kind=3, pixels=pixels, basins=basins, box=(minlon, minlat, maxlon, maxlat))
                    for n in range(1, 101)]
        before = [r.region_id for r in regions]
        changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
        check("the ids were left alone", changed, False)
        check("not one of the %d regions was renumbered" % len(regions), [r.region_id for r in regions], before)
        check("no Level-01 was written", sorted({r.level1_code for r in regions[1:]}), [0])
    finally:
        fd1_partition.ISLAND_GROUP_CONTINENTS = kept


def an_island_carries_the_level1_it_is_numbered_under():
    print("the Level-01 column of an island group")
    regions = [region_of(11101, 111), island_of(200001), island_of(200023)]
    _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("Hawaii's column", (regions[1].region_id, regions[1].level1_code), (20501, 5))
    check("north-east Greenland's column", (regions[2].region_id, regions[2].level1_code), (20901, 9))


def a_divided_level3_unit_keeps_the_build_ids():
    print("a Level-03 unit divided into parts")
    regions = [region_of(62201, 622), region_of(62202, 622), region_of(61101, 611)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids were left alone", changed, False)
    check("the build ids", [r.region_id for r in regions], [62201, 62202, 61101])


def a_region_of_pieces_keeps_the_build_ids():
    print("a region holding the pieces of a cut basin")
    regions = [region_of(62201, 622, kind=5, cut_basin_id=7), region_of(61101, 611)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids were left alone", changed, False)
    check("the build ids", [r.region_id for r in regions], [62201, 61101])


def a_region_of_more_than_one_unit_keeps_the_build_ids():
    print("a region that took in another Level-02 unit")
    regions = [region_of(11101, 111), region_of(21101, 211)]
    regions[0].level2_code = 0                      # what a join across the units leaves behind
    regions[0].level1_code = 0
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    check("the ids were left alone", changed, False)
    check("the build ids", [r.region_id for r in regions], [11101, 21101])


def the_merge_records_follow_the_region_that_was_kept():
    print("the merge records")
    regions = [region_of(11101, 111)]
    merges = [{"kept_region_id": 11101, "absorbed_region_id": 11201}]
    _renumber_the_regions_for_publication(regions, merges, GRID, "test")
    check("the kept id is the published one", merges[0]["kept_region_id"], 11)
    check("the absorbed id is left as it was built", merges[0]["absorbed_region_id"], 11201)


def the_rows_are_ordered_by_the_unit():
    print("the order of the rows")
    ids = [11, 34, 3513, 3546, 36, 91, 9112, 9134, 10001, 20101, 20501]
    check("a part sits where its unit sits", sorted(ids, key=region_row_order_key), ids)


def the_table_of_islands_is_whole():
    print("the table of island groups")
    check("one row per island group of the run",
          sorted(ISLAND_GROUP_CONTINENTS), sorted(k for k in UNCODED_BUILD if k >= 200000))
    check("every row names a Level-01 region",
          sorted({row[0] for row in ISLAND_GROUP_CONTINENTS.values()}), [1, 2, 3, 4, 5, 6, 7, 9])
    by_hand = sorted(k for k, row in ISLAND_GROUP_CONTINENTS.items() if "by hand" in row[7])
    check("the three decided by hand", by_hand, [200010, 200011, 200016])


def the_real_global_partition_comes_out_as_specified():
    print("the global 90 m partition at 1400 square degrees")
    regions = [region_of(build_id, build_id // 100) for build_id in CODED_BUILD_IDS]
    regions += [region_of(build_id, build_id // 100, members=members) for build_id, members in sorted(DIVIDED_UNITS.items())]
    regions += [island_of(build_id) for build_id in sorted(UNCODED_BUILD)]
    changed = _renumber_the_regions_for_publication(regions, [], GRID, "test")
    published = [r.region_id for r in sorted(regions, key=lambda r: region_row_order_key(r.region_id))]
    check("the ids changed", changed, True)
    check("every id is used once", len(set(published)), len(published))
    check("the whole units", sum(1 for r in published if r < 100), 57)
    check("the parts of the divided units", [r for r in published if 100 <= r < 10000],
          [3513, 3546, 5618, 5699, 7711, 7724, 9112, 9134])
    check("the groups astride the antimeridian", [r for r in published if 10000 < r < 20000], [10001, 10002])
    check("Africa's islands", [r for r in published if 20100 < r < 20200], [20101, 20102, 20103, 20104, 20105, 20106])
    check("Europe's island", [r for r in published if 20200 < r < 20300], [20201])
    check("Oceania's islands", [r for r in published if 20500 < r < 20600],
          [20501, 20502, 20503, 20504, 20505, 20506, 20507, 20508, 20509])
    check("Greenland's island", [r for r in published if 20900 < r < 21000], [20901])
    ids = {r.region_id for r in regions}
    check("Hawaii is the first of Oceania", [r.region_id for r in regions if r.region_id == 20501], [20501])
    check("Kerguelen is the first of Africa",
          next(r.region_id for r in regions if r.pixels == UNCODED_BUILD[200003][0]), 20101)
    check("the Azores are Europe's", next(r.region_id for r in regions if r.pixels == UNCODED_BUILD[200010][0]), 20201)
    check("Pitcairn is Oceania's", next(r.region_id for r in regions if r.pixels == UNCODED_BUILD[200011][0]), 20507)
    check("St Helena is Africa's", next(r.region_id for r in regions if r.pixels == UNCODED_BUILD[200016][0]), 20106)
    check("nothing is longer than five digits", max(len(str(i)) for i in ids), 5)


def main():
    a_unit_that_is_one_region_takes_its_two_digits()
    a_divided_unit_gives_its_parts_four_digits()
    a_part_holding_one_unit_repeats_its_digit()
    a_part_whose_codes_are_not_a_run_keeps_the_build_ids()
    an_island_carries_its_level1_and_its_size()
    a_seam_group_keeps_the_plain_numbers()
    an_island_that_moved_keeps_the_build_ids()
    an_island_not_in_the_table_keeps_the_build_ids()
    an_island_that_holds_other_basins_keeps_the_build_ids()
    a_failure_after_the_numbering_leaves_every_field_alone()
    an_island_carries_the_level1_it_is_numbered_under()
    a_divided_level3_unit_keeps_the_build_ids()
    a_region_of_pieces_keeps_the_build_ids()
    a_region_of_more_than_one_unit_keeps_the_build_ids()
    the_merge_records_follow_the_region_that_was_kept()
    the_rows_are_ordered_by_the_unit()
    the_table_of_islands_is_whole()
    the_real_global_partition_comes_out_as_specified()
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
