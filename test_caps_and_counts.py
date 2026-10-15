"""The limits the package shares with the C code, on made-up values: the upstream count read from a raster (whole
counts, halves up), the top tier of the capacities, the highest stream order (254) and the tiled distance's nodata
(-1).  Synthetic: it checks that the code holds; no number from it goes anywhere.

    python3 test_caps_and_counts.py
"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import fd1_partition as fd1
import fd3_attributes as fd3
import fd_tables
import tile_kernels

FAILED = []


def check(what, got, want):
    if got != want:
        FAILED.append("%s: got %r, expected %r" % (what, got, want))
        print("  FAIL  %-58s got %r, expected %r" % (what, got, want))
    else:
        print("  ok    %-58s %r" % (what, got))


def the_upstream_count():
    print("an upstream count read from a raster")
    for value, want in ((1.0, 1), (1.4, 1), (1.5, 2), (1.6, 2), (2.5, 3), (0.9, 0), (0.0, 0), (-1.0, 0),
                        (float("nan"), 0), (9.0e18, 0), (1.79e308, 0), (7, 7)):
        check("acc_count(%r)" % value, fd1.acc_count(value), want)
    floats = np.array([1.0, 1.6, 2.5, 0.4, -9999.0, np.nan, 3.0e18], np.float32)
    check("acc_counts on Float32", fd1.acc_counts(floats).tolist(), [fd1.acc_count(v) for v in floats.tolist()])
    whole = np.array([0, 1, 5, 4294967295], np.uint32)
    check("acc_counts on UInt32", fd1.acc_counts(whole).tolist(), [0, 1, 5, 4294967295])
    check("acc_counts gives int64", fd1.acc_counts(whole).dtype, np.dtype(np.int64))
    big = np.array([9007199254740993], np.int64)              # 2^53 + 1: a double holds 2^53
    check("an Int64 count read as a double", fd1.acc_counts(big).tolist(), [fd1.acc_count(int(big[0]))])


def the_top_tier():
    print("the top tier of the capacities, 2^31")
    check("3600-pixel blocks", fd1.top_tier_blocks(3600), 160)
    check("1200-pixel blocks", fd1.top_tier_blocks(1200), 1400)
    key = fd_tables.region_map_key(160 * 3600 * 3600, 3600, "l3")
    check("the key of 2^31 on 3600-pixel blocks", fd1.level3_top_tier_key(key), key)
    check("the same from another key on that block",
          fd1.level3_top_tier_key(fd_tables.region_map_key(80 * 3600 * 3600, 3600, "hilbert")), key)
    for what, text in (("a key without a block", "capacity_px=1,groups=l3"), ("a block alone", "block_px=3600"),
                       ("a key with a field more", key + ",index_bits=31"),
                       ("a block of which 2^31 holds none", "capacity_px=0,block_px=50000,groups=l3")):
        try:
            fd1.level3_top_tier_key(text)
            check(what, "accepted", "refused")
        except fd1.FlowDivideError:
            check(what, "refused", "refused")


def the_highest_order():
    print("the highest stream order, %d" % fd1.ORDER_CAP)
    check("ORDER_CAP", fd1.ORDER_CAP, 254)
    # Hack: a chain of channel pixels, each the smaller of two donors, so each takes one more than the pixel below
    for length, want in ((254, 0), (255, 1)):
        value = np.zeros(length, np.uint8)
        value[0] = 1
        status = fd3.sweep_hack_order(length, np.arange(length, dtype=np.int64), np.zeros(length, np.int64),
                                      np.arange(length, dtype=np.int64) - 1, value, np.full(length, -1, np.int64),
                                      np.zeros(1, np.int64), np.zeros(1, np.int64))
        check("Hack, a chain of %d pixels (the top one %d)" % (length, length), status, want)
    # Hack: an order of 255 carried in at the outlet and copied up the main stem
    value = np.zeros(2, np.uint8)
    value[0] = 255
    status = fd3.sweep_hack_order(2, np.arange(2, dtype=np.int64), np.zeros(2, np.int64), np.arange(2, dtype=np.int64) - 1,
                                  value, np.array([1, -1], np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64))
    check("Hack, 255 carried in at the outlet", status, 1)
    # Strahler: one order of 255 arrives, alone
    status = fd3.sweep_strahler_order(1, np.zeros(1, np.int64), np.full(1, -1, np.int64), np.zeros(1, np.uint8),
                                      np.full(1, 255, np.uint8), np.zeros(1, np.uint8), np.zeros(1, np.int64),
                                      np.zeros(1, np.int64))
    check("Strahler, 255 arrives alone", status, 1)
    # Strahler: two orders arrive at a pixel, each of 253 or 254
    for arriving, want in ((253, 0), (254, 1)):
        status = fd3.sweep_strahler_order(1, np.zeros(1, np.int64), np.full(1, -1, np.int64), np.zeros(1, np.uint8),
                                          np.full(1, arriving, np.uint8), np.full(1, arriving, np.uint8),
                                          np.zeros(1, np.int64), np.zeros(1, np.int64))
        check("Strahler, two of %d arrive" % arriving, status, want)


def the_tiled_distance():
    print("the tiled distance to the outlet")
    default = inspect.signature(tile_kernels.ldn_on_tiles).parameters["nodata"].default
    check("its nodata is fd3's", default, fd3.ATTRIBUTES["ldn"]["nodata"])


if __name__ == "__main__":
    the_upstream_count()
    the_top_tier()
    the_highest_order()
    the_tiled_distance()
    print()
    print("ALL PASSED" if not FAILED else "%d FAILED" % len(FAILED))
    sys.exit(1 if FAILED else 0)
