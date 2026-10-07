"""test_fd3_memory.py -- the three changes of 0.7.4 that bound the memory of FD3, checked on made-up data.

    python3 test_fd3_memory.py

1. HeldBlocks: a block comes back bit for bit from memory and from disk, in every dtype the attributes write; the
   blocks in memory never pass the budget; the scratch directory goes at the end and is cleared at the start.
2. fd_tables.read_basin_table_columns: on a made-up table read in chunks of a few rows, the columns equal those of
   read_basin_table, and every damaged table is refused with the very message read_basin_table gives, the damage
   placed across a chunk boundary.
3. sweep_main_stem_donor (int32 donor, the donor's area read from the window): the same donors as the 0.7.3 rule (an
   int64 donor and a Float32 array of the best areas) on random windows full of tied areas.

Synthetic: it checks that the code holds; no number from it goes anywhere.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from numba import njit

import fd_tables
import fd3_attributes as fd3

FAILURES = []


def check(what, condition):
    print(("ok    " if condition else "FAIL  ") + what)
    if not condition:
        FAILURES.append(what)


# =============================================================================
#  1. HeldBlocks
# =============================================================================

def blocks_of_every_kind(rng):
    """blocks as write_back holds them: the raster's own type, 512 x 512 or the narrower last block of a grid"""
    float_block = rng.standard_normal((512, 512)).astype(np.float32) * 1e5
    float_block[0, :4] = [np.nan, -0.0, np.inf, -9999.0]
    return [("float32", float_block),
            ("uint32", rng.integers(0, 2 ** 32 - 1, (512, 512), dtype=np.uint32)),
            ("uint8", rng.integers(0, 256, (512, 512), dtype=np.uint8)),
            ("uint8 narrow", rng.integers(0, 256, (512, 37), dtype=np.uint8)),
            ("float32 short", rng.standard_normal((211, 512)).astype(np.float32))]


def test_held_blocks():
    rng = np.random.default_rng(7)
    scratch = tempfile.mkdtemp(prefix="held_blocks_test_")
    try:
        directory = os.path.join(scratch, "out.tif.held_blocks")
        os.makedirs(directory)
        with open(os.path.join(directory, "block_9_9.npy"), "w") as handle:
            handle.write("left over by a run that stopped")
        for budget in (0, 3 * 2 ** 20, 2 ** 40):
            held = fd3.HeldBlocks(directory, budget)
            check("budget %d: a scratch directory left over is cleared at the start" % budget, not os.path.exists(directory))
            kinds = blocks_of_every_kind(rng)
            kept = {}
            for number, (name, block) in enumerate(kinds * 3):
                index = (number, 2 * number + 1)
                values = block.copy()
                held[index] = values
                kept[index] = block.copy()
                check("budget %d, block %s: held, the memory at most the budget or one block" % (budget, index),
                      index in held and held.memory_bytes <= max(budget, 0))
            check("budget %d: len counts the blocks in memory and on disk" % budget, len(held) == len(kinds) * 3)
            if budget == 0:
                check("budget 0: every block went to disk", len(held.memory) == 0 and len(held.on_disk) == len(kinds) * 3)
            if budget == 2 ** 40:
                check("budget 1 TB: no block went to disk", held.written_to_disk == 0 and not os.path.exists(directory))
            check("budget %d: the largest memory never passed the budget" % budget, held.largest_memory_bytes <= budget)
            try:
                held[(0, 1)] = kinds[0][1]
                check("budget %d: a block held twice is refused" % budget, False)
            except fd3.FlowDivideError:
                check("budget %d: a block held twice is refused" % budget, True)
            # half the blocks popped one by one, the other half read by items() as the end of the run does
            indices = sorted(kept)
            for index in indices[: len(indices) // 2]:
                back = held.pop(index)
                same = back.dtype == kept[index].dtype and back.shape == kept[index].shape and \
                    back.tobytes() == kept[index].tobytes()
                check("budget %d, block %s: popped back bit for bit (%s)" % (budget, index, back.dtype), same)
                check("budget %d, block %s: gone once popped" % (budget, index), index not in held)
            check("budget %d: pop of a block not held gives the default" % budget, held.pop((99, 99), None) is None)
            try:
                held.pop((99, 99))
                check("budget %d: pop of a block not held without a default raises KeyError" % budget, False)
            except KeyError:
                check("budget %d: pop of a block not held without a default raises KeyError" % budget, True)
            rest = dict(held.items())
            check("budget %d: items gives the rest" % budget, sorted(rest) == indices[len(indices) // 2:])
            check("budget %d: items gives every block bit for bit" % budget,
                  all(rest[index].tobytes() == kept[index].tobytes() and rest[index].dtype == kept[index].dtype for index in rest))
            held.clear()
            check("budget %d: clear leaves nothing, the scratch directory gone" % budget,
                  len(held) == 0 and not os.path.exists(directory))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# =============================================================================
#  2. read_basin_table_columns
# =============================================================================

def made_up_basin_table(count, rng):
    table = fd_tables.new_basin_table(count)
    area = np.sort(rng.uniform(0.001, 5000.0, count))[::-1].round(6)
    table["basin_area_km2"] = area
    table["basin_grid_count"] = np.maximum(1, (area * 100).astype(np.int64))
    table["outlet_row"] = rng.integers(10, 5000, count)
    table["outlet_col"] = rng.integers(10, 5000, count)
    table["outlet_flag"] = rng.integers(0, 2, count)
    table["basin_row_min"] = table["outlet_row"] - rng.integers(0, 10, count)
    table["basin_row_max"] = table["outlet_row"] + rng.integers(1, 10, count)
    table["basin_col_min"] = table["outlet_col"] - rng.integers(0, 10, count)
    table["basin_col_max"] = table["outlet_col"] + rng.integers(1, 10, count)
    table["outlet_lon"] = rng.uniform(-180, 180, count)
    table["outlet_lat"] = rng.uniform(-60, 85, count)
    for name in ("bbox_minlon", "bbox_minlat", "bbox_maxlon", "bbox_maxlat"):
        table[name] = rng.uniform(-60, 60, count)
    table["level3_id"] = rng.integers(1, 999, count)
    table["region_id"] = rng.integers(1, 99, count)
    unset = rng.random(count) < 0.05                  # some boxes not set yet
    for name in ("basin_row_min", "basin_row_max", "basin_col_min", "basin_col_max"):
        table.loc[unset, name] = -1
    return table


def message_of(reader):
    try:
        reader()
        return None
    except (fd_tables.TableError, ValueError) as error:
        return "%s: %s" % (type(error).__name__, error)


def damaged(path, damage):
    """the table's text damaged by damage(lines), its .done marker kept"""
    with open(path) as handle:
        lines = handle.read().split("\n")
    damage(lines)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def set_field(lines, row, column, text):
    """row 1 is the first row after the header"""
    words = lines[row].split(" ")
    words[fd_tables.BASIN_TABLE_COLUMNS.index(column)] = text
    lines[row] = " ".join(words)


def test_read_basin_table_columns():
    rng = np.random.default_rng(11)
    scratch = tempfile.mkdtemp(prefix="basin_table_test_")
    try:
        path = os.path.join(scratch, "global", "table", "basin_table_fine_test.csv")
        table = made_up_basin_table(10007, rng)
        fd_tables.write_basin_table(table, path, "fd1.6")
        whole = fd_tables.read_basin_table(path)
        for chunk_rows in (1, 7, 1000, 10007, 2000000):
            part = fd_tables.read_basin_table_columns(path, fd3.PARTITION_BASIN_COLUMNS, chunk_rows=chunk_rows)
            check("chunks of %d rows: the columns kept are the ones asked for, in the table's order" % chunk_rows,
                  list(part.columns) == [name for name in fd_tables.BASIN_TABLE_COLUMNS if name in fd3.PARTITION_BASIN_COLUMNS])
            check("chunks of %d rows: every value and dtype equal to read_basin_table" % chunk_rows,
                  all(part[name].dtype == whole[name].dtype and np.array_equal(part[name].to_numpy(), whole[name].to_numpy())
                      for name in part.columns))
        # the damages, each across the boundary of chunks of 1000 rows (rows 1000 and 1001)
        damages = {
            "basin_id out of order at a chunk boundary": lambda lines: (set_field(lines, 1000, "basin_id", "1001"), set_field(lines, 1001, "basin_id", "1000")),
            "area rising at a chunk boundary": lambda lines: set_field(lines, 1001, "basin_area_km2", "%.6f" % 9999.0),
            "a fraction in an integer column in a later chunk": lambda lines: set_field(lines, 5001, "outlet_row", "12.5"),
            "a word in an integer column": lambda lines: set_field(lines, 9999, "level3_id", "abc"),
            "inf in a float column": lambda lines: set_field(lines, 3001, "outlet_lon", "inf"),
            "an id past 2^32 - 1": lambda lines: set_field(lines, 7001, "region_id", "4294967296"),
            "a negative id in a column the partition does not read": lambda lines: set_field(lines, 1001, "global_basin_id", "-3"),
            "basin_nrow not the box": lambda lines: set_field(lines, 1001, "basin_nrow", "999999"),
            "an outlet outside its box": lambda lines: set_field(lines, 1001, "outlet_row", "999999"),
            "a row cut off the end": lambda lines: lines.pop(len(lines) - 2),
            "a header column renamed": lambda lines: lines.__setitem__(0, lines[0].replace("outlet_flag", "flag")),
            "a header column missing": lambda lines: lines.__setitem__(0, " ".join(lines[0].split(" ")[:-1])),
        }
        for name, damage in damages.items():
            copy = os.path.join(scratch, "damaged", "global", "table", "basin_table_fine_test.csv")
            os.makedirs(os.path.dirname(copy), exist_ok=True)
            shutil.copy(path, copy)
            shutil.copy(path + ".done", copy + ".done")
            damaged(copy, damage)
            expected = message_of(lambda: fd_tables.read_basin_table(copy))
            found = message_of(lambda: fd_tables.read_basin_table_columns(copy, fd3.PARTITION_BASIN_COLUMNS, chunk_rows=1000))
            check("%s: refused%s" % (name, "" if expected == found else "\n          read_basin_table: %s\n          the columns:      %s" % (expected, found)),
                  expected is not None and expected == found)
        with open(path + ".done") as handle:
            marker = handle.read()
        for promised in (10008, 4000000000):
            with open(path + ".done", "w") as handle:
                handle.write(marker.replace("basins=10007", "basins=%d" % promised))
            check("a marker promising %d rows for 10007: refused as read_basin_table refuses" % promised,
                  message_of(lambda: fd_tables.read_basin_table(path)) is not None and
                  message_of(lambda: fd_tables.read_basin_table(path)) == message_of(lambda: fd_tables.read_basin_table_columns(path, ["basin_id"])))
        os.remove(path + ".done")
        check("no marker: refused as read_basin_table refuses",
              message_of(lambda: fd_tables.read_basin_table(path)) == message_of(lambda: fd_tables.read_basin_table_columns(path, ["basin_id"])))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# =============================================================================
#  3. The main-stem donor of the Hack order
# =============================================================================

@njit(cache=False)
def donor_as_in_1_0_3(order, downstream, member, channel_window, area_window, ncol, best_donor, best_area):
    """sweep_main_stem_donor of 0.7.3, word for word"""
    for position in range(order.size):
        pixel = order[position]
        if member[pixel] < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        downstream_row = downstream_pixel // ncol
        if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] == 0:
            continue
        area_here = area_window[row, column]
        if best_donor[downstream_pixel] < 0 or area_here > best_area[downstream_pixel]:
            best_area[downstream_pixel] = area_here
            best_donor[downstream_pixel] = pixel
        elif area_here == best_area[downstream_pixel] and pixel < best_donor[downstream_pixel]:
            best_donor[downstream_pixel] = pixel
    return 0


def test_main_stem_donor():
    rng = np.random.default_rng(3)
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], np.uint8)
    for trial in range(40):
        nrow, ncol = int(rng.integers(3, 60)), int(rng.integers(3, 60))
        directions = codes[rng.integers(0, 8, (nrow, ncol))]
        directions[rng.random((nrow, ncol)) < 0.02] = 0          # a few mouths
        directions[0, :] = 0
        downstream, order, taken, _ = fd3.window_flow_structure(directions, np.zeros((1, 1), np.uint8), False)
        if taken < 0:
            continue                                            # the random directions hold a cycle: drawn again
        pixel_count = nrow * ncol
        member = np.where(rng.random(pixel_count) < 0.9, 0, -1).astype(np.int32)
        channel = (rng.random((nrow, ncol)) < 0.7).astype(np.uint8)
        area = rng.choice(np.array([1.5, 2.25, 7.0, 7.0, 11.125], np.float32), (nrow, ncol))   # ties on purpose
        old_donor = np.full(pixel_count, -1, np.int64)
        old_area = np.zeros(pixel_count, np.float32)
        donor_as_in_1_0_3(order, downstream, member, channel, area, ncol, old_donor, old_area)
        new_donor = np.full(pixel_count, -1, np.int32)
        fd3.sweep_main_stem_donor(order, downstream, member, channel, area, ncol, new_donor)
        check("random window %d (%d x %d): the same main-stem donors as 0.7.3" % (trial, nrow, ncol),
              np.array_equal(old_donor, new_donor.astype(np.int64)))


if __name__ == "__main__":
    test_held_blocks()
    test_read_basin_table_columns()
    test_main_stem_donor()
    print("%d failures" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
