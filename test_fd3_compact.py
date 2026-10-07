"""test_fd3_compact.py -- the numbering of 0.7.7 (fd3_attributes section [2c]) against the whole-window order of 0.7.6.

    python3 test_fd3_compact.py

On random windows of flow directions (no cycle by construction: every pixel flows to its lowest lower neighbour of a
random surface, or ends as a mouth or a sink, or leaves the window), with random members' outlets, outlets of child
pieces another region works, and, for the network, a channel mask made from the upstream count:

1. the structure: the pixels numbered are exactly the pixels 0.7.6 gives a member (window_flow_structure and
   window_member_of_pixel, which stay in the package for the registered rules), each with the same member, each
   numbered after the pixel it flows into, linked to the same downstream pixel; the outlets outside the order last;
2. the six sweeps: the 0.7.6 kernels, copied here as they were, over the whole window, and the 0.7.7 kernels over
   the numbered pixels, from the same starting values (values handed across cuts put at random pixels), give the same
   values bit for bit and the same counts, sources, maxima, heads and donors;
3. the refusals: a cycle (one through the outlet of a child of another region too), two members on one outlet, a member
   numbered before the member it flows into, more pixels than the capacity.

Synthetic: it checks that the code holds; no number from it goes anywhere.
"""
import sys

import numpy as np
from numba import njit

import fd3_attributes as fd3
from fd1_partition import DROW, DCOL, IS_LAND

FAILURES = []


def check(what, condition):
    print(("ok    " if condition else "FAIL  ") + what)
    if not condition:
        FAILURES.append(what)


# =============================================================================
#  The 0.7.6 sweeps over the whole window, as they were (fd3_attributes 0.7.6, section [2b])
# =============================================================================

@njit(cache=True)
def ref_sweep_distance_to_outlet(order, downstream, member, lengths, value, ncol,
                             visited_of_member, farthest_of_member, head_pixel_of_member):
    """The labelling class: a pixel's distance is the distance of the pixel it flows into plus the step
    between them.  The order is swept downstream first, so the value a pixel needs is already there; an
    outlet of ours whose parent is in another region was given its value before the sweep."""
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        downstream_pixel = downstream[pixel]
        row = pixel // ncol
        if downstream_pixel >= 0 and member[downstream_pixel] >= 0:
            downstream_row = downstream_pixel // ncol
            step = fd3._step_length(lengths, row, downstream_row - row,
                                (downstream_pixel - downstream_row * ncol) - (pixel - row * ncol))
            value[pixel] = value[downstream_pixel] + step
        visited_of_member[mine] += 1
        # the farthest pixel of the member, and on a tie the smaller row and then the smaller column,
        # which is the smaller index in this window: the rule compares the pixel's place in the rectangle
        # it read, and on a periodic grid that rectangle is unrolled, so folding the column back onto
        # the grid would pick the other pixel at the seam.  A member
        # of one pixel takes that pixel as its head
        if value[pixel] > farthest_of_member[mine] or head_pixel_of_member[mine] < 0:
            farthest_of_member[mine] = value[pixel]
            head_pixel_of_member[mine] = pixel
        elif value[pixel] == farthest_of_member[mine] and pixel < head_pixel_of_member[mine]:
            head_pixel_of_member[mine] = pixel
    return 0


@njit(cache=True)
def ref_sweep_maximum_length(order, downstream, member, lengths, value, ncol, visited_of_member,
                         value_at_outlet_of_member, outlet_pixels):
    """The maximum class: a pixel hands its own value plus the step to the pixel it flows into, which keeps
    the largest of what arrives.  Swept upstream first; a divide keeps the zero it starts with."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        row = pixel // ncol
        downstream_row = downstream_pixel // ncol
        step = fd3._step_length(lengths, row, downstream_row - row,
                            (downstream_pixel - downstream_row * ncol) - (pixel - row * ncol))
        candidate = value[pixel] + step
        if candidate > value[downstream_pixel]:
            value[downstream_pixel] = candidate
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


@njit(cache=True)
def ref_sweep_shreve_magnitude(order, downstream, member, channel_window, value, ncol, is_outlet,
                           visited_of_member, sources_of_member, value_at_outlet_of_member, outlet_pixels):
    """The accumulation class on the channel network: a channel head counts 1 and every other channel
    pixel counts what arrives at it.  Swept upstream first, so what arrives at a pixel is complete when
    the pixel is reached; `value` carries the arriving sum until the pixel is finished and its own
    magnitude afterwards."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        if value[pixel] == 0:
            value[pixel] = 1                                # a channel head
            sources_of_member[mine] += 1
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        downstream_row = downstream_pixel // ncol
        if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] == 0:
            if is_outlet[pixel] == 0:
                return 3                                    # the channel mask breaks along the flow
            continue
        if value[downstream_pixel] > 4294967295 - value[pixel]:
            return 1
        value[downstream_pixel] += value[pixel]
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


@njit(cache=True)
def ref_sweep_main_stem_donor(order, downstream, member, channel_window, area_window, ncol, best_donor):
    """Which channel pixel that flows into a channel pixel carries the most upstream area, and so keeps
    the Hack order of the pixel below.  A tie goes to the smaller index in this window, which is the
    the rule (it compares the donor's place in the rectangle it read; for two neighbours of one pixel
    the row decides, and the rectangle is at least three columns wide).  One sweep upstream first;
    nothing depends on the order here, but the sweep is the cheapest way over the member's pixels.
    best_donor is int32 (the window holds at most 2^31 - 1 pixels), and the best donor's area is read from
    the window where it lies, the same Float32 value an array of the best areas would hold: 8 bytes a pixel
    fewer than an int64 donor and a Float32 area beside it."""
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
        best = best_donor[downstream_pixel]
        if best < 0:
            best_donor[downstream_pixel] = pixel
            continue
        best_row = best // ncol
        best_area = area_window[best_row, best - best_row * ncol]
        if area_here > best_area:
            best_donor[downstream_pixel] = pixel
        elif area_here == best_area and pixel < best:
            best_donor[downstream_pixel] = pixel
    return 0


@njit(cache=True)
def ref_sweep_hack_order(order, downstream, member, channel_window, value, ncol, best_donor, visited_of_member,
                     largest_of_member):
    """The labelling class on the channel network: the pixel that keeps the most upstream area keeps the
    order of the pixel below it, every other channel pixel takes one more.  Swept downstream first, so the
    order a pixel reads is already final; a member's outlet was given its order before the sweep."""
    for position in range(order.size - 1, -1, -1):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        downstream_pixel = downstream[pixel]
        if downstream_pixel >= 0 and member[downstream_pixel] >= 0:
            downstream_row = downstream_pixel // ncol
            if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] != 0:
                if best_donor[downstream_pixel] == pixel:
                    value[pixel] = value[downstream_pixel]
                elif value[downstream_pixel] >= 255:
                    return 1                                # an order the raster cannot carry
                else:
                    value[pixel] = value[downstream_pixel] + 1
        if value[pixel] == 0:
            return 2                                        # nothing gave this channel pixel an order
        visited_of_member[mine] += 1
        if value[pixel] > largest_of_member[mine]:
            largest_of_member[mine] = value[pixel]
    return 0


@njit(cache=True)
def ref_sweep_strahler_order(order, downstream, member, channel_window, value, ncol, is_outlet,
                         arriving_largest, arriving_second, visited_of_member, sources_of_member,
                         value_at_outlet_of_member, outlet_pixels):
    """The stream-order class, on the channel pixels only: a pixel's order is the largest order that
    arrives, plus one when that largest arrives at least twice; a channel head is 1.  Swept upstream
    first, so every order that arrives at a pixel is final when the pixel is reached."""
    for position in range(order.size):
        pixel = order[position]
        mine = member[pixel]
        if mine < 0:
            continue
        row = pixel // ncol
        column = pixel - row * ncol
        if channel_window[row, column] == 0:
            continue
        largest = arriving_largest[pixel]
        if largest == 0:
            here = 1                                        # a channel head
            sources_of_member[mine] += 1
        elif largest == arriving_second[pixel]:
            here = largest + 1
            if here > 255:
                return 1                                    # an order the raster cannot carry
        else:
            here = largest
        value[pixel] = here
        visited_of_member[mine] += 1
        downstream_pixel = downstream[pixel]
        if downstream_pixel < 0 or member[downstream_pixel] < 0:
            continue
        downstream_row = downstream_pixel // ncol
        if channel_window[downstream_row, downstream_pixel - downstream_row * ncol] == 0:
            if is_outlet[pixel] == 0:
                return 3                                    # the channel mask breaks along the flow
            continue
        if here > arriving_largest[downstream_pixel]:
            arriving_second[downstream_pixel] = arriving_largest[downstream_pixel]
            arriving_largest[downstream_pixel] = here
        elif here > arriving_second[downstream_pixel]:
            arriving_second[downstream_pixel] = here
    for index in range(outlet_pixels.size):
        value_at_outlet_of_member[index] = value[outlet_pixels[index]]
    return 0


# =============================================================================
#  Random windows
# =============================================================================

CODES = [(1, 0, 1), (2, 1, 1), (4, 1, 0), (8, 1, -1), (16, 0, -1), (32, -1, -1), (64, -1, 0), (128, -1, 1)]


def random_window(rng, nrow, ncol):
    """flow directions without a cycle: each land pixel flows to its lowest lower neighbour of a random surface; a pixel
    with none is a mouth, a sink, or, on the edge, flows out of the window; a few pixels are no data"""
    surface = rng.random((nrow, ncol))
    directions = np.zeros((nrow, ncol), np.uint8)
    for row in range(nrow):
        for column in range(ncol):
            best = None
            for code, row_step, column_step in CODES:
                r, c = row + row_step, column + column_step
                if 0 <= r < nrow and 0 <= c < ncol and surface[r, c] < surface[row, column]:
                    if best is None or surface[r, c] < surface[best[1], best[2]]:
                        best = (code, r, c)
            if best is not None and rng.random() < 0.97:
                directions[row, column] = best[0]
                continue
            outward = [code for code, row_step, column_step in CODES
                       if not (0 <= row + row_step < nrow and 0 <= column + column_step < ncol)]
            draw = rng.random()
            if outward and draw < 0.5:
                directions[row, column] = outward[int(rng.integers(len(outward)))]
            else:
                directions[row, column] = 0 if draw < 0.8 else 255       # a mouth, or an inland sink
    directions[rng.random((nrow, ncol)) < 0.04] = 247                    # no data
    return directions


def upstream_count(directions):
    downstream, order, taken, _ = fd3.window_flow_structure(directions, np.zeros((1, 1), np.uint8), False)
    count = np.zeros(directions.size, np.int64)
    for pixel in order:
        count[pixel] += 1
        if downstream[pixel] >= 0:
            count[downstream[pixel]] += count[pixel]
    return count.reshape(directions.shape)


def flows_into(directions, mask, use_mask, pixel):
    return fd3._flows_into(directions, mask, use_mask, directions.shape[0], directions.shape[1], pixel)


def outlet_order_of(directions, mask, use_mask, outlets):
    """every outlet after the outlets below it on its way down: by the number of outlets below it"""
    is_outlet = set(int(p) for p in outlets)
    depth = []
    for pixel in outlets:
        below = 0
        walking = flows_into(directions, mask, use_mask, int(pixel))
        steps = 0
        while walking >= 0 and steps < directions.size:
            below += walking in is_outlet
            walking = flows_into(directions, mask, use_mask, walking)
            steps += 1
        depth.append(below)
    return np.asarray(sorted(range(len(outlets)), key=lambda i: (depth[i], i)), np.int64)


def a_case(rng, use_mask):
    nrow, ncol = int(rng.integers(3, 70)), int(rng.integers(3, 70))
    directions = random_window(rng, nrow, ncol)
    land = np.nonzero(IS_LAND[directions.reshape(-1)])[0]
    if land.size < 3:
        return None
    count = upstream_count(directions)
    mask = (count >= int(rng.integers(1, 6))).astype(np.uint8) if use_mask else np.zeros((1, 1), np.uint8)
    picked = rng.choice(land, size=min(land.size, int(rng.integers(2, 14))), replace=False)
    split = int(rng.integers(1, len(picked)))
    outlets = np.asarray(picked[:split], np.int64)
    blocked = np.sort(np.asarray(picked[split:], np.int64))
    if use_mask and rng.random() < 0.5:
        # an outlet outside the network too (a basin under the channel threshold): a pixel of the region all the same
        off = np.nonzero((mask.reshape(-1) == 0) & (IS_LAND[directions.reshape(-1)] != 0))[0]
        off = np.setdiff1d(off, np.concatenate([outlets, blocked]))
        if off.size:
            outlets = np.append(outlets, rng.choice(off))
    return directions, mask, outlets, blocked


# =============================================================================
#  1 and 2. The structure and the sweeps
# =============================================================================

def compare(rng, use_mask, trial):
    case = a_case(rng, use_mask)
    if case is None:
        return
    directions, mask, outlets, blocked = case
    nrow, ncol = directions.shape
    pixel_count = nrow * ncol
    what = "%s window %d (%d x %d, %d outlets, %d blocked)" % ("network" if use_mask else "land", trial, nrow, ncol,
                                                              outlets.size, blocked.size)
    downstream, order, taken, broken = fd3.window_flow_structure(directions, mask, use_mask)
    member = fd3.window_member_of_pixel(order, downstream, outlets.astype(np.int32), blocked, pixel_count)
    ours = member >= 0
    in_order = np.zeros(pixel_count, bool)
    in_order[order] = True
    outlet_order = outlet_order_of(directions, mask, use_mask, outlets)
    (status, count, count_in_order, state, pixel_of_compact, member_of_compact, downstream_of_compact,
     compact_of_outlet) = fd3.compact_the_pixels_of_the_region(directions, mask, use_mask, outlets, outlet_order,
                                                               blocked, pixel_count)
    if status != 0:
        check(what + ": numbered (status %d)" % status, False)
        return
    pixel_of_compact = pixel_of_compact[:count].astype(np.int64)
    member_of_compact = member_of_compact[:count]
    downstream_of_compact = downstream_of_compact[:count]
    good = (count == int(ours.sum()) and np.array_equal(np.sort(pixel_of_compact), np.nonzero(ours)[0])
            and np.array_equal(member_of_compact, member[pixel_of_compact])
            and np.array_equal(np.sort(pixel_of_compact[:count_in_order]), np.nonzero(ours & in_order)[0])
            and np.array_equal(pixel_of_compact[compact_of_outlet], outlets)
            and bool(np.all(downstream_of_compact[count_in_order:] == -1)))
    below_ref = downstream[pixel_of_compact[:count_in_order]].astype(np.int64)
    below_ref = np.where((below_ref >= 0) & (member[np.maximum(below_ref, 0)] >= 0), below_ref, -1)
    below_new = np.where(downstream_of_compact[:count_in_order] >= 0,
                         pixel_of_compact[np.maximum(downstream_of_compact[:count_in_order], 0)], -1)
    good = good and np.array_equal(below_ref, below_new)
    good = good and bool(np.all(downstream_of_compact[:count_in_order] < np.arange(count_in_order)))
    check(what + ": the same pixels, members and links, each after the pixel below it", good)
    if not good:
        return
    lengths = rng.uniform(20.0, 130.0, (nrow, 5))
    members = outlets.size
    pick = lambda k: rng.choice(np.nonzero(ours)[0], size=min(k, int(ours.sum())), replace=False) if ours.any() else np.zeros(0, np.int64)
    compact_of = {int(p): k for k, p in enumerate(pixel_of_compact)}
    if not use_mask:
        # the distance: zeros on ours, a state at some outlets
        start = np.zeros(pixel_count, np.float64)
        for pixel in pick(3):
            start[pixel] = rng.uniform(0.0, 5000.0)
        reference = np.where(ours, start, -1.0)
        visited_ref, farthest_ref, head_ref = np.zeros(members, np.int64), np.zeros(members), np.full(members, -1, np.int64)
        ref_sweep_distance_to_outlet(order, downstream, member, lengths, reference, ncol, visited_ref, farthest_ref, head_ref)
        value = start[pixel_of_compact].copy()
        visited, farthest, head = np.zeros(members, np.int64), np.zeros(members), np.full(members, -1, np.int64)
        fd3.sweep_distance_to_outlet(count_in_order, pixel_of_compact.astype(np.int32), member_of_compact,
                                     downstream_of_compact, lengths, value, ncol, visited, farthest, head)
        check(what + ": ldn the same values, counts, farthest pixels and heads",
              np.array_equal(value.view(np.uint64), reference[pixel_of_compact].view(np.uint64))
              and np.array_equal(visited, visited_ref) and np.array_equal(farthest, farthest_ref) and np.array_equal(head, head_ref))
        # the upstream flow length: zeros on ours, values handed down at some pixels
        start = np.zeros(pixel_count, np.float64)
        for pixel in pick(3):
            start[pixel] = rng.uniform(0.0, 5000.0)
        reference = np.where(ours, start, -9999.0)
        visited_ref, at_outlet_ref = np.zeros(members, np.int64), np.zeros(members)
        ref_sweep_maximum_length(order, downstream, member, lengths, reference, ncol, visited_ref, at_outlet_ref, outlets.astype(np.int32))
        value = start[pixel_of_compact].copy()
        visited = np.zeros(members, np.int64)
        fd3.sweep_maximum_length(count_in_order, pixel_of_compact.astype(np.int32), member_of_compact,
                                 downstream_of_compact, lengths, value, ncol, visited)
        check(what + ": lup the same values, counts and values at the outlets",
              np.array_equal(value.view(np.uint64), reference[pixel_of_compact].view(np.uint64))
              and np.array_equal(visited, visited_ref) and np.array_equal(value[compact_of_outlet], at_outlet_ref))
        return
    is_outlet = np.zeros(pixel_count, np.uint8)
    is_outlet[outlets] = 1
    on_network = lambda pixel: mask.reshape(-1)[pixel] != 0
    # the Shreve magnitude: what earlier regions left at some pixels of the network
    reference = np.zeros(pixel_count, np.uint32)
    for pixel in pick(3):
        if on_network(pixel):
            reference[pixel] = rng.integers(1, 50)
    value = reference[pixel_of_compact].copy()
    visited_ref, sources_ref, at_outlet_ref = np.zeros(members, np.int64), np.zeros(members, np.int64), np.zeros(members, np.int64)
    status_ref = ref_sweep_shreve_magnitude(order, downstream, member, mask, reference, ncol, is_outlet, visited_ref, sources_ref,
                                            at_outlet_ref, outlets.astype(np.int32))
    visited, sources = np.zeros(members, np.int64), np.zeros(members, np.int64)
    status_new = fd3.sweep_shreve_magnitude(count_in_order, member_of_compact, downstream_of_compact, value, visited, sources)
    check(what + ": shv the same values, counts, sources and values at the outlets",
          status_ref == status_new == 0 and np.array_equal(value, reference[pixel_of_compact])
          and np.array_equal(visited, visited_ref) and np.array_equal(sources, sources_ref)
          and np.array_equal(value[compact_of_outlet].astype(np.int64), at_outlet_ref))
    # the Strahler order: orders arriving from earlier regions at some pixels of the network
    largest_ref, second_ref = np.zeros(pixel_count, np.uint8), np.zeros(pixel_count, np.uint8)
    for pixel in pick(4):
        if on_network(pixel):
            came = np.uint8(rng.integers(1, 6))
            if came > largest_ref[pixel]:
                second_ref[pixel], largest_ref[pixel] = largest_ref[pixel], came
            elif came > second_ref[pixel]:
                second_ref[pixel] = came
    largest, second = largest_ref[pixel_of_compact].copy(), second_ref[pixel_of_compact].copy()
    reference = np.zeros(pixel_count, np.uint8)
    visited_ref, sources_ref, at_outlet_ref = np.zeros(members, np.int64), np.zeros(members, np.int64), np.zeros(members, np.int64)
    status_ref = ref_sweep_strahler_order(order, downstream, member, mask, reference, ncol, is_outlet, largest_ref, second_ref,
                                          visited_ref, sources_ref, at_outlet_ref, outlets.astype(np.int32))
    value = np.zeros(count, np.uint8)
    visited, sources = np.zeros(members, np.int64), np.zeros(members, np.int64)
    status_new = fd3.sweep_strahler_order(count_in_order, member_of_compact, downstream_of_compact, value, largest, second,
                                          visited, sources)
    check(what + ": ord the same orders, counts, sources and orders at the outlets",
          status_ref == status_new == 0 and np.array_equal(value, reference[pixel_of_compact])
          and np.array_equal(visited, visited_ref) and np.array_equal(sources, sources_ref)
          and np.array_equal(value[compact_of_outlet].astype(np.int64), at_outlet_ref))
    # the Hack order: areas full of ties; the blocked outlets on the network are donors of the pixel they flow into
    area = rng.choice(np.array([1.5, 2.25, 7.0, 7.0, 11.125], np.float32), (nrow, ncol))
    donor_ref = np.full(pixel_count, -1, np.int32)
    ref_sweep_main_stem_donor(order, downstream, member, mask, area, ncol, donor_ref)
    donor = np.full(count, -1, np.int32)
    fd3.sweep_main_stem_donor(count_in_order, pixel_of_compact.astype(np.int32), downstream_of_compact, area.reshape(-1), donor)
    area_flat = area.reshape(-1)
    for outlet in blocked:
        inlet = flows_into(directions, np.zeros((1, 1), np.uint8), False, int(outlet))
        if inlet < 0 or not on_network(outlet) or not on_network(inlet) or member[inlet] < 0:
            continue
        for array, index in ((donor_ref, inlet), (donor, compact_of[inlet])):
            best = int(array[index])
            if best < 0 or area_flat[outlet] > area_flat[best]:
                array[index] = outlet
            elif area_flat[outlet] == area_flat[best] and outlet < best:
                array[index] = outlet
    reference = np.zeros(pixel_count, np.uint8)
    for index, pixel in enumerate(outlets):
        if on_network(pixel):
            reference[pixel] = 1 if index % 2 == 0 else rng.integers(1, 9)
    value = reference[pixel_of_compact].copy()
    visited_ref, largest_ref = np.zeros(members, np.int64), np.zeros(members, np.int64)
    status_ref = ref_sweep_hack_order(order, downstream, member, mask, reference, ncol, donor_ref, visited_ref, largest_ref)
    visited, largest = np.zeros(members, np.int64), np.zeros(members, np.int64)
    status_new = fd3.sweep_hack_order(count_in_order, pixel_of_compact.astype(np.int32), member_of_compact,
                                      downstream_of_compact, value, donor, visited, largest)
    check(what + ": hck the same donors, orders, counts and largest orders (status %d / %d)" % (status_ref, status_new),
          status_ref == status_new and np.array_equal(donor, donor_ref[pixel_of_compact])
          and (status_ref != 0 or (np.array_equal(value, reference[pixel_of_compact])
                                   and np.array_equal(visited, visited_ref) and np.array_equal(largest, largest_ref))))


# =============================================================================
#  3. The refusals
# =============================================================================

def test_refusals():
    # a 2 x 3 window: (0,0) -> (0,1) -> (0,2) a mouth; (1,0) -> (1,1) and (1,1) -> (1,0): a cycle
    directions = np.array([[1, 1, 0], [1, 16, 0]], np.uint8)
    nothing = np.zeros((1, 1), np.uint8)
    status = fd3.compact_the_pixels_of_the_region(directions, nothing, False, np.array([2], np.int64),
                                                  np.array([0], np.int64), np.zeros(0, np.int64), 6)[0]
    _, _, taken, _ = fd3.window_flow_structure(directions, nothing, False)
    check("a cycle: refused (status %d), as 0.7.6 refuses it (taken %d)" % (status, taken), status == 2 and taken < 0)
    directions = np.array([[1, 1, 0], [64, 64, 64]], np.uint8)
    status = fd3.compact_the_pixels_of_the_region(directions, nothing, False, np.array([2, 2], np.int64),
                                                  np.array([0, 1], np.int64), np.zeros(0, np.int64), 6)[0]
    check("two members on one outlet: refused (status %d)" % status, status == 3)
    # the outlet at (0,1) flows into the outlet at (0,2); numbered first, it would come before the pixel below it
    status = fd3.compact_the_pixels_of_the_region(directions, nothing, False, np.array([1, 2], np.int64),
                                                  np.array([0, 1], np.int64), np.zeros(0, np.int64), 6)[0]
    check("a member numbered before the member it flows into: refused (status %d)" % status, status == 4)
    status, count = fd3.compact_the_pixels_of_the_region(directions, nothing, False, np.array([1, 2], np.int64),
                                                         np.array([1, 0], np.int64), np.zeros(0, np.int64), 6)[:2]
    check("the same members in the right order: numbered, all six pixels (%d)" % count, status == 0 and count == 6)
    status = fd3.compact_the_pixels_of_the_region(directions, nothing, False, np.array([2], np.int64),
                                                  np.array([0], np.int64), np.zeros(0, np.int64), 5)[0]
    check("more pixels than the capacity: refused (status %d)" % status, status == 1)
    blocked = np.array([1], np.int64)
    status, count, in_order = fd3.compact_the_pixels_of_the_region(directions, nothing, False, np.array([2], np.int64),
                                                                   np.array([0], np.int64), blocked, 6)[:3]
    check("a child's outlet of another region: it and the pixels above it are not ours (%d pixels)" % count,
          status == 0 and count == 2 and in_order == 2)
    # a child's outlet of another region on a cycle (Codex, round 1): the cycle is refused as 0.7.6 refuses it
    directions = np.array([[1, 1, 0], [1, 16, 0]], np.uint8)
    for mask, use_mask, what in ((nothing, False, "the land"), (np.ones((2, 3), np.uint8), True, "a mask of every pixel")):
        status = fd3.compact_the_pixels_of_the_region(directions, mask, use_mask, np.array([2], np.int64),
                                                      np.array([0], np.int64), np.array([3], np.int64), 6)[0]
        check("a cycle through a child's outlet of another region, on %s: refused (status %d)" % (what, status), status == 2)
    # a child's outlet in the middle of a walk: the pixels from the start to it are another region's, those below ours
    directions = np.array([[1, 1, 1, 0]], np.uint8)
    status, count, in_order, state, pixel_of_compact = fd3.compact_the_pixels_of_the_region(
        directions, nothing, False, np.array([3], np.int64), np.array([0], np.int64), np.array([1], np.int64), 4)[:5]
    check("a child's outlet in the middle of a walk: pixels 2 and 3 ours, 0 and 1 not (%s)" % sorted(pixel_of_compact[:count].tolist()),
          status == 0 and sorted(pixel_of_compact[:count].tolist()) == [2, 3] and state[0] < 0 and state[1] < 0)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for trial in range(150):
        compare(rng, False, trial)
    for trial in range(150):
        compare(rng, True, trial)
    test_refusals()
    print("%d failures" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
