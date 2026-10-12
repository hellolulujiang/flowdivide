"""Figure A3: the Strahler order computed in parts on the 10 x 10 test grid of Figure 1, on the channel network
(upstream count of at least CHANNEL_THRESHOLD), as the package computes it: a channel head is order 1.
The order of a pixel is the largest order among its inflows, plus one when that largest occurs twice; it can be
neither summed nor maximised across a cut, so a part can only be finished once the orders at its inlets are final.
    top row, the four 5 x 5 tiles, swept in reading order, an inlet whose order is not yet known counted as 1, until a
        sweep changes nothing: (a) sweep 1, (b) sweep 2 (bold: changed), (c) the last sweep;
    bottom row, the pieces cut below the two channel junctions of Figure 1c, in downstream order, each once:
        (d) pieces 1 and 4, which have no inlet; (e) piece 2; (f) piece 3; the crossing read in a panel as a red arrow.
Reads data_01_test_grid_10x10.py; writes fig_a3_strahler_tiles_vs_pieces.png into ../figures.

    python3 fig_a3_strahler_tiles_vs_pieces.py
"""
import collections
import os
import paths
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_01_test_grid_10x10 import DIR, BSN, OUTLETS, CUT_NODES
import style
from matplotlib import patheffects

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
N = DIR.shape[0]
TILE = 5
GAP = 0.6
D8_OFFSET = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}

OUTSIDE = "#e9ebee"
EXIT_GREY = "#b9bfc8"
ARROW = "#d7301f"
CHANNEL_THRESHOLD = 5
PIECE_FACE = style.PIECE_FACE
FOOT_COLOUR = "#2b3038"


# ------------------------------------------------------------------ the computations

def downstream_of():
    downstream = {}
    for row in range(N):
        for col in range(N):
            if BSN[row, col] == 0:
                continue
            offset = D8_OFFSET.get(int(DIR[row, col]))
            downstream[(row, col)] = None if offset is None else (row + offset[0], col + offset[1])
    return downstream


def pieces_by_nodes(downstream):
    stop_ids = {CUT_NODES[0]: 1, CUT_NODES[1]: 2, OUTLETS[1]: 3, OUTLETS[2]: 4}
    piece = {}
    for cell in downstream:
        walk = cell
        while walk not in stop_ids:
            walk = downstream[walk]
        piece[cell] = stop_ids[walk]
    return piece


def inflows_of(downstream):
    inflow = collections.defaultdict(list)
    for cell, target in downstream.items():
        if target is not None:
            inflow[target].append(cell)
    return inflow


def upstream_count(downstream):
    inflow = inflows_of(downstream)
    in_degree = {cell: len(inflow[cell]) for cell in downstream}
    count = {cell: 1 for cell in downstream}
    queue = collections.deque(cell for cell in downstream if in_degree[cell] == 0)
    while queue:
        cell = queue.popleft()
        target = downstream[cell]
        if target is not None:
            count[target] += count[cell]
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    return count


def strahler_of(inflow_orders):
    ranked = sorted(inflow_orders, reverse=True)
    if not ranked:
        return 1
    if len(ranked) > 1 and ranked[0] == ranked[1]:
        return ranked[0] + 1
    return ranked[0]


def order_one_part(downstream, inflow, group_of, group, order):
    """Strahler order of every pixel of one part, from the orders now held at its inlets (an unknown
    inlet, order 0, counts as 1); updates `order` in place, returns how many pixels changed"""
    cells = [cell for cell in downstream if group_of(cell) == group]
    in_degree = collections.Counter()
    for cell in cells:
        target = downstream[cell]
        if target is not None and group_of(target) == group:
            in_degree[target] += 1
    queue = collections.deque(cell for cell in cells if in_degree[cell] == 0)
    changed = 0
    while queue:
        cell = queue.popleft()
        new_order = strahler_of([max(order[u], 1) for u in inflow[cell]])
        if new_order != order[cell]:
            order[cell] = new_order
            changed += 1
        target = downstream[cell]
        if target is not None and group_of(target) == group:
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    return changed


def sweep_tiles(downstream, inflow, tile_of):
    """sweep the tiles in reading order until nothing changes; returns the order after every sweep"""
    order = {cell: 0 for cell in downstream}
    after_sweep = []
    while True:
        changed = 0
        for tile_row in range(N // TILE):
            for tile_col in range(N // TILE):
                changed += order_one_part(downstream, inflow, tile_of, (tile_row, tile_col), order)
        after_sweep.append((dict(order), changed))
        if changed == 0:
            return after_sweep


def pieces_in_order(downstream, inflow, piece_of, stages):
    """the pieces done once each, in downstream order; returns the order after every stage"""
    order = {cell: 0 for cell in downstream}
    after_stage = []
    for stage in stages:
        for piece_id in stage:
            order_one_part(downstream, inflow, piece_of, piece_id, order)
        after_stage.append(dict(order))
    return after_stage


# ------------------------------------------------------------------ drawing

def make_xy(group_of, apart):
    def xy(cell):
        x, y = cell[1], N - cell[0] - 1
        if apart:
            tile_row, tile_col = group_of(cell)
            x += GAP if tile_col == 1 else 0.0
            y += GAP if tile_row == 0 else 0.0
        return x, y
    return xy


def draw(axis, letter, title_text, foot_text, text_of, group_of, apart, exit_pixels, bold=(), hand_over=None,
         text_size=style.SIZE_CELL, path=(), face_of=lambda cell: "white"):
    xy = make_xy(group_of, apart)
    exit_pixels = set(exit_pixels)
    bold = set(bold)
    # a crossing inside one grid: the two numbers of an edge-sharing pair move apart so the arrow fits
    nudge_of = {}
    if hand_over and not apart:
        for exit_cell, entry in hand_over.items():
            drow, dcol = entry[0] - exit_cell[0], entry[1] - exit_cell[1]
            if drow == 0 or dcol == 0:
                nudge_of[exit_cell] = (-dcol * style.CROSSING_NUDGE, drow * style.CROSSING_NUDGE)
                nudge_of[entry] = (dcol * style.CROSSING_NUDGE, -drow * style.CROSSING_NUDGE)
    for row in range(N):
        for col in range(N):
            cell = (row, col)
            x, y = xy(cell)
            if BSN[cell] == 0:
                axis.add_patch(Rectangle((x, y), 1, 1, facecolor=OUTSIDE, edgecolor="#c8ccd2", linewidth=0.35))
                continue
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=EXIT_GREY if cell in exit_pixels else face_of(cell),
                                     edgecolor="#c8ccd2", linewidth=0.35))
            text = text_of(cell)
            if text is not None:
                is_outlet = cell in OUTLETS.values()
                dx, dy = nudge_of.get(cell, (0.0, 0.0))
                axis.text(x + 0.5 + dx, y + 0.5 + dy, str(text), fontsize=text_size, ha="center", va="center",
                          color=style.BASIN_INK[int(BSN[cell])],
                          fontweight="bold" if (cell in bold or is_outlet) else "normal", zorder=6,
                          path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")])
    if hand_over:
        for exit_cell in hand_over:
            entry = hand_over[exit_cell]
            (x0, y0), (x1, y1) = xy(exit_cell), xy(entry)
            if apart:
                shrink_a, shrink_b = 5, 3
            elif exit_cell in nudge_of:
                shrink_a = shrink_b = style.CROSSING_SHRINK_EDGE
            else:
                shrink_a = shrink_b = style.CROSSING_SHRINK_CORNER
            axis.add_patch(FancyArrowPatch((x0 + 0.5, y0 + 0.5), (x1 + 0.5, y1 + 0.5), arrowstyle="-|>",
                                           mutation_scale=5, linewidth=0.8, color=ARROW, shrinkA=shrink_a, shrinkB=shrink_b,
                                           zorder=7))
    for one_path in path:
        for cell, target in zip(one_path, one_path[1:]):
            (x0, y0), (x1, y1) = xy(cell), xy(target)
            axis.add_patch(PathPatch(Path([(x0 + 0.5, y0 + 0.5), (x1 + 0.5, y1 + 0.5)]), edgecolor=style.CHANNEL_LINE[int(BSN[cell])],
                                     facecolor="none", linewidth=2.4, capstyle="round", zorder=2))
    # outlines
    if apart:
        for tile_row in range(N // TILE):
            for tile_col in range(N // TILE):
                x, y = xy((tile_row * TILE, tile_col * TILE))
                axis.add_patch(Rectangle((x, y - TILE + 1), TILE, TILE, facecolor="none", edgecolor="#12161c",
                                         linewidth=1.1, zorder=5))
    else:
        for row in range(N):
            for col in range(N):
                if BSN[row, col] == 0:
                    continue
                group = group_of((row, col))
                x, y = xy((row, col))
                for drow, dcol, seg in ((-1, 0, [(x, y + 1), (x + 1, y + 1)]), (1, 0, [(x, y), (x + 1, y)]),
                                        (0, -1, [(x, y), (x, y + 1)]), (0, 1, [(x + 1, y), (x + 1, y + 1)])):
                    nb = (row + drow, col + dcol)
                    if not (0 <= nb[0] < N and 0 <= nb[1] < N) or BSN[nb] == 0 or group_of(nb) != group:
                        axis.add_patch(PathPatch(Path(seg), edgecolor="#12161c", facecolor="none", linewidth=1.1,
                                                 capstyle="projecting", zorder=5))
    extra = GAP if apart else 0.0
    axis.set_xlim(-0.02, N + extra + 0.02)
    axis.set_ylim(-0.02, N + extra + 0.02)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.axis("off")
    style.panel_heading(axis, letter, title_text)
    style.panel_foot(axis, foot_text)


def main():
    downstream = downstream_of()
    inflow = inflows_of(downstream)
    count = upstream_count(downstream)
    piece = pieces_by_nodes(downstream)
    tile_of = lambda cell: (cell[0] // TILE, cell[1] // TILE)
    piece_of = lambda cell: piece[cell]
    channel = [cell for cell in downstream if count[cell] >= CHANNEL_THRESHOLD]
    channel_links = [[cell, downstream[cell]] for cell in channel if downstream[cell] is not None]
    # the order lives on the channel network alone: a channel pixel flows into a channel pixel (the count only grows
    # downstream), and only channel pixels flow into it
    channel_set = set(channel)
    network = {cell: downstream[cell] for cell in channel}
    if any(target is not None and target not in channel_set for target in network.values()):
        raise SystemExit("a channel pixel flows into a pixel off the channel")
    network_inflow = collections.defaultdict(list)
    for cell in channel:
        network_inflow[cell] = [u for u in inflow[cell] if u in channel_set]

    # the true order, for the check
    truth = {cell: 0 for cell in network}
    order_one_part(network, network_inflow, lambda cell: 0, 0, truth)

    after_sweep = sweep_tiles(network, network_inflow, tile_of)
    stages = [(1, 4), (2,), (3,)]
    after_stage = pieces_in_order(network, network_inflow, piece_of, stages)
    assert after_sweep[-1][0] == truth and after_stage[-1] == truth
    print("sweeps", len(after_sweep), "changed per sweep", [c for _, c in after_sweep], "max order",
          max(truth.values()), "outlets", {k: truth[v] for k, v in OUTLETS.items()})

    style.apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.48, 5.45))
    plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.05, wspace=0.14, hspace=0.26)
    on_channel = lambda order_now: (lambda c: order_now[c] if (c in set(channel) and order_now[c] > 0) else None)

    # top row: sweep 1, sweep 2 and the last sweep, which changes nothing
    n_sweeps = len(after_sweep)
    if n_sweeps < 3:
        raise SystemExit("the tiles finished in %d sweeps; the top row shows three" % n_sweeps)
    orders = [o for o, _ in after_sweep]
    changes = [c for _, c in after_sweep]
    zero = {cell: 0 for cell in network}
    between = (" (sweeps 3 to %d: %s more)" % (n_sweeps - 1, " + ".join(str(c) for c in changes[2:n_sweeps - 1]))
               if n_sweeps > 3 else "")
    panels = [(0, zero, "Sweep 1 of the tiles", "sweep 1: every tile, inlets unknown"),
              (1, orders[0], "Sweep 2 of the tiles", "sweep 2: %d orders change, bold" % changes[1]),
              (n_sweeps - 1, orders[n_sweeps - 2], "Sweep %d of the tiles" % n_sweeps,
               "sweep %d: none change%s,\nso the tiles are done" % (n_sweeps, between))]
    for k, (index, previous, title, foot) in enumerate(panels):
        order_now = orders[index]
        bold = [c for c in network if order_now[c] != previous[c]] if k > 0 else []
        draw(axes[0][k], "(%s)" % "abc"[k], title, foot, on_channel(order_now), tile_of, True, (), bold=bold,
             path=channel_links)
    # bottom row: the pieces in downstream order
    titles = ["Pieces 1 and 4: no inlet", "Piece 2: after piece 1", "Piece 3: after piece 2"]
    # the inlet holds the order of the exit that feeds it (no tie at either inlet, so no plus one)
    assert after_stage[1][(4, 5)] == after_stage[1][(4, 4)] and after_stage[2][(8, 4)] == after_stage[2][(7, 5)]
    foots = ["each piece read once",
             "A\u2032 takes order %d from A;\nits inlet order is final" % after_stage[1][(4, 5)],
             "B\u2032 takes order %d from B;\nevery piece read once" % after_stage[2][(8, 4)]]
    # the crossing read in a stage, exit to inlet, drawn as the red arrow of Figure 1e
    piece_exits = [cell for cell, target in downstream.items() if target is not None and piece[target] != piece[cell]]
    done = set()
    for k, order_now in enumerate(after_stage):
        done |= set(stages[k])
        face = lambda c, D=set(done): PIECE_FACE[piece[c]] if piece[c] in D else "white"
        hand_over = {cell: downstream[cell] for cell in piece_exits if piece[downstream[cell]] in stages[k]}
        draw(axes[1][k], "(%s)" % "def"[k], titles[k], foots[k], on_channel(order_now), piece_of, False, (),
             hand_over=hand_over, path=channel_links, face_of=face)
        print("pieces stage", stages[k], "crossing", hand_over)

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_a3_strahler_tiles_vs_pieces")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
