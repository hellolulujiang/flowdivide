"""Rows two and three of Figure 1: the upstream pixel count (UPG) computed in parts on the 10 x 10 test grid.
    above, the four 5 x 5 tiles, drawn apart: (a) pass 1, each tile alone, the exit pixels shaded; (b) the exit graph:
        every exit hands its total to the pixel it drains into, solved once; (c) pass 2, each tile again with what
        arrived, bold where the count changed;
    below, the pieces cut below the two channel junctions A and B, each read once in downstream order: (d) pieces 1 and 4,
        which have no inlet; (e) piece 2, its inlet A' holding the count at A; (f) piece 3, its inlet B' holding the count
        at B.  The crossing read in a panel is drawn as the same red arrow as the exits of (b).
Imported by fig_01_test_grid_upg.py (draw_rows); reads data_01_test_grid_10x10.py.
"""
import collections
import os
import paths
import sys
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_01_test_grid_10x10 import DIR, BSN, OUTLETS, CUT_NODES
import style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY
N = DIR.shape[0]
TILE = 5
GAP = 0.6
D8_OFFSET = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}

OUTSIDE = "#e9ebee"
EXIT_GREY = "#b9bfc8"
ARROW = "#d7301f"
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


def accumulate_inside(downstream, group_of, arriving=None):
    """UPG summed inside each group only; returns the count and the exit pixels"""
    in_degree = collections.Counter()
    for cell, target in downstream.items():
        if target is not None and group_of(target) == group_of(cell):
            in_degree[target] += 1
    count = {cell: 1 + (arriving.get(cell, 0) if arriving else 0) for cell in downstream}
    queue = collections.deque(cell for cell in downstream if in_degree[cell] == 0)
    while queue:
        cell = queue.popleft()
        target = downstream[cell]
        if target is not None and group_of(target) == group_of(cell):
            count[target] += count[cell]
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    exits = [cell for cell, target in downstream.items() if target is not None and group_of(target) != group_of(cell)]
    return count, exits


def upstream_count_whole(downstream):
    """UPG of the whole grid in one go, the check for the piece-wise result"""
    count, _ = accumulate_inside(downstream, lambda cell: 0)
    return count


def next_exit_of(cell, downstream, group_of):
    walk = cell
    while True:
        target = downstream[walk]
        if target is None:
            return None
        if group_of(target) != group_of(walk):
            return walk
        walk = target


def solve_exit_graph(downstream, group_of, local_count, exits):
    into = collections.defaultdict(list)
    for exit_cell in exits:
        next_exit = next_exit_of(downstream[exit_cell], downstream, group_of)
        if next_exit is not None:
            into[next_exit].append(exit_cell)
    total = {}
    remaining = set(exits)
    while remaining:
        ready = [x for x in remaining if all(y in total for y in into[x])]
        assert ready, "the exit graph holds a cycle"
        for x in ready:
            total[x] = local_count[x] + sum(total[y] for y in into[x])
            remaining.remove(x)
    arriving = collections.Counter()
    for exit_cell in exits:
        arriving[downstream[exit_cell]] += total[exit_cell]
    return total, arriving


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
         text_size=style.SIZE_CELL, face_of=lambda cell: "white"):
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
                          fontweight="bold" if (cell in bold or is_outlet) else "normal", zorder=6)
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


def draw_rows(axes, letters="abcdef"):
    """the six panels on a 2 x 3 block of axes: tiles above, pieces below"""
    downstream = downstream_of()
    piece = pieces_by_nodes(downstream)
    tile_of = lambda cell: (cell[0] // TILE, cell[1] // TILE)
    piece_of = lambda cell: piece[cell]
    letters = iter(letters)
    for row_axes, group_of, apart, word in ((axes[0], tile_of, True, "tile"),):
        local_count, exits = accumulate_inside(downstream, group_of)
        exit_total, arriving = solve_exit_graph(downstream, group_of, local_count, exits)
        final_count, _ = accumulate_inside(downstream, group_of, arriving)
        changed = [c for c in downstream if final_count[c] != local_count[c]]
        hand_over = {c: downstream[c] for c in exits}
        print(word, "exits", len(exits), "changed in pass 2", len(changed), "UPG at the outlets",
              {k: final_count[v] for k, v in OUTLETS.items()})
        draw(row_axes[0], "(%s)" % next(letters), "Pass 1: each %s alone" % word,
             "%d exit pixels, shaded" % len(exits),
             lambda c: local_count[c], group_of, apart, exits)
        draw(row_axes[1], "(%s)" % next(letters), "The exit graph, solved once",
             "%d totals handed on" % len(exits),
             lambda c: exit_total[c] if c in exit_total else None, group_of, apart, exits, hand_over=hand_over)
        draw(row_axes[2], "(%s)" % next(letters), "Pass 2: each %s again" % word,
             "%d counts changed, bold" % len(changed),
             lambda c: final_count[c], group_of, apart, exits, bold=changed)

    # bottom row: the pieces read once each in downstream order; a piece read later shows nothing yet
    local_count, exits = accumulate_inside(downstream, piece_of)
    exit_total, arriving = solve_exit_graph(downstream, piece_of, local_count, exits)
    final_count, _ = accumulate_inside(downstream, piece_of, arriving)
    assert all(final_count[c] == upstream_count_whole(downstream)[c] for c in downstream)
    entry_of = {downstream[exit_cell]: exit_cell for exit_cell in exits}
    titles = ["Pieces 1 and 4: no inlet", "Piece 2: after piece 1", "Piece 3: after piece 2"]
    foots = ["each piece read once",
             "A\u2032 takes %d from A, bold" % final_count[(4, 4)],
             "B\u2032 takes %d from B, bold" % final_count[(7, 5)]]
    done = set()
    for k, stage in enumerate(style.PIECE_STAGES_DOWN):
        done |= set(stage)
        face = lambda c, D=set(done): style.PIECE_FACE[piece[c]] if piece[c] in D else "white"
        text = lambda c, D=set(done): final_count[c] if piece[c] in D else None
        bold = [entry for entry in entry_of if piece[entry] in stage]
        # the one crossing this stage reads: exit to inlet, the same red arrow as the exit graph in (b)
        hand_over = {entry_of[entry]: entry for entry in bold}
        draw(axes[1][k], "(%s)" % next(letters), titles[k], foots[k], text, piece_of, False, exits, bold=bold,
             hand_over=hand_over, face_of=face)
        print("pieces stage", stage, "handed", {entry_of[e]: final_count[e] for e in bold})

