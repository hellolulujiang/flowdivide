"""Figure A2: the upstream flow length (LUP) and the longest flow path (LFP) computed in parts on the 10 x 10 test
grid of Figure 1.  The LUP of a pixel is the longest path, in pixel steps, from a divide down to it (0 at a head); the
head that gives it travels with it, so the longest flow path of a basin is the walk from the outlet's head to the outlet.
    top row, the four 5 x 5 tiles: (a) pass 1, each tile alone; (b) the exit graph: every exit hands its length and
        head to the pixel it drains into, the larger kept; (c) pass 2, bold where the length changed, the longest flow
        paths drawn;
    bottom row, the pieces cut below the two channel junctions of Figure 1c, in downstream order: (d) pieces 1 and 4, no
        inlet; (e) piece 2, its inlet A' holding the length at A plus one step; (f) piece 3, its inlet B' likewise, the
        longest flow paths drawn.  The crossing read in a panel is drawn as a red arrow.
Reads data_01_test_grid_10x10.py; writes fig_a2_upstream_length_tiles_vs_pieces.png into ../figures.

    python3 fig_a2_upstream_length_tiles_vs_pieces.py
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
LFP_COLOUR = "#b4b9c1"
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


def longest_inside(downstream, group_of, arriving=None):
    """LUP (steps) and head of every pixel, taking only the inflows inside the same part; `arriving`
    gives (length, head) handed in at entry pixels.  Returns length, head, exits."""
    in_degree = collections.Counter()
    for cell, target in downstream.items():
        if target is not None and group_of(target) == group_of(cell):
            in_degree[target] += 1
    length = {cell: 0 for cell in downstream}
    head = {cell: cell for cell in downstream}
    if arriving:
        for cell, (arrived_length, arrived_head) in arriving.items():
            length[cell] = arrived_length
            head[cell] = arrived_head
    queue = collections.deque(cell for cell in downstream if in_degree[cell] == 0)
    while queue:
        cell = queue.popleft()
        target = downstream[cell]
        if target is not None and group_of(target) == group_of(cell):
            if length[cell] + 1 > length[target]:
                length[target] = length[cell] + 1
                head[target] = head[cell]
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    exits = [cell for cell, target in downstream.items() if target is not None and group_of(target) != group_of(cell)]
    return length, head, exits


def next_exit_of(cell, downstream, group_of):
    walk = cell
    while True:
        target = downstream[walk]
        if target is None:
            return None
        if group_of(target) != group_of(walk):
            return walk
        walk = target


def steps_between(cell, stop, downstream):
    steps = 0
    walk = cell
    while walk != stop:
        walk = downstream[walk]
        steps += 1
    return steps


def solve_exit_graph(downstream, group_of, local_length, local_head, exits):
    """the length at every exit once the exits upstream of it are known: an exit's length is the max
    of its own part's length and, for every exit feeding it, that exit's length + the steps between"""
    into = collections.defaultdict(list)
    for exit_cell in exits:
        next_exit = next_exit_of(downstream[exit_cell], downstream, group_of)
        if next_exit is not None:
            into[next_exit].append(exit_cell)
    total = {}
    head = {}
    remaining = set(exits)
    while remaining:
        ready = [x for x in remaining if all(y in total for y in into[x])]
        assert ready, "the exit graph holds a cycle"
        for x in ready:
            total[x] = local_length[x]
            head[x] = local_head[x]
            for y in into[x]:
                candidate = total[y] + steps_between(y, x, downstream)
                if candidate > total[x]:
                    total[x] = candidate
                    head[x] = head[y]
            remaining.remove(x)
    arriving = {}
    for exit_cell in exits:
        entry = downstream[exit_cell]
        candidate = (total[exit_cell] + 1, head[exit_cell])
        if entry not in arriving or candidate[0] > arriving[entry][0]:
            arriving[entry] = candidate
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
                on_path = any(cell in one_path for one_path in path)
                dx, dy = nudge_of.get(cell, (0.0, 0.0))
                axis.text(x + 0.5 + dx, y + 0.5 + dy, str(text), fontsize=text_size, ha="center", va="center",
                          color=style.BASIN_INK[int(BSN[cell])],
                          fontweight="bold" if (cell in bold or is_outlet) else "normal", zorder=6,
                          path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")] if on_path else None)
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
            axis.add_patch(PathPatch(Path([(x0 + 0.5, y0 + 0.5), (x1 + 0.5, y1 + 0.5)]), edgecolor=LFP_COLOUR,
                                     facecolor="none", linewidth=1.9, capstyle="round", zorder=2))
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
    piece = pieces_by_nodes(downstream)
    tile_of = lambda cell: (cell[0] // TILE, cell[1] // TILE)
    piece_of = lambda cell: piece[cell]

    style.apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.48, 5.45))
    plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.05, wspace=0.14, hspace=0.26)
    letters = iter("abcdef")
    for row_axes, group_of, apart, word in ((axes[0], tile_of, True, "tile"),):
        local_length, local_head, exits = longest_inside(downstream, group_of)
        exit_total, arriving = solve_exit_graph(downstream, group_of, local_length, local_head, exits)
        final_length, final_head, _ = longest_inside(downstream, group_of, arriving)
        changed = [c for c in downstream if final_length[c] != local_length[c]]
        hand_over = {c: downstream[c] for c in exits}
        # the longest flow path of each basin: from the head recorded at the outlet down to the outlet
        paths = []
        for outlet in OUTLETS.values():
            one_path = [final_head[outlet]]
            while one_path[-1] != outlet:
                one_path.append(downstream[one_path[-1]])
            assert len(one_path) - 1 == final_length[outlet]
            paths.append(one_path)
        outlet = OUTLETS[1]
        print(word, "exits", len(exits), "changed", len(changed), "LUP at the outlets",
              {k: final_length[v] for k, v in OUTLETS.items()}, "LFP heads", [p[0] for p in paths])
        draw(row_axes[0], "(%s)" % next(letters), "Pass 1: each %s alone" % word,
             "%d exit pixels, shaded" % len(exits), lambda c: local_length[c], group_of, apart, exits)
        draw(row_axes[1], "(%s)" % next(letters), "The exit graph, solved once",
             "%d lengths handed on, the larger kept" % len(exits),
             lambda c: exit_total[c] if c in exit_total else None, group_of, apart, exits, hand_over=hand_over)
        draw(row_axes[2], "(%s)" % next(letters), "Pass 2: each %s again" % word,
             "%d lengths changed, bold;\ngrey: the longest flow paths, %d and %d steps"
             % (len(changed), final_length[OUTLETS[1]], final_length[OUTLETS[2]]),
             lambda c: final_length[c], group_of, apart, exits, bold=changed, path=paths)

    # bottom row: the pieces read once each in downstream order; a piece read later shows nothing yet
    local_length, local_head, exits = longest_inside(downstream, piece_of)
    exit_total, arriving = solve_exit_graph(downstream, piece_of, local_length, local_head, exits)
    final_length, final_head, _ = longest_inside(downstream, piece_of, arriving)
    whole_length, _, _ = longest_inside(downstream, lambda cell: 0)
    assert all(final_length[c] == whole_length[c] for c in downstream)
    entry_of = {downstream[exit_cell]: exit_cell for exit_cell in exits}
    titles = ["Pieces 1 and 4: no inlet", "Piece 2: after piece 1", "Piece 3: after piece 2"]
    foots = ["each piece read once",
             "A\u2032 takes %d + 1 from A, bold" % final_length[(4, 4)],
             "B\u2032 takes %d + 1 from B, bold;\ngrey: the longest flow paths, %d and %d steps"
             % (final_length[(7, 5)], final_length[OUTLETS[1]], final_length[OUTLETS[2]])]
    done = set()
    for k, stage in enumerate(style.PIECE_STAGES_DOWN):
        done |= set(stage)
        face = lambda c, D=set(done): style.PIECE_FACE[piece[c]] if piece[c] in D else "white"
        text = lambda c, D=set(done): final_length[c] if piece[c] in D else None
        bold = [entry for entry in entry_of if piece[entry] in stage]
        # the one crossing this stage reads: exit to inlet, the same red arrow as the exit graph in (b)
        hand_over = {entry_of[entry]: entry for entry in bold}
        draw(axes[1][k], "(%s)" % next(letters), titles[k], foots[k], text, piece_of, False, exits, bold=bold,
             hand_over=hand_over, path=paths if k == 2 else (), face_of=face)

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_a2_upstream_length_tiles_vs_pieces")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
