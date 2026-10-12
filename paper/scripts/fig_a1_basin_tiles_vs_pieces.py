"""Figure A1: the basin label (BSN) computed in parts on the 10 x 10 test grid of Figure 1.
    top row, the four 5 x 5 tiles: (a) pass 1, each tile alone: a pixel takes its basin when its flow reaches an outlet
        inside the tile and stays blank when it leaves the tile; (b) the exit graph: each exit is followed to its outlet
        and told its basin; (c) pass 2, each tile again with the exits known;
    bottom row, the pieces cut below the two channel junctions of Figure 1c, downstream first, because the label travels
        from the outlet up: (d) pieces 3 and 4, which hold the outlets; (e) piece 2, whose exit B takes the label found
        at B'; (f) piece 1, whose exit A takes it from A'.  The crossing read in a panel is drawn as a red arrow.
Reads data_01_test_grid_10x10.py; writes fig_a1_basin_tiles_vs_pieces.png into ../figures.

    python3 fig_a1_basin_tiles_vs_pieces.py
"""
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


def label_inside(downstream, group_of, known=None):
    """basin of every pixel found inside its own part: walk downstream until an outlet (or a pixel
    whose basin is already known); None when the walk leaves the part first.  Returns the labels
    and the exit pixels."""
    label = {}
    for cell in downstream:
        walk = cell
        while True:
            if known and walk in known:
                label[cell] = known[walk]
                break
            target = downstream[walk]
            if target is None:
                label[cell] = BSN[walk]
                break
            if group_of(target) != group_of(walk):
                label[cell] = ("exit", walk)
                break
            walk = target
    exits = [cell for cell, target in downstream.items() if target is not None and group_of(target) != group_of(cell)]
    return label, exits


def chase_exits(downstream, group_of, local_label, exits):
    """each exit followed to the next exit downstream until a pixel with a known label"""
    known = {}
    for exit_cell in exits:
        walk = downstream[exit_cell]
        while isinstance(local_label[walk], tuple):
            walk = downstream[local_label[walk][1]]
        known[exit_cell] = local_label[walk]
    return known


def next_exit_of(cell, downstream, group_of):
    walk = cell
    while True:
        target = downstream[walk]
        if group_of(target) != group_of(walk):
            return walk
        walk = target


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


def draw(axis, letter, title_text, foot_text, text_of, group_of, apart, exit_pixels, face_of, bold=(), hand_over=None,
         text_size=style.SIZE_CELL):
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
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=face_of(cell), edgecolor="#c8ccd2", linewidth=0.35))
            if cell in exit_pixels:
                # the exit pixel keeps a grey band on top of its colour
                axis.add_patch(Rectangle((x, y), 1, 1, facecolor=EXIT_GREY, edgecolor="none", alpha=0.85, zorder=1))
            text = text_of(cell)
            if text is not None:
                is_outlet = cell in OUTLETS.values()
                dx, dy = nudge_of.get(cell, (0.0, 0.0))
                axis.text(x + 0.5 + dx, y + 0.5 + dy, str(text), fontsize=text_size, ha="center", va="center",
                          color="#6b7280" if isinstance(text, str) else style.BASIN_INK[int(text)],
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


def main():
    downstream = downstream_of()
    piece = pieces_by_nodes(downstream)
    tile_of = lambda cell: (cell[0] // TILE, cell[1] // TILE)
    piece_of = lambda cell: piece[cell]

    style.apply_style()
    figure, axes = plt.subplots(2, 3, figsize=(7.48, 5.3))
    plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.05, wspace=0.14, hspace=0.22)
    letters = iter("abcdef")
    for row_axes, group_of, apart, word in ((axes[0], tile_of, True, "tile"),):
        local_label, exits = label_inside(downstream, group_of)
        known = chase_exits(downstream, group_of, local_label, exits)
        final_label, _ = label_inside(downstream, group_of, known)
        assert all(final_label[c] == BSN[c] for c in downstream)
        unknown = [c for c in downstream if isinstance(local_label[c], tuple)]
        hand_over = {c: downstream[c] for c in exits}
        print(word, "exits", len(exits), "unknown after pass 1", len(unknown))
        face_local = lambda c: "white"
        face_final = lambda c: "white"
        text_local = lambda c, L=local_label: None if isinstance(L[c], tuple) else L[c]
        text_final = lambda c, L=final_label: L[c]
        draw(row_axes[0], "(%s)" % next(letters), "Pass 1: each %s alone" % word,
             "blank: %d pixels leave the %s before an outlet" % (len(unknown), word), text_local, group_of, apart, exits, face_local)
        draw(row_axes[1], "(%s)" % next(letters), "The exit graph, solved once",
             "the %d exits told their basin" % len(exits),
             lambda c, K=known: K.get(c), group_of, apart, exits, face_local, hand_over=hand_over)
        draw(row_axes[2], "(%s)" % next(letters), "Pass 2: each %s again" % word,
             "the %d blank pixels take the basin of their exit" % len(unknown), text_final, group_of, apart, (), face_final)

    # bottom row: the pieces read once each in upstream order; a piece read later shows nothing yet
    local_label, exits = label_inside(downstream, piece_of)
    known = chase_exits(downstream, piece_of, local_label, exits)
    final_label, _ = label_inside(downstream, piece_of, known)
    assert all(final_label[c] == BSN[c] for c in downstream)
    titles = ["Pieces 3 and 4: hold the outlets", "Piece 2: after piece 3", "Piece 1: after piece 2"]
    foots = ["each piece read once",
             "B takes basin %d from B\u2032, bold" % final_label[(7, 5)],
             "A takes basin %d from A\u2032, bold" % final_label[(4, 4)]]
    done = set()
    for k, stage in enumerate(style.PIECE_STAGES_UP):
        done |= set(stage)
        face = lambda c, D=set(done): style.PIECE_FACE[piece[c]] if piece[c] in D else "white"
        text = lambda c, D=set(done): final_label[c] if piece[c] in D else None
        bold = [exit_cell for exit_cell in exits if piece[exit_cell] in stage]
        # the one crossing this stage reads, drawn exit to inlet (the flow) as in the exit graph of (b);
        # the label travels the other way, which the footer says
        hand_over = {exit_cell: downstream[exit_cell] for exit_cell in bold}
        draw(axes[1][k], "(%s)" % next(letters), titles[k], foots[k], text, piece_of, False, exits, face, bold=bold,
             hand_over=hand_over)
        print("pieces stage", stage, "labels at exits", {e: final_label[e] for e in bold})

    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_a1_basin_tiles_vs_pieces")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
