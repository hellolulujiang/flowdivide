"""The top row of Figure 1: the 10 x 10 test grid and the two ways of cutting it.
    (a) the D8 flow directions, written in the colour of their basin, with red arrows; the channel (upstream count of
        at least CHANNEL_THRESHOLD) as a line;
    (b) four regular tiles of 5 x 5 pixels, drawn apart; the pixels whose flow leaves the tile in grey;
    (c) basin 1 cut below two channel junctions of its main stem, A and B, into three pieces, basin 2 the fourth; the
        pieces are told apart by their fill, the exits and the pixels they drain into lettered A, A', B, B', the basin
        outlets O1, O2.
Imported by fig_01_test_grid_upg.py (draw_row); reads data_01_test_grid_10x10.py.
"""
import collections
import os
import paths
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
import sys
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
CHANNEL_THRESHOLD = 5

PIECE_COLOURS = style.PIECE_FACE
EXIT_GREY = "#b9bfc8"
OUTSIDE = "#e9ebee"
RIVER = style.CHANNEL_LINE[1]
RIVER_2 = style.CHANNEL_LINE[2]
ARROW = "#d7301f"
FOOT_COLOUR = "#2b3038"


def downstream_of():
    downstream = {}
    for row in range(N):
        for col in range(N):
            if BSN[row, col] == 0:
                continue
            offset = D8_OFFSET.get(int(DIR[row, col]))
            downstream[(row, col)] = None if offset is None else (row + offset[0], col + offset[1])
    return downstream


def upstream_count(downstream):
    in_degree = collections.Counter(d for d in downstream.values() if d is not None)
    count = {c: 1 for c in downstream}
    queue = collections.deque(c for c in downstream if in_degree[c] == 0)
    while queue:
        cell = queue.popleft()
        target = downstream[cell]
        if target is not None:
            count[target] += count[cell]
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)
    return count


def pieces_by_nodes(downstream):
    """piece id per pixel: walk downstream until a cut node (or the outlet); basin 1 pieces 1..3
    numbered from upstream, basin 2 is piece 4"""
    piece = {}
    stop_ids = {CUT_NODES[0]: 1, CUT_NODES[1]: 2, OUTLETS[1]: 3, OUTLETS[2]: 4}
    for cell in downstream:
        walk = cell
        while walk not in stop_ids:
            walk = downstream[walk]
        piece[cell] = stop_ids[walk]
    return piece


def label_pixel_of_pieces(piece, channel, exits):
    """for every piece the pixel with the most neighbours in the same piece (an interior pixel,
    not a one-pixel nub), off the channel and off the exits; ties go to the pixel farthest from
    the channel"""
    label_pixel = {}
    channel_cells = list(channel)
    for number in sorted(set(piece.values())):
        best, best_score = None, -1.0
        for cell in piece:
            if piece[cell] != number or cell in channel or cell in exits:
                continue
            same_piece_neighbours = sum(1 for drow in (-1, 0, 1) for dcol in (-1, 0, 1)
                                        if (drow or dcol) and piece.get((cell[0] + drow, cell[1] + dcol)) == number)
            distance_channel = min(max(abs(cell[0] - c[0]), abs(cell[1] - c[1])) for c in channel_cells)
            score = same_piece_neighbours * 10 + distance_channel
            if score > best_score:
                best, best_score = cell, score
        label_pixel[number] = best
    return label_pixel


def shift_of(group, gap):
    """offset (dx, dy) in plot units for a group id in an exploded layout (2 x 2 tiles)"""
    if group is None:
        return (0.0, 0.0)
    tile_row, tile_col = group
    return (gap if tile_col == 1 else 0.0, gap if tile_row == 0 else 0.0)


def draw(axis, letter, title_text, foot_text, face_of, group_of=None, shift_of_group=None, codes=False,
         river=None, exit_pixels=(), text_size=8.0, outline_groups=None, piece_numbers=None, letters=None):
    def xy(cell):
        x, y = cell[1], N - cell[0] - 1
        if group_of is not None and shift_of_group is not None:
            dx, dy = shift_of_group(group_of(cell))
            x, y = x + dx, y + dy
        return x, y

    for row in range(N):
        for col in range(N):
            cell = (row, col)
            x, y = xy(cell)
            if BSN[cell] == 0:
                axis.add_patch(Rectangle((x, y), 1, 1, facecolor=OUTSIDE, edgecolor="#c8ccd2", linewidth=0.35))
                if codes:
                    axis.text(x + 0.5, y + 0.5, "247", fontsize=text_size, ha="center", va="center",
                              color="#9aa0a8")
                continue
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=EXIT_GREY if cell in exit_pixels else face_of(cell),
                                     edgecolor="#c8ccd2", linewidth=0.35))
            if codes:
                is_outlet = cell in OUTLETS.values()
                axis.text(x + 0.5, y + 0.38, str(int(DIR[cell])), fontsize=text_size, ha="center", va="center",
                          color=style.BASIN_INK[int(BSN[cell])], fontweight="bold" if is_outlet else "normal",
                          zorder=6)
    for row in range(N):
        for col in range(N):
            offset = D8_OFFSET.get(int(DIR[row, col]))
            if offset is None or not codes:
                continue
            x, y = xy((row, col))
            cx, cy = (x + 0.70, y + 0.70) if codes else (x + 0.5, y + 0.5)
            axis.add_patch(FancyArrowPatch((cx - 0.17 * offset[1], cy + 0.17 * offset[0]),
                                           (cx + 0.24 * offset[1], cy - 0.24 * offset[0]), arrowstyle="-|>",
                                           mutation_scale=4.5, linewidth=0.6, color=ARROW, shrinkA=0, shrinkB=0,
                                           zorder=4))
    if river:
        for cell, target in river.items():
            if target is None:
                continue
            (x0, y0), (x1, y1) = xy(cell), xy(target)
            axis.add_patch(PathPatch(Path([(x0 + 0.5, y0 + 0.5), (x1 + 0.5, y1 + 0.5)]),
                                     edgecolor=RIVER if BSN[cell] == 1 else RIVER_2,
                                     facecolor="none", linewidth=2.4, capstyle="round", zorder=2))
    # outlines: whole grid, or each group
    if outline_groups is None:
        axis.add_patch(Rectangle((0, 0), N, N, facecolor="none", edgecolor="#12161c", linewidth=1.2, zorder=5))
    elif outline_groups == "tiles":
        for tile_row in range(N // TILE):
            for tile_col in range(N // TILE):
                x, y = xy((tile_row * TILE, tile_col * TILE))
                axis.add_patch(Rectangle((x, y - TILE + 1), TILE, TILE, facecolor="none", edgecolor="#12161c",
                                         linewidth=1.1, zorder=5))
    else:
        for group in outline_groups:
            for row in range(N):
                for col in range(N):
                    if BSN[row, col] == 0 or group_of((row, col)) != group:
                        continue
                    x, y = xy((row, col))
                    for drow, dcol, seg in ((-1, 0, [(x, y + 1), (x + 1, y + 1)]), (1, 0, [(x, y), (x + 1, y)]),
                                            (0, -1, [(x, y), (x, y + 1)]), (0, 1, [(x + 1, y), (x + 1, y + 1)])):
                        nb = (row + drow, col + dcol)
                        if not (0 <= nb[0] < N and 0 <= nb[1] < N) or BSN[nb] == 0 or group_of(nb) != group:
                            axis.add_patch(PathPatch(Path(seg), edgecolor="#12161c", facecolor="none",
                                                     linewidth=1.1, capstyle="projecting", zorder=5))
    extra = GAP if shift_of_group is not None else 0.0
    axis.set_xlim(-0.02, N + extra + 0.02)
    axis.set_ylim(-0.02, N + extra + 0.02)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.axis("off")
    if piece_numbers:
        # one bold number per piece, on the pixel farthest from the channel and from the piece edge
        for number, cell in piece_numbers.items():
            x, y = xy(cell)
            axis.text(x + 0.5, y + 0.5, str(number), fontsize=style.SIZE_LETTER, fontweight="bold",
                      ha="center", va="center", color="#12161c", zorder=7,
                      path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])
    if letters:
        for cell, mark in letters.items():
            x, y = xy(cell)
            axis.text(x + 0.5, y + 0.5, mark, fontsize=style.SIZE_CELL, fontweight="bold", ha="center",
                      va="center", color="#12161c", zorder=7,
                      path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")])
    style.panel_heading(axis, letter, title_text)
    style.panel_foot(axis, foot_text)


def draw_row(axes, letters="abc"):
    """the three panels on three axes: (a) DIR, (b) tiles, (c) pieces"""
    downstream = downstream_of()
    count = upstream_count(downstream)
    channel = {cell: downstream[cell] for cell in downstream if count[cell] >= CHANNEL_THRESHOLD}
    channel = {cell: (target if target in channel else None) for cell, target in channel.items()}
    piece = pieces_by_nodes(downstream)

    tile_of = lambda cell: (cell[0] // TILE, cell[1] // TILE)
    tile_exits = [c for c, d in downstream.items() if d is not None and tile_of(c) != tile_of(d)]
    tile_edges = {(tile_of(c), tile_of(downstream[c])) for c in tile_exits}
    cyclic = any((b, a) in tile_edges for a, b in tile_edges)
    piece_exits = [c for c, d in downstream.items() if d is not None and piece[c] != piece[d]]
    sizes = collections.Counter(piece.values())
    print("basins", collections.Counter(BSN[BSN > 0].tolist()), "channel pixels", len(channel),
          "tile exits", len(tile_exits), "tile graph cyclic", cyclic, "pieces", dict(sizes),
          "piece exits", len(piece_exits))

    basin_face = lambda c: "white"
    piece_face = lambda c: PIECE_COLOURS[piece[c]]

    draw(axes[0], "(%s)" % letters[0], "Flow direction",
         "basin 1 blue numbers, %d pixels; basin 2 green, %d\n247 = nodata; outlets bold; channel: upstream count \u2265 %d"
         % ((BSN == 1).sum(), (BSN == 2).sum(), CHANNEL_THRESHOLD), basin_face, codes=True, river=channel)
    draw(axes[1], "(%s)" % letters[1], "Four regular tiles (5 \u00d7 5 pixels)",
         "%d pixels drain into another tile (grey)\nevery two neighbouring tiles feed each other: cycles" % len(tile_exits),
         basin_face, group_of=tile_of, shift_of_group=lambda g: shift_of(g, GAP), river=channel,
         exit_pixels=set(tile_exits), outline_groups="tiles")
    draw(axes[2], "(%s)" % letters[2], "Cut at two channel junctions",
         "%d exits, A and B (grey); no cycle\n$O_1$, $O_2$: the basin outlets" % len(piece_exits),
         piece_face, group_of=lambda c: piece.get(c), shift_of_group=None, river=channel,
         exit_pixels=set(piece_exits), outline_groups=sorted(sizes), letters=style.INTERFACE_LETTER)

