"""Figure 2: the FlowDivide workflow.
The data producer's work, DEM to flow direction, sits outside in a dashed column.  Inside the FlowDivide boundary,
FD1 builds the partition in six steps (FD1.0 assembly and recoding, FD1.1 flow accumulation, FD1.2 outlet indexing,
FD1.3 watershed delineation, FD1.4 basin grouping, FD1.5 fitting to memory: a group over the capacity is cut or
divided, groups under it are merged); FD2 makes the views delivered with the partition (resampling, vectorisation and
colouring, every polygon one of eight colours, no two neighbours alike); FD3 is the user's run, the attributes region by
region.  FD1 and FD2 share one field, what is delivered; FD3 lies outside it.  Every text is measured against its box
after drawing; a text wider than its box is reported and the figure is not saved.
Writes fig_02_workflow.png into ../figures.

    python3 fig_02_workflow.py
"""
import os
import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY

INK = "#12161c"
BODY_INK = "#2b3038"
GREY_FACE, GREY_EDGE = "#f1f2f4", "#7d838c"
BLUE_FACE, BLUE_EDGE = "#e6eef9", "#2b4c7e"
ORANGE_FACE, ORANGE_EDGE = "#fff0da", "#b26a00"
FRAME_EDGE = "#4a5160"
DELIVERED_FACE = "#eef3fa"
TITLE_SIZE = 7.5     # box titles; nothing in the figure is below 7 pt on paper
BODY_SIZE = 7.0
HEAD_SIZE = 7.5
LINE_TITLE = 0.34    # height of one title line in axis units (7.5 pt, linespacing 1.15)
LINE_BODY = 0.325    # height of one body line in axis units (7 pt, linespacing 1.25)

fitted_texts = []    # (text artist, x, y, w, h) for the fit check after drawing


def box(axis, x, y, w, h, number, title, product, face, edge, dashed=False, note=None, title_top=None):
    """A rounded box with a number at the top left (or a two-line italic note in its place), the title and
    the product text below it.  The block of title and text is centred in the space under the number line,
    or its top is put at title_top so that the titles of a row of boxes stand on one line."""
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=face,
                                  edgecolor=edge, linewidth=1.1, linestyle=(0, (3, 2)) if dashed else "-", zorder=2))
    top = y + h
    if number:
        t = axis.text(x + 0.16, top - 0.14, number, ha="left", va="top", fontsize=TITLE_SIZE, fontweight="bold",
                      color=edge, zorder=3)
        fitted_texts.append((t, x, y, w, h))
        top -= 0.42
    elif note:
        t = axis.text(x + 0.16, top - 0.14, note, ha="left", va="top", fontsize=BODY_SIZE, color=edge,
                      style="italic", linespacing=1.2, fontfamily=["Arial", "DejaVu Sans"], zorder=3)
        fitted_texts.append((t, x, y, w, h))
        top -= 0.14 + LINE_BODY * (note.count("\n") + 1) + 0.30
    title_h = LINE_TITLE * (title.count("\n") + 1) if title else 0.0
    body_h = LINE_BODY * (product.count("\n") + 1) if product else 0.0
    gap_h = 0.16 if (title and product) else 0.0
    block_h = title_h + gap_h + body_h
    block_top = title_top if title_top is not None else y + (top - y + block_h) / 2.0
    if title:
        t = axis.text(x + w / 2, block_top, title, ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold",
                      color=INK, linespacing=1.15, zorder=3)
        fitted_texts.append((t, x, y, w, h))
    if product:
        t = axis.text(x + w / 2, block_top - title_h - gap_h, product, ha="center", va="top", fontsize=BODY_SIZE,
                      color=BODY_INK, linespacing=1.25, zorder=3)
        fitted_texts.append((t, x, y, w, h))


def line(axis, points, dashed=False, colour=INK, head=True):
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        axis.plot([x0, x1], [y0, y1], color=colour, linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                  solid_capstyle="projecting", zorder=1)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    if head:
        axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8.5, linewidth=1.0,
                                       color=colour, linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0,
                                       zorder=1))
    else:
        axis.plot([x0, x1], [y0, y1], color=colour, linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                  solid_capstyle="projecting", zorder=1)


def check_fit(figure, axis, margin=0.10):
    """Every box text must lie inside its box with a margin, in axis units.  Returns the list of offenders."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    offenders = []
    for text, x, y, w, h in fitted_texts:
        bbox = text.get_window_extent(renderer=renderer).transformed(axis.transData.inverted())
        if bbox.x0 < x + margin or bbox.x1 > x + w - margin or bbox.y0 < y + margin or bbox.y1 > y + h - margin:
            offenders.append((text.get_text().replace("\n", " / "), round(bbox.x0 - x, 2), round(x + w - bbox.x1, 2),
                              round(bbox.y0 - y, 2), round(y + h - bbox.y1, 2)))
    return offenders


def main():
    style.apply_style()
    figure, axis = plt.subplots(figsize=(7.48, 4.11))
    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 11.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.axis("off")

    # geometry: the producer column on the left, the FlowDivide frame on the right
    x0, w0 = 0.15, 3.1                          # the producer column
    fx, fy = x0 + w0 + 0.37, 0.1                # the FlowDivide frame
    fw, fh = 19.9 - fx, 10.75
    xs0 = fx + 0.38                             # inner left
    inner_w = fw - 0.76
    gap_top, gap_low = 0.28, 0.30
    n_top = 6
    w6 = (inner_w - (n_top - 1) * gap_top) / n_top
    x6 = [xs0 + k * (w6 + gap_top) for k in range(n_top)]     # FD1.0 .. FD1.5
    wl = (inner_w - gap_low) / 2.0
    xl = [xs0, xs0 + wl + gap_low]                            # FD2, FD3
    y1, h = 7.1, 3.2                            # top tier
    y2, hl = 0.55, 2.95                         # lower tier
    # the FD1 titles stand on one line: the tallest block of the row (FD1.5, 2 title lines, 5 body lines)
    # is centred in the space under the number line, and every box takes the same block top
    tallest = 2 * LINE_TITLE + 0.16 + 5 * LINE_BODY
    row_title_top = y1 + h - 0.42 - ((h - 0.42) - tallest) / 2.0

    axis.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0,rounding_size=0.25", facecolor="none",
                                  edgecolor=FRAME_EDGE, linewidth=1.2, zorder=0))
    axis.text(fx + 0.35, fy + fh - 0.12, "FlowDivide", ha="left", va="top", fontsize=8.5, fontweight="bold",
              color=FRAME_EDGE)
    axis.text(fx + 2.35, fy + fh - 0.16, "⚙ automatic", ha="left", va="top", fontsize=BODY_SIZE,
              color=FRAME_EDGE, style="italic", fontfamily=["Arial", "DejaVu Sans"])
    # the delivery: one pale field behind FD1 (top) and FD2 (lower left); FD3, the user's choice, lies outside it
    axis.add_patch(FancyBboxPatch((xs0 - 0.22, y1 - 0.22), inner_w + 0.44, h + 0.22 + 0.62,
                                  boxstyle="round,pad=0,rounding_size=0.18", facecolor=DELIVERED_FACE, edgecolor="none",
                                  zorder=0.5))
    axis.add_patch(FancyBboxPatch((xs0 - 0.22, y2 - 0.22), wl + 0.44, hl + 0.22 + 0.62,
                                  boxstyle="round,pad=0,rounding_size=0.18", facecolor=DELIVERED_FACE, edgecolor="none",
                                  zorder=0.5))
    axis.text(xs0 + 4.2, y1 + h + 0.1, "FD1  the partition", ha="left", va="bottom", fontsize=HEAD_SIZE,
              fontweight="bold", color=BLUE_EDGE)
    axis.text(xs0 + inner_w, y1 + h + 0.1, "FD1 and FD2 are delivered with the grid", ha="right", va="bottom",
              fontsize=BODY_SIZE, color=BLUE_EDGE, style="italic", zorder=4)

    # the data producer, outside: a dashed column, the DEM at the bottom and the flow direction at the top
    axis.add_patch(FancyBboxPatch((x0 - 0.12, fy), w0 + 0.24, fh, boxstyle="round,pad=0,rounding_size=0.25",
                                  facecolor="none", edgecolor=GREY_EDGE, linewidth=1.1, linestyle=(0, (4, 2.5)),
                                  zorder=0))
    axis.text(x0 + 0.2, fy + fh - 0.12, "the data producer", ha="left", va="top", fontsize=HEAD_SIZE,
              fontweight="bold", color=GREY_EDGE)
    yd, hd = y2, 1.7                            # DEM box
    ym, hm = yd + hd + 0.5, y1 - 0.5 - (yd + hd + 0.5)   # conditioning box fills the middle
    box(axis, x0, yd, w0, hd, "", "DEM acquisition", "satellite radar,\noptical stereo or lidar", GREY_FACE, GREY_EDGE)
    box(axis, x0, ym, w0, hm, "", "Hydrological\nconditioning",
        "buildings and\nvegetation removed;\ndepressions filled\nor breached;\nstreams burned;\nnatural sinks\nchecked by hand",
        GREY_FACE, GREY_EDGE, note="✎ extensive\nmanual correction")
    box(axis, x0, y1, w0, h, "", "Flow-direction\nassignment", "D8 flow direction", GREY_FACE, GREY_EDGE)
    line(axis, [(x0 + w0 / 2, yd + hd), (x0 + w0 / 2, ym)], colour=GREY_EDGE)
    line(axis, [(x0 + w0 / 2, ym + hm), (x0 + w0 / 2, y1)], colour=GREY_EDGE)
    line(axis, [(x0 + w0, y1 + h / 2), (x6[0], y1 + h / 2)], colour=GREY_EDGE)

    # FD1.0 .. FD1.5
    box(axis, x6[0], y1, w6, h, "FD1.0", "Assembly and\nrecoding", "one continental\nfile, one code\nconvention",
        BLUE_FACE, BLUE_EDGE, title_top=row_title_top)
    box(axis, x6[1], y1, w6, h, "FD1.1", "Flow\naccumulation", "upstream pixel\ncount and area", BLUE_FACE, BLUE_EDGE, title_top=row_title_top)
    box(axis, x6[2], y1, w6, h, "FD1.2", "Outlet\nindexing", "river mouths and\ninland sinks,\nranked by area;\nbasin table",
        BLUE_FACE, BLUE_EDGE, title_top=row_title_top)
    box(axis, x6[3], y1, w6, h, "FD1.3", "Watershed\ndelineation", "basin raster:\nevery pixel\nlabelled with\nits outlet",
        BLUE_FACE, BLUE_EDGE, title_top=row_title_top)
    box(axis, x6[4], y1, w6, h, "FD1.4", "Basin\ngrouping", "basin groups,\nfrom given units\nor automatically", BLUE_FACE, BLUE_EDGE, title_top=row_title_top)
    # FD1.5 fits every group to the memory capacity from either side: a group over the capacity
    # is cut or divided, groups under it are merged; the step is always in the chain, so the box is solid now
    box(axis, x6[5], y1, w6, h, "FD1.5", "Fitting to\nmemory", "over capacity:\na basin cut at\ntributary outlets,\na group divided;\nunder it: merged",
        BLUE_FACE, BLUE_EDGE, title_top=row_title_top)
    for k in range(5):
        line(axis, [(x6[k] + w6, y1 + h / 2), (x6[k + 1], y1 + h / 2)])

    # the partition leaves FD1.5 and feeds the lower tier through one bus with a dot
    ybus = 5.85
    xj = x6[5] + w6 / 2
    line(axis, [(xj, y1), (xj, ybus)], head=False)
    axis.add_patch(Circle((xj, ybus), 0.11, facecolor=INK, edgecolor="none", zorder=3))
    axis.text(xj - 0.25, ybus + 0.12, "the partition, regions numbered in downstream order", ha="right", va="bottom",
              fontsize=BODY_SIZE, color=BODY_INK, style="italic")
    xe2 = xl[0] + wl * 0.5                      # into FD2 (blue)
    xe3 = xl[1] + wl * 0.9                      # into FD3 (orange)
    line(axis, [(xj, ybus), (xe2, ybus), (xe2, y2 + hl)], colour=BLUE_EDGE)
    line(axis, [(xj, ybus), (xe3, ybus), (xe3, y2 + hl)], colour=ORANGE_EDGE)

    # lower tier: FD2 the views, FD3 the attributes
    axis.text(xl[0], y2 + hl + 0.12, "FD2  the views", ha="left", va="bottom", fontsize=HEAD_SIZE, fontweight="bold",
              color=BLUE_EDGE)
    box(axis, xl[0], y2, wl, hl, "FD2", "Folding, vectorisation, colouring",
        "the basin and region masks folded to a coarse\nresolution, one object per cell; vectorised into\n"
        "polygons carrying the tables; each polygon given\none of eight colours, no two neighbours alike,\n"
        "saved as the default style",
        BLUE_FACE, BLUE_EDGE)
    axis.text(xl[1], y2 + hl + 0.12, "FD3  the user's choice", ha="left", va="bottom", fontsize=HEAD_SIZE,
              fontweight="bold", color=ORANGE_EDGE)
    box(axis, xl[1], y2, wl, hl, "FD3", "", "", ORANGE_FACE, ORANGE_EDGE)
    t = axis.text(xl[1] + wl / 2, y2 + hl - 0.42, "Attribute derivation, region by region", ha="center", va="center",
                  fontsize=TITLE_SIZE, fontweight="bold", color=INK, zorder=3)
    fitted_texts.append((t, xl[1], y2, wl, hl))
    rows = [("accumulate", "Shreve magnitude"),
            ("label", "distance to the outlet, Hack order"),
            ("maximum", "upstream flow length, longest flow path"),
            ("order", "Strahler order"),
            ("user-defined", "a rule of one of the four classes")]
    for k, (name, examples) in enumerate(rows):
        yy = y2 + hl - 0.95 - 0.36 * k
        t = axis.text(xl[1] + 0.45, yy, name, ha="left", va="center", fontsize=BODY_SIZE, color=BODY_INK, zorder=3)
        fitted_texts.append((t, xl[1], y2, wl, hl))
        t = axis.text(xl[1] + 2.35, yy, examples, ha="left", va="center", fontsize=BODY_SIZE, color=BODY_INK, zorder=3)
        fitted_texts.append((t, xl[1], y2, wl, hl))
    t = axis.text(xl[1] + wl - 0.2, y2 + 0.14, "the first four provided", ha="right", va="bottom",
                  fontsize=BODY_SIZE, color=ORANGE_EDGE, style="italic", zorder=3)
    fitted_texts.append((t, xl[1], y2, wl, hl))

    offenders = check_fit(figure, axis)
    for text, left, right, bottom, top in offenders:
        print("DOES NOT FIT:", text, "| margins left/right/bottom/top:", left, right, bottom, top)
    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    if offenders:
        raise SystemExit("texts outside their boxes; not saved")
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_02_workflow")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
