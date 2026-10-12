"""One style for every FlowDivide figure: one font, one ladder of sizes, one way of labelling panels.
Every figure is drawn at the double-column width of 7.48 inch, so the point size in the file is the point size on paper;
nothing goes below 7 pt.

    import style
    style.apply_style()
    style.panel_heading(axis, "(a)", "Flow direction")
    style.panel_foot(axis, "84 pixels")
"""
import matplotlib

# Arial, not Helvetica: macOS ships Helvetica as one .ttc and matplotlib only opens its first face,
# so "Helvetica bold" renders regular; Arial has separate regular and bold files and the same look.
FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]

SIZE_LETTER = 8.5      # "(a)", bold
SIZE_TITLE = 8.0       # panel title, regular, same line as the letter
SIZE_SUBTITLE = 7.2    # optional second line under the title
SIZE_FOOT = 7.2        # one-line note under a panel
SIZE_TICK = 7.5        # map tick labels
SIZE_CELL = 8.5        # numbers inside test-grid pixels (Figure 1a, three-digit D8 codes, uses 8.0)
SIZE_LABEL = 7.0       # region / piece numbers on maps
SIZE_LEGEND = 7.5

INK = "#12161c"
TEXT_COLOUR = "#2b3038"
FOOT_COLOUR = "#2b3038"

# test grid (Figures 1 and A1 to A3): basin 1 and basin 2 are told apart by the colour of the number
# written in the pixel, blue and green, on a white pixel; a pixel fill is reserved for the pieces
# and for the exit grey, so a fill never means a basin.
BASIN_INK = {1: "#1f5fbf", 2: "#2a8a3e"}
BASIN_WORD = {1: "blue", 2: "green"}
# the channel line on the test grid, the same two colours, lighter so the numbers stay readable
CHANNEL_LINE = {1: "#9dc3ee", 2: "#9fd4a8"}
# the four pieces of Figure 1c: no blue, no green
PIECE_FACE = {1: "#f6dfc2", 2: "#e3d5f5", 3: "#fbeaa8", 4: "#f7d3d3"}

# the interface of the test grid (Figure 1c): the two cross-piece exits and their entry pixels,
# and the two basin outlets, named by letters so that the figures need no pixel coordinates
INTERFACE_LETTER = {(4, 4): "A", (4, 5): "A\u2032", (7, 5): "B", (8, 4): "B\u2032", (9, 0): "$O_1$", (9, 9): "$O_2$"}
# the pieces read in order, once each: upstream-to-downstream variables (UPG, LUP, ORD), and the
# reverse for the basin label, which travels from the outlet upwards
PIECE_STAGES_DOWN = [(1, 4), (2,), (3,)]
PIECE_STAGES_UP = [(3, 4), (2,), (1,)]

# a crossing drawn inside one grid (the pieces of Figures 1 and A1 to A3): the exit pixel and its inlet
# share an edge or a corner, so the red arrow has to fit between their two numbers.  Along an edge the
# two numbers move this far apart (pixel fraction) and the arrow is shrunk to the gap left; across a
# corner the diagonal already clears the numbers.  The tiles are drawn apart and need neither.
CROSSING_NUDGE = 0.12
CROSSING_SHRINK_EDGE = 3.5      # points, both ends
CROSSING_SHRINK_CORNER = 5.0    # points, both ends

# basin outlet on a real map: red downward triangle, black edge (the same in every figure)
OUTLET_MARKER = dict(marker="v", markerfacecolor="#d7301f", markeredgecolor="#12161c",
                     markeredgewidth=0.6, markersize=6.0, linestyle="none")


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": SIZE_TITLE,
        "axes.titlesize": SIZE_TITLE,
        "axes.labelsize": SIZE_TITLE,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "legend.fontsize": SIZE_LEGEND,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })


def panel_heading(axis, letter, title_text, subtitle_text=None, pad_points=4.0):
    """'(a) Title' on one line above the panel, left aligned: the letter bold, the title regular.
    An optional subtitle goes on a second, smaller line between the title and the panel.
    Everything is anchored to the panel's top-left corner in points, so a later change of the
    axes box (aspect, tight bbox) moves the heading with the panel and keeps the gap after the
    letter."""
    figure = axis.figure
    renderer = figure.canvas.get_renderer()
    probe = axis.text(0.0, 1.0, letter + " ", transform=axis.transAxes, fontsize=SIZE_LETTER,
                      fontweight="bold", va="bottom", ha="left")
    letter_width_points = probe.get_window_extent(renderer=renderer).width * 72.0 / figure.dpi
    probe.remove()
    y_points = pad_points
    if subtitle_text is not None:
        axis.annotate(subtitle_text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, y_points),
                      textcoords="offset points", fontsize=SIZE_SUBTITLE, va="bottom", ha="left",
                      color=TEXT_COLOUR, annotation_clip=False)
        y_points += SIZE_SUBTITLE * 1.35
    axis.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, y_points),
                  textcoords="offset points", fontsize=SIZE_LETTER, fontweight="bold", va="bottom",
                  ha="left", color=INK, annotation_clip=False)
    axis.annotate(title_text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(letter_width_points, y_points),
                  textcoords="offset points", fontsize=SIZE_TITLE, va="bottom", ha="left", color=INK,
                  annotation_clip=False)


def panel_foot(axis, foot_text, y=-0.03):
    """one short line under the panel, centred"""
    axis.text(0.5, y, foot_text, transform=axis.transAxes, fontsize=SIZE_FOOT, ha="center", va="top",
              color=FOOT_COLOUR)


def figure_path(figure_directory, stem):
    """<figure_directory>/<stem>.png; running a script again draws its figure again in place"""
    import os
    os.makedirs(figure_directory, exist_ok=True)
    return os.path.join(figure_directory, stem + ".png")
