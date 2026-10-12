"""Figure 1: the 10 x 10 test grid, the two ways of cutting it, and the upstream pixel count (UPG) computed on
each cut.  Nine panels in three rows:
    (a) the D8 flow directions  (b) four regular tiles  (c) the cut at two channel junctions   [panels_test_grid]
    (d) tiles, pass 1  (e) the exit graph  (f) tiles, pass 2                                   [panels_upg_tiles_vs_pieces]
    (g) pieces 1 and 4  (h) piece 2  (i) piece 3, each read once, in order
The other three attribute classes (label, maximum, order) are Figures A1 to A3.
Reads data_01_test_grid_10x10.py; writes fig_01_test_grid_upg.png into ../figures.

    python3 fig_01_test_grid_upg.py
"""
import os
import paths
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
import panels_test_grid as test_grid
import panels_upg_tiles_vs_pieces as upg

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIRECTORY = paths.FIGURE_DIRECTORY


def main():
    style.apply_style()
    figure, axes = plt.subplots(3, 3, figsize=(7.48, 8.0))
    plt.subplots_adjust(left=0.02, right=0.99, top=0.965, bottom=0.035, wspace=0.14, hspace=0.24)
    test_grid.draw_row(axes[0], "abc")
    upg.draw_rows(axes[1:], "defghi")
    if os.environ.get("FLOWDIVIDE_NO_SAVE"):
        return
    out_path = style.figure_path(FIGURE_DIRECTORY, "fig_01_test_grid_upg")
    figure.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(out_path)


if __name__ == "__main__":
    main()
