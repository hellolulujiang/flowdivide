"""The 10 x 10 test grid of Figures 1 and A1 to A3.
Two basins: basin 1, 84 pixels, drains to the bottom-left corner; basin 2, 8 pixels, along the right edge to the
bottom-right corner; 8 pixels are outside both (code 247).  D8 codes 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE,
0 the outlet.  Cut into 2 x 2 tiles of 5 x 5 pixels, 15 pixels drain into another tile.  Basin 1 is cut at CUT_NODES,
the two channel junctions A and B of its main stem (upstream counts 40 and 65), each kept in the piece upstream of it.

    from data_01_test_grid_10x10 import DIR, BSN, OUTLETS, CUT_NODES
"""
import numpy as np

DIR = np.array([
    [  2, 247,   2, 247, 247,   2,  16,   8,   4, 247],
    [247,   4,  64,   4, 128,  64,   2,  64,   8,  16],
    [  1,   2,  32,   2,  32,   8,  16,  16,  64,   8],
    [ 64,  32,   2,  64,   4,  64,  32,  32,   4,   8],
    [  2, 128,  64,   1,   1,   2,  16, 128,   8,   4],
    [247,   4, 128,  64,  32,  32,   8,  16,  64,   4],
    [247,   2,   4,  16,   2,  16,  64,  32,  32,   4],
    [128,  64,   4,  32,  64,   8,  16,  16,   8,   4],
    [  4, 128,   8,  16,  16,  64,  32,  32,  16,   4],
    [  0,  16,  64,  32,  32, 247, 128,   1,   1,   0],
], dtype=np.uint8)

BSN = np.array([
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 0, 1, 2, 2, 2],
], dtype=np.int32)

MSK = (BSN > 0).astype(np.uint8)
OUTLETS = {1: (9, 0), 2: (9, 9)}
# the two channel junctions of the main stem where basin 1 is cut into three pieces, A at (4, 4) and B at (7, 5);
# the piece upstream keeps the junction pixel
CUT_NODES = [(4, 4), (7, 5)]
