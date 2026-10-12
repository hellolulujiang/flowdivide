"""data_04_amazon_pieces: the Amazon's Pfafstetter pieces for Figures 4 and 6.
Copies from the 160 deg2 partition of South America into data/:
    pfafstetter_pieces_south-america_basin1.csv   the pieces on three levels, which are used, and the region of each
                                                  used piece (as pfafstetter_pieces_south-america_basin1_160deg2.csv)
    pfafstetter_pieces_south-america_basin1_30s.tif
                                                  the deepest-level piece index of every 30th pixel
                                                  (as pfafstetter_pieces_south-america_basin1_30s_160deg2.tif)
    piece_fine_south-america.csv, Amazon rows     one row per used piece, with its outlet and the piece it drains into
                                                  (as piece_fine_south-america_basin1_160deg2.csv)
"""
import os
import sys
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ROOT = os.path.join(paths.HYDROSHEDS30M_ROOT, "south-america", "partitions", "160deg2", "global")
DATA_DIRECTORY = paths.DATA_DIRECTORY

shutil.copyfile(os.path.join(ROOT, "table", "pfafstetter_pieces_south-america_basin1.csv"),
                os.path.join(DATA_DIRECTORY, "pfafstetter_pieces_south-america_basin1_160deg2.csv"))
shutil.copyfile(os.path.join(ROOT, "preview", "pfafstetter_pieces_south-america_basin1_30s.tif"),
                os.path.join(DATA_DIRECTORY, "pfafstetter_pieces_south-america_basin1_30s_160deg2.tif"))
with open(os.path.join(ROOT, "table", "piece_fine_south-america.csv")) as source, \
     open(os.path.join(DATA_DIRECTORY, "piece_fine_south-america_basin1_160deg2.csv"), "w") as target:
    header = source.readline()
    target.write(header)
    column_of_basin = header.split().index("basin_id")
    kept = 0
    for line in source:
        if line.split()[column_of_basin] == "1":
            target.write(line)
            kept += 1
print("pieces table, 30 arc-second preview and %d unit rows of the Amazon copied" % kept)
