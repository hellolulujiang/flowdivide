"""data_a6_network_lines: the stream-segment lines of Figure A6 (the continent).
Reads partitions/160deg2/global/vector/net_south-america_400th.gpkg of the published products (one line per stream
segment, carrying its Strahler order, Shreve magnitude and Hack order) and writes into data/
    net_south-america_400th_ord_ge4.npz   every segment of Strahler order 4 and above (1,273,922 of 9,431,876 lines):
                                          coordinates (n, 2) lon/lat; offsets (m+1), line i is
                                          coordinates[offsets[i]:offsets[i+1]]; strahler_order, shreve_magnitude,
                                          hack_order, basin_id (m)
The example basin is extracted by data_a5_example_basin.py.

    python3 data_a6_network_lines.py
"""
import os
import sys
import time
import numpy as np
import pyogrio
import shapely
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
PARTITION_VECTOR = os.path.join(paths.HYDROSHEDS30M_ROOT, "south-america", "partitions", "160deg2", "global", "vector")
CONTINENT_MIN_ORDER = 4


def lines_to_npz(gpkg_path, npz_path, where=None):
    started = time.time()
    frame = pyogrio.read_dataframe(gpkg_path, layer="stream_segments", where=where,
                                   columns=["segment_id", "basin_id", "strahler_order", "shreve_magnitude", "hack_order"])
    geometry = frame.geometry.values
    coordinates = shapely.get_coordinates(geometry)
    counts = shapely.get_num_coordinates(geometry)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.savez_compressed(npz_path, coordinates=coordinates, offsets=offsets,
                        strahler_order=frame["strahler_order"].to_numpy(np.int16),
                        shreve_magnitude=frame["shreve_magnitude"].to_numpy(np.int64),
                        hack_order=frame["hack_order"].to_numpy(np.int16),
                        basin_id=frame["basin_id"].to_numpy(np.int64))
    print("%s: %d lines, %d vertices, %.0f s" % (os.path.basename(npz_path), len(frame), len(coordinates),
                                                 time.time() - started))


def main():
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    lines_to_npz(os.path.join(PARTITION_VECTOR, "net_south-america_400th.gpkg"),
                 os.path.join(DATA_DIRECTORY, "net_south-america_400th_ord_ge%d.npz" % CONTINENT_MIN_ORDER),
                 where="strahler_order >= %d" % CONTINENT_MIN_ORDER)


if __name__ == "__main__":
    main()
