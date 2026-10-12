"""data_04_amazon_longest_flow_path: the Amazon's longest flow path for Figure 4.
The 1/400 degree GeoPackage of the longest flow paths keeps one vertex per 1/400 degree cell the path enters; this script
copies the Amazon's line out of it into data/ as lfp_amazon_400th.geojson.
"""
import json
import os
import sys
import fiona
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

SOURCE = os.path.join(paths.HYDROSHEDS30M_ROOT, "south-america", "partitions", "160deg2",
                      "global", "vector", "lfp_south-america_400th.gpkg")
DATA_DIRECTORY = paths.DATA_DIRECTORY
BASIN_ID = 1

with fiona.open(SOURCE) as source:
    feature = next(f for f in source if f["properties"]["basin_id"] == BASIN_ID)
out_path = os.path.join(DATA_DIRECTORY, "lfp_amazon_400th.geojson")
with open(out_path, "w") as handle:
    json.dump({"type": "FeatureCollection",
               "features": [{"type": "Feature", "properties": dict(feature["properties"]),
                             "geometry": dict(feature["geometry"])}]}, handle)
print("%s: %d vertices, path_grid_count %d" % (out_path, len(feature["geometry"]["coordinates"]),
                                          feature["properties"]["path_grid_count"]))
