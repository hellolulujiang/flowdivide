"""data_a5_length_views: the continental maps of Figure A5 (LDN, LUP, LFP).
Reads partitions/160deg2/global/coarseresolution/<attribute>/<attribute>_south-america_400th.tif of the published
products and reads them down for a printed panel: LDN by 10 with nearest neighbour, LUP to 20 cells a degree by the
block maximum, since its large values sit on the channel pixels only.  Copies the longest flow paths of the basins of at least 100 km2 out of the
1/400 degree GeoPackage.

Writes into data/:
    ldn_south-america_400th_view.npy   float32, m, -1 = nodata (read down by 10, nearest neighbour)
    lup_south-america_20th_view.npy    float32, m, -9999 = nodata (block maximum down to 20 cells a degree)
    lfp_south-america_400th_ge100km2.geojson

    python3 data_a5_length_views.py
"""
import json
import os
import sys
import time
import numpy as np
from osgeo import gdal, ogr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

gdal.UseExceptions()
ROOT = os.path.join(paths.HYDROSHEDS30M_ROOT, "south-america", "partitions", "160deg2", "global")
DATA_DIRECTORY = paths.DATA_DIRECTORY
READ_DOWN = 10
LFP_MINIMUM_AREA_KM2 = 100.0       # the longest flow path of every basin of at least 100 km2

# LDN varies smoothly, so one sample per block is faithful
MOSAIC = (-93.0, -32.0, -57.0, 15.0)
for attribute in ("ldn",):
    source = "%s/coarseresolution/%s/%s_south-america_400th.tif" % (ROOT, attribute, attribute)
    started = time.time()
    dataset = gdal.Open(source, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
    buf_x, buf_y = nx // READ_DOWN, ny // READ_DOWN
    nodata = band.GetNoDataValue()
    array = band.ReadAsArray(0, 0, buf_x * READ_DOWN, buf_y * READ_DOWN, buf_xsize=buf_x, buf_ysize=buf_y,
                             resample_alg=gdal.GRIORA_NearestNeighbour)
    out = os.path.join(DATA_DIRECTORY, "%s_south-america_400th_view.npy" % attribute)
    np.save(out, array)
    valid = array != nodata
    print("%s  nearest  %d x %d -> %s  nodata %s  min %.4g  max %.4g  %.0f s" % (
        attribute, nx, ny, array.shape, nodata, array[valid].min(), array[valid].max(),
        time.time() - started))
    dataset = None

# LUP is large only on the channel pixels, which one sample per block would miss, so it is read down to 20 cells a
# degree by the block maximum: a river is one cell wide in the 1/400-degree product and a panel 2.2 inch wide cannot
# show 2,440 columns, so imshow would drop most river cells; at 20 cells a degree every river cell is a printed pixel.
# gdal.Warp with resampleAlg="max" does that (gdal.Translate falls back to nearest for "max" in GDAL 3.6)
started = time.time()
dataset = gdal.Open("%s/coarseresolution/lup/lup_south-america_400th.tif" % ROOT, gdal.GA_ReadOnly)
small = gdal.Warp("/vsimem/lup_20th.tif", dataset, xRes=1 / 20.0, yRes=1 / 20.0, resampleAlg="max",
                  outputBounds=(MOSAIC[0], MOSAIC[2], MOSAIC[1], MOSAIC[3]), srcNodata=-9999, dstNodata=-9999)
array = small.GetRasterBand(1).ReadAsArray()
small = None
gdal.Unlink("/vsimem/lup_20th.tif")
np.save(os.path.join(DATA_DIRECTORY, "lup_south-america_20th_view.npy"), array)
print("lup  max to 20 cells a degree  %s  %.0f s" % (array.shape, time.time() - started))
dataset = None

# the longest flow paths of the large basins, as GeoJSON lines
started = time.time()
source = ogr.Open("%s/vector/lfp_south-america_400th.gpkg" % ROOT)
layer = source.GetLayer(0)
features = []
for feature in layer:
    if feature.GetField("basin_area_km2") < LFP_MINIMUM_AREA_KM2:
        continue
    geometry = json.loads(feature.GetGeometryRef().ExportToJson())
    features.append({"type": "Feature", "geometry": geometry,
                     "properties": {"basin_id": int(feature.GetField("basin_id")),
                                    "basin_area_km2": float(feature.GetField("basin_area_km2")),
                                    "longest_flow_path_m": float(feature.GetField("longest_flow_path_m"))}})
with open(os.path.join(DATA_DIRECTORY, "lfp_south-america_400th_ge100km2.geojson"), "w") as handle:
    json.dump({"type": "FeatureCollection", "features": features}, handle)
print("lfp  %d paths of basins >= %.0f km2  %.0f s" % (len(features), LFP_MINIMUM_AREA_KM2, time.time() - started))
