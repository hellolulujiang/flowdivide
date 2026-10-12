"""data_03_basin_view: the basin raster of South America for the continental maps of Figures 3, 5, 6 and A5.
Reads the published global/coarseresolution/bsn/bsn_south-america_400th.tif and reads it down by 10 with nearest
neighbour, which keeps whole basin ids: a 7.48 inch two-column panel at 400 dpi holds about 22 pixels a degree.
Writes bsn_south-america_400th_view.npy into data/.

    python3 data_03_basin_view.py
"""
import os
import sys
import time
import numpy as np
from osgeo import gdal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

gdal.UseExceptions()
ROOT = paths.HYDROSHEDS30M_ROOT
DATA_DIRECTORY = paths.DATA_DIRECTORY
READ_DOWN = {"south-america": 10}

for continent in ["south-america"]:
    source = "%s/%s/global/coarseresolution/bsn/bsn_%s_400th.tif" % (ROOT, continent, continent)
    started = time.time()
    dataset = gdal.Open(source, gdal.GA_ReadOnly)
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
    step = READ_DOWN[continent]
    buf_x, buf_y = nx // step, ny // step
    array = dataset.GetRasterBand(1).ReadAsArray(
        0, 0, buf_x * step, buf_y * step, buf_xsize=buf_x, buf_ysize=buf_y,
        resample_alg=gdal.GRIORA_NearestNeighbour)
    np.save(os.path.join(DATA_DIRECTORY, "bsn_%s_400th_view.npy" % continent), array.astype(np.int32))
    geotransform = dataset.GetGeoTransform()
    print("%s  source %d x %d at 1/400  ->  %s in %.0f s  lon %.1f..%.1f lat %.1f..%.1f  land %d"
          % (continent, nx, ny, array.shape, time.time() - started,
             geotransform[0], geotransform[0] + nx * geotransform[1],
             geotransform[3] + ny * geotransform[5], geotransform[3], int((array > 0).sum())))
