"""data_05_region_views: the region rasters and region tables of Figure 5, at the three capacities (also read by Figures 6
and A4).
Reads partitions/<cap>deg2/global/coarseresolution/rgn/rgn_south-america_400th.tif of the published products and reads
it down as data_03_basin_view does (nearest neighbour, one pixel in ten), so that the region view lies on the grid of the
basin view; copies the region table of each partition.  Writes into data/, for cap = 160, 80, 40:
    rgn_south-america_400th_view_<cap>deg2.npy
    region_fine_south-america_<cap>deg2.csv

    python3 data_05_region_views.py
"""
import os
import shutil
import sys
import time
import numpy as np
from osgeo import gdal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

gdal.UseExceptions()
ROOT = paths.HYDROSHEDS30M_ROOT
DATA_DIRECTORY = paths.DATA_DIRECTORY
continent = "south-america"
step = 10

for square_degrees in (160, 80, 40):
    partition = "%s/%s/partitions/%ddeg2/global" % (ROOT, continent, square_degrees)
    source = "%s/coarseresolution/rgn/rgn_%s_400th.tif" % (partition, continent)
    started = time.time()
    dataset = gdal.Open(source, gdal.GA_ReadOnly)
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
    buf_x, buf_y = nx // step, ny // step
    array = dataset.GetRasterBand(1).ReadAsArray(
        0, 0, buf_x * step, buf_y * step, buf_xsize=buf_x, buf_ysize=buf_y,
        resample_alg=gdal.GRIORA_NearestNeighbour)
    out_path = os.path.join(DATA_DIRECTORY,
                            "rgn_%s_400th_view_%ddeg2.npy" % (continent, square_degrees))
    np.save(out_path, array.astype(np.int32))
    table_source = "%s/table/region_fine_%s.csv" % (partition, continent)
    table_copy = os.path.join(DATA_DIRECTORY,
                              "region_fine_%s_%ddeg2.csv" % (continent, square_degrees))
    shutil.copyfile(table_source, table_copy)
    print("%d deg2  source %d x %d  ->  %s in %.0f s  regions %d  land %d"
          % (square_degrees, nx, ny, array.shape, time.time() - started,
             len(np.unique(array[array > 0])), int((array > 0).sum())))
