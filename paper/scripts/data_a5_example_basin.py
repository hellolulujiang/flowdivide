"""data_a5_example_basin: the example basin of Figures A5 and A6, drawn at 1 arc-second so that its pixels can be seen.
Basin 10827: 10.3 km2 on the Atlantic coast of Brazil south of Florianopolis (outlet 48.60 W, 27.87 S), Strahler order
3 at the outlet, 9 stream segments at the 1 km2 channel threshold.

Reads the published products of South America (the basin table, the 1-arc-second flow directions and basin raster, the
LDN, LUP and LFP rasters and the per-basin tables of the 160 deg2 partition, and the stream segments of the basin) and
writes into data/:
    example_basin_10827_outlet.json        outlet, area, bounding box, the orders at the outlet, the longest flow path
    example_basin_10827_boundary.geojson   the outer boundary, polygonised from the basin raster
    example_basin_10827_lines.npz          every stream segment with its Strahler order, Shreve magnitude, Hack order
    example_basin_10827_ldn_lup.npz        the distance to the outlet and the upstream flow length, m, over the basin's
                                           rectangle (NaN outside the basin), with the extent
    example_basin_10827_lfp.npz            the longest flow path as an ordered run of pixel centres (lon, lat)

    python3 data_a5_example_basin.py [basin_id]      (default 10827)
"""
import json
import os
import sys
import time
import numpy as np
import pyogrio
import shapely
from osgeo import gdal, ogr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = paths.DATA_DIRECTORY
ROOT = os.path.join(paths.HYDROSHEDS30M_ROOT, "south-america")
PARTITION = os.path.join(ROOT, "partitions", "160deg2", "global")
EXAMPLE_BASIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10827
BASIN_TABLE = os.path.join(ROOT, "global", "table", "basin_table_fine_south-america.csv")
DIR_MOSAIC = os.path.join(ROOT, "global", "fineresolution", "dir", "dir_south-america_1s_merit.tif")
BASIN_MASK = os.path.join(ROOT, "global", "fineresolution", "bsn", "bsn_south-america_1s.tif")
# the MERIT convention of the flow direction: 1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE; 0 river mouth, 255 sink
D8_OFFSET = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def table_row(path, basin_id):
    with open(path) as handle:
        header = handle.readline().split()
        for line in handle:
            fields = line.split()
            if int(fields[0]) == basin_id:
                return header, fields
    raise SystemExit("basin %d is not in %s" % (basin_id, path))


def per_basin_value(name, basin_id, column):
    with open(os.path.join(PARTITION, "table", "%s_basin_south-america.csv" % name)) as handle:
        header = handle.readline().split()
        for line in handle:
            fields = line.split()
            if int(fields[0]) == basin_id:
                return float(fields[header.index(column)])
    raise SystemExit("basin %d is not in the %s table" % (basin_id, name))


def read_window(path, row0, col0, nrow, ncol):
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray(col0, row0, ncol, nrow)
    no_data = band.GetNoDataValue()
    geotransform = dataset.GetGeoTransform()
    dataset = None
    return values, no_data, geotransform


def main():
    started = time.time()
    header, fields = table_row(BASIN_TABLE, EXAMPLE_BASIN)
    row = dict(zip(header, fields))
    row0, row1 = int(row["basin_row_min"]), int(row["basin_row_max"])
    col0, col1 = int(row["basin_col_min"]), int(row["basin_col_max"])
    nrow, ncol = row1 - row0, col1 - col0          # the table's rectangles are left closed and right open

    # the basin mask of the rectangle, its outer boundary as a polygon (8-connected, as the basin views draw it)
    mask, _, geotransform = read_window(BASIN_MASK, row0, col0, nrow, ncol)
    inside = (mask == EXAMPLE_BASIN)
    print("basin %d: rectangle rows %d..%d, cols %d..%d (%d x %d), %d pixels in the basin, table says %s" % (
        EXAMPLE_BASIN, row0, row1, col0, col1, nrow, ncol, inside.sum(), row["basin_grid_count"]))
    assert int(inside.sum()) == int(row["basin_grid_count"])
    window_geotransform = (geotransform[0] + col0 * geotransform[1], geotransform[1], 0.0,
                           geotransform[3] + row0 * geotransform[5], 0.0, geotransform[5])
    extent = (window_geotransform[0], window_geotransform[0] + ncol * geotransform[1],
              window_geotransform[3] + nrow * geotransform[5], window_geotransform[3])   # lon min, lon max, lat min, lat max
    memory = gdal.GetDriverByName("MEM").Create("", ncol, nrow, 1, gdal.GDT_Byte)
    memory.SetGeoTransform(window_geotransform)
    memory.GetRasterBand(1).WriteArray(inside.astype(np.uint8))
    vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = vector.CreateLayer("parts", None, ogr.wkbPolygon)
    gdal.Polygonize(memory.GetRasterBand(1), memory.GetRasterBand(1), layer, -1, ["8CONNECTED=8"])
    parts = [shapely.from_wkb(bytes(feature.GetGeometryRef().ExportToWkb())) for feature in layer]
    outline = shapely.union_all(parts)
    boundary_path = os.path.join(DATA_DIRECTORY, "example_basin_%d_boundary.geojson" % EXAMPLE_BASIN)
    with open(boundary_path, "w") as handle:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"basin_id": EXAMPLE_BASIN}, "geometry": json.loads(shapely.to_geojson(outline))}]},
            handle)
    print("boundary: %d part(s), %s, %.0f vertices" % (len(parts), outline.geom_type, shapely.get_num_coordinates(outline)))

    # the two length layers over the rectangle, NaN outside the basin
    ldn, ldn_no_data, _ = read_window(os.path.join(PARTITION, "fineresolution", "ldn", "ldn_south-america_1s.tif"),
                                      row0, col0, nrow, ncol)
    lup, lup_no_data, _ = read_window(os.path.join(PARTITION, "fineresolution", "lup", "lup_south-america_1s.tif"),
                                      row0, col0, nrow, ncol)
    ldn = np.where(inside & (ldn != ldn_no_data), ldn, np.nan).astype(np.float32)
    lup = np.where(inside & (lup != lup_no_data), lup, np.nan).astype(np.float32)
    print("ldn 0..%.0f m, lup 0..%.0f m inside the basin; no data outside: %d / %d pixels of the basin" % (
        np.nanmax(ldn), np.nanmax(lup), np.isnan(ldn[inside]).sum(), np.isnan(lup[inside]).sum()))
    np.savez_compressed(os.path.join(DATA_DIRECTORY, "example_basin_%d_ldn_lup.npz" % EXAMPLE_BASIN),
                        ldn_m=ldn, lup_m=lup, inside=inside, extent=np.array(extent))

    # the longest flow path, walked from its head down the flow directions to the outlet
    head_row = int(per_basin_value("lfp", EXAMPLE_BASIN, "head_row"))
    head_col = int(per_basin_value("lfp", EXAMPLE_BASIN, "head_col"))
    path_pixels = int(per_basin_value("lfp", EXAMPLE_BASIN, "path_grid_count"))
    direction, _, _ = read_window(DIR_MOSAIC, row0, col0, nrow, ncol)
    lfp_raster, _, _ = read_window(os.path.join(PARTITION, "fineresolution", "lfp", "lfp_south-america_1s.tif"),
                                   row0, col0, nrow, ncol)
    outlet_row, outlet_col = int(row["outlet_row"]), int(row["outlet_col"])
    walk = []
    at_row, at_col = head_row - row0, head_col - col0
    while True:
        assert lfp_raster[at_row, at_col] == EXAMPLE_BASIN, "a walked pixel is not on the lfp raster"
        walk.append((at_row, at_col))
        if (at_row + row0, at_col + col0) == (outlet_row, outlet_col):
            break
        code = int(direction[at_row, at_col])
        assert code in D8_OFFSET, "the path stops at code %d before the outlet" % code
        at_row += D8_OFFSET[code][0]
        at_col += D8_OFFSET[code][1]
        assert len(walk) <= path_pixels, "walked past the table's path_grid_count"
    assert len(walk) == path_pixels, "walked %d pixels, the table says %d" % (len(walk), path_pixels)
    assert int((lfp_raster == EXAMPLE_BASIN).sum()) == path_pixels
    walk = np.array(walk)
    lon = window_geotransform[0] + (walk[:, 1] + 0.5) * window_geotransform[1]
    lat = window_geotransform[3] + (walk[:, 0] + 0.5) * window_geotransform[5]
    np.savez_compressed(os.path.join(DATA_DIRECTORY, "example_basin_%d_lfp.npz" % EXAMPLE_BASIN),
                        lon=lon, lat=lat)
    print("longest flow path: %d pixels, head to outlet along DIR, equal to the table" % len(walk))

    # the stream segments with their three orders
    lines = pyogrio.read_dataframe(os.path.join(PARTITION, "vector", "net_south-america_1s_basin%d.gpkg" % EXAMPLE_BASIN),
                                   layer="stream_segments")
    geometry = lines.geometry.values
    coordinates = shapely.get_coordinates(geometry)
    offsets = np.concatenate([[0], np.cumsum(shapely.get_num_coordinates(geometry))]).astype(np.int64)
    np.savez_compressed(os.path.join(DATA_DIRECTORY, "example_basin_%d_lines.npz" % EXAMPLE_BASIN),
                        coordinates=coordinates, offsets=offsets,
                        strahler_order=lines["strahler_order"].to_numpy(np.int16),
                        shreve_magnitude=lines["shreve_magnitude"].to_numpy(np.int64),
                        hack_order=lines["hack_order"].to_numpy(np.int16))
    print("%d segments, %d vertices" % (len(lines), len(coordinates)))

    outlet = {"basin_id": EXAMPLE_BASIN, "outlet_lon": float(row["outlet_lon"]), "outlet_lat": float(row["outlet_lat"]),
              "basin_area_km2": float(row["basin_area_km2"]), "basin_grid_count": int(row["basin_grid_count"]),
              "bbox": [float(row["bbox_minlon"]), float(row["bbox_minlat"]), float(row["bbox_maxlon"]), float(row["bbox_maxlat"])],
              "extent": list(extent),
              "order_at_outlet": int(per_basin_value("strahler", EXAMPLE_BASIN, "order_at_outlet")),
              "magnitude_at_outlet": int(per_basin_value("shreve", EXAMPLE_BASIN, "magnitude_at_outlet")),
              "largest_hack_order": int(per_basin_value("hack", EXAMPLE_BASIN, "largest_order")),
              "longest_flow_path_m": per_basin_value("lfp", EXAMPLE_BASIN, "longest_flow_path_m"),
              "segments": int(len(lines))}
    with open(os.path.join(DATA_DIRECTORY, "example_basin_%d_outlet.json" % EXAMPLE_BASIN), "w") as handle:
        json.dump(outlet, handle, indent=1)
    print(json.dumps(outlet), "%.0f s" % (time.time() - started))


if __name__ == "__main__":
    main()
