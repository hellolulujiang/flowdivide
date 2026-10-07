"""fd1_partition.py -- FlowDivide stage 1, the partition (FD1.0 to FD1.5 of Figure 2).

FlowDivide (Jiang, EMS manuscript 2026) takes a D8 flow-direction grid that is too large for the
memory of the machine and turns it into regions that fit a chosen capacity, so that every
hydrographic attribute can be computed region by region, exactly, without any boundary work by
the user.  This file is the first of the three stages:

    FD1.0  assembly and recoding      one continental file, the MERIT code convention
    FD1.1  flow accumulation          upstream pixel count and upstream area, on work tiles
    FD1.2  outlet indexing            every river mouth and inland sink, the basins numbered by area
    FD1.3  watershed delineation      the basin of every pixel, the basin rectangles
    FD1.4  basin grouping             the HydroBASINS Level-03 unit of every basin, whole basins
                                      gathered into basin groups; and the automatic groups along a
                                      Hilbert curve, from the basins alone
    FD1.5  fitting to memory          groups over the capacity cut (a basin at its tributary
                                      outlets, a group along a block line), groups under it merged;
                                      the region mask, the region and piece tables

The grid is never held in memory as a whole.  Every step reads it window by window from a tiled
GeoTIFF and writes its result window by window.  The loops over pixels are compiled with Numba.

Conventions used throughout the three stage files:
    flow directions   the eight ESRI D8 codes 1, 2, 4, ..., 128 (east, south-east, south, ...),
                      0 = river mouth, 255 = inland sink, 247 = nodata  (the MERIT Hydro convention)
    pixel index       row * ncol + col of the continental grid, as int64
    rectangles        row_min, row_max, col_min, col_max, left closed and right open (row_max and col_max
                      are one past the last pixel, nrow = row_max - row_min); on a grid that is periodic
                      in longitude (MERIT Hydro, the whole globe) a rectangle across the antimeridian is
                      "unrolled": col_max may be larger than ncol, and a column c >= ncol means the
                      column c - ncol
    windows           a rectangle rounded out to whole blocks of block_pixels x block_pixels pixels
                      (one degree: 3600 pixels at 1 arc-second, 1200 at 3 arc-seconds); the capacity
                      limits the window, in pixels
    tables            space-separated text with one header line, the columns of the
                      paper, read and written with pandas (read_table, write_table)
"""
import csv
import json
import math
import os
import struct
import time

import numpy as np
import pandas as pd
import rasterio

import fd_tables
from numba import njit
from rasterio.windows import Window

# =============================================================================
#  [0] Constants, the D8 tables, the earth model, raster and table helpers
# =============================================================================

MERIT_MOUTH = 0            # a river mouth: the water reaches the sea here
MERIT_SINK = 255           # an inland sink: the water ends in a closed depression here
# the largest accumulation a raster can carry before it is nodata or garbage
ACCUMULATION_MAX = 9.0e18
MERIT_NODATA = 247         # no data: sea, or outside the continent
HYDROSHEDS_SINK = 0        # HydroSHEDS v2 marks an inland sink with 0 ...
HYDROSHEDS_NODATA = 255    # ... and nodata with 255; a coastal outlet keeps its direction code

D8_OFFSETS = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}

# The eight directions in one order, east then clockwise, as (row step, column step).  Everything below
# works on the MERIT codes of D8_OFFSETS; a grid in another coding is translated to them by FD1.0, so a
# baseline hydrography of any producer needs no change anywhere else.
DIRECTION_ORDER = [("E", 0, 1), ("SE", 1, 1), ("S", 1, 0), ("SW", 1, -1), ("W", 0, -1), ("NW", -1, -1), ("N", -1, 0), ("NE", -1, 1)]
CODE_LIMIT = 65535         # a code of the native grid: any whole number a 16-bit raster holds (the degrees go to 360)
UNKNOWN_CODE = 254         # what merit_code_table_of gives a value the convention does not name; FD1.0 refuses it
MERIT_CODE_OF_DIRECTION = {"E": 1, "SE": 2, "S": 4, "SW": 8, "W": 16, "NW": 32, "N": 64, "NE": 128}

# The codings this package knows by name: the eight codes, the value of an inland sink, the value of no
# data, and the value of a river mouth (None when the coding has none and a mouth is a direction pointing
# out of the grid).  A coding not listed here is given on the command line as --direction-codes with
# --sink-code, --nodata-code and --mouth-code.
DIRECTION_CONVENTIONS = {
    # MERIT Hydro, and the ESRI order every raster GIS writes: 1 E, 2 SE, 4 S ... 128 NE
    "merit": {"codes": dict(MERIT_CODE_OF_DIRECTION), "sink": MERIT_SINK, "nodata": MERIT_NODATA, "mouth": MERIT_MOUTH},
    "esri": {"codes": dict(MERIT_CODE_OF_DIRECTION), "sink": None, "nodata": 255, "mouth": 0},
    # HydroSHEDS v2: the same eight codes, but 0 is an inland sink, 255 is no data and a coastal outlet
    # keeps the direction code it flows out on
    "hydrosheds": {"codes": dict(MERIT_CODE_OF_DIRECTION), "sink": HYDROSHEDS_SINK, "nodata": HYDROSHEDS_NODATA, "mouth": None},
    # TauDEM: 1 east, then counter-clockwise
    "taudem": {"codes": {"E": 1, "NE": 2, "N": 3, "NW": 4, "W": 5, "SW": 6, "S": 7, "SE": 8}, "sink": None, "nodata": 0, "mouth": None},
    # GRASS GIS r.watershed and the r.stream.* add-ons: 1 north-east, then counter-clockwise, 8 east
    "grass": {"codes": {"NE": 1, "N": 2, "NW": 3, "W": 4, "SW": 5, "S": 6, "SE": 7, "E": 8}, "sink": None, "nodata": 0, "mouth": None},
    # the degrees of GRASS's "degree" format and of Cho's add-ons: 45 north-east, then counter-clockwise
    "degrees": {"codes": {"NE": 45, "N": 90, "NW": 135, "W": 180, "SW": 225, "S": 270, "SE": 315, "E": 360}, "sink": None, "nodata": 0, "mouth": None},
}


def direction_convention(name, codes=None, sink=None, nodata=None, mouth=None):
    """The coding of a flow-direction grid: a name of DIRECTION_CONVENTIONS, or "custom" with codes given
    as "E=1,SE=2,..." (the eight directions E SE S SW W NW N NE, every code a byte).  sink, nodata and
    mouth override the named coding's values, so a producer who writes the ESRI codes with a nodata of
    247 and no sink needs no new name in this file.  Returns a dict with the eight codes, sink, nodata
    and mouth, every value an int or None."""
    if name == "custom":
        if not codes:
            raise FlowDivideError("the custom convention needs --direction-codes, e.g. E=1,SE=2,S=4,SW=8,W=16,NW=32,N=64,NE=128")
        convention = {"codes": {}, "sink": None, "nodata": None, "mouth": None}
    elif name in DIRECTION_CONVENTIONS:
        convention = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DIRECTION_CONVENTIONS[name].items()}
    else:
        raise FlowDivideError("unknown flow-direction convention '%s'; the ones known by name are %s, or custom with --direction-codes"
                              % (name, ", ".join(sorted(DIRECTION_CONVENTIONS))))
    if codes:
        given = {}
        for pair in str(codes).replace(";", ",").split(","):
            if not pair.strip():
                continue
            direction, _, value = pair.partition("=")
            direction = direction.strip().upper()
            if direction not in MERIT_CODE_OF_DIRECTION or not value.strip().isdigit():
                raise FlowDivideError("a direction code must read like E=1 with a direction among E SE S SW W NW N NE, got %r" % pair)
            given[direction] = int(value)
        convention["codes"].update(given)
    if sorted(convention["codes"]) != sorted(MERIT_CODE_OF_DIRECTION):
        raise FlowDivideError("the convention '%s' must give a code for each of the eight directions, it gives %s" % (name, sorted(convention["codes"])))
    for key, value in (("sink", sink), ("nodata", nodata), ("mouth", mouth)):
        if value is not None:
            convention[key] = int(value)
    for key, value in convention.items():
        if key == "codes":
            for direction, code in value.items():
                if int(code) < 0 or int(code) > CODE_LIMIT:
                    raise FlowDivideError("the code of %s is %s; a code must be a whole number between 0 and %d" % (direction, code, CODE_LIMIT))
        elif value is not None and (int(value) < 0 or int(value) > CODE_LIMIT):
            raise FlowDivideError("the %s value is %s; it must be a whole number between 0 and %d" % (key, value, CODE_LIMIT))
    codes_used = list(convention["codes"].values())
    if len(set(codes_used)) != 8:
        raise FlowDivideError("the eight direction codes of '%s' are not distinct: %s" % (name, convention["codes"]))
    for key in ("sink", "nodata", "mouth"):
        if convention[key] is not None and convention[key] in codes_used:
            raise FlowDivideError("the %s value %d of '%s' is also a direction code" % (key, convention[key], name))
    if len({convention[key] for key in ("sink", "nodata", "mouth") if convention[key] is not None}) != len([key for key in ("sink", "nodata", "mouth") if convention[key] is not None]):
        raise FlowDivideError("the sink, nodata and mouth values of '%s' are not distinct: %s" % (name, convention))
    if convention["nodata"] is None:
        raise FlowDivideError("the convention '%s' gives no nodata value; give one with --nodata-code" % name)
    return convention


def merit_code_table_of(convention):
    """A table that turns a value of the native grid into the MERIT code of FD1.0's output: the eight
    directions to 1 .. 128, the sink to 255, the mouth to 0, no data to 247, and every other value to 254,
    which FD1.0 counts as a value it does not know and refuses to write.  As long as the table's own
    length (a coding in degrees goes up to 360, GRASS writes it in a CELL raster), so the grid need not be
    a Byte raster."""
    largest = max([int(code) for code in convention["codes"].values()]
                  + [int(convention[key]) for key in ("sink", "nodata", "mouth") if convention[key] is not None])
    table = np.full(max(largest + 1, 256), UNKNOWN_CODE, np.uint8)
    for direction, code in convention["codes"].items():
        table[int(code)] = MERIT_CODE_OF_DIRECTION[direction]
    if convention["mouth"] is not None:
        table[int(convention["mouth"])] = MERIT_MOUTH
    if convention["sink"] is not None:
        table[int(convention["sink"])] = MERIT_SINK
    table[int(convention["nodata"])] = MERIT_NODATA
    return table

# lookup tables indexed by the byte value of a pixel: the row and column step of its code, whether it
# is land (a direction code, a mouth or a sink), whether the flow ends there (a mouth or a sink)
DROW = np.zeros(256, np.int8)
DCOL = np.zeros(256, np.int8)
IS_LAND = np.zeros(256, np.uint8)
IS_TERMINAL = np.zeros(256, np.uint8)
IS_D8_CODE = np.zeros(256, np.uint8)
for _code, (_drow, _dcol) in D8_OFFSETS.items():
    DROW[_code] = _drow
    DCOL[_code] = _dcol
    IS_LAND[_code] = 1
    IS_D8_CODE[_code] = 1
IS_LAND[MERIT_MOUTH] = 1
IS_LAND[MERIT_SINK] = 1
IS_TERMINAL[MERIT_MOUTH] = 1
IS_TERMINAL[MERIT_SINK] = 1

# the pass over a tile marks a pixel whose path leaves the tile with this plus the exit number,
# so basin ids must stay below it (they do: a continent has tens of millions of basins at most)
EXIT_LABEL_BASE = np.uint32(2 ** 31)

# How many threads GDAL may use to compress a GeoTIFF we write.  Every core by default, which is what
# a reader wants; the timing experiment of Figure 7 sets FLOWDIVIDE_GDAL_NUM_THREADS=1, because every
# tool it compares runs on one thread.
GDAL_WRITE_THREADS = os.environ.get("FLOWDIVIDE_GDAL_NUM_THREADS", "ALL_CPUS")
RASTER_BLOCK = 512         # every raster written here is tiled in 512 x 512 blocks
BLOCK_METRES_DEFAULT = 100000.0    # the block of a projected grid when none is given: 100 km, whatever the resolution
WORK_TILE_DEFAULT = 16384  # the square work tile of FD1.1, FD1.3 and FD1.5; a multiple of RASTER_BLOCK

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
R_AUTHALIC = 6371007.181


class FlowDivideError(Exception):
    """A check failed or an input is not what the step needs.  Nothing is published after it."""


def log(tag, message):
    """one line of log: the step tag, the wall clock, the message"""
    print("[%s] %s  %s" % (tag, time.strftime("%H:%M:%S"), message), flush=True)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path


# ---- the earth model: WGS84 ellipsoid ----

@njit(cache=True)
def _wgs84_zone_function(lat_rad):
    e = math.sqrt(WGS84_E2)
    sl = math.sin(lat_rad)
    return sl / (1.0 - WGS84_E2 * sl * sl) + (1.0 / (2.0 * e)) * math.log((1.0 + e * sl) / (1.0 - e * sl))


@njit(cache=True)
def pixel_area_m2_at_latitude(lat_deg, pixel_width_deg, pixel_height_deg):
    """the exact area of one geographic pixel on the WGS84 ellipsoid, from the zone between its two
    latitude edges (a strip of the ellipsoid) cut to the pixel's width in longitude"""
    lat_low = (lat_deg - abs(pixel_height_deg) / 2.0) * math.pi / 180.0
    lat_high = (lat_deg + abs(pixel_height_deg) / 2.0) * math.pi / 180.0
    dx = abs(pixel_width_deg) * math.pi / 180.0
    return WGS84_A * WGS84_A * (1.0 - WGS84_E2) * dx * 0.5 * (_wgs84_zone_function(lat_high) - _wgs84_zone_function(lat_low))


MERIT_EARTH_RADIUS = 6378136.0       # rgetara's drad
MERIT_ECCENTRICITY2 = 0.00669447     # rgetara's de2, not WGS84's 0.00669437999014


@njit(cache=True)
def _merit_zone_function(lat_degrees):
    sin_lat = math.sin(lat_degrees * math.pi / 180.0)
    return sin_lat * (1.0 + MERIT_ECCENTRICITY2 * sin_lat * sin_lat / 2.0)


@njit(cache=True)
def pixel_area_m2_at_latitude_merit(lat_deg, pixel_width_deg, pixel_height_deg):
    """the area of one geographic pixel the way MERIT Hydro computes it: rgetara of CaMa-Flood
    (map/src/src_region/set_map.F90, "algorithm by T. Oki, mathematics by S. Kanae, mod by nhanasaki"),
    which is what MERIT Hydro's upstream area was built with.

        A = pi R^2 (1 - e^2) / 180 * [ f(lat_north) - f(lat_south) ] * dlon_degrees,
        f(lat) = sin(lat) (1 + e^2 sin^2(lat) / 2),   R = 6378136 m,  e^2 = 0.00669447

    f is the first two terms of the ellipsoid's exact zone function, whose series is
    sin(lat) (1 + (2/3) e^2 sin^2(lat) + ...): the 1/2 in place of 2/3 is that code's approximation, and
    it is the approximation MERIT is on.  The Fortran returns a single-precision number and multiplies by
    the longitude width in single precision, which is why both roundings are here.

    Measured against MERIT's own Float32 upstream area at the pixels whose upstream count
    is 1 (so the value is one pixel's area), 63,750 latitudes from 50 S to 73 N: this formula is
    +1.29e-5 out, flat with latitude; M(lat) dlat x a cos(lat) dlon is +1.29e-5 at the equator but
    +6.0e-5 at 60 degrees; the exact WGS84 zone strip +2.3e-3.  The 1.29e-5 that remains is a constant
    scale (0.11 m2 in 8548) not accounted for -- not the series, the radius or the eccentricity, all
    tried -- so our upstream area is MERIT's formula to 13 parts per million and not bit for bit."""
    zone = (math.pi * MERIT_EARTH_RADIUS * MERIT_EARTH_RADIUS * (1.0 - MERIT_ECCENTRICITY2) / 180.0
            * (_merit_zone_function(lat_deg + abs(pixel_height_deg) / 2.0)
               - _merit_zone_function(lat_deg - abs(pixel_height_deg) / 2.0)))
    return np.float64(np.float32(np.float32(zone) * np.float32(abs(pixel_width_deg))))


# The earth models the package computes pixel areas under.  Only the area differs: lengths and distances
# (lfp, lup, the perimeter) stay on the exact ellipsoid in both.
#   "wgs84-zone"  the exact area of the pixel on the WGS84 ellipsoid: the most correct number, and what
#                 every product of ours is published with
#   "merit"       MERIT Hydro's own, above: kept for comparing our upstream area with MERIT's upa
#                 (2.3e-3 smaller than the exact ellipsoid), used by no product
EARTH_MODEL_WGS84_ZONE = "wgs84-zone"
EARTH_MODEL_MERIT = "merit"
EARTH_MODELS = (EARTH_MODEL_WGS84_ZONE, EARTH_MODEL_MERIT)


def pixel_area_m2(lat_deg, pixel_width_deg, pixel_height_deg, earth_model=EARTH_MODEL_WGS84_ZONE):
    """the area of one geographic pixel under one of EARTH_MODELS"""
    if earth_model == EARTH_MODEL_MERIT:
        return pixel_area_m2_at_latitude_merit(lat_deg, pixel_width_deg, pixel_height_deg)
    if earth_model != EARTH_MODEL_WGS84_ZONE:
        raise FlowDivideError("unknown earth model '%s': it must be one of %s" % (earth_model, ", ".join(EARTH_MODELS)))
    return pixel_area_m2_at_latitude(lat_deg, pixel_width_deg, pixel_height_deg)


@njit(cache=True)
def earth_distance_m(lat1, lon1, lat2, lon2):
    """the distance between two points: the haversine distance on the authalic sphere, scaled by the
    ratio of the ellipsoidal to the spherical arc length at the mid-latitude for the actual bearing
    (earth_distance_m)"""
    to_rad = math.pi / 180.0
    dlat = (lat2 - lat1) * to_rad
    dlon_deg = lon2 - lon1
    if dlon_deg > 180.0:
        dlon_deg -= 360.0
    elif dlon_deg < -180.0:
        dlon_deg += 360.0
    dlon = dlon_deg * to_rad
    la1 = lat1 * to_rad
    la2 = lat2 * to_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2.0) ** 2
    d_sphere = R_AUTHALIC * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    if d_sphere <= 0.0:
        return d_sphere
    phim = 0.5 * (la1 + la2)
    s2 = math.sin(phim) ** 2
    w = math.sqrt(1.0 - WGS84_E2 * s2)
    m = WGS84_A * (1.0 - WGS84_E2) / (w * w * w)
    n = WGS84_A / w
    ex = n * math.cos(phim) * dlon
    ey = m * dlat
    sx = R_AUTHALIC * math.cos(phim) * dlon
    sy = R_AUTHALIC * dlat
    num = math.sqrt(ex * ex + ey * ey)
    den = math.sqrt(sx * sx + sy * sy)
    if den > 0.0:
        return d_sphere * (num / den)
    return d_sphere


class Grid:
    """What every step needs to know about the continental grid: its size, its geotransform,
    whether it is geographic (degrees) or projected (metres), whether it is periodic in longitude,
    and the size of the blocks the windows lie on."""

    def __init__(self, path, periodic=False, block_pixels=None, earth_model=EARTH_MODEL_WGS84_ZONE):
        self.path = path                  # the flow directions the grid was read from (regions_final reads them)
        with rasterio.open(path) as dataset:
            self.nrow = dataset.height
            self.ncol = dataset.width
            self.transform = dataset.transform
            self.crs = dataset.crs
            # what the coordinates are: degrees, or metres.  A grid with no CRS, one whose axes are
            # rotated, or a projected one whose unit is not the metre is refused instead of being read
            # as one of the two: the lengths and the areas below are degrees or metres and nothing else,
            # and a grid in feet would come out labelled metres
            if dataset.crs is None:
                raise FlowDivideError("%s carries no coordinate reference system; a geographic grid (degrees) "
                                      "or a projected one in metres is needed" % path)
            if dataset.transform.b != 0.0 or dataset.transform.d != 0.0:
                raise FlowDivideError("%s is rotated; a north-up grid is needed" % path)
            self.geographic = bool(dataset.crs.is_geographic)
            if not self.geographic:
                metres_per_unit = dataset.crs.linear_units_factor[1] if dataset.crs.linear_units_factor else 0.0
                if abs(metres_per_unit - 1.0) > 1e-9:
                    raise FlowDivideError("%s is projected in units of %.6g m ('%s'); the lengths and areas of "
                                          "this package are metres, so a grid in another unit has to be "
                                          "reprojected first" % (path, metres_per_unit, dataset.crs.linear_units))
        self.periodic = bool(periodic)
        # which pixel area this grid's products are published with: the exact ellipsoid, or MERIT Hydro's
        # own on the MERIT 90 m grid
        if earth_model not in EARTH_MODELS:
            raise FlowDivideError("unknown earth model '%s': it must be one of %s" % (earth_model, ", ".join(EARTH_MODELS)))
        self.earth_model = earth_model
        self.pixel_width = float(self.transform.a)
        self.pixel_height = float(self.transform.e)       # negative on a north-up grid
        if block_pixels is None:
            if self.geographic:
                block_pixels = int(round(1.0 / abs(self.pixel_width)))      # one degree
            else:
                # a projected grid: a block of about 100 km, so that a capacity of a few 10^9 pixels is a round
                # number of blocks at any resolution (10,000 blocks of 100 km at 1 m, 100 at 10 m; a fixed number
                # of pixels would be 5 km at 1 m and 50 km at 10 m), so that grids of 10 m and 1 m are supported too
                block_pixels = max(1, int(round(BLOCK_METRES_DEFAULT / abs(self.pixel_width))))
        self.block_pixels = int(block_pixels)
        if self.periodic:
            span = self.ncol * abs(self.pixel_width)
            if abs(span - 360.0) > 1e-6:
                log("grid", "warning: the grid is treated as periodic in longitude although it spans %.6f degrees, not 360" % span)

    def latitude_of_row(self, row):
        return self.transform.f + (row + 0.5) * self.pixel_height

    def longitude_of_col(self, col):
        return self.transform.c + (col + 0.5) * self.pixel_width

    def wrap_col(self, col):
        if self.periodic:
            return col % self.ncol
        return col

    def lon_lat(self, x, y):
        """the grid's own coordinates as longitude and latitude: unchanged on a geographic grid,
        transformed to WGS84 on a projected one (the tables carry longitude and latitude whatever the grid)"""
        if self.geographic:
            return np.asarray(x, np.float64), np.asarray(y, np.float64)
        from rasterio.warp import transform as warp_transform
        xs = np.atleast_1d(np.asarray(x, np.float64))
        ys = np.atleast_1d(np.asarray(y, np.float64))
        lon, lat = warp_transform(self.crs, "EPSG:4326", xs.ravel().tolist(), ys.ravel().tolist())
        lon = np.asarray(lon, np.float64).reshape(xs.shape)
        lat = np.asarray(lat, np.float64).reshape(ys.shape)
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            raise FlowDivideError("a point of the grid does not transform to longitude and latitude")
        return lon.reshape(np.shape(x)), lat.reshape(np.shape(y))

    def pixel_centre_lon_lat(self, row, col):
        """the centre of a pixel (or of arrays of pixels) in longitude and latitude"""
        x = self.transform.c + (np.asarray(col, np.float64) + 0.5) * self.pixel_width
        y = self.transform.f + (np.asarray(row, np.float64) + 0.5) * self.pixel_height
        if self.periodic:
            x = np.where(x >= self.transform.c + 360.0, x - 360.0, x)
        return self.lon_lat(x, y)

    def row_pixel_areas_m2(self, row0, nrow):
        """the area of one pixel in each of the rows row0 .. row0 + nrow - 1"""
        areas = np.empty(nrow, np.float64)
        if self.geographic:
            area_of = (pixel_area_m2_at_latitude_merit if self.earth_model == EARTH_MODEL_MERIT
                       else pixel_area_m2_at_latitude)
            for k in range(nrow):
                areas[k] = area_of(self.latitude_of_row(row0 + k), self.pixel_width, self.pixel_height)
        else:
            areas[:] = abs(self.pixel_width * self.pixel_height)
        return areas

    def row_step_lengths_m(self, row0, nrow):
        """the five step lengths of each row: east-west, north, south, north-diagonal, south-diagonal
        (a step is measured from the row it starts in)"""
        lengths = np.empty((nrow, 5), np.float64)
        if not self.geographic:
            dx = abs(self.pixel_width)
            dy = abs(self.pixel_height)
            lengths[:, 0] = dx
            lengths[:, 1] = dy
            lengths[:, 2] = dy
            lengths[:, 3] = math.hypot(dx, dy)
            lengths[:, 4] = math.hypot(dx, dy)
            return lengths
        lon_here = self.longitude_of_col(0)
        lon_next = self.longitude_of_col(1)
        for k in range(nrow):
            row = row0 + k
            row_above = row - 1 if row > 0 else row
            row_below = row + 1 if row < self.nrow - 1 else row
            lat_here = self.latitude_of_row(row)
            lat_above = self.latitude_of_row(row_above)
            lat_below = self.latitude_of_row(row_below)
            lengths[k, 0] = earth_distance_m(lat_here, lon_here, lat_here, lon_next)
            lengths[k, 1] = earth_distance_m(lat_here, lon_here, lat_above, lon_here)
            lengths[k, 2] = earth_distance_m(lat_here, lon_here, lat_below, lon_here)
            lengths[k, 3] = earth_distance_m(lat_here, lon_here, lat_above, lon_next)
            lengths[k, 4] = earth_distance_m(lat_here, lon_here, lat_below, lon_next)
        return lengths

    def window_of_rectangle(self, row_min, row_max, col_min, col_max):
        """the rectangle rounded out to whole blocks: (row_min, row_max, col_min, col_max) of the window"""
        block = self.block_pixels
        return (row_min // block * block, ((row_max - 1) // block + 1) * block,
                col_min // block * block, ((col_max - 1) // block + 1) * block)

    def window_pixels(self, row_min, row_max, col_min, col_max):
        """how many pixels the window that holds this rectangle has"""
        window = self.window_of_rectangle(row_min, row_max, col_min, col_max)
        return (window[1] - window[0]) * (window[3] - window[2])

    def pixel_boxes_lon_lat(self, row_min, row_max, col_min, col_max):
        """pixel_box_lon_lat for arrays of rectangles at once"""
        x_left = self.transform.c + np.asarray(col_min, np.float64) * self.pixel_width
        x_right = self.transform.c + np.asarray(col_max, np.float64) * self.pixel_width
        y_top = self.transform.f + np.asarray(row_min, np.float64) * self.pixel_height
        y_bottom = self.transform.f + np.asarray(row_max, np.float64) * self.pixel_height
        # a rectangle unrolled past the last column keeps its longitudes past 180 degrees (182.6 for a basin across the
        # antimeridian), so that the tables are the same bytes on every run (a fold back by 360 degrees would give the west edge of such a basin as -184)
        if self.geographic:
            return np.minimum(x_left, x_right), np.minimum(y_top, y_bottom), np.maximum(x_left, x_right), np.maximum(y_top, y_bottom)
        # a projected grid: every edge of the rectangle sampled at 21 points, all transformed, the box that holds them; in chunks, so that a table of millions of
        # rectangles does not need them all at once
        count = x_left.size
        minlon = np.empty(count)
        minlat = np.empty(count)
        maxlon = np.empty(count)
        maxlat = np.empty(count)
        fractions = np.linspace(0.0, 1.0, 21)
        for start in range(0, count, 50000):
            stop = min(start + 50000, count)
            xl = x_left[start:stop][:, None]
            xr = x_right[start:stop][:, None]
            yt = y_top[start:stop][:, None]
            yb = y_bottom[start:stop][:, None]
            sample_x = np.concatenate([xl + (xr - xl) * fractions, xl + (xr - xl) * fractions, np.repeat(xl, 21, axis=1), np.repeat(xr, 21, axis=1)], axis=1)
            sample_y = np.concatenate([np.repeat(yt, 21, axis=1), np.repeat(yb, 21, axis=1), yt + (yb - yt) * fractions, yt + (yb - yt) * fractions], axis=1)
            lon, lat = self.lon_lat(sample_x, sample_y)
            minlon[start:stop] = lon.min(axis=1)
            minlat[start:stop] = lat.min(axis=1)
            maxlon[start:stop] = lon.max(axis=1)
            maxlat[start:stop] = lat.max(axis=1)
        return minlon, minlat, maxlon, maxlat

    def pixel_box_lon_lat(self, row_min, row_max, col_min, col_max):
        """the four edges of a rectangle of pixels in longitude and latitude (the grid's own coordinates on
        a geographic grid, transformed on a projected one); an unrolled column past the last one gives a longitude past the grid's east edge"""
        minlon, minlat, maxlon, maxlat = self.pixel_boxes_lon_lat(np.asarray([row_min]), np.asarray([row_max]), np.asarray([col_min]), np.asarray([col_max]))
        return (float(minlon[0]), float(minlat[0]), float(maxlon[0]), float(maxlat[0]))


# ---- raster helpers: every raster is a tiled, compressed, BigTIFF GeoTIFF on the grid of DIR ----

def raster_profile(grid, dtype, nodata, predictor=True):
    profile = {
        "driver": "GTiff", "dtype": dtype, "count": 1, "width": grid.ncol, "height": grid.nrow,
        "crs": grid.crs, "transform": grid.transform, "nodata": nodata, "tiled": True,
        "blockxsize": RASTER_BLOCK, "blockysize": RASTER_BLOCK, "compress": "DEFLATE", "zlevel": 6,
        "BIGTIFF": "YES", "NUM_THREADS": GDAL_WRITE_THREADS,
    }
    if predictor and dtype not in ("uint8", "int8"):
        profile["predictor"] = 2
    return profile


def read_block(dataset, row0, nrow, col0, ncol, periodic, fill, dtype=None):
    """A rectangle of rows row0 .. row0 + nrow - 1 and columns col0 .. col0 + ncol - 1 of band 1, as an
    array of exactly (nrow, ncol).  Rows outside the grid come back as `fill`.  Columns outside the
    grid come back as `fill` on an ordinary grid; on a periodic grid they wrap round (a negative column
    or one past the last is read from the other edge), which is how a halo or an unrolled rectangle
    across the antimeridian is read in one call."""
    grid_nrow = dataset.height
    grid_ncol = dataset.width
    if dtype is None:
        dtype = dataset.dtypes[0]
    out = np.full((nrow, ncol), fill, dtype=dtype)
    read_row0 = max(row0, 0)
    read_row1 = min(row0 + nrow, grid_nrow)
    if read_row1 <= read_row0:
        return out
    if not periodic:
        read_col0 = max(col0, 0)
        read_col1 = min(col0 + ncol, grid_ncol)
        if read_col1 <= read_col0:
            return out
        window = Window(read_col0, read_row0, read_col1 - read_col0, read_row1 - read_row0)
        out[read_row0 - row0:read_row1 - row0, read_col0 - col0:read_col1 - col0] = dataset.read(1, window=window, out_dtype=dtype)
        return out
    # periodic: split the column range into pieces that lie inside 0 .. grid_ncol - 1
    col = col0
    while col < col0 + ncol:
        wrapped = col % grid_ncol
        length = min(grid_ncol - wrapped, col0 + ncol - col)
        window = Window(wrapped, read_row0, length, read_row1 - read_row0)
        out[read_row0 - row0:read_row1 - row0, col - col0:col - col0 + length] = dataset.read(1, window=window, out_dtype=dtype)
        col += length
    return out


def write_block(dataset, array, row0, col0, periodic):
    """the counterpart of read_block for a rectangle that lies inside the grid in rows; on a periodic
    grid the columns may run past the last one and are folded back"""
    nrow, ncol = array.shape
    grid_ncol = dataset.width
    if not periodic:
        dataset.write(array, 1, window=Window(col0, row0, ncol, nrow))
        return
    col = col0
    while col < col0 + ncol:
        wrapped = col % grid_ncol
        length = min(grid_ncol - wrapped, col0 + ncol - col)
        dataset.write(np.ascontiguousarray(array[:, col - col0:col - col0 + length]), 1, window=Window(wrapped, row0, length, nrow))
        col += length


@njit(cache=True)
def _first_code_that_is_not_merit(block):
    """the flat index of the first value that is none of the eight directions, a mouth (0), an inland sink (255) or
    the nodata (247); -1 when there is none"""
    flat = block.ravel()
    for index in range(flat.size):
        code = flat[index]
        if IS_LAND[code] == 0 and code != MERIT_NODATA:
            return index
    return -1


def check_merit_flow_directions(block, where):
    """a flow direction raster in the MERIT convention holds uint8 and only its eleven codes: another
    code would be taken as sea (not land) and cut the network there without a word, and a signed raster could index
    the tables with a negative value"""
    if block.dtype != np.uint8:
        raise FlowDivideError("the flow directions of %s are %s, not uint8" % (where, block.dtype))
    bad = int(_first_code_that_is_not_merit(block))
    if bad >= 0:
        raise FlowDivideError("the flow directions of %s hold the code %d, which is none of the eight directions, 0, "
                              "255 or 247" % (where, int(block.ravel()[bad])))


def read_halo_tile(dataset, row0, nrow, col0, ncol, periodic, fill, dtype=None):
    """the tile plus a one-pixel ring around it, so that every pixel of the tile can look at its
    downstream neighbour and every neighbour outside can be seen flowing in.  A tile of the MERIT
    convention (fill 247 and no dtype asked for) has its codes checked"""
    block = read_block(dataset, row0 - 1, nrow + 2, col0 - 1, ncol + 2, periodic, fill, dtype=dtype)
    if dtype is None and fill == MERIT_NODATA:
        check_merit_flow_directions(block, "%s rows %d .. %d" % (dataset.name, row0, row0 + nrow - 1))
    return block


def publish(temporary_path, final_path):
    """a step writes to a temporary name and renames it when every check has passed, so that a file
    under its final name is always a complete one"""
    if os.path.exists(final_path):
        os.remove(final_path)
    os.replace(temporary_path, final_path)


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, default=str)


# ---- tables: space separated with one header line, as the paper writes every table ----

def write_table(frame, path, float_format="%.6f"):
    """a table written under a temporary name and published when the write is complete"""
    temporary = path + ".partial"
    with timing("output"):
        frame.to_csv(temporary, index=False, sep=" ", float_format=float_format)
    publish(temporary, path)


def read_table(path, usecols=None, dtype=None, exact_floats=False):
    """exact_floats: the round-trip float converter, for a table whose values must come back to the bit"""
    with timing("input"):
        return pd.read_csv(path, sep=r"\s+", usecols=usecols, dtype=dtype, float_precision="round_trip" if exact_floats else None)


# ---- the time spent on checks, kept apart from the time of the work itself ----
#  A step verifies what it wrote (the recoded grid read back and counted again, the colouring read
#  back from the polygon file and set against the mask).  That time is added up here on its own, so
#  that the time of the workflow can be reported without it; a step's process starts at zero and the
#  chain reads the total when the step is done.
CHECK_SECONDS = [0.0]


class checking:
    """with checking(): ... adds the time the block took to CHECK_SECONDS"""

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exception):
        CHECK_SECONDS[0] += time.perf_counter() - self.started
        return False


#  The time spent in the files themselves, added up on its own like the checks: the input (reading
#  windows of the rasters, the decompression; the tables) and the output (writing them, the compression,
#  and the flush when a file is closed; the tables; the polygon files).  A step's time is then
#  input + computing + output (+ its checks), and the paper can say how much of it is the disk
#  (the time recorded in detail: input, compute and output).  The rasters are timed at rasterio's Python
#  classes, so that every read and write of the package counts wherever it is written; a nested timed
#  call (close inside __exit__) counts once.  Timing is always measured (a few microseconds a call);
#  --timing off only leaves it out of the markers, the log and the timings table.
IO_SECONDS = {"input": 0.0, "output": 0.0}
_IO_DEPTH = {"input": 0, "output": 0}


class timing:
    """with timing("input"): ... adds the time the block took to IO_SECONDS["input"] (or "output")"""

    def __init__(self, kind):
        self.kind = kind

    def __enter__(self):
        _IO_DEPTH[self.kind] += 1
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exception):
        _IO_DEPTH[self.kind] -= 1
        if _IO_DEPTH[self.kind] == 0:
            IO_SECONDS[self.kind] += time.perf_counter() - self.started
        return False


def reset_timing():
    CHECK_SECONDS[0] = 0.0
    IO_SECONDS["input"] = IO_SECONDS["output"] = 0.0


def _timed(kind, function):
    def timed(*args, **kwargs):
        with timing(kind):
            return function(*args, **kwargs)
    timed.__name__ = getattr(function, "__name__", "timed")
    timed.__doc__ = getattr(function, "__doc__", None)
    return timed


def _time_rasterio():
    """rasterio's reads and writes timed: the Python classes DatasetReader and DatasetWriter get a
    timed read, write, close and __exit__ (the compiled bases cannot be changed)"""
    from rasterio.io import DatasetReader, DatasetWriter, BufferedDatasetWriter
    from rasterio._io import DatasetReaderBase, DatasetWriterBase
    if getattr(DatasetReader, "_flowdivide_timed", False):
        return
    DatasetReader.read = _timed("input", DatasetReaderBase.read)
    for writer in (DatasetWriter, BufferedDatasetWriter):
        writer.write = _timed("output", DatasetWriterBase.write)
        writer.close = _timed("output", DatasetWriterBase.close)
        writer.__exit__ = _timed("output", DatasetWriterBase.__exit__)
    DatasetReader._flowdivide_timed = True


_time_rasterio()


def strip_rows_for(ncol, pixels_per_strip=200000000, at_least=RASTER_BLOCK, at_most=4096):
    """how many rows a strip may have so that it holds about pixels_per_strip pixels: the strips of the
    passes over the whole grid are sized this way, so that a wide grid does not take a strip of many gigabytes"""
    return int(max(at_least, min(at_most, pixels_per_strip // max(ncol, 1))))


def check_work_tile(tile):
    """a work tile is a positive multiple of the raster block, at most 32768 pixels a side (its pixels
    are indexed with 32-bit integers inside the tile)"""
    if tile <= 0 or tile % RASTER_BLOCK != 0 or tile > 32768:
        raise FlowDivideError("the work tile must be a positive multiple of %d pixels and at most 32768; %d was given" % (RASTER_BLOCK, tile))


def tiles_of_grid(grid, tile, row_min=0, row_max=None, col_min=0, col_max=None):
    """the work tiles that cover the rectangle (default: the whole grid), as (row0, nrow, col0, ncol);
    the tiles start at multiples of `tile` from the rectangle's origin, and the last ones are cut short"""
    if row_max is None:
        row_max = grid.nrow
    if col_max is None:
        col_max = grid.ncol
    tiles = []
    row0 = row_min
    while row0 < row_max:
        nrow = min(tile, row_max - row0)
        col0 = col_min
        while col0 < col_max:
            ncol = min(tile, col_max - col0)
            tiles.append((row0, nrow, col0, ncol))
            col0 += ncol
        row0 += nrow
    return tiles


# =============================================================================
#  [1] FD1.0  assembly and recoding
# =============================================================================
#
#  The producer's tiles are joined into one continental file, and the three values that are not
#  directions are recoded to the MERIT convention: 0 river mouth, 255 inland sink, 247 nodata.
#  HydroSHEDS v2 marks an inland sink with 0 and nodata with 255, and a coastal outlet keeps its
#  direction code and points into nodata or off the grid.  So three rules (rule 3 needs the neighbour):
#      native 255            -> 247   nodata
#      native 0              -> 255   inland sink
#      native direction code -> 0     river mouth, when the pixel one step downstream is nodata or off the grid
#                            -> same  otherwise
#  Every value is counted before and after, and the counts must agree: the directions in equal the
#  directions out plus the river mouths, the sinks in equal the sinks out, the nodata in equal the
#  nodata out.  A grid already in the MERIT convention (MERIT Hydro) is only verified and copied.

@njit(cache=True)
def _recode_strip(halo, out, merit_code_of_byte, counts):
    """The rows 1 .. nrow of `halo` recoded into `out` under the three rules.  merit_code_of_byte is the
    256-entry table of merit_code_table_of: the native coding's eight direction codes to the MERIT codes,
    its sink to 255, its mouth to 0, its nodata to 247, everything else to 254, a value this package does
    not know.  Whatever the producer's coding, only this table changes, so that any baseline
    hydrography is supported.  counts: [direction in, sink in, nodata in, mouth in, unknown in, direction
    out, mouth out, sink out, nodata out]."""
    halo_nrow, halo_ncol = halo.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    limit = merit_code_of_byte.size
    for row in range(nrow):
        for col in range(ncol):
            native = halo[row + 1, col + 1]
            code = merit_code_of_byte[native] if 0 <= native < limit else UNKNOWN_CODE
            if code == MERIT_NODATA:
                counts[2] += 1
                counts[8] += 1
                out[row, col] = MERIT_NODATA
            elif code == MERIT_SINK:
                counts[1] += 1
                counts[7] += 1
                out[row, col] = MERIT_SINK
            elif IS_D8_CODE[code] != 0:
                counts[0] += 1
                native_downstream = halo[row + 1 + DROW[code], col + 1 + DCOL[code]]
                downstream = merit_code_of_byte[native_downstream] if 0 <= native_downstream < limit else UNKNOWN_CODE
                if downstream == MERIT_NODATA:
                    counts[6] += 1                      # a direction pointing into nodata or off the grid: a river mouth
                    out[row, col] = MERIT_MOUTH
                else:
                    counts[5] += 1
                    out[row, col] = code
            elif code == MERIT_MOUTH:
                counts[3] += 1
                counts[6] += 1
                out[row, col] = MERIT_MOUTH
            else:
                counts[4] += 1
                out[row, col] = MERIT_NODATA


@njit(cache=True)
def _count_merit_strip(strip, counts):
    """counts: [direction, mouth, sink, nodata, unknown]"""
    nrow, ncol = strip.shape
    for row in range(nrow):
        for col in range(ncol):
            value = strip[row, col]
            if IS_D8_CODE[value] != 0:
                counts[0] += 1
            elif value == MERIT_MOUTH:
                counts[1] += 1
            elif value == MERIT_SINK:
                counts[2] += 1
            elif value == MERIT_NODATA:
                counts[3] += 1
            else:
                counts[4] += 1


def assemble_tiles(tile_paths, out_path, nodata_value, tag="fd1.0"):
    """The producer's tiles joined into one file on their common grid: the union of their extents at
    their common pixel size, filled with the native nodata where no tile is.  The tiles must share the
    pixel size and the projection and lie on the same lattice.  Returns the path written."""
    extents = []
    pixel_width = None
    pixel_height = None
    crs = None
    tile_dtype = None
    for path in tile_paths:
        with rasterio.open(path) as dataset:
            if dataset.transform.b != 0.0 or dataset.transform.d != 0.0 or dataset.transform.e >= 0.0 or dataset.transform.a <= 0.0:
                raise FlowDivideError("the tile %s is rotated or not north-up; only north-up tiles are assembled" % path)
            # any integer type fd1.0 takes, kept as it is and recoded after, as for a single file: tiles held to
            # Byte could not be joined when they are in degrees (360 does not fit a byte)
            if dataset.dtypes[0] not in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "uint64"):
                raise FlowDivideError("the tile %s is not an integer raster" % path)
            if tile_dtype is None:
                tile_dtype = dataset.dtypes[0]
            elif dataset.dtypes[0] != tile_dtype:
                raise FlowDivideError("the tiles do not share one data type: %s is %s, the first %s" % (path, dataset.dtypes[0], tile_dtype))
            if pixel_width is None:
                pixel_width = dataset.transform.a
                pixel_height = dataset.transform.e
                crs = dataset.crs
            if abs(dataset.transform.a - pixel_width) > 1e-12 or abs(dataset.transform.e - pixel_height) > 1e-12:
                raise FlowDivideError("the tiles do not share one pixel size: %s" % path)
            if dataset.crs != crs:
                raise FlowDivideError("the tiles do not share one projection: %s" % path)
            extents.append((dataset.transform.c, dataset.transform.f, dataset.width, dataset.height))
    x_left = min(e[0] for e in extents)
    y_top = max(e[1] for e in extents)
    x_right = max(e[0] + e[2] * pixel_width for e in extents)
    y_bottom = min(e[1] + e[3] * pixel_height for e in extents)
    ncol = int(round((x_right - x_left) / pixel_width))
    nrow = int(round((y_bottom - y_top) / pixel_height))
    transform = rasterio.transform.Affine(pixel_width, 0.0, x_left, 0.0, pixel_height, y_top)
    tile_range = np.iinfo(np.dtype(tile_dtype))
    if nodata_value is not None and not (tile_range.min <= nodata_value <= tile_range.max):
        raise FlowDivideError("the nodata %s of the coding does not fit the tiles' type %s" % (nodata_value, tile_dtype))
    profile = {"driver": "GTiff", "dtype": tile_dtype, "count": 1, "width": ncol, "height": nrow, "crs": crs,
               "transform": transform, "nodata": nodata_value, "tiled": True, "blockxsize": RASTER_BLOCK,
               "blockysize": RASTER_BLOCK, "compress": "DEFLATE", "BIGTIFF": "YES", "NUM_THREADS": GDAL_WRITE_THREADS}
    temporary = out_path + ".partial.tif"
    log(tag, "assembling %d tiles into %d x %d pixels" % (len(tile_paths), ncol, nrow))
    with rasterio.open(temporary, "w", **profile) as out:
        # the file starts as all nodata; every tile is copied in strips onto its place in the lattice
        for path in tile_paths:
            with rasterio.open(path) as dataset:
                col0 = int(round((dataset.transform.c - x_left) / pixel_width))
                row0 = int(round((dataset.transform.f - y_top) / pixel_height))
                if abs((dataset.transform.c - x_left) / pixel_width - col0) > 1e-6 or abs((dataset.transform.f - y_top) / pixel_height - row0) > 1e-6:
                    raise FlowDivideError("the tile %s does not lie on the lattice of the others" % path)
                for strip_row0 in range(0, dataset.height, 4096):
                    strip_nrow = min(4096, dataset.height - strip_row0)
                    strip = dataset.read(1, window=Window(0, strip_row0, dataset.width, strip_nrow))
                    out.write(strip, 1, window=Window(col0, row0 + strip_row0, dataset.width, strip_nrow))
    publish(temporary, out_path)
    return out_path


def fd1_0_recode(native_path, out_path, convention, periodic=False, strip_rows=2048, tag="fd1.0", direction_codes=None,
                 sink_code=None, nodata_code=None, mouth_code=None):
    """The grid as its producer wrote it, recoded to the MERIT convention (1 E ... 128 NE, 0 a river mouth,
    255 an inland sink, 247 no data), strip by strip with a one-pixel halo.  Every later step reads only
    that, so a baseline hydrography of any producer and any resolution enters here and nowhere else.
    convention is a name of DIRECTION_CONVENTIONS (merit, esri, hydrosheds, taudem, grass, degrees) or
    "custom" with direction_codes ("E=1,SE=2,..."); sink_code, nodata_code and mouth_code override the
    named coding's values.  A pixel whose direction points into nodata or off the grid becomes a river
    mouth, whatever the coding.  Writes <out_path> and <out_path>.report.json."""
    started = time.time()
    coding = direction_convention(convention, codes=direction_codes, sink=sink_code, nodata=nodata_code, mouth=mouth_code)
    merit_code_of_byte = merit_code_table_of(coding)
    native_nodata = coding["nodata"]
    grid = Grid(native_path, periodic=periodic)
    with rasterio.open(native_path) as source:
        # any integer raster: a coding in degrees does not fit a byte, and GRASS writes its directions in a CELL
        # raster, and any baseline hydrography is supported
        if source.dtypes[0] not in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "uint64"):
            raise FlowDivideError("the flow directions must be an integer raster; %s is %s" % (native_path, source.dtypes[0]))
        if source.transform.b != 0.0 or source.transform.d != 0.0 or source.transform.e >= 0.0 or source.transform.a <= 0.0:
            raise FlowDivideError("the flow directions must be a north-up, unrotated raster: %s" % native_path)
    profile = raster_profile(grid, "uint8", MERIT_NODATA, predictor=False)
    temporary = out_path + ".partial.tif"
    counts = np.zeros(9, np.int64)
    with rasterio.open(native_path) as source, rasterio.open(temporary, "w", **profile) as out:
        for row0 in range(0, grid.nrow, strip_rows):
            nrow = min(strip_rows, grid.nrow - row0)
            halo = read_halo_tile(source, row0, nrow, 0, grid.ncol, periodic, native_nodata, dtype=np.int32)
            strip = np.empty((nrow, grid.ncol), np.uint8)
            _recode_strip(halo, strip, merit_code_of_byte, counts)
            out.write(strip, 1, window=Window(0, row0, grid.ncol, nrow))
            if (row0 // strip_rows) % 20 == 0:
                log(tag, "recoded rows %d .. %d of %d" % (row0, row0 + nrow - 1, grid.nrow))
    report = {"convention": convention, "codes": {direction: int(code) for direction, code in coding["codes"].items()},
              "sink_code": coding["sink"], "nodata_code": coding["nodata"], "mouth_code": coding["mouth"], "direction_in": int(counts[0]), "sink_in": int(counts[1]), "nodata_in": int(counts[2]),
              "mouth_in": int(counts[3]), "unknown_in": int(counts[4]), "direction_out": int(counts[5]), "mouth_out": int(counts[6]),
              "sink_out": int(counts[7]), "nodata_out": int(counts[8]), "mouths_made_from_directions": int(counts[6] - counts[3])}
    if counts[4] != 0:
        raise FlowDivideError("%d pixels carry a value that is no direction, no outlet and no nodata under the %s convention (%s)" % (counts[4], convention, coding))
    if counts[0] != counts[5] + counts[6] - counts[3] or counts[1] != counts[7] or counts[2] != counts[8]:
        raise FlowDivideError("the recode counts do not add up: %s" % report)
    # the output read back and counted again (a check, timed on its own)
    read_back = np.zeros(5, np.int64)
    with checking(), rasterio.open(temporary) as check:
        for row0 in range(0, grid.nrow, strip_rows):
            nrow = min(strip_rows, grid.nrow - row0)
            _count_merit_strip(check.read(1, window=Window(0, row0, grid.ncol, nrow)), read_back)
    report["read_back"] = {"direction": int(read_back[0]), "mouth": int(read_back[1]), "sink": int(read_back[2]), "nodata": int(read_back[3]), "unknown": int(read_back[4])}
    if read_back[4] != 0 or int(read_back[1]) != int(counts[6]) or int(read_back[2]) != int(counts[7]) or int(read_back[0]) != int(counts[5]):
        raise FlowDivideError("the written file does not read back with the counts that were written: %s" % report)
    report["land_pixels"] = int(read_back[0] + read_back[1] + read_back[2])
    report["outlets"] = int(read_back[1] + read_back[2])
    report["seconds"] = round(time.time() - started, 1)
    publish(temporary, out_path)
    write_json(out_path + ".report.json", report)
    log(tag, "written %s: %d land pixels, %d river mouths (%d made from directions pointing into nodata), %d inland sinks" % (
        out_path, report["land_pixels"], int(counts[6]), report["mouths_made_from_directions"], int(counts[7])))
    return report


# =============================================================================
#  [2] The work tiles: one order for every pass over a tile, the exits and the inlets
# =============================================================================
#
#  Flow leaves a work tile only through pixels on its edge and enters it only at pixels on its edge,
#  so the whole grid can be worked tile by tile in three steps once the values arriving at the edge
#  pixels are known: every tile on its own (pass 1), the exit graph solved (the exits of all tiles,
#  each flowing into an inlet pixel of a neighbouring tile), every tile again (pass 2).  FD1.1 passes
#  sums this way, FD1.3 and FD1.5 pass labels.  What the passes share is here.

@njit(cache=True)
def tile_topological_order(halo_dir):
    """The land pixels of one tile in an order in which every pixel comes after every pixel that flows
    into it from inside the tile (Kahn's algorithm: a pixel is taken when its in-tile in-degree has
    dropped to zero).  Returns (order, land_count, taken_count); taken_count < land_count means the
    directions hold a cycle inside the tile.  Local pixel index = row * ncol + col of the tile."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    pixel_count = nrow * ncol
    in_degree = np.zeros(pixel_count, np.uint8)
    land_count = 0
    for row in range(nrow):
        for col in range(ncol):
            code = halo_dir[row + 1, col + 1]
            if IS_LAND[code] == 0:
                continue
            land_count += 1
            drow = DROW[code]
            dcol = DCOL[code]
            if drow == 0 and dcol == 0:
                continue
            down_row = row + drow
            down_col = col + dcol
            if down_row >= 0 and down_row < nrow and down_col >= 0 and down_col < ncol and IS_LAND[halo_dir[down_row + 1, down_col + 1]] != 0:
                in_degree[down_row * ncol + down_col] += 1
    order = np.empty(land_count, np.int32)
    tail = 0
    for pixel in range(pixel_count):
        row = pixel // ncol
        col = pixel - row * ncol
        if IS_LAND[halo_dir[row + 1, col + 1]] != 0 and in_degree[pixel] == 0:
            order[tail] = pixel
            tail += 1
    head = 0
    while head < tail:
        pixel = order[head]
        head += 1
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        drow = DROW[code]
        dcol = DCOL[code]
        if drow == 0 and dcol == 0:
            continue
        down_row = row + drow
        down_col = col + dcol
        if down_row >= 0 and down_row < nrow and down_col >= 0 and down_col < ncol and IS_LAND[halo_dir[down_row + 1, down_col + 1]] != 0:
            down = down_row * ncol + down_col
            in_degree[down] -= 1
            if in_degree[down] == 0:
                order[tail] = down
                tail += 1
    return order, land_count, tail


@njit(cache=True)
def _ring_position(row, col, nrow, ncol):
    """a position for every pixel on the edge of the tile, so that an inlet pixel is recorded once"""
    if row == 0:
        return col
    if row == nrow - 1:
        return ncol + col
    if col == 0:
        return 2 * ncol + row
    return 2 * ncol + nrow + row


@njit(cache=True)
def _global_index(row, col, grid_ncol, periodic):
    if periodic:
        col = col % grid_ncol
    return row * grid_ncol + col


@njit(cache=True)
def tile_exits_and_drains(halo_dir, order, tile_row0, tile_col0, grid_nrow, grid_ncol, periodic, drain, exit_pixel_global, exit_destination_global):
    """Pass 1 of any tile scheme: read the order backwards and give every land pixel the exit its path
    leaves the tile through (drain[pixel] = exit number, -1 when the path ends inside the tile).  A
    pixel drains to what its downstream pixel drains to, and the downstream pixel was taken later.
    Returns (exit_count, off_grid_count, into_nodata_count)."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    exit_count = 0
    off_grid_count = 0
    into_nodata_count = 0
    for k in range(order.size - 1, -1, -1):
        pixel = order[k]
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        drow = DROW[code]
        dcol = DCOL[code]
        if drow == 0 and dcol == 0:
            drain[pixel] = -1
            continue
        down_row = row + drow
        down_col = col + dcol
        if down_row >= 0 and down_row < nrow and down_col >= 0 and down_col < ncol:
            if IS_LAND[halo_dir[down_row + 1, down_col + 1]] == 0:
                into_nodata_count += 1
                drain[pixel] = -1
            else:
                drain[pixel] = drain[down_row * ncol + down_col]
            continue
        grid_row = tile_row0 + down_row
        grid_col = tile_col0 + down_col
        if periodic:
            grid_col = grid_col % grid_ncol
        if grid_row < 0 or grid_row >= grid_nrow or grid_col < 0 or grid_col >= grid_ncol:
            off_grid_count += 1
            drain[pixel] = -1
            continue
        if IS_LAND[halo_dir[down_row + 1, down_col + 1]] == 0:
            into_nodata_count += 1
            drain[pixel] = -1
            continue
        exit_pixel_global[exit_count] = _global_index(tile_row0 + row, tile_col0 + col, grid_ncol, periodic)
        exit_destination_global[exit_count] = grid_row * grid_ncol + grid_col
        drain[pixel] = exit_count
        exit_count += 1
    return exit_count, off_grid_count, into_nodata_count


@njit(cache=True)
def tile_inlets(halo_dir, tile_row0, tile_col0, grid_nrow, grid_ncol, periodic, inlet_local, inlet_pixel_global):
    """the inlet pixels of a tile: pixels just inside its edge that a pixel outside flows into, each
    recorded once.  Returns the count."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    seen = np.zeros(2 * nrow + 2 * ncol + 4, np.uint8)
    inlet_count = 0
    for halo_row in range(halo_nrow):
        for halo_col in range(halo_ncol):
            if halo_row != 0 and halo_row != halo_nrow - 1 and halo_col != 0 and halo_col != halo_ncol - 1:
                continue
            code = halo_dir[halo_row, halo_col]
            if IS_D8_CODE[code] == 0:
                continue
            target_row = halo_row - 1 + DROW[code]
            target_col = halo_col - 1 + DCOL[code]
            if target_row < 0 or target_row >= nrow or target_col < 0 or target_col >= ncol:
                continue
            if IS_LAND[halo_dir[target_row + 1, target_col + 1]] == 0:
                continue
            position = _ring_position(target_row, target_col, nrow, ncol)
            if seen[position] != 0:
                continue
            seen[position] = 1
            inlet_local[inlet_count] = target_row * ncol + target_col
            inlet_pixel_global[inlet_count] = _global_index(tile_row0 + target_row, tile_col0 + target_col, grid_ncol, periodic)
            inlet_count += 1
    return inlet_count


class TilePassRecords:
    """what pass 1 of all tiles leaves behind: the exits and the inlets of every tile, kept per tile
    so that pass 2 can take its own slice"""

    def __init__(self):
        self.exit_global = []
        self.exit_destination = []
        self.exit_value_count = []
        self.exit_value_area = []
        self.exit_label = []
        self.inlet_local = []
        self.inlet_global = []
        self.inlet_exit_global = []
        self.inlet_label = []
        self.tile_exit_offsets = [0]
        self.tile_inlet_offsets = [0]

    def concatenate(self):
        def cat(parts, dtype):
            if len(parts) == 0:
                return np.zeros(0, dtype)
            return np.concatenate(parts).astype(dtype, copy=False)
        self.exit_global = cat(self.exit_global, np.int64)
        self.exit_destination = cat(self.exit_destination, np.int64)
        self.exit_value_count = cat(self.exit_value_count, np.int64)
        self.exit_value_area = cat(self.exit_value_area, np.float64)
        self.exit_label = cat(self.exit_label, np.int64)
        self.inlet_local = cat(self.inlet_local, np.int32)
        self.inlet_global = cat(self.inlet_global, np.int64)
        self.inlet_exit_global = cat(self.inlet_exit_global, np.int64)
        self.inlet_label = cat(self.inlet_label, np.int64)
        self.tile_exit_offsets = np.asarray(self.tile_exit_offsets, np.int64)
        self.tile_inlet_offsets = np.asarray(self.tile_inlet_offsets, np.int64)


@njit(cache=True)
def _link_exits_to_inlets(exit_global, exit_destination, inlet_global, inlet_exit_global):
    """every exit flows into an inlet pixel of a neighbouring tile; that inlet drains to one of its
    tile's exits, or its path ends inside its tile.  Returns (status, inlet_of_exit, next_exit):
    status 0 fine, 1 a destination is no inlet, 2 an inlet names an exit that does not exist."""
    exit_count = exit_global.size
    inlet_count = inlet_global.size
    inlet_order = np.argsort(inlet_global)
    inlet_sorted = inlet_global[inlet_order]
    exit_order = np.argsort(exit_global)
    exit_sorted = exit_global[exit_order]
    inlet_of_exit = np.full(exit_count, -1, np.int64)
    next_exit = np.full(exit_count, -1, np.int64)
    for e in range(exit_count):
        position = np.searchsorted(inlet_sorted, exit_destination[e])
        if position >= inlet_count or inlet_sorted[position] != exit_destination[e]:
            return 1, inlet_of_exit, next_exit
        inlet_index = inlet_order[position]
        inlet_of_exit[e] = inlet_index
        target = inlet_exit_global[inlet_index]
        if target >= 0:
            position2 = np.searchsorted(exit_sorted, target)
            if position2 >= exit_count or exit_sorted[position2] != target:
                return 2, inlet_of_exit, next_exit
            next_exit[e] = exit_order[position2]
    return 0, inlet_of_exit, next_exit


@njit(cache=True)
def _accumulate_over_exit_graph(next_exit, inlet_of_exit, inlet_count, local_count, local_area):
    """the totals of every exit in topological order of the exit graph (a forest: every exit flows into
    at most one other exit), then what arrives at every inlet.  Returns (status, arriving_count,
    arriving_area); status 1 means the exit graph has a cycle."""
    exit_count = next_exit.size
    in_degree = np.zeros(exit_count, np.int32)
    for e in range(exit_count):
        if next_exit[e] >= 0:
            in_degree[next_exit[e]] += 1
    total_count = local_count.copy()
    total_area = local_area.copy()
    queue = np.empty(exit_count, np.int64)
    tail = 0
    for e in range(exit_count):
        if in_degree[e] == 0:
            queue[tail] = e
            tail += 1
    head = 0
    while head < tail:
        e = queue[head]
        head += 1
        target = next_exit[e]
        if target >= 0:
            total_count[target] += total_count[e]
            total_area[target] += total_area[e]
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue[tail] = target
                tail += 1
    arriving_count = np.zeros(inlet_count, np.int64)
    arriving_area = np.zeros(inlet_count, np.float64)
    if tail != exit_count:
        return 1, arriving_count, arriving_area
    for e in range(exit_count):
        arriving_count[inlet_of_exit[e]] += total_count[e]
        arriving_area[inlet_of_exit[e]] += total_area[e]
    return 0, arriving_count, arriving_area


@njit(cache=True)
def _resolve_labels_over_exit_graph(next_exit, inlet_of_exit, inlet_label, exit_count):
    """every exit chased to the label at the end of its chain of exits (a basin id, or a piece code),
    with memoisation.  Returns (status, label_of_exit); status 1 means a cycle, 2 an inlet that neither
    carries a label nor names an exit."""
    label_of_exit = np.zeros(exit_count, np.int64)
    for e in range(exit_count):
        if next_exit[e] < 0:
            label_of_exit[e] = inlet_label[inlet_of_exit[e]]
            if label_of_exit[e] <= 0:
                return 2, label_of_exit
    path = np.empty(exit_count, np.int64)
    on_path = np.zeros(exit_count, np.uint8)
    for e in range(exit_count):
        if label_of_exit[e] > 0:
            continue
        depth = 0
        current = e
        while label_of_exit[current] <= 0:
            if on_path[current] != 0:
                return 1, label_of_exit
            on_path[current] = 1
            path[depth] = current
            depth += 1
            current = next_exit[current]
            if current < 0:
                return 2, label_of_exit
        found = label_of_exit[current]
        for k in range(depth):
            label_of_exit[path[k]] = found
            on_path[path[k]] = 0
    return 0, label_of_exit


# =============================================================================
#  [3] FD1.1  flow accumulation
# =============================================================================
#
#  Every pixel gets its upstream pixel count (itself included) and its upstream area.  No basin is
#  known yet, so it is done on the work tiles in the three steps above: pass 1 gives every exit its
#  local total (what drains to it from inside its tile); the exit graph gives every exit its full
#  total and every inlet what arrives there; pass 2 accumulates every tile again with the arrivals
#  added at the inlets, and the result is final everywhere in the tile.  The channel mask (upstream
#  area at least the threshold) is written at the same time, so that the area raster can be dropped
#  as soon as the outlets have been read.

@njit(cache=True)
def _pass_one_local_totals(drain, exit_count, ncol, row_area_m2):
    exit_local_count = np.zeros(exit_count, np.int64)
    exit_local_area = np.zeros(exit_count, np.float64)
    for pixel in range(drain.size):
        e = drain[pixel]
        if e >= 0:
            exit_local_count[e] += 1
            exit_local_area[e] += row_area_m2[pixel // ncol]
    return exit_local_count, exit_local_area


@njit(cache=True)
def _pass_one_local_totals_area_only(drain, exit_count, ncol, row_area_m2):
    exit_local_area = np.zeros(exit_count, np.float64)
    for pixel in range(drain.size):
        e = drain[pixel]
        if e >= 0:
            exit_local_area[e] += row_area_m2[pixel // ncol]
    return exit_local_area


@njit(cache=True)
def _accumulate_over_exit_graph_area_only(next_exit, inlet_of_exit, inlet_count, local_area):
    """the area-only twin of _accumulate_over_exit_graph, for fd1_1_upstream_area: the totals of every
    exit in topological order of the exit graph, then what arrives at every inlet.  Returns (status,
    arriving_area); status 1 means the exit graph has a cycle."""
    exit_count = next_exit.size
    in_degree = np.zeros(exit_count, np.int32)
    for e in range(exit_count):
        if next_exit[e] >= 0:
            in_degree[next_exit[e]] += 1
    total_area = local_area.copy()
    queue = np.empty(exit_count, np.int64)
    tail = 0
    for e in range(exit_count):
        if in_degree[e] == 0:
            queue[tail] = e
            tail += 1
    head = 0
    while head < tail:
        e = queue[head]
        head += 1
        target = next_exit[e]
        if target >= 0:
            total_area[target] += total_area[e]
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue[tail] = target
                tail += 1
    if head != exit_count:
        return 1, total_area
    # one inlet pixel is recorded once (tile_inlets) but several exits can flow into it -- two tiles that
    # meet at a corner both flow into the same pixel of a third -- so what arrives there is the SUM.  An
    # assignment would lose every contribution but one (on basin 141, test_tile_kernels.py counts 68,379
    # of 6.9 million pixels wrong with it).  The production twin _accumulate_over_exit_graph sums too.
    arriving_area = np.zeros(inlet_count, np.float64)
    for e in range(exit_count):
        i = inlet_of_exit[e]
        if i >= 0:
            arriving_area[i] += total_area[e]
    return 0, arriving_area


@njit(cache=True)
def _pass_two_accumulate_area_only(halo_dir, order, inlet_local, arriving_area, row_area_m2, area_out):
    """the area-only twin of _pass_two_accumulate, for fd1_1_upstream_area: every land pixel starts
    with its own area (plus what arrives at an inlet) and adds it to its downstream pixel, in order.
    Returns the area that ends at the terminals of the tile (mouths, sinks and paths into nodata or
    off the grid)."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    for k in range(order.size):
        pixel = order[k]
        row = pixel // ncol
        col = pixel - row * ncol
        area_out[row, col] = row_area_m2[row]
    for i in range(inlet_local.size):
        pixel = inlet_local[i]
        row = pixel // ncol
        col = pixel - row * ncol
        area_out[row, col] += arriving_area[i]
    ended_area = 0.0
    for k in range(order.size):
        pixel = order[k]
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        drow = DROW[code]
        dcol = DCOL[code]
        if drow == 0 and dcol == 0:
            ended_area += area_out[row, col]
            continue
        down_row = row + drow
        down_col = col + dcol
        if IS_LAND[halo_dir[down_row + 1, down_col + 1]] == 0:
            ended_area += area_out[row, col]
        elif down_row >= 0 and down_row < nrow and down_col >= 0 and down_col < ncol:
            area_out[down_row, down_col] += area_out[row, col]
    return ended_area


def fd1_1_upstream_area(dir_path, aca_path, str_path, grid, tile=WORK_TILE_DEFAULT, channel_threshold_km2=1.0, tag="fd1.1_aca"):
    """the upstream area alone (Int64 square metres, nodata 0) and the channel mask, on the grid of DIR:
    the accumulation class of Table 1/3 as one standalone, separately timed kernel (for Figure 8's
    four featured kernels: accumulation, labelling, maximum, stream order).  Everything
    but the upstream pixel count of fd1_1_flow_accumulation, so that this kernel's own time is not the
    combined acc+aca+str pass but the accumulation alone; the tiling, the exit graph and the topological
    order inside a tile are shared in spirit with fd1_1_flow_accumulation but written out again here
    rather than parameterised over what to accumulate (no shared kernel, project style)."""
    started = time.time()
    check_work_tile(tile)
    tiles = tiles_of_grid(grid, tile)
    records = TilePassRecords()
    land_total = 0
    off_grid_total = 0
    into_nodata_total = 0
    # pass 1
    with rasterio.open(dir_path) as dir_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            if taken != land_count:
                raise FlowDivideError("the flow directions hold a cycle inside the tile at row %d col %d" % (row0, col0))
            land_total += land_count
            drain = np.full(nrow * ncol, -1, np.int32)       # -1: water, or a path that ends inside the tile
            max_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(max_edge, np.int64)
            exit_destination = np.empty(max_edge, np.int64)
            exit_count, off_grid, into_nodata = tile_exits_and_drains(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, drain, exit_global, exit_destination)
            off_grid_total += off_grid
            into_nodata_total += into_nodata
            local_area = _pass_one_local_totals_area_only(drain, exit_count, ncol, grid.row_pixel_areas_m2(row0, nrow))
            inlet_local = np.empty(max_edge, np.int32)
            inlet_global = np.empty(max_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            for i in range(inlet_count):
                e = drain[inlet_local[i]]
                inlet_exit_global[i] = exit_global[e] if e >= 0 else -1
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.exit_value_area.append(local_area)
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            del halo, order, drain
            log(tag, "pass 1 tile %d of %d (row %d col %d): %d land pixels, %d exits, %d inlets" % (tile_index + 1, len(tiles), row0, col0, land_count, exit_count, inlet_count))
    records.concatenate()
    if off_grid_total != 0 or into_nodata_total != 0:
        raise FlowDivideError("%d pixels flow off the grid and %d into nodata; the grid is not in the MERIT convention (run FD1.0)" % (off_grid_total, into_nodata_total))
    # the exit graph
    status, inlet_of_exit, next_exit = _link_exits_to_inlets(records.exit_global, records.exit_destination, records.inlet_global, records.inlet_exit_global)
    if status != 0:
        raise FlowDivideError("the exit graph cannot be linked (status %d): a destination is no inlet or an inlet names no exit" % status)
    status, arriving_area = _accumulate_over_exit_graph_area_only(next_exit, inlet_of_exit, records.inlet_global.size, records.exit_value_area)
    if status != 0:
        raise FlowDivideError("the exit graph of the tiles has a cycle")
    log(tag, "exit graph: %d exits, %d inlets, %d land pixels in all" % (records.exit_global.size, records.inlet_global.size, land_total))
    # pass 2
    ended_area_total = 0.0
    # the threshold in float32 and in the raster's unit, compared with the area as float32, as FD3
    # compares them; a double threshold against the float32 area comes out otherwise at 1.00000001 km2
    threshold_m2 = np.float32(channel_threshold_km2 * 1e6)
    aca_temporary = aca_path + ".partial.tif"
    str_temporary = str_path + ".partial.tif"
    with rasterio.open(dir_path) as dir_dataset, \
            rasterio.open(aca_temporary, "w", **raster_profile(grid, "int64", 0)) as aca_out, \
            rasterio.open(str_temporary, "w", **raster_profile(grid, "uint8", 0, predictor=False)) as str_out:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            first = records.tile_inlet_offsets[tile_index]
            last = records.tile_inlet_offsets[tile_index + 1]
            area_out = np.zeros((nrow, ncol), np.float64)
            ended_area = _pass_two_accumulate_area_only(halo, order, records.inlet_local[first:last], arriving_area[first:last], grid.row_pixel_areas_m2(row0, nrow), area_out)
            ended_area_total += ended_area
            area_m2 = np.floor(area_out + 0.5).astype(np.int64)                     # to the nearest square metre, halves up
            aca_out.write(area_m2, 1, window=Window(col0, row0, ncol, nrow))
            channel = (area_m2.astype(np.float32) >= threshold_m2).astype(np.uint8)
            str_out.write(channel, 1, window=Window(col0, row0, ncol, nrow))
            del halo, order, area_out, area_m2, channel
            log(tag, "pass 2 tile %d of %d written" % (tile_index + 1, len(tiles)))
    publish(aca_temporary, aca_path)
    publish(str_temporary, str_path)
    report = {"land_pixels": int(land_total), "land_area_km2": ended_area_total / 1e6, "exits": int(records.exit_global.size),
              "inlets": int(records.inlet_global.size), "work_tile": tile, "channel_threshold_km2": channel_threshold_km2,
              "seconds": round(time.time() - started, 1)}
    write_json(aca_path + ".report.json", report)
    log(tag, "written %s, %s: %d land pixels, %.1f km2" % (aca_path, str_path, land_total, report["land_area_km2"]))
    return report


@njit(cache=True)
def _pass_two_accumulate(halo_dir, order, inlet_local, arriving_count, arriving_area, row_area_m2, count_out, area_out):
    """every land pixel starts with its own count and area (plus what arrives at an inlet) and adds
    them to its downstream pixel, in order.  Returns the count and area that end at the terminals of
    the tile (mouths, sinks and paths into nodata or off the grid)."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    for k in range(order.size):
        pixel = order[k]
        row = pixel // ncol
        col = pixel - row * ncol
        count_out[row, col] = 1
        area_out[row, col] = row_area_m2[row]
    for i in range(inlet_local.size):
        pixel = inlet_local[i]
        row = pixel // ncol
        col = pixel - row * ncol
        count_out[row, col] += arriving_count[i]
        area_out[row, col] += arriving_area[i]
    ended_count = 0
    ended_area = 0.0
    for k in range(order.size):
        pixel = order[k]
        row = pixel // ncol
        col = pixel - row * ncol
        code = halo_dir[row + 1, col + 1]
        drow = DROW[code]
        dcol = DCOL[code]
        if drow == 0 and dcol == 0:
            ended_count += count_out[row, col]
            ended_area += area_out[row, col]
            continue
        down_row = row + drow
        down_col = col + dcol
        if IS_LAND[halo_dir[down_row + 1, down_col + 1]] == 0:
            ended_count += count_out[row, col]
            ended_area += area_out[row, col]
        elif down_row >= 0 and down_row < nrow and down_col >= 0 and down_col < ncol:
            count_out[down_row, down_col] += count_out[row, col]
            area_out[down_row, down_col] += area_out[row, col]
    return ended_count, ended_area


def fd1_1_flow_accumulation(dir_path, acc_path, aca_path, str_path, grid, tile=WORK_TILE_DEFAULT,
                            channel_threshold_km2=1.0, tag="fd1.1"):
    """upstream pixel count (Int64, nodata 0), upstream area (Int64 square metres, nodata 0) and the
    channel mask (Byte: 1 where the upstream area is at least channel_threshold_km2) on the grid of DIR"""
    started = time.time()
    check_work_tile(tile)
    tiles = tiles_of_grid(grid, tile)
    records = TilePassRecords()
    land_total = 0
    off_grid_total = 0
    into_nodata_total = 0
    # pass 1
    with rasterio.open(dir_path) as dir_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            if taken != land_count:
                raise FlowDivideError("the flow directions hold a cycle inside the tile at row %d col %d" % (row0, col0))
            land_total += land_count
            drain = np.full(nrow * ncol, -1, np.int32)       # -1: water, or a path that ends inside the tile
            max_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(max_edge, np.int64)
            exit_destination = np.empty(max_edge, np.int64)
            exit_count, off_grid, into_nodata = tile_exits_and_drains(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, drain, exit_global, exit_destination)
            off_grid_total += off_grid
            into_nodata_total += into_nodata
            local_count, local_area = _pass_one_local_totals(drain, exit_count, ncol, grid.row_pixel_areas_m2(row0, nrow))
            inlet_local = np.empty(max_edge, np.int32)
            inlet_global = np.empty(max_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            for i in range(inlet_count):
                e = drain[inlet_local[i]]
                inlet_exit_global[i] = exit_global[e] if e >= 0 else -1
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.exit_value_count.append(local_count)
            records.exit_value_area.append(local_area)
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            del halo, order, drain
            log(tag, "pass 1 tile %d of %d (row %d col %d): %d land pixels, %d exits, %d inlets" % (tile_index + 1, len(tiles), row0, col0, land_count, exit_count, inlet_count))
    records.concatenate()
    if off_grid_total != 0 or into_nodata_total != 0:
        raise FlowDivideError("%d pixels flow off the grid and %d into nodata; the grid is not in the MERIT convention (run FD1.0)" % (off_grid_total, into_nodata_total))
    # the exit graph
    status, inlet_of_exit, next_exit = _link_exits_to_inlets(records.exit_global, records.exit_destination, records.inlet_global, records.inlet_exit_global)
    if status != 0:
        raise FlowDivideError("the exit graph cannot be linked (status %d): a destination is no inlet or an inlet names no exit" % status)
    status, arriving_count, arriving_area = _accumulate_over_exit_graph(next_exit, inlet_of_exit, records.inlet_global.size, records.exit_value_count, records.exit_value_area)
    if status != 0:
        raise FlowDivideError("the exit graph of the tiles has a cycle")
    log(tag, "exit graph: %d exits, %d inlets, %d land pixels in all" % (records.exit_global.size, records.inlet_global.size, land_total))
    # pass 2
    ended_count_total = 0
    ended_area_total = 0.0
    # the threshold in float32 and in the raster's unit, compared with the area as float32, as FD3
    # compares them; a double threshold against the float32 area comes out otherwise at 1.00000001 km2
    threshold_m2 = np.float32(channel_threshold_km2 * 1e6)
    acc_temporary = acc_path + ".partial.tif"
    aca_temporary = aca_path + ".partial.tif"
    str_temporary = str_path + ".partial.tif"
    with rasterio.open(dir_path) as dir_dataset, \
            rasterio.open(acc_temporary, "w", **raster_profile(grid, "int64", 0)) as acc_out, \
            rasterio.open(aca_temporary, "w", **raster_profile(grid, "int64", 0)) as aca_out, \
            rasterio.open(str_temporary, "w", **raster_profile(grid, "uint8", 0, predictor=False)) as str_out:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            first = records.tile_inlet_offsets[tile_index]
            last = records.tile_inlet_offsets[tile_index + 1]
            count_out = np.zeros((nrow, ncol), np.int64)
            area_out = np.zeros((nrow, ncol), np.float64)
            ended_count, ended_area = _pass_two_accumulate(halo, order, records.inlet_local[first:last], arriving_count[first:last], arriving_area[first:last], grid.row_pixel_areas_m2(row0, nrow), count_out, area_out)
            ended_count_total += ended_count
            ended_area_total += ended_area
            acc_out.write(count_out, 1, window=Window(col0, row0, ncol, nrow))
            area_m2 = np.floor(area_out + 0.5).astype(np.int64)                     # to the nearest square metre, halves up
            aca_out.write(area_m2, 1, window=Window(col0, row0, ncol, nrow))
            # the channel mask from the saved area read as Float32, which is how the attributes of FD3
            # threshold the area they read
            channel = (area_m2.astype(np.float32) >= threshold_m2).astype(np.uint8)
            str_out.write(channel, 1, window=Window(col0, row0, ncol, nrow))
            del halo, order, count_out, area_out, area_m2, channel
            log(tag, "pass 2 tile %d of %d written" % (tile_index + 1, len(tiles)))
    if ended_count_total != land_total:
        raise FlowDivideError("the counts ending at the terminals (%d) do not add up to the land pixels (%d)" % (ended_count_total, land_total))
    publish(acc_temporary, acc_path)
    publish(aca_temporary, aca_path)
    publish(str_temporary, str_path)
    report = {"land_pixels": int(land_total), "land_area_km2": ended_area_total / 1e6, "exits": int(records.exit_global.size),
              "inlets": int(records.inlet_global.size), "work_tile": tile, "channel_threshold_km2": channel_threshold_km2,
              "seconds": round(time.time() - started, 1)}
    write_json(acc_path + ".report.json", report)
    log(tag, "written %s, %s, %s: %d land pixels, %.1f km2" % (acc_path, aca_path, str_path, land_total, report["land_area_km2"]))
    return report


def fd1_1_channel_mask_from_area(aca_path, str_path, grid, channel_threshold_km2=1.0, aca_to_km2=1.0, strip_rows=None, tag="fd1.1"):
    """the channel mask (Byte: 1 where the upstream area is at least channel_threshold_km2) from an
    upstream area raster given instead of fd1.1's own (a provider's, in the unit aca_to_km2 converts to
    square kilometres); its nodata pixels are 0"""
    started = time.time()
    if strip_rows is None:
        strip_rows = strip_rows_for(grid.ncol)
    temporary = str_path + ".partial.tif"
    channel_pixels = 0
    with rasterio.open(aca_path) as aca_dataset:
        # on the grid, not only of its size: the origin, the pixel and the CRS (an ACA one pixel to the east would
        # give a channel mask one pixel off)
        check_raster_on_the_grid(aca_dataset, grid, aca_path)
        nodata = aca_dataset.nodata
        # the threshold put into the raster's unit and into float32 once, and the float32 area compared with it,
        # as FD3 does; the area in float64 times aca_to_km2 against the double
        # threshold decides otherwise at 1.00000001 km2
        if aca_to_km2 == 1.0:
            threshold_in_the_unit = np.float32(channel_threshold_km2)
        elif aca_to_km2 == 1e-6:
            threshold_in_the_unit = np.float32(channel_threshold_km2 * 1.0e6)
        else:
            threshold_in_the_unit = np.float32(channel_threshold_km2 / aca_to_km2)
        with rasterio.open(temporary, "w", **raster_profile(grid, "uint8", 0, predictor=False)) as out:
            for row0 in range(0, grid.nrow, strip_rows):
                nrow = min(strip_rows, grid.nrow - row0)
                strip = aca_dataset.read(1, window=Window(0, row0, grid.ncol, nrow)).astype(np.float32)     # as Float32
                channel = strip >= threshold_in_the_unit
                if nodata is not None:
                    channel &= strip != nodata
                channel &= np.isfinite(strip) & (strip < np.float32(3.0e38))     # a river pixel: finite and below 3e38
                channel_pixels += int(channel.sum())
                out.write(channel.astype(np.uint8), 1, window=Window(0, row0, grid.ncol, nrow))
    publish(temporary, str_path)
    report = {"channel_pixels": channel_pixels, "threshold_km2": channel_threshold_km2, "area_raster": aca_path, "seconds": round(time.time() - started, 1)}
    write_json(str_path + ".report.json", report)
    log(tag, "written %s: %d channel pixels of at least %g km2 from %s" % (str_path, channel_pixels, channel_threshold_km2, aca_path))
    return report


# =============================================================================
#  [4] FD1.2  outlet indexing
# =============================================================================
#
#  Every pixel where the flow ends (a river mouth or an inland sink) is an outlet and has a basin.
#  The outlets are found in one pass over DIR; their upstream count and area are read from the two
#  rasters of FD1.1, in the raster blocks that hold an outlet only (a few percent of the blocks);
#  and the basins are numbered from the largest area down, so that basin 1 is the largest basin of the
#  continent and every basin of at least a given area is a prefix of the table.

@njit(cache=True)
def _collect_terminals_in_strip(strip, row0, out_row, out_col, out_kind, start):
    nrow, ncol = strip.shape
    count = start
    for row in range(nrow):
        for col in range(ncol):
            value = strip[row, col]
            if value == MERIT_MOUTH:
                out_row[count] = row0 + row
                out_col[count] = col
                out_kind[count] = 0
                count += 1
            elif value == MERIT_SINK:
                out_row[count] = row0 + row
                out_col[count] = col
                out_kind[count] = 1
                count += 1
    return count


BASIN_ORDER_RULES = ("area-then-count", "from-a-published-table")
# How the basins are numbered, which decides the basin ids the products carry:
#   "area-then-count"  the area in double precision, the largest first; ties by the pixel count, then by
#                      the outlet's place in the grid.  What the 30 m products are published with.
#   "from-a-published-table"
#                      the ids are not worked out at all: every outlet takes the id the published table
#                      gives its pixel, and the run is refused when an outlet is missing from that table,
#                      when two outlets would take the same id, or when the ids are not 1 .. N.  What the
#                      MERIT 90 m line uses, always: a re-run gives every basin the id v1.0 and v1.1 published.
#                      A rule of the area rounded to Float32, ties by the scan, does not reproduce v1.0: over all
#                      24,587,290 outlets it leaves 23,907,729 ids unlike v1.0's, and the area rounded to six
#                      decimals, the order v1.0's ids keep, 22,784,002 (the order among equal areas is not the scan
#                      order).  So no rule computed from today's rasters stands in for the published table.


def same_coordinate_system(first, second):
    """GDAL's OSRIsSame: EPSG:4326 and a WGS84 WKT written longitude first are the same system
    (rasterio's == tells them apart)"""
    if first is None or second is None:
        return first is None and second is None
    from osgeo import osr
    first_reference = osr.SpatialReference()
    second_reference = osr.SpatialReference()
    if first_reference.ImportFromWkt(first.to_wkt()) != 0 or second_reference.ImportFromWkt(second.to_wkt()) != 0:
        return False                                    # a system GDAL cannot read is not taken as the same
    return bool(first_reference.IsSame(second_reference))


def check_raster_on_the_grid(dataset, grid, path):
    """a raster read beside the flow directions lies on their grid: the same size, the same coordinate system and a
    geotransform whose corners drift less than a hundredth of a pixel over the grid (fd1.2 and the cut read ACC and ACA by row and
    column without asking)"""
    expected = grid.transform
    found = dataset.transform
    if not all(np.isfinite(term) for term in tuple(found)[:6] + tuple(expected)[:6]):
        raise FlowDivideError("%s has a geotransform that is not finite" % path)
    east_west = abs(found.c - expected.c) + abs(found.a - expected.a) * grid.ncol + abs(found.b - expected.b) * grid.nrow
    north_south = abs(found.f - expected.f) + abs(found.d - expected.d) * grid.ncol + abs(found.e - expected.e) * grid.nrow
    if (dataset.width, dataset.height) != (grid.ncol, grid.nrow) or east_west > 0.01 * abs(expected.a) \
            or north_south > 0.01 * abs(expected.e) or not same_coordinate_system(dataset.crs, grid.crs):
        raise FlowDivideError("%s does not lie on the grid of the flow directions (size, geotransform or coordinate system)" % path)


def fd1_2_outlet_indexing(dir_path, acc_path, aca_path, out_table_path, grid, cut_cap_pixels=2 ** 31 - 1, aca_to_km2=1e-6, strip_rows=2048, tag="fd1.2", basin_order="area-then-count", basin_id_table=None, global_id_is_basin_id=False,
                           single_pixel_tolerance=1.0e-3):
    """the basin table begins (fd_tables.BASIN_TABLE_COLUMNS): basin_id, outlet_row, outlet_col,
    outlet_flag (0 exorheic, 1 endorheic), basin_grid_count and basin_area_km2 of every outlet, sorted by area, the
    largest first (basin_order: one of BASIN_ORDER_RULES); every other column unset for the later steps, and
    global_basin_id = basin_id when the grid is the globe (global_id_is_basin_id, MERIT).  acc_path may be the
    upstream count of fd1.1 or a provider's (both count the pixel itself); aca_path the upstream area of fd1.1 (square
    metres, aca_to_km2 = 1e-6) or a provider's (square kilometres: aca_to_km2 = 1).  cut_cap_pixels is only reported:
    the table has no needs_cut column."""
    if basin_order not in BASIN_ORDER_RULES:
        raise FlowDivideError("unknown basin order '%s': it must be one of %s" % (basin_order, ", ".join(BASIN_ORDER_RULES)))
    if basin_order == "from-a-published-table" and not basin_id_table:
        raise FlowDivideError("the basin order 'from-a-published-table' needs basin_id_table: the table whose ids to carry over")
    started = time.time()
    rows_list = []
    cols_list = []
    kinds_list = []
    with rasterio.open(dir_path) as dir_dataset:
        for row0 in range(0, grid.nrow, strip_rows):
            nrow = min(strip_rows, grid.nrow - row0)
            strip = dir_dataset.read(1, window=Window(0, row0, grid.ncol, nrow))
            capacity = int((strip == MERIT_MOUTH).sum() + (strip == MERIT_SINK).sum())
            if capacity == 0:
                continue
            out_row = np.empty(capacity, np.int64)
            out_col = np.empty(capacity, np.int64)
            out_kind = np.empty(capacity, np.uint8)
            found = _collect_terminals_in_strip(strip, row0, out_row, out_col, out_kind, 0)
            rows_list.append(out_row[:found])
            cols_list.append(out_col[:found])
            kinds_list.append(out_kind[:found])
    outlet_row = np.concatenate(rows_list) if rows_list else np.zeros(0, np.int64)
    outlet_col = np.concatenate(cols_list) if cols_list else np.zeros(0, np.int64)
    outlet_kind = np.concatenate(kinds_list) if kinds_list else np.zeros(0, np.uint8)
    outlet_count = outlet_row.size
    if outlet_count == 0:
        raise FlowDivideError("no river mouth and no inland sink in the flow directions")
    log(tag, "%d outlets: %d river mouths, %d inland sinks" % (outlet_count, int((outlet_kind == 0).sum()), int((outlet_kind == 1).sum())))
    # the count and area at every outlet, one raster block at a time, in file order
    block_row = outlet_row // RASTER_BLOCK
    block_col = outlet_col // RASTER_BLOCK
    block_key = block_row * (grid.ncol // RASTER_BLOCK + 2) + block_col
    order = np.argsort(block_key, kind="stable")
    run_starts = np.nonzero(np.diff(block_key[order], prepend=-1))[0]
    run_ends = np.append(run_starts[1:], outlet_count)
    count_read_at_outlet = np.zeros(outlet_count, np.float64)    # as the raster holds it
    area_at_outlet = np.zeros(outlet_count, np.float64)
    with rasterio.open(acc_path) as acc_dataset, rasterio.open(aca_path) as aca_dataset:
        check_raster_on_the_grid(acc_dataset, grid, acc_path)
        check_raster_on_the_grid(aca_dataset, grid, aca_path)
        blocks_read = 0
        for first, last in zip(run_starts, run_ends):
            b_row = block_row[order[first]]
            b_col = block_col[order[first]]
            members = order[first:last]
            window = Window(b_col * RASTER_BLOCK, b_row * RASTER_BLOCK, min(RASTER_BLOCK, grid.ncol - b_col * RASTER_BLOCK), min(RASTER_BLOCK, grid.nrow - b_row * RASTER_BLOCK))
            acc_block = acc_dataset.read(1, window=window)
            aca_block = aca_dataset.read(1, window=window)
            local_rows = outlet_row[members] - b_row * RASTER_BLOCK
            local_cols = outlet_col[members] - b_col * RASTER_BLOCK
            # the count as it is read, rounded only after it has been checked: a provider's count may be
            # a Float32 raster, and the raw value is tested against 1 and then a half added before it
            # is taken as an integer, so 0.6 is refused rather than rounded up to 1
            count_read_at_outlet[members] = acc_block[local_rows, local_cols].astype(np.float64)
            area_at_outlet[members] = aca_block[local_rows, local_cols].astype(np.float64) * aca_to_km2
            blocks_read += 1
    log(tag, "read the count and area at every outlet from %d raster blocks" % blocks_read)
    # written the way round that refuses a NaN instead of letting it pass; a
    # nodata of 1.79e308 in the provider's rasters shows up here as well
    good_count = (count_read_at_outlet >= 1) & (count_read_at_outlet <= ACCUMULATION_MAX)
    if not good_count.all():
        raise FlowDivideError("%d outlets carry an upstream count that is not between 1 and %g"
                              % (int((~good_count).sum()), ACCUMULATION_MAX))
    count_at_outlet = np.floor(count_read_at_outlet + 0.5).astype(np.int64)
    good_area = (area_at_outlet > 0) & (area_at_outlet <= ACCUMULATION_MAX * aca_to_km2)
    if not good_area.all():
        raise FlowDivideError("%d outlets carry an upstream area that is not above 0 and below the largest a "
                              "raster can hold" % int((~good_area).sum()))
    # the mean pixel area of a basin must lie between the pixel areas at 85 degrees and at the equator
    mean_pixel_km2 = area_at_outlet / count_at_outlet
    if grid.geographic:
        # the bounds: the pixel area at 85 degrees and at the equator, 1e-4 either side.  A basin of one pixel
        # is held to its own row just below; tighter bounds at the grid's own latitude edges would refuse a
        # one-pixel basin the 0.5 % of the MERIT grid lets through
        lowest = pixel_area_m2(85.0, grid.pixel_width, grid.pixel_height, grid.earth_model) / 1e6 * (1.0 - 1.0e-4)
        highest = pixel_area_m2(0.0, grid.pixel_width, grid.pixel_height, grid.earth_model) / 1e6 * (1.0 + 1.0e-4)
    else:
        lowest = abs(grid.pixel_width * grid.pixel_height) / 1e6 * 0.999
        highest = abs(grid.pixel_width * grid.pixel_height) / 1e6 * 1.001
    # the area raster holds whole square metres, so a basin's total may be off by half a square metre; a
    # provider's raster in square kilometres is read as it is (Float32: its own rounding is finer than this)
    allowance_km2 = 1e-6 / count_at_outlet
    if (mean_pixel_km2 < lowest - allowance_km2).any() or (mean_pixel_km2 > highest + allowance_km2).any():
        raise FlowDivideError("the ratio of upstream area to upstream count is outside the pixel areas of the grid at some outlets; the rasters do not line up")
    # a basin of one pixel: its area must be the area of a pixel of its own row, which ties the area raster, the earth
    # model and the geotransform together in one number (fd1.2; the range over the grid above lets a
    # one-pixel basin at 60 degrees through with the area of a pixel at the equator).  The tolerance is
    # 0.1 %, and 0.5 % on the MERIT grid (single_pixel_tolerance), whose provider has an
    # area formula of its own
    one_pixel = np.flatnonzero(count_at_outlet == 1)
    if one_pixel.size:
        if grid.geographic:
            # one area per row, not per basin: a continent holds millions of one-pixel basins
            rows, row_of_basin = np.unique(outlet_row[one_pixel], return_inverse=True)
            row_latitudes = grid.transform.f + (rows.astype(np.float64) + 0.5) * grid.pixel_height
            expected_km2 = np.array([pixel_area_m2(latitude, grid.pixel_width, grid.pixel_height, grid.earth_model) / 1e6
                                     for latitude in row_latitudes])[row_of_basin]
        else:
            expected_km2 = np.full(one_pixel.size, abs(grid.pixel_width * grid.pixel_height) / 1e6)
        rounding_km2 = 0.5e-6 if aca_to_km2 < 1.0 else 0.0
        difference_km2 = np.maximum(np.abs(area_at_outlet[one_pixel] - expected_km2) - rounding_km2, 0.0)
        tolerance = single_pixel_tolerance
        off = int((difference_km2 / expected_km2 > tolerance).sum())
        if off:
            raise FlowDivideError("%d of the %d basins of one pixel carry an upstream area more than %g off the area of a pixel of "
                                  "their row; the area raster does not line up with the grid" % (off, one_pixel.size, tolerance))
    # the basins numbered by area, the largest first
    pixel_index = outlet_row * grid.ncol + outlet_col
    if basin_order == "from-a-published-table":
        order = _basin_ids_from_a_published_table(basin_id_table, pixel_index, grid, tag)
    else:
        order = np.lexsort((pixel_index, -count_at_outlet, -area_at_outlet))
    log(tag, "the basins are numbered by the '%s' rule" % basin_order)
    table = fd_tables.new_basin_table(outlet_count)
    table["outlet_row"] = outlet_row[order]
    table["outlet_col"] = outlet_col[order]
    table["outlet_flag"] = outlet_kind[order].astype(np.int64)
    table["basin_grid_count"] = count_at_outlet[order]
    table["basin_area_km2"] = area_at_outlet[order]
    if global_id_is_basin_id:
        table["global_basin_id"] = table["basin_id"]
    fd_tables.write_basin_table(table, out_table_path, "fd1.2",
                                "%d outlets, sorted by the '%s' rule" % (outlet_count, basin_order))
    over_the_cap = int((count_at_outlet > cut_cap_pixels).sum())
    report = {"outlets": int(outlet_count), "river_mouths": int((outlet_kind == 0).sum()), "inland_sinks": int((outlet_kind == 1).sum()),
              "largest_basin_km2": float(table["basin_area_km2"].iloc[0]), "basins_at_least_1km2": int((table["basin_area_km2"] >= 1.0).sum()),
              "needs_cut": over_the_cap, "cut_cap_pixels": int(cut_cap_pixels), "seconds": round(time.time() - started, 1)}
    write_json(out_table_path + ".report.json", report)
    log(tag, "written %s: largest basin %.1f km2, %d basins of at least 1 km2, %d over %d pixels" % (out_table_path, report["largest_basin_km2"], report["basins_at_least_1km2"], report["needs_cut"], cut_cap_pixels))
    return report


# =============================================================================
#  [5] FD1.3  watershed delineation
# =============================================================================
#
#  Every land pixel gets the basin id of the outlet its flow ends at, in the three tile steps with a
#  label travelling upstream: pass A gives every inlet the label its path reaches inside the tile (a
#  basin id at a terminal, or the exit the path leaves through); the exit graph resolves every exit to
#  a basin id; pass C labels every pixel once.  The rectangle and the pixel count of every basin are
#  collected in pass C, and the count must equal the upstream count at the basin's outlet.

@njit(cache=True)
def _label_tile_backwards(halo_dir, order, tile_row0, tile_col0, grid_nrow, grid_ncol, periodic, terminal_global_sorted, terminal_label_sorted,
                          mask, mask_value, label, exit_pixel_global, exit_destination_global, exit_label_known, use_known):
    """the labels of one tile from the order read backwards: a terminal pixel takes its label from the
    list (without a mask the terminals are the mouths and sinks, so only those pixels are looked up;
    with a mask they are the piece outlets, ordinary pixels, marked in the tile first), a pixel whose
    path leaves the tile takes EXIT_LABEL_BASE + exit number (or the resolved basin id from
    exit_label_known in pass C), every other pixel takes the label of its downstream pixel.  With a
    mask, only the pixels where mask == mask_value are labelled (the others get 0).
    Returns (status, exit_count): status 1 a terminal not in the list, 2 a path leaves the grid or
    enters nodata, 3 a pixel flows into a masked-out pixel, 4 the exits differ from pass A."""
    halo_nrow, halo_ncol = halo_dir.shape
    nrow = halo_nrow - 2
    ncol = halo_ncol - 2
    exit_count = 0
    terminal_count = terminal_global_sorted.size
    # with a mask the terminals are marked in the tile once, so that no pixel has to search the list
    terminal_mark = np.zeros(nrow * ncol if mask_value != 0 else 1, np.uint8)
    if mask_value != 0:
        first_global = _global_index(tile_row0, 0, grid_ncol, 0)
        last_global = _global_index(tile_row0 + nrow - 1, grid_ncol - 1, grid_ncol, 0)
        start = np.searchsorted(terminal_global_sorted, first_global)
        for t in range(start, terminal_count):
            value = terminal_global_sorted[t]
            if value > last_global:
                break
            t_row = value // grid_ncol - tile_row0
            t_col = value - (value // grid_ncol) * grid_ncol - tile_col0
            if periodic and t_col < 0:
                t_col += grid_ncol
            if t_row >= 0 and t_row < nrow and t_col >= 0 and t_col < ncol:
                terminal_mark[t_row * ncol + t_col] = 1
    for k in range(order.size - 1, -1, -1):
        pixel = order[k]
        row = pixel // ncol
        col = pixel - row * ncol
        if mask_value != 0 and mask[row, col] != mask_value:
            label[pixel] = 0
            continue
        code = halo_dir[row + 1, col + 1]
        this_global = _global_index(tile_row0 + row, tile_col0 + col, grid_ncol, periodic)
        if (mask_value == 0 and IS_TERMINAL[code] != 0) or (mask_value != 0 and terminal_mark[pixel] != 0):
            position = np.searchsorted(terminal_global_sorted, this_global)
            if position < terminal_count and terminal_global_sorted[position] == this_global:
                label[pixel] = terminal_label_sorted[position]
                continue
        drow = DROW[code]
        dcol = DCOL[code]
        if drow == 0 and dcol == 0:
            if mask_value != 0:
                return 1, exit_count            # a mouth or sink inside the masked basin that is no piece outlet
            return 1, exit_count                # a mouth or sink that the outlet table does not list
        down_row = row + drow
        down_col = col + dcol
        if IS_LAND[halo_dir[down_row + 1, down_col + 1]] == 0:
            return 2, exit_count
        if down_row >= 0 and down_row < nrow and down_col >= 0 and down_col < ncol:
            if mask_value != 0 and mask[down_row, down_col] != mask_value:
                return 3, exit_count
            label[pixel] = label[down_row * ncol + down_col]
            continue
        grid_row = tile_row0 + down_row
        grid_col = tile_col0 + down_col
        if periodic:
            grid_col = grid_col % grid_ncol
        if grid_row < 0 or grid_row >= grid_nrow or grid_col < 0 or grid_col >= grid_ncol:
            return 2, exit_count
        if use_known != 0:
            if exit_count >= exit_pixel_global.size or exit_pixel_global[exit_count] != this_global:
                return 4, exit_count
            label[pixel] = np.uint32(exit_label_known[exit_count])
        else:
            exit_pixel_global[exit_count] = this_global
            exit_destination_global[exit_count] = grid_row * grid_ncol + grid_col
            label[pixel] = EXIT_LABEL_BASE + np.uint32(exit_count)
        exit_count += 1
    return 0, exit_count


@njit(cache=True)
def _inlet_labels(inlet_local, inlet_count, label, exit_pixel_global, inlet_label, inlet_exit_global):
    """what pass A found at every inlet: a label (basin id or piece code, > 0) or the exit it drains to"""
    for i in range(inlet_count):
        value = label[inlet_local[i]]
        if value >= EXIT_LABEL_BASE:
            inlet_label[i] = 0
            inlet_exit_global[i] = exit_pixel_global[np.int64(value - EXIT_LABEL_BASE)]
        else:
            inlet_label[i] = np.int64(value)
            inlet_exit_global[i] = -1


@njit(cache=True)
def _collect_basin_statistics(label, nrow, ncol, tile_row0, tile_col0, grid_ncol, periodic, count, row_min, row_max, col_min, col_max, col_min_shift, col_max_shift):
    """the pixel count and the rectangle of every label, in two column frames on a periodic grid: the
    grid's own, and one shifted by half the width, so that a basin across the antimeridian can be
    unrolled afterwards"""
    half = grid_ncol // 2
    for pixel in range(label.size):
        value = label[pixel]
        if value == 0:
            continue
        row = pixel // ncol
        col = pixel - row * ncol
        grid_row = tile_row0 + row
        grid_col = tile_col0 + col
        count[value] += 1
        if grid_row < row_min[value]:
            row_min[value] = grid_row
        if grid_row >= row_max[value]:
            row_max[value] = grid_row + 1          # one past: left closed, right open
        if grid_col < col_min[value]:
            col_min[value] = grid_col
        if grid_col >= col_max[value]:
            col_max[value] = grid_col + 1
        if periodic:
            shifted = (grid_col + half) % grid_ncol
            if shifted < col_min_shift[value]:
                col_min_shift[value] = shifted
            if shifted >= col_max_shift[value]:
                col_max_shift[value] = shifted + 1


def unroll_rectangles(grid, col_min, col_max, col_min_shift, col_max_shift):
    """on a periodic grid, an object whose columns span more than half the grid in the grid's own frame
    lies across the antimeridian; its rectangle is taken from the shifted frame and unrolled past the
    last column.  Returns (col_min, col_max, astride) with col_max possibly >= ncol."""
    if not grid.periodic:
        return col_min, col_max, np.zeros(col_min.shape, np.uint8)
    half = grid.ncol // 2
    astride = (col_max - col_min > half) & (col_max > col_min)
    new_min = col_min.copy()
    new_max = col_max.copy()
    unrolled_min = col_min_shift.astype(np.int64) - half
    unrolled_min[unrolled_min < 0] += grid.ncol
    width = col_max_shift.astype(np.int64) - col_min_shift.astype(np.int64)
    new_min[astride] = unrolled_min[astride]
    new_max[astride] = unrolled_min[astride] + width[astride]
    return new_min, new_max, astride.astype(np.uint8)


def read_outlet_table(path):
    return read_table(path)


def basin_rectangles(basins):
    """the rectangles of the basin table as an (N, 4) int64 array: basin_row_min, basin_row_max,
    basin_col_min, basin_col_max (unrolled past the last column for a basin across the antimeridian)"""
    return basins[["basin_row_min", "basin_row_max", "basin_col_min", "basin_col_max"]].to_numpy(np.int64)


def _basin_ids_from_a_published_table(path, pixel_index, grid, tag):
    """the order that gives every outlet the id the published table gives its pixel.

    The table is one of ours from an earlier run: it carries the basin number and the outlet's row and
    column, whatever the column names of that version.  Returns the order that puts the outlet with id 1
    first, so the table this step writes is in id order, as the published one is.  Refuses when an outlet
    of this run is not in the table, when two of ours would take the same id, or when the ids are not
    1 .. N: a product that carries the ids of an earlier one must carry all of them or none."""
    started = time.time()
    header = pd.read_csv(path, sep=" ", nrows=0).columns.tolist()
    id_column = next((name for name in ("basin_number", "basin_id", "Basin_ID") if name in header), None)
    row_column = next((name for name in ("idxs_outlet_row", "outlet_row") if name in header), None)
    col_column = next((name for name in ("idxs_outlet_col", "outlet_col") if name in header), None)
    if not id_column or not row_column or not col_column:
        raise FlowDivideError("the published table %s has no basin id and outlet columns I know (it has %s)"
                              % (path, ", ".join(header[:8])))
    published = pd.read_csv(path, sep=" ", usecols=[id_column, row_column, col_column])
    published_index = (published[row_column].to_numpy(np.int64) * grid.ncol
                       + published[col_column].to_numpy(np.int64))
    published_id = published[id_column].to_numpy(np.int64)
    if published_index.size != pixel_index.size:
        raise FlowDivideError("%s holds %d outlets and this run found %d: the ids of the published table "
                              "cannot be carried over" % (path, published_index.size, pixel_index.size))
    order_of_published = np.argsort(published_index, kind="stable")
    sorted_index = published_index[order_of_published]
    place = np.searchsorted(sorted_index, pixel_index)
    not_found = (place >= sorted_index.size)
    place_inside = np.minimum(place, sorted_index.size - 1)
    not_found |= sorted_index[place_inside] != pixel_index
    if not_found.any():
        raise FlowDivideError("%d outlets of this run are not in the published table %s: the ids cannot be "
                              "carried over" % (int(not_found.sum()), path))
    id_of_outlet = published_id[order_of_published][place_inside]
    if (id_of_outlet.min() != 1 or id_of_outlet.max() != id_of_outlet.size
            or np.unique(id_of_outlet).size != id_of_outlet.size):
        raise FlowDivideError("the ids the published table %s gives this run's outlets are not 1 .. %d without "
                              "repetition (smallest %d, largest %d, distinct %d)"
                              % (path, id_of_outlet.size, id_of_outlet.min(), id_of_outlet.max(),
                                 np.unique(id_of_outlet).size))
    log(tag, "the basin ids are carried over from %s: %d outlets, every one found, ids 1 .. %d, %.0f s"
        % (os.path.basename(path), id_of_outlet.size, id_of_outlet.size, time.time() - started))
    return np.argsort(id_of_outlet, kind="stable")


def fd1_3_watershed_delineation(dir_path, basin_table_path, bsn_path, grid, tile=WORK_TILE_DEFAULT, tag="fd1.3"):
    """the basin raster (UInt32, 0 = water), and the basin table of fd1.2 again with the box of every basin and the
    outlet and the box in longitude and latitude (in place; the group and region
    columns are set back to unset, fd1.4 and fd1.6 fill them)"""
    started = time.time()
    check_work_tile(tile)
    outlets = fd_tables.read_basin_table(basin_table_path)
    basin_count = len(outlets)
    terminal_global = (outlets["outlet_row"].to_numpy(np.int64) * grid.ncol + outlets["outlet_col"].to_numpy(np.int64))
    terminal_basin = outlets["basin_id"].to_numpy(np.int64)
    # the pass over a tile marks the pixels that leave it with EXIT_LABEL_BASE + the exit number, so a basin
    # id at or above it would be read as an exit (and index past the exits); refused (far
    # beyond any continent)
    if terminal_basin.size and int(terminal_basin.max()) >= int(EXIT_LABEL_BASE):
        raise FlowDivideError("basin %d is at or above %d, which the tile pass keeps for its exits" % (int(terminal_basin.max()), int(EXIT_LABEL_BASE)))
    sort = np.argsort(terminal_global)
    terminal_global_sorted = terminal_global[sort]
    terminal_basin_sorted = terminal_basin[sort]
    if np.unique(terminal_global_sorted).size != basin_count:
        raise FlowDivideError("the outlet table lists one pixel twice")
    tiles = tiles_of_grid(grid, tile)
    records = TilePassRecords()
    empty_mask = np.zeros((1, 1), np.uint32)
    # pass A
    with rasterio.open(dir_path) as dir_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            if taken != land_count:
                raise FlowDivideError("the flow directions hold a cycle inside the tile at row %d col %d" % (row0, col0))
            label = np.zeros(nrow * ncol, np.uint32)
            max_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(max_edge, np.int64)
            exit_destination = np.empty(max_edge, np.int64)
            status, exit_count = _label_tile_backwards(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, terminal_global_sorted, terminal_basin_sorted,
                                                       empty_mask, 0, label, exit_global, exit_destination, np.zeros(1, np.int64), 0)
            if status != 0:
                raise FlowDivideError("pass A failed with status %d in the tile at row %d col %d (1: an outlet the table does not list, 2: a path leaves the land)" % (status, row0, col0))
            inlet_local = np.empty(max_edge, np.int32)
            inlet_global = np.empty(max_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_label = np.empty(inlet_count, np.int64)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            _inlet_labels(inlet_local, inlet_count, label, exit_global, inlet_label, inlet_exit_global)
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.inlet_label.append(inlet_label)
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            del halo, order, label
            log(tag, "pass A tile %d of %d: %d exits, %d inlets" % (tile_index + 1, len(tiles), exit_count, inlet_count))
    records.concatenate()
    # pass B
    status, inlet_of_exit, next_exit = _link_exits_to_inlets(records.exit_global, records.exit_destination, records.inlet_global, records.inlet_exit_global)
    if status != 0:
        raise FlowDivideError("the exit graph cannot be linked (status %d)" % status)
    status, label_of_exit = _resolve_labels_over_exit_graph(next_exit, inlet_of_exit, records.inlet_label, records.exit_global.size)
    if status != 0:
        raise FlowDivideError("the exit graph cannot be resolved to basins (status %d)" % status)
    log(tag, "pass B: %d exits resolved" % records.exit_global.size)
    # pass C
    count = np.zeros(basin_count + 1, np.int64)
    row_min = np.full(basin_count + 1, np.iinfo(np.int32).max, np.int32)
    row_max = np.full(basin_count + 1, -1, np.int32)
    col_min = np.full(basin_count + 1, np.iinfo(np.int32).max, np.int32)
    col_max = np.full(basin_count + 1, -1, np.int32)
    col_min_shift = np.full(basin_count + 1, np.iinfo(np.int32).max, np.int32)
    col_max_shift = np.full(basin_count + 1, -1, np.int32)
    temporary = bsn_path + ".partial.tif"
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(temporary, "w", **raster_profile(grid, "uint32", 0)) as bsn_out:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            order, land_count, taken = tile_topological_order(halo)
            label = np.zeros(nrow * ncol, np.uint32)
            first = records.tile_exit_offsets[tile_index]
            last = records.tile_exit_offsets[tile_index + 1]
            status, exit_count = _label_tile_backwards(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, terminal_global_sorted, terminal_basin_sorted,
                                                       empty_mask, 0, label, records.exit_global[first:last], np.zeros(1, np.int64), label_of_exit[first:last], 1)
            if status != 0 or exit_count != last - first:
                raise FlowDivideError("pass C failed with status %d in the tile at row %d col %d" % (status, row0, col0))
            _collect_basin_statistics(label, nrow, ncol, row0, col0, grid.ncol, grid.periodic, count, row_min, row_max, col_min, col_max, col_min_shift, col_max_shift)
            bsn_out.write(label.reshape(nrow, ncol), 1, window=Window(col0, row0, ncol, nrow))
            del halo, order, label
            log(tag, "pass C tile %d of %d written" % (tile_index + 1, len(tiles)))
    # the checks: every basin's pixel count equals the upstream count at its outlet
    expected = np.zeros(basin_count + 1, np.int64)
    expected[outlets["basin_id"].to_numpy(np.int64)] = outlets["basin_grid_count"].to_numpy(np.int64)
    differing = np.nonzero(count[1:] != expected[1:])[0]
    if differing.size != 0:
        first_bad = int(differing[0]) + 1
        raise FlowDivideError("%d basins have a pixel count that differs from the upstream count at their outlet (first: basin %d, %d labelled, %d expected)" % (differing.size, first_bad, count[first_bad], expected[first_bad]))
    col_min_unrolled, col_max_unrolled, astride = unroll_rectangles(grid, col_min.astype(np.int64), col_max.astype(np.int64), col_min_shift, col_max_shift)
    # the extended outlet table: the outlet table, the rectangle of every
    # basin (unrolled past the last column for a basin across the antimeridian, so basin_col_max > ncol
    # says which those are), the outlet and the rectangle in longitude and latitude
    basin_ids = outlets["basin_id"].to_numpy(np.int64)
    table = outlets.copy()
    table["basin_row_min"] = row_min[basin_ids].astype(np.int64)
    table["basin_row_max"] = row_max[basin_ids].astype(np.int64)
    table["basin_col_min"] = col_min_unrolled[basin_ids]
    table["basin_col_max"] = col_max_unrolled[basin_ids]
    for column in ("level1_id", "basin_id_in_level1", "level2_id", "basin_id_in_level2", "level3_id", "basin_id_in_level3",
                   "region_id", "basin_id_in_region"):
        table[column] = fd_tables.UNSET_ID
    # the outlet lies in its own box: in the columns as they are, or one width east on a periodic grid
    outlet_col_values = table["outlet_col"].to_numpy(np.int64)
    inside_rows = (table["outlet_row"] >= table["basin_row_min"]) & (table["outlet_row"] < table["basin_row_max"])
    inside_cols = (outlet_col_values >= table["basin_col_min"]) & (outlet_col_values < table["basin_col_max"])
    if grid.periodic:
        inside_cols |= (outlet_col_values + grid.ncol >= table["basin_col_min"]) & (outlet_col_values + grid.ncol < table["basin_col_max"])
    if not (inside_rows & inside_cols).all():
        raise FlowDivideError("the outlet of basin %d lies outside its own box" % int(table["basin_id"][~(inside_rows & inside_cols)].iloc[0]))
    table["outlet_lon"], table["outlet_lat"] = grid.pixel_centre_lon_lat(table["outlet_row"].to_numpy(np.int64), table["outlet_col"].to_numpy(np.int64))
    minlon, minlat, maxlon, maxlat = grid.pixel_boxes_lon_lat(table["basin_row_min"].to_numpy(np.int64), table["basin_row_max"].to_numpy(np.int64),
                                                              table["basin_col_min"].to_numpy(np.int64), table["basin_col_max"].to_numpy(np.int64))
    table["bbox_minlon"] = minlon
    table["bbox_minlat"] = minlat
    table["bbox_maxlon"] = maxlon
    table["bbox_maxlat"] = maxlat
    # the basin table stops counting as done before the new raster replaces the old one
    if os.path.exists(basin_table_path + ".done"):
        os.remove(basin_table_path + ".done")
    publish(temporary, bsn_path)
    fd_tables.write_done_marker(bsn_path, "%d basins, every basin's pixel count equals the upstream count at its outlet" % basin_count)
    fd_tables.write_basin_table(table, basin_table_path, "fd1.3", "the box of every basin and the coordinates of fd1.3")
    report = {"basins": int(basin_count), "land_pixels": int(count[1:].sum()), "basins_astride_the_seam": int(astride.sum()),
              "exits": int(records.exit_global.size), "work_tile": tile, "seconds": round(time.time() - started, 1)}
    write_json(bsn_path + ".report.json", report)
    log(tag, "written %s and %s: %d basins, %d land pixels, every count equals its outlet's upstream count" % (bsn_path, basin_table_path, basin_count, report["land_pixels"]))
    return report


# =============================================================================
#  [6] FD1.4  basin grouping: the Level-03 raster and the basin groups of whole basins
# =============================================================================
#
#  A basin group is a set of whole basins read together; nothing flows between two basins, so nothing
#  crosses the boundary of a group.  Each basin is given the HydroBASINS Level-03 unit that covers
#  most of its pixels, counted on the Level-03 polygons burned onto the grid (the outlet pixel alone
#  is not enough: the HydroBASINS land mask does not reach the river mouth of the largest rivers).
#  A small doubtful basin (a fragment of coast the vectors map as water) takes the code of its longest
#  shared boundary; a basin still without a code takes the code of the coded basin whose outlet is
#  nearest, within one block; what remains are islands, gathered around seeds from the largest down,
#  within a reach of a few blocks and a window well under the capacity.  On a periodic grid the basins
#  across the antimeridian are kept out of the Level-03 groups and gathered among themselves the same
#  way.  The rules, the numbering and the tables are those of the paper.

LEVEL3_CODE_LIMIT = 1000                                # a Level-03 code has three digits, 100 .. 999


def _open_level3_layer(path, query, codes_seen, tag):
    """one polygon file opened read-only, its filter set and the codes it will burn added to codes_seen: a file without a layer, a spatial reference or the field PFAF_ID, a filter GDAL does
    not accept and a code outside 100 .. 999 are refused (going on after a refused filter would burn every
    polygon).  Returns the file, its layer
    and its polygon count; the file stays open until the burn is done."""
    from osgeo import ogr
    source = ogr.Open(path)
    if source is None:
        raise FlowDivideError("cannot open %s" % path)
    layer = source.GetLayer(0)
    if layer is None:
        raise FlowDivideError("%s holds no layer" % path)
    # GDAL's burn carries the polygons from the layer's spatial reference to the grid's; a layer without one
    # would be burned untransformed with only a warning
    if layer.GetSpatialRef() is None:
        raise FlowDivideError("%s carries no spatial reference" % path)
    field_index = layer.GetLayerDefn().GetFieldIndex("PFAF_ID")
    if field_index < 0:
        raise FlowDivideError("%s has no field PFAF_ID" % path)
    condition = query.replace("!=", "<>") if query else None      # the dataset's pandas query as an OGR condition
    if condition and layer.SetAttributeFilter(condition) != 0:
        raise FlowDivideError("the filter '%s' is not accepted on %s" % (condition, path))
    polygon_count = 0
    layer.ResetReading()
    for feature in layer:
        code = feature.GetFieldAsInteger64(field_index)
        if code < 100 or code >= LEVEL3_CODE_LIMIT:
            # the groups of levels 2 and 1 are the code's first two digits and its first (fd1_4_group_whole_basins)
            raise FlowDivideError("%s carries the code %d, which is not a three-digit Level-03 code" % (path, code))
        codes_seen.add(code)
        polygon_count += 1
    layer.ResetReading()
    log(tag, "  in  %s%s: %d polygons" % (path, ", filter " + condition if condition else "", polygon_count))
    return source, layer, polygon_count


def _burn_level3_layers(dir_path, temporary, layers, tag):
    """the empty output on the grid of DIR
    (DIR's size, geotransform and spatial reference, 512 x 512 tiles, UInt16, nodata 0), and every layer burned
    into it in the order given.  The output and the polygon files are closed on the way out, the output first.
    Returns the codes the layers hold and their polygon count."""
    from osgeo import gdal
    codes_seen = set()
    polygon_count = 0
    sources = []
    output = None
    try:
        for path, query in layers:
            source, layer, layer_polygons = _open_level3_layer(path, query, codes_seen, tag)
            sources.append((source, layer))
            polygon_count += layer_polygons
        if not codes_seen:
            raise FlowDivideError("no Level-03 polygon to burn")
        reference = gdal.Open(dir_path)
        if reference is None:
            raise FlowDivideError("cannot open the reference grid %s" % dir_path)
        ncol = reference.RasterXSize
        nrow = reference.RasterYSize
        geotransform = reference.GetGeoTransform(can_return_null=True)
        projection = reference.GetProjection()
        reference = None
        if geotransform is None:
            raise FlowDivideError("%s has no geotransform" % dir_path)
        if not projection:
            raise FlowDivideError("the reference grid %s carries no spatial reference" % dir_path)
        output = gdal.GetDriverByName("GTiff").Create(
            temporary, ncol, nrow, 1, gdal.GDT_UInt16,
            options=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6",
                     "BIGTIFF=YES", "SPARSE_OK=TRUE", "NUM_THREADS=ALL_CPUS"])
        if output is None:
            raise FlowDivideError("cannot create %s" % temporary)
        if (output.SetGeoTransform(geotransform) != 0 or output.SetProjection(projection) != 0 or
                output.GetRasterBand(1).SetNoDataValue(0) != 0):
            raise FlowDivideError("cannot set the grid, the spatial reference or the nodata value of %s" % temporary)
        log(tag, "burning %d polygons of %d layer%s" % (polygon_count, len(layers), "" if len(layers) == 1 else "s"))
        for (source, layer), (path, _) in zip(sources, layers):
            if gdal.RasterizeLayer(output, [1], layer, options=["ATTRIBUTE=PFAF_ID"]) != 0:
                raise FlowDivideError("the burn of %s failed" % path)
    finally:
        output = None                                   # the close writes the tiles out
        source = layer = None                           # the loop's names hold the last file too
        sources = None
    return codes_seen, polygon_count


def _written_level3_on_the_grid(dir_path, temporary):
    """the written file reopened: DIR's size, geotransform and
    spatial reference (GDAL's OSRIsSame), UInt16, nodata 0"""
    from osgeo import gdal, osr
    reference = gdal.Open(dir_path)
    written = gdal.Open(temporary)
    if reference is None or written is None:
        raise FlowDivideError("cannot reopen %s or %s" % (dir_path, temporary))
    reference_system = osr.SpatialReference()
    written_system = osr.SpatialReference()
    reference_wkt = reference.GetProjection()
    written_wkt = written.GetProjection()
    same_system = (bool(reference_wkt) and bool(written_wkt) and reference_system.ImportFromWkt(reference_wkt) == 0 and
                   written_system.ImportFromWkt(written_wkt) == 0 and bool(reference_system.IsSame(written_system)))
    band = written.GetRasterBand(1)
    reference_geotransform = reference.GetGeoTransform(can_return_null=True)
    written_geotransform = written.GetGeoTransform(can_return_null=True)
    # byte for byte: -0.0 and 0.0 are not the same here either
    same_geotransform = (reference_geotransform is not None and written_geotransform is not None and
                         struct.pack("<6d", *reference_geotransform) == struct.pack("<6d", *written_geotransform))
    on_the_grid = (written.RasterXSize == reference.RasterXSize and written.RasterYSize == reference.RasterYSize and
                   same_geotransform and same_system and band.DataType == gdal.GDT_UInt16 and band.GetNoDataValue() == 0)
    band = None
    written = None
    reference = None
    if not on_the_grid:
        raise FlowDivideError("%s is not on the grid of %s (size, geotransform, spatial reference), or is not UInt16 with nodata 0"
                              % (temporary, dir_path))


def fd1_4_rasterise_level3(dir_path, l3_path, grid, layers, strip_rows=None, tag="fd1.4"):
    """the Level-03 polygons burned onto the grid of DIR: every pixel gets the PFAF_ID of the polygon
    that holds its centre, 0 where none does.  `layers` is a list of (shapefile path, attribute filter
    or None); the filter is a pandas query string on the attribute table, e.g. "PFAF_ID != 353".
    Burned as the script that made the raster first burned them: GDAL's RasterizeLayer on the whole output, a GTiff of 512 x 512
    tiles on the grid of DIR, the layers in the order given, GDAL's cache left at its default.  GDAL
    burns a large output in chunks of rows, and the chunks follow the tiles and the cache;
    rasterio.features.rasterize over strips of its own, each with its own origin, gives a few dozen
    pixels of MERIT's raster otherwise at the polygons' edges.  The checks are: a
    GDAL failure while burning, closing or reopening is an error, not a warning, and the written file
    must lie on DIR's grid.  `strip_rows` is the height of the strips the written file is read back in."""
    from osgeo import gdal
    started = time.time()
    temporary = l3_path + ".partial.tif"
    gdal_failures = []

    def count_gdal_failures(error_class, error_number, message):
        # a failure is counted, every warning and failure is shown
        if error_class >= gdal.CE_Failure:
            gdal_failures.append(message)
        if error_class >= gdal.CE_Warning:
            log(tag, "GDAL %s %d: %s" % ("failure" if error_class >= gdal.CE_Failure else "warning", error_number, message))

    gdal.PushErrorHandler(count_gdal_failures)
    try:
        codes_seen, polygon_count = _burn_level3_layers(dir_path, temporary, layers, tag)
        if gdal_failures:
            raise FlowDivideError("%d GDAL failures were recorded while burning or closing, the first: %s" % (len(gdal_failures), gdal_failures[0]))
        _written_level3_on_the_grid(dir_path, temporary)
        if gdal_failures:
            raise FlowDivideError("%d GDAL failures were recorded while reopening, the first: %s" % (len(gdal_failures), gdal_failures[0]))
    finally:
        gdal.PopErrorHandler()
    # the written file read back in strips: the pixels of every code (the handler is off by now: rasterio raises on a
    # read GDAL fails)
    if strip_rows is None:
        strip_rows = max(RASTER_BLOCK, min(4096, 500000000 // grid.ncol))
    counts = {}
    with rasterio.open(temporary) as written:
        for row0 in range(0, grid.nrow, strip_rows):
            nrow = min(strip_rows, grid.nrow - row0)
            burned = written.read(1, window=Window(0, row0, grid.ncol, nrow))
            values, value_counts = np.unique(burned, return_counts=True)
            for value, value_count in zip(values.tolist(), value_counts.tolist()):
                counts[value] = counts.get(value, 0) + value_count
    burned_codes = {code for code in counts if code != 0}
    if not burned_codes:
        raise FlowDivideError("the Level-03 polygons cover no pixel of the grid")
    unknown = burned_codes - codes_seen
    if unknown:
        raise FlowDivideError("codes appear in the raster that no layer holds: %s" % sorted(unknown)[:10])
    publish(temporary, l3_path)
    report = {"polygons_burned": polygon_count, "codes_burned": len(burned_codes), "codes_without_a_pixel": sorted(codes_seen - burned_codes),
              "pixels_per_code": {str(k): v for k, v in sorted(counts.items())}, "seconds": round(time.time() - started, 1)}
    write_json(l3_path + ".report.json", report)
    log(tag, "written %s: %d codes on %d pixels, %d pixels of no code; %d burned codes without a pixel" % (
        l3_path, len(burned_codes), sum(v for k, v in counts.items() if k != 0), counts.get(0, 0), len(report["codes_without_a_pixel"])))
    return report

VOTE_SLOTS = 8


@njit(cache=True)
def _vote_level3_exact_strip(bsn_strip, l3_strip, flagged_index, exact):
    """the second pass for the few basins whose slots overflowed: every code of theirs counted exactly
    in exact[flagged index, code]"""
    nrow, ncol = bsn_strip.shape
    for row in range(nrow):
        for col in range(ncol):
            basin = bsn_strip[row, col]
            if basin == 0:
                continue
            index = flagged_index[basin]
            if index < 0:
                continue
            code = l3_strip[row, col]
            if code != 0:
                exact[index, code] += 1


@njit(cache=True)
def _vote_level3_strip(bsn_strip, l3_strip, slot_code, slot_count, overflow, pixels_seen, rectangle_of_basin, row0,
                       grid_ncol, outside_rectangle):
    """for every basin, the pixels of each Level-03 code it holds, kept in eight slots per basin; a
    ninth code evicts the slot with the fewest pixels and is counted in overflow[basin]; such a basin
    is counted again exactly in a second pass (see fd1_4_group_whole_basins).  Every pixel of a
    basin is counted in pixels_seen, for the check against the table, and a basin id past the table
    stops the strip: its id is returned (0 when the strip is sound), before anything is indexed with it
    (Numba does not check bounds).  A pixel outside its basin's rectangle of the table
    (rectangle_of_basin[basin] = row_min, row_max, col_min, col_max, unrolled past the last column for a
    basin astride the antimeridian) is counted in outside_rectangle[0]"""
    nrow, ncol = bsn_strip.shape
    basin_limit = slot_code.shape[0]
    for row in range(nrow):
        for col in range(ncol):
            basin = bsn_strip[row, col]
            if basin == 0:
                continue
            if basin >= basin_limit:
                return np.int64(basin)
            pixels_seen[basin] += 1
            frame_col = col
            if rectangle_of_basin[basin, 3] > grid_ncol and col < rectangle_of_basin[basin, 2]:
                frame_col = col + grid_ncol          # a basin astride the seam, in its unrolled frame
            grid_row = row0 + row
            if grid_row < rectangle_of_basin[basin, 0] or grid_row >= rectangle_of_basin[basin, 1] or \
                    frame_col < rectangle_of_basin[basin, 2] or frame_col >= rectangle_of_basin[basin, 3]:
                outside_rectangle[0] += 1
            code = l3_strip[row, col]
            if code == 0:
                continue
            placed = False
            for slot in range(VOTE_SLOTS):
                if slot_code[basin, slot] == code:
                    slot_count[basin, slot] += 1
                    placed = True
                    break
                if slot_code[basin, slot] == 0:
                    slot_code[basin, slot] = code
                    slot_count[basin, slot] = 1
                    placed = True
                    break
            if not placed:
                overflow[basin] += 1
                weakest = 0
                for slot in range(1, VOTE_SLOTS):
                    if slot_count[basin, slot] < slot_count[basin, weakest]:
                        weakest = slot
                slot_code[basin, weakest] = code
                slot_count[basin, weakest] = 1
    return np.int64(0)


def _vote_winners(slot_code, slot_count):
    """the code with the most pixels per basin (ties: the smaller code), and its pixel count"""
    best = np.argmax(slot_count, axis=1)
    rows = np.arange(slot_code.shape[0])
    winner = slot_code[rows, best].astype(np.int64)
    winner_count = slot_count[rows, best]
    # ties broken by the smaller code
    for slot in range(VOTE_SLOTS):
        tie = (slot_count[:, slot] == winner_count) & (slot_code[:, slot] != 0) & (slot_code[:, slot] < winner)
        winner[tie] = slot_code[tie, slot]
    winner[winner_count == 0] = 0
    return winner, winner_count.astype(np.int64)


def _union_window(grid, rectangles):
    """the block window that holds all the rectangles (row_min, row_max, col_min, col_max), each already
    unrolled where it crosses the seam; on a periodic grid the union is taken in whichever of the two
    frames gives the narrower window (a unit on both sides of the antimeridian, like the Bering Strait,
    is unrolled rather than stretched across the whole grid)"""
    rectangles = np.asarray(rectangles, np.int64).reshape(-1, 4)
    row_min = int(rectangles[:, 0].min())
    row_max = int(rectangles[:, 1].max())
    col_min = int(rectangles[:, 2].min())
    col_max = int(rectangles[:, 3].max())
    if grid.periodic:
        # the rule: a rectangle not astride the seam is in the
        # western half when its centre is, and the western ones are moved past the seam when the span across the
        # seam, measured on the rectangles not astride it, is the shorter.  Moving a rectangle only when all of it
        # lies in the western half would give a group holding a basin across 0 degrees and one by 180 another
        # window (no MERIT Level-03 group is such a one)
        half = grid.ncol // 2
        not_astride = rectangles[:, 3] <= grid.ncol
        in_the_west = not_astride & ((rectangles[:, 2] + rectangles[:, 3] - 1) // 2 < half)
        in_the_east = not_astride & ~in_the_west
        if in_the_west.any() and in_the_east.any():
            span_as_it_is = int(rectangles[not_astride, 3].max()) - int(rectangles[not_astride, 2].min())
            span_across_the_seam = int(rectangles[in_the_west, 3].max()) + grid.ncol - int(rectangles[in_the_east, 2].min())
            if span_across_the_seam < span_as_it_is:
                moved_min = np.where(in_the_west, rectangles[:, 2] + grid.ncol, rectangles[:, 2])
                moved_max = np.where(in_the_west, rectangles[:, 3] + grid.ncol, rectangles[:, 3])
                col_min = int(moved_min.min())
                col_max = int(moved_max.max())
                if col_min >= grid.ncol:
                    col_min -= grid.ncol
                    col_max -= grid.ncol
    return grid.window_of_rectangle(row_min, row_max, col_min, col_max)


# ---- [C] a small doubtful basin takes the code of its longest shared boundary (the rule of the C code) ----
#
#  A basin is doubtful when it holds at most SMALL_BASIN_MAX_PIXELS pixels and either has no code or its
#  winning code covers less than half of them.  One pass over the basin mask counts, for every doubtful
#  basin, the pixels of shared boundary with each code around it (every pixel against its right
#  neighbour and the pixel above it); the longest wins, the smaller code breaks a tie.  A doubtful basin
#  whose neighbours are all doubtful too follows the largest of them until the run reaches a basin with
#  a code; what the run never reaches falls back on its own weak vote, and a basin that never had a
#  code is left to the rules after this one.

SMALL_BASIN_MAX_PIXELS = 10000     # a doubtful basin holds at most this many pixels
NEIGHBOUR_CODE_SLOTS = 8           # codes kept per doubtful basin; a fragment of coast touches one or two


@njit(cache=True)
def _offer_code_of_neighbour(slot_code, slot_pixels, overflow, slot, code):
    """one pixel of shared boundary between a doubtful basin (its slot) and a neighbour holding code"""
    for k in range(NEIGHBOUR_CODE_SLOTS):
        if slot_code[slot, k] == code:
            slot_pixels[slot, k] += 1
            return
    for k in range(NEIGHBOUR_CODE_SLOTS):
        if slot_code[slot, k] == 0:
            slot_code[slot, k] = code
            slot_pixels[slot, k] = 1
            return
    overflow[slot] = 1        # more codes around the basin than slots: reported, never guessed


@njit(cache=True)
def _offer_doubtful_neighbour(largest_id, largest_pixels, slot, other_id, other_pixels):
    """the link a run of fragments follows: the largest doubtful neighbour (ties: the smaller id)"""
    if other_pixels > largest_pixels[slot] or (other_pixels == largest_pixels[slot] and (largest_id[slot] == 0 or other_id < largest_id[slot])):
        largest_id[slot] = other_id
        largest_pixels[slot] = other_pixels


@njit(cache=True)
def _count_one_adjacent_pair(first, second, doubtful_slot, level3, land, slot_code, slot_pixels, overflow, largest_id, largest_pixels):
    """one adjacent pair of the mask, each side offered to the other"""
    if first == second or first == 0 or second == 0:
        return
    first_slot = doubtful_slot[first]
    second_slot = doubtful_slot[second]
    if first_slot >= 0:
        if second_slot < 0:
            if level3[second] > 0:
                _offer_code_of_neighbour(slot_code, slot_pixels, overflow, first_slot, level3[second])
        else:
            _offer_doubtful_neighbour(largest_id, largest_pixels, first_slot, second, land[second])
    if second_slot >= 0:
        if first_slot < 0:
            if level3[first] > 0:
                _offer_code_of_neighbour(slot_code, slot_pixels, overflow, second_slot, level3[first])
        else:
            _offer_doubtful_neighbour(largest_id, largest_pixels, second_slot, first, land[first])


@njit(cache=True)
def _count_adjacent_pairs_strip(bsn_strip, row_above, has_row_above, periodic, doubtful_slot, level3, land, slot_code, slot_pixels, overflow, largest_id, largest_pixels):
    """every pixel of the strip against its right neighbour and the pixel above it (the last column
    against the first on a periodic grid)"""
    nrow, ncol = bsn_strip.shape
    for row in range(nrow):
        for col in range(ncol):
            here = np.int64(bsn_strip[row, col])
            if here == 0:
                continue
            if col + 1 < ncol:
                _count_one_adjacent_pair(here, np.int64(bsn_strip[row, col + 1]), doubtful_slot, level3, land, slot_code, slot_pixels, overflow, largest_id, largest_pixels)
            elif periodic:
                _count_one_adjacent_pair(here, np.int64(bsn_strip[row, 0]), doubtful_slot, level3, land, slot_code, slot_pixels, overflow, largest_id, largest_pixels)
            if row > 0:
                _count_one_adjacent_pair(here, np.int64(bsn_strip[row - 1, col]), doubtful_slot, level3, land, slot_code, slot_pixels, overflow, largest_id, largest_pixels)
            elif has_row_above:
                _count_one_adjacent_pair(here, np.int64(row_above[col]), doubtful_slot, level3, land, slot_code, slot_pixels, overflow, largest_id, largest_pixels)


@njit(cache=True)
def _next_basin_along_the_run(doubtful_slot, largest_id, basin):
    """one step along a run: the largest doubtful neighbour of this basin, or 0 at the end of the run"""
    if basin <= 0:
        return np.int64(0)
    slot = doubtful_slot[basin]
    if slot < 0:
        return np.int64(0)
    return largest_id[slot]


@njit(cache=True)
def _follow_the_runs(doubtful_of_slot, doubtful_slot, level3, land, largest_id, own_vote, code_along_run):
    """A run of doubtful basins with no coded neighbour of its own follows its largest doubtful
    neighbour until it meets a basin that now has a code.  Two walkers, one twice as fast as the other,
    so a run of any length is followed to its end and a run that closes on itself is recognised: such a
    run takes the weak vote of the largest basin in the loop (ties: the smaller id), the same for every
    member of the run.  The walk reads the codes as they are and writes into code_along_run, so no basin
    picks up an answer another basin of the same run was given a moment earlier."""
    closed_runs = 0
    for slot in range(doubtful_of_slot.size):
        basin = doubtful_of_slot[slot]
        if level3[basin] > 0:
            continue
        slow = basin
        fast = basin
        run_is_open = True
        while run_is_open:
            for hop in range(2):
                if fast <= 0 or not run_is_open:
                    break
                fast = _next_basin_along_the_run(doubtful_slot, largest_id, fast)
                if fast <= 0:
                    break
                if level3[fast] > 0:
                    code_along_run[slot] = level3[fast]
                    run_is_open = False
            if not run_is_open:
                break
            slow = _next_basin_along_the_run(doubtful_slot, largest_id, slow)
            if slow <= 0:
                break                                    # the run ends without a code
            if level3[slow] > 0:
                code_along_run[slot] = level3[slow]
                break
            if fast > 0 and slow == fast:
                loop = slow
                largest_in_loop = np.int64(0)
                largest_in_loop_pixels = np.int64(0)
                while True:
                    if land[loop] > largest_in_loop_pixels or (land[loop] == largest_in_loop_pixels and (largest_in_loop == 0 or loop < largest_in_loop)):
                        largest_in_loop = loop
                        largest_in_loop_pixels = land[loop]
                    loop = _next_basin_along_the_run(doubtful_slot, largest_id, loop)
                    if loop <= 0 or loop == slow:
                        break
                if largest_in_loop > 0:
                    largest_slot = doubtful_slot[largest_in_loop]
                    if largest_slot >= 0 and own_vote[largest_slot] > 0:
                        code_along_run[slot] = own_vote[largest_slot]
                closed_runs += 1
                break
    return closed_runs


def _resolve_doubtful_basins_by_longest_shared_boundary(bsn_path, grid, level3, vote_pixels, land, astride, strip_rows, tag):
    """the rule [C] over the whole grid; level3 (by basin index) is changed in place, and the array
    from_neighbour says which basins took a code here.  Returns (from_neighbour, report)"""
    basin_count = level3.size
    # everything indexed by basin id here (the mask holds ids), so index 0 is unused
    level3_by_id = np.zeros(basin_count + 1, np.int64)
    level3_by_id[1:] = level3
    land_by_id = np.zeros(basin_count + 1, np.int64)
    land_by_id[1:] = land
    doubtful = (land <= SMALL_BASIN_MAX_PIXELS) & ((level3 == 0) | (vote_pixels * 2 < land)) & (astride == 0)
    doubtful_of_slot = np.nonzero(doubtful)[0].astype(np.int64) + 1
    doubtful_slot = np.full(basin_count + 1, -1, np.int64)
    doubtful_slot[doubtful_of_slot] = np.arange(doubtful_of_slot.size)
    slots = doubtful_of_slot.size
    log(tag, "neighbour rule: %d of %d basins are doubtful (at most %d pixels, no code or a vote under half)" % (slots, basin_count, SMALL_BASIN_MAX_PIXELS))
    from_neighbour = np.zeros(basin_count, np.int64)
    if slots == 0:
        return from_neighbour, {"doubtful": 0, "taken_from_a_coded_neighbour": 0, "taken_along_a_run": 0, "closed_runs": 0}
    slot_code = np.zeros((slots, NEIGHBOUR_CODE_SLOTS), np.int64)
    slot_pixels = np.zeros((slots, NEIGHBOUR_CODE_SLOTS), np.int64)
    overflow = np.zeros(slots, np.uint8)
    largest_id = np.zeros(slots, np.int64)
    largest_pixels = np.zeros(slots, np.int64)
    row_above = np.zeros(grid.ncol, np.uint32)
    with rasterio.open(bsn_path) as bsn_dataset:
        for row0 in range(0, grid.nrow, strip_rows):
            nrow = min(strip_rows, grid.nrow - row0)
            strip = bsn_dataset.read(1, window=Window(0, row0, grid.ncol, nrow))
            _count_adjacent_pairs_strip(strip, row_above, row0 > 0, grid.periodic, doubtful_slot, level3_by_id, land_by_id, slot_code, slot_pixels, overflow, largest_id, largest_pixels)
            row_above[:] = strip[-1]
    if overflow.any():
        raise FlowDivideError("%d doubtful basins touch more than %d Level-03 codes; the neighbour rule cannot name a winner without keeping them all" % (int(overflow.sum()), NEIGHBOUR_CODE_SLOTS))
    # the code with the most pixels of shared boundary, the smaller code when two are equally long; the
    # pixels are counted per code, not per neighbouring basin
    own_vote = level3_by_id[doubtful_of_slot].copy()
    best_slot = np.argmax(slot_pixels, axis=1)
    best_pixels = slot_pixels[np.arange(slots), best_slot]
    best_code = slot_code[np.arange(slots), best_slot]
    for k in range(NEIGHBOUR_CODE_SLOTS):
        tie = (slot_pixels[:, k] == best_pixels) & (slot_code[:, k] != 0) & (slot_code[:, k] < best_code)
        best_code[tie] = slot_code[tie, k]
    best_code[best_pixels == 0] = 0
    taken_from_a_coded_neighbour = int((best_code > 0).sum())
    level3_by_id[doubtful_of_slot[best_code > 0]] = best_code[best_code > 0]
    from_neighbour[doubtful_of_slot[best_code > 0] - 1] = 1
    # a basin with nothing around it keeps its own weak vote; one that only touches doubtful basins
    # waits for the run below
    waits = (best_code == 0) & (largest_id != 0)
    level3_by_id[doubtful_of_slot[waits]] = 0
    code_along_run = np.zeros(slots, np.int64)
    closed_runs = _follow_the_runs(doubtful_of_slot, doubtful_slot, level3_by_id, land_by_id, largest_id, own_vote, code_along_run)   # basins whose run closes on itself
    still_open = level3_by_id[doubtful_of_slot] == 0
    reached = still_open & (code_along_run > 0)
    level3_by_id[doubtful_of_slot[reached]] = code_along_run[reached]
    from_neighbour[doubtful_of_slot[reached] - 1] = 1
    fell_back = still_open & (code_along_run == 0)
    level3_by_id[doubtful_of_slot[fell_back]] = own_vote[fell_back]
    level3[:] = level3_by_id[1:]
    report = {"doubtful": int(slots), "taken_from_a_coded_neighbour": taken_from_a_coded_neighbour, "taken_along_a_run": int(reached.sum()), "basins_in_a_closed_run": int(closed_runs)}
    log(tag, "neighbour rule: %d basins took the code of their longest shared boundary, %d the code at the end of a run of fragments; %d basins are in a run that closes on itself; %d basins remain without a code" % (
        taken_from_a_coded_neighbour, report["taken_along_a_run"], closed_runs, int((level3 == 0).sum())))
    return from_neighbour, report


def group_windows_in_pixels(groups, grid, block=None):
    """the block windows of a group table as (row_min, row_max, col_min, col_max) in pixels, from the window_* columns
    (in blocks); col_max may run past the last column on a periodic grid"""
    block = block or fd_tables.group_block_pixels(groups)
    return np.column_stack([groups["window_row_min"].to_numpy(np.int64) * block, groups["window_row_max"].to_numpy(np.int64) * block,
                            groups["window_col_min"].to_numpy(np.int64) * block, groups["window_col_max"].to_numpy(np.int64) * block])


def _fill_group_window_columns(table, grid, block):
    """window_nrow .. window_maxlat of a group table whose window_* block columns are set
    fill_the_window_columns_of_a_group makes them"""
    table["window_nrow"] = (table["window_row_max"] - table["window_row_min"]) * block
    table["window_ncol"] = (table["window_col_max"] - table["window_col_min"]) * block
    table["window_grid_count"] = table["window_nrow"] * table["window_ncol"]
    table["fill_percent"] = np.where(table["window_grid_count"] > 0, 100.0 * table["land_grid_count"] / table["window_grid_count"], 0.0)
    boxes = [grid.pixel_box_lon_lat(int(r0) * block, int(r1) * block, int(c0) * block, int(c1) * block)
             for r0, r1, c0, c1 in zip(table["window_row_min"], table["window_row_max"], table["window_col_min"], table["window_col_max"])]
    table["window_minlon"] = [box[0] for box in boxes]
    table["window_minlat"] = [box[1] for box in boxes]
    table["window_maxlon"] = [box[2] for box in boxes]
    table["window_maxlat"] = [box[3] for box in boxes]
    return table[fd_tables.GROUP_TABLE_COLUMNS]


def group_table_rows(grid, windows, group_columns, level):
    """the group table (fd_tables.GROUP_TABLE_COLUMNS) from the windows in pixels and the columns of every group:
    group_id, group_kind, level_code, level3_count, basin_count, coded_basin_count, neighbour_basin_count,
    land_grid_count"""
    block = grid.block_pixels
    windows = np.asarray(windows, np.int64).reshape(-1, 4)
    table = pd.DataFrame(group_columns)
    table["group_level"] = level
    table["window_row_min"] = windows[:, 0] // block
    table["window_row_max"] = (windows[:, 1] - 1) // block + 1        # one past, in blocks
    table["window_col_min"] = windows[:, 2] // block
    table["window_col_max"] = (windows[:, 3] - 1) // block + 1
    return _fill_group_window_columns(table, grid, block)


def check_groups_against_basins(group_table, group_of_basin, land):
    """the checks every grouping must pass: every basin in a group the table lists, the land of the groups the land
    of the basins, every group holding at least one basin; group_of_basin and land by basin index"""
    if (group_of_basin == 0).any():
        raise FlowDivideError("%d basins landed in no group" % int((group_of_basin == 0).sum()))
    if not set(np.unique(group_of_basin).tolist()) <= set(group_table["group_id"].tolist()):
        raise FlowDivideError("a basin names a group the group table does not list")
    if group_table["group_id"].duplicated().any():
        raise FlowDivideError("two groups share an id")
    if (group_table["basin_count"] < 1).any():
        raise FlowDivideError("a group holds no basin")
    if int(group_table["land_grid_count"].sum()) != int(land.sum()):
        raise FlowDivideError("the land pixels of the groups do not add up to the land pixels of the basins")
    counted = pd.DataFrame({"group_id": group_of_basin, "land": land}).groupby("group_id")["land"].agg(["count", "sum"])
    for row in group_table.itertuples(index=False):
        if int(counted.loc[row.group_id, "count"]) != int(row.basin_count) or int(counted.loc[row.group_id, "sum"]) != int(row.land_grid_count):
            raise FlowDivideError("group %d lists %d basins and %d pixels, but its basins make %d and %d"
                                  % (row.group_id, row.basin_count, row.land_grid_count, counted.loc[row.group_id, "count"], counted.loc[row.group_id, "sum"]))


def level_code_of_level3_code(level3_code, level):
    """the Level-02 (level 2) or Level-01 (level 1) code of a Level-03 code; 0 for a group without a code"""
    level3_code = np.asarray(level3_code, np.int64)
    return np.where(level3_code <= 0, 0, level3_code // (10 if level == 2 else 100))


def fold_level3_groups(level3_groups, level, grid):
    """the Level-03 groups folded into the groups of `level`: one group per code of
    that level, in code order, its window the smallest block window around its Level-03 windows (on a periodic grid
    the narrower of that and the one with the western half moved one width east), its counts summed, kind 2; then
    every group without a code as it is, with the level changed"""
    block = grid.block_pixels
    codes = level_code_of_level3_code(level3_groups["level_code"].to_numpy(np.int64), level)
    rows = []
    periodic_width_blocks = grid.ncol // block if grid.periodic else 0
    for code in sorted(set(int(c) for c in codes if c > 0)):
        if code >= 100:
            raise FlowDivideError("the Level-%02d code %d is out of range" % (level, code))
        members = level3_groups[codes == code]
        row = {"group_id": code, "group_level": level, "group_kind": fd_tables.GROUP_KIND_MANY, "level_code": code,
               "level3_count": len(members), "basin_count": int(members["basin_count"].sum()),
               "coded_basin_count": int(members["coded_basin_count"].sum()),
               "neighbour_basin_count": int(members["neighbour_basin_count"].sum()),
               "land_grid_count": int(members["land_grid_count"].sum()),
               "window_row_min": int(members["window_row_min"].min()), "window_row_max": int(members["window_row_max"].max()),
               "window_col_min": int(members["window_col_min"].min()), "window_col_max": int(members["window_col_max"].max())}
        if periodic_width_blocks > 0:
            shift = np.where(members["window_col_max"].to_numpy(np.int64) <= periodic_width_blocks // 2, periodic_width_blocks, 0)
            moved_min = int((members["window_col_min"].to_numpy(np.int64) + shift).min())
            moved_max = int((members["window_col_max"].to_numpy(np.int64) + shift).max())
            if moved_max - moved_min < row["window_col_max"] - row["window_col_min"]:
                row["window_col_min"] = moved_min
                row["window_col_max"] = moved_max
        rows.append(row)
    folded = pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else None)
    uncoded = level3_groups[level3_groups["level_code"] <= 0].copy()
    uncoded["group_level"] = level
    folded = pd.concat([folded, uncoded[folded.columns if len(folded.columns) else uncoded.columns]], ignore_index=True)
    return _fill_group_window_columns(folded, grid, block)


def write_level3_groupings(grid, group_windows, group_columns, basins, group_of_basin, vote_columns, basin_table_path, tag):
    """fd1.4 of the Level-03 units: the vote table, the Level-03 groups and the
    Level-02 and Level-01 groups folded from them, and the basin table again with every basin's three group ids (its
    places and its region set back to unset: fd1.6's).  The basin table stops counting as done before the first group
    table is replaced and is written last."""
    root, run = fd_tables.root_and_run_of_basin_table(basin_table_path)
    land = basins["basin_grid_count"].to_numpy(np.int64)
    level3 = group_table_rows(grid, group_windows, group_columns, 3)
    check_groups_against_basins(level3, group_of_basin, land)
    level2 = fold_level3_groups(level3, 2, grid)
    level1 = fold_level3_groups(level3, 1, grid)
    # the vote table: the evidence of the Level-03 code of every basin
    vote_path = os.path.join(root, "global", "table", "level3_vote_fine_%s.csv" % run)
    vote = pd.DataFrame(vote_columns)[["basin_id", "level3_code", "level3_from_neighbour", "level3_vote_grid_count", "level3_coded_grid_count"]]
    temporary = "%s.tmp.%d" % (vote_path, os.getpid())
    vote.to_csv(temporary, sep=" ", index=False, lineterminator="\n")
    fd_tables.publish_with_marker(temporary, vote_path, "%d basins, %d of them given their code by the neighbour rule"
                                  % (len(vote), int((vote["level3_from_neighbour"] != 0).sum())))
    # the three group ids of every basin; a group without a code keeps its id at every level
    level3_code_of_group = dict(zip(level3["group_id"].tolist(), level3["level_code"].tolist()))
    code3 = np.array([level3_code_of_group[g] for g in group_of_basin], np.int64)
    table = basins.copy()
    table["level3_id"] = group_of_basin
    table["level2_id"] = np.where(code3 > 0, code3 // 10, group_of_basin)
    table["level1_id"] = np.where(code3 > 0, code3 // 100, group_of_basin)
    for column in ("basin_id_in_level1", "basin_id_in_level2", "basin_id_in_level3", "region_id", "basin_id_in_region"):
        table[column] = fd_tables.UNSET_ID
    for level, groups in ((2, level2), (1, level1)):
        check_groups_against_basins(groups, table["level%d_id" % level].to_numpy(np.int64), land)
    if os.path.exists(basin_table_path + ".done"):
        os.remove(basin_table_path + ".done")
    summary = ("%d Level-03 groups on whole blocks, %d Level-02 and %d Level-01 groups folded from them; every basin in "
               "exactly one group of each level" % (len(level3), len(level2), len(level1)))
    for grouping, groups in (("l3", level3), ("l2", level2), ("l1", level1)):
        fd_tables.write_group_table(groups, fd_tables.group_table_path(root, run, grouping), summary)
    fd_tables.write_basin_table(table, basin_table_path, "fd1.4", summary)
    log(tag, "written the group tables l3, l2, l1 (%d, %d, %d groups) and the basin table" % (len(level3), len(level2), len(level1)))
    return level3


def fd1_6_basin_table(basin_table_path, basin_region_path, region_table_path, piece_table_path, region_map_key, tag="fd1.6"):
    """The last columns of the basin table, in place:

      region_id            the region of the partition whose windows fit 2^31 pixels (160 square degrees on the 30 m
                           grid, 1400 on MERIT) that holds the basin: the value of the basin-to-region map, or, for a
                           basin the capacity cut, the region of its outlet piece
      basin_id_in_level1   1 .. n within the basin's Level-01, Level-02, Level-03 group and region, in basin_id order,
      basin_id_in_level2   so also largest first; a cut basin counts in the region of its outlet
      basin_id_in_level3
      basin_id_in_region

    region_map_key: fd_tables.region_map_key of the partition, which the map must carry."""
    started = time.time()
    basins = fd_tables.read_basin_table(basin_table_path)
    basin_count = len(basins)
    root, run = fd_tables.root_and_run_of_basin_table(basin_table_path)
    for column in ("level1_id", "level2_id", "level3_id"):
        if (basins[column] == fd_tables.UNSET_ID).any():
            raise FlowDivideError("%s has basins without a %s: run fd1.4 first" % (basin_table_path, column))
    region = fd_tables.region_of_basin(basin_region_path, piece_table_path, run, region_map_key, basin_count)[1:].astype(np.int64)
    known_regions = set(fd_tables.read_region_table(region_table_path)["region_id"].astype(np.int64).tolist())
    unknown = sorted(set(np.unique(region).tolist()) - known_regions)
    if unknown:
        raise FlowDivideError("basins are given the regions %s, which %s does not list" % (unknown[:5], region_table_path))
    table = basins.copy()
    table["region_id"] = region
    # the place of every basin in its unit: the rows are in basin_id order, so a running count within the unit
    for unit_column, place_column in (("level1_id", "basin_id_in_level1"), ("level2_id", "basin_id_in_level2"),
                                      ("level3_id", "basin_id_in_level3"), ("region_id", "basin_id_in_region")):
        table[place_column] = table.groupby(unit_column, sort=False).cumcount().to_numpy(np.int64) + 1
    fd_tables.write_basin_table(table, basin_table_path, "fd1.6",
                                "places in the Level-01/02/03 groups and in the regions of %s" % region_map_key)
    report = {"basins": basin_count, "regions": int(np.unique(region).size), "seconds": round(time.time() - started, 1)}
    log(tag, "%s now carries the places of every basin and region_id: %d basins in %d regions"
        % (basin_table_path, basin_count, report["regions"]))
    return report


def fd1_4_group_whole_basins(bsn_path, l3_path, basin_table_path, grid,
                             island_reach_blocks, island_max_blocks, level1=None, strip_rows=None, tag="fd1.4"):
    """The basin groups along the Level-03 units.  [A] every basin takes the
    code covering most of its pixels; [C] a small doubtful basin takes the code of its longest shared
    boundary; [C2] a basin still without a code takes the code of the coded basin whose outlet is
    nearest, within one block; [B] one group per code, its window the union of its basins' rectangles
    rounded out to blocks; [D] the rest are islands, gathered around seeds from the largest down; on a
    periodic grid the basins across the antimeridian are gathered among themselves the same way.
    Writes the vote table level3_vote_fine, the group tables group_l3/l2/l1_fine and
    the basin table again with the three group ids of every basin (write_level3_groupings).  group_kind 1: one basin voted for the code (a great river that is its own
    unit); 2: several basins; 3: an island group; 4: a group of basins across the antimeridian.  level1:
    the level-1 number of the island and seam groups (200000 + level1 * 1000 + k, 100000 + ...); the
    smallest code's first digit when None (the whole MERIT grid takes 0)."""
    from scipy.spatial import cKDTree
    started = time.time()
    if strip_rows is None:
        strip_rows = strip_rows_for(grid.ncol)
    basins = fd_tables.read_basin_table(basin_table_path)
    basin_count = len(basins)
    basin_id = basins["basin_id"].to_numpy(np.int64)
    if not np.array_equal(basin_id, np.arange(1, basin_count + 1)):
        raise FlowDivideError("the basin table must list the basins 1 .. N in order")
    # [A] the vote
    slot_code = np.zeros((basin_count + 1, VOTE_SLOTS), np.uint16)
    slot_count = np.zeros((basin_count + 1, VOTE_SLOTS), np.int64)
    overflow = np.zeros(basin_count + 1, np.int64)
    pixels_seen = np.zeros(basin_count + 1, np.int64)
    rectangle_of_basin = np.zeros((basin_count + 1, 4), np.int32)     # 16 B a basin; rows and unrolled columns fit int32
    rectangle_of_basin[1:] = basin_rectangles(basins)
    outside_rectangle = np.zeros(1, np.int64)
    with rasterio.open(bsn_path) as bsn_dataset, rasterio.open(l3_path) as l3_dataset:
        # the two rasters on the grid of the run, checked before they are read: the same size, the
        # geotransforms within 1e-12 term by term, the same CRS; and the basin mask unsigned 32-bit, so that no id
        # read from it is negative
        for name, dataset in (("the basin mask", bsn_dataset), ("the Level-03 raster", l3_dataset)):
            same_transform = all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(tuple(dataset.transform)[:6], tuple(grid.transform)[:6]))
            if dataset.width != grid.ncol or dataset.height != grid.nrow or not same_transform or dataset.crs != grid.crs:
                raise FlowDivideError("%s is not on the grid of the run (%d x %d, %s)" % (name, grid.ncol, grid.nrow, grid.transform))
        if bsn_dataset.dtypes[0] != "uint32":
            raise FlowDivideError("the basin mask %s is %s, not uint32" % (bsn_path, bsn_dataset.dtypes[0]))
        # the Level-03 codes fit 16 bits, as fd1.4's raster writes them: the vote keeps them in uint16 slots and the
        # exact recount indexes a row of 65536 with them (a code of 65536 would write past it)
        if l3_dataset.dtypes[0] not in ("uint8", "uint16"):
            raise FlowDivideError("the Level-03 raster %s is %s, not an unsigned raster of at most 16 bits" % (l3_path, l3_dataset.dtypes[0]))
        for row0 in range(0, grid.nrow, strip_rows):
            nrow = min(strip_rows, grid.nrow - row0)
            window = Window(0, row0, grid.ncol, nrow)
            past_the_table = _vote_level3_strip(bsn_dataset.read(1, window=window), l3_dataset.read(1, window=window), slot_code, slot_count,
                                                overflow, pixels_seen, rectangle_of_basin, row0, grid.ncol, outside_rectangle)
            if past_the_table:
                raise FlowDivideError("the basin mask holds basin %d in rows %d .. %d, past the %d basins of the table"
                                      % (int(past_the_table), row0, row0 + nrow - 1, basin_count))
    if outside_rectangle[0]:
        raise FlowDivideError("%d pixels of the basin mask lie outside their basin's rectangle of the table" % int(outside_rectangle[0]))
    del rectangle_of_basin
    # the pixels of every basin in the mask are its table row's
    table_pixels = basins["basin_grid_count"].to_numpy(np.int64)
    disagreeing = np.nonzero(pixels_seen[1:] != table_pixels)[0]
    if disagreeing.size:
        first = int(disagreeing[0])
        raise FlowDivideError("%d basins hold another number of pixels in the mask than in the table; basin %d: %d in the mask, %d in the table"
                              % (disagreeing.size, first + 1, int(pixels_seen[first + 1]), int(table_pixels[first])))
    del pixels_seen
    winner, winner_count = _vote_winners(slot_code, slot_count)
    coded_pixels = slot_count.sum(axis=1)
    del slot_code, slot_count
    flagged = np.nonzero(overflow > 0)[0]
    if flagged.size:
        # the basins that touch more than eight units are counted again, exactly, in a second pass
        log(tag, "%d basins touch more than %d Level-03 units; a second pass counts their codes exactly" % (flagged.size, VOTE_SLOTS))
        flagged_index = np.full(basin_count + 1, -1, np.int64)
        flagged_index[flagged] = np.arange(flagged.size)
        exact = np.zeros((flagged.size, 65536), np.int64)
        with rasterio.open(bsn_path) as bsn_dataset, rasterio.open(l3_path) as l3_dataset:
            for row0 in range(0, grid.nrow, strip_rows):
                nrow = min(strip_rows, grid.nrow - row0)
                window = Window(0, row0, grid.ncol, nrow)
                _vote_level3_exact_strip(bsn_dataset.read(1, window=window), l3_dataset.read(1, window=window), flagged_index, exact)
        for k, basin in enumerate(flagged):
            best = int(np.argmax(exact[k]))          # the smallest code among equals, argmax takes the first
            winner[basin] = best
            winner_count[basin] = int(exact[k, best])
            coded_pixels[basin] = int(exact[k].sum())
        del exact
    level3 = winner[basin_id]
    vote_pixels = winner_count[basin_id]
    coded_pixels = coded_pixels[basin_id]
    rectangles = basin_rectangles(basins)
    land = basins["basin_grid_count"].to_numpy(np.int64)
    area = basins["basin_area_km2"].to_numpy(np.float64)
    astride = (rectangles[:, 3] > grid.ncol).astype(np.int64)       # across the antimeridian: an unrolled rectangle (col_max past the width)
    level3[astride == 1] = 0                     # a basin across the seam takes no code
    coded_by_vote = level3 > 0
    log(tag, "vote: %d of %d basins hold a Level-03 code, %d have none" % (int(coded_by_vote.sum()), basin_count, int((~coded_by_vote).sum())))
    # [C] the longest shared boundary
    from_neighbour, neighbour_report = _resolve_doubtful_basins_by_longest_shared_boundary(bsn_path, grid, level3, vote_pixels, land, astride, strip_rows, tag)
    # [C2] the nearest coded outlet within one block for the basins still without a code
    outlet_row = basins["outlet_row"].to_numpy(np.int64)
    outlet_col = basins["outlet_col"].to_numpy(np.int64)
    coded = np.nonzero(level3 > 0)[0]
    without = np.nonzero((level3 == 0) & (astride == 0))[0]
    taken_from_the_nearest = 0
    if without.size and coded.size:
        points_row = outlet_row[coded]
        points_col = outlet_col[coded]
        source = coded
        if grid.periodic:
            # the coded outlets within a block of either edge are seen from the other side as well
            near_left = coded[outlet_col[coded] < grid.block_pixels]
            near_right = coded[outlet_col[coded] >= grid.ncol - grid.block_pixels]
            points_row = np.concatenate([points_row, outlet_row[near_left], outlet_row[near_right]])
            points_col = np.concatenate([points_col, outlet_col[near_left] + grid.ncol, outlet_col[near_right] - grid.ncol])
            source = np.concatenate([coded, near_left, near_right])
        tree = cKDTree(np.column_stack([points_row, points_col]).astype(np.float64))
        queries = np.column_stack([outlet_row[without], outlet_col[without]]).astype(np.float64)
        limit_squared = grid.block_pixels * grid.block_pixels
        chosen_source = np.full(without.size, -1, np.int64)
        unresolved = np.arange(without.size)
        neighbours = 8
        while unresolved.size:
            # the nearest by the integer squared distance, ties to the smaller basin id, within the block
            # inclusive; more neighbours are asked for where the last one is as near as the first
            distance, nearest = tree.query(queries[unresolved], k=neighbours, distance_upper_bound=grid.block_pixels + 1.0)
            found = np.isfinite(distance)
            nearest = np.where(found, nearest, 0)
            gap_row = points_row[nearest] - queries[unresolved][:, [0]]
            gap_col = points_col[nearest] - queries[unresolved][:, [1]]
            squared = np.where(found, (gap_row * gap_row + gap_col * gap_col).astype(np.int64), np.iinfo(np.int64).max)
            squared = np.where(squared <= limit_squared, squared, np.iinfo(np.int64).max)
            ids = np.where(found, source[nearest] + 1, np.iinfo(np.int64).max)
            order = np.lexsort((ids, squared))                           # per query (the last axis): by squared distance, then by basin id
            best = order[:, 0]
            best_squared = squared[np.arange(unresolved.size), best]
            resolved = best_squared < np.iinfo(np.int64).max
            # settled when the farthest neighbour asked for is farther than the best (on the exact squared
            # distances), or fewer were found than asked for, or every outlet was asked for
            exact_squared = np.where(found, (gap_row * gap_row + gap_col * gap_col).astype(np.int64), -1)
            farthest_squared = exact_squared.max(axis=1)
            settled = (~found.all(axis=1)) | (farthest_squared > best_squared) | (neighbours >= source.size)
            chosen_source[unresolved[settled & resolved]] = source[nearest[np.arange(unresolved.size), best]][settled & resolved]
            chosen_source[unresolved[settled & ~resolved]] = -2
            unresolved = unresolved[~settled]
            neighbours = min(neighbours * 4, max(source.size, 1))
        within = chosen_source >= 0
        level3[without[within]] = level3[chosen_source[within]]
        from_neighbour[without[within]] = 1
        taken_from_the_nearest = int(within.sum())
        log(tag, "%d basins without a code took the code of the nearest coded outlet within one block; %d remain" % (taken_from_the_nearest, int((~within).sum())))
    # [B] one group per code, its window the union of its basins' rectangles rounded out to blocks; the
    #     kind is decided by the basins that voted, not by those that took a code from a neighbour
    group_of_basin = np.zeros(basin_count, np.int64)
    group_columns = []
    group_windows = []
    for code in sorted(set(level3[level3 > 0].tolist())):
        members = np.nonzero(level3 == code)[0]
        voted = members[from_neighbour[members] == 0]
        group_columns.append({"group_id": int(code), "group_kind": 1 if voted.size == 1 else 2, "level_code": int(code), "level3_count": 1, "basin_count": int(members.size),
                              "coded_basin_count": int(voted.size), "neighbour_basin_count": int(members.size - voted.size), "land_grid_count": int(land[members].sum())})
        group_windows.append(_union_window(grid, rectangles[members]))
        group_of_basin[members] = code
    if level1 is None:
        level1 = min(code // 100 for code in set(level3[level3 > 0].tolist())) if (level3 > 0).any() else 0
    # [D] the islands, and the basins across the seam, gathered around seeds from the largest down
    def gather_around_seeds(candidates, kind, first_id):
        """the seeds from the largest basin down; the block cells around a seed are visited ring by ring
        out to the reach (wrapping round on a periodic grid), and a free basin joins while the group's
        window stays under the island limit.  The window grows incrementally: a basin is compared with
        the running rectangle, never with all the members again."""
        free = np.zeros(basin_count, np.uint8)
        free[candidates] = 1
        block = grid.block_pixels
        cells_across = -(-grid.ncol // block)
        # a basin is indexed by the block cell of the centre of its rectangle; the basins of a cell are
        # visited in the reverse order of their ids
        cell_row = ((rectangles[:, 0] + rectangles[:, 1] - 1) // 2 // block).astype(np.int64)     # the centre pixel: row_max is one past
        cell_col = ((rectangles[:, 2] + rectangles[:, 3] - 1) // 2 // block).astype(np.int64)
        if grid.periodic:
            cell_col %= cells_across
        by_cell = {}
        for index in candidates[::-1]:
            by_cell.setdefault((int(cell_row[index]), int(cell_col[index])), []).append(int(index))
        seeds = candidates[np.lexsort((candidates, -land[candidates], -area[candidates]))]      # the largest first: area, then count, then id
        next_id = first_id
        limit = island_max_blocks * block * block - 1

        def window_pixels(rect):
            window = grid.window_of_rectangle(*rect)
            return (window[1] - window[0]) * (window[3] - window[2])

        made = 0
        for seed in seeds:
            if free[seed] == 0:
                continue
            free[seed] = 0
            members = [int(seed)]
            running = tuple(int(v) for v in rectangles[seed])
            if window_pixels(running) > limit:
                raise FlowDivideError("island seed basin %d has %d pixels in its block window; an island group must be strictly smaller than %d blocks"
                                      % (seed + 1, window_pixels(running), island_max_blocks))
            for ring in range(0, island_reach_blocks + 1):
                for d_row in range(-ring, ring + 1):
                    for d_col in range(-ring, ring + 1):
                        if max(abs(d_row), abs(d_col)) != ring:
                            continue
                        key_col = int(cell_col[seed]) + d_col
                        if grid.periodic:
                            key_col %= cells_across
                        for index in by_cell.get((int(cell_row[seed]) + d_row, key_col), []):
                            if free[index] == 0:
                                continue
                            # the rectangles are joined as they are, unrolled where a basin lies across the seam and
                            # not otherwise: two islands on either side of the seam make a
                            # window across the whole width and are refused here
                            candidate_rect = tuple(int(v) for v in rectangles[index])
                            trial = _rectangle_union(running, candidate_rect)
                            if window_pixels(trial) > limit:
                                continue
                            members.append(index)
                            free[index] = 0
                            running = trial
            window = grid.window_of_rectangle(*running)
            if window[2] >= grid.ncol:
                window = (window[0], window[1], window[2] - grid.ncol, window[3] - grid.ncol)
            members = np.asarray(members, np.int64)
            group_columns.append({"group_id": next_id, "group_kind": kind, "level_code": 0, "level3_count": 0, "basin_count": int(members.size), "coded_basin_count": 0,
                                  "neighbour_basin_count": 0, "land_grid_count": int(land[members].sum())})
            group_windows.append(window)
            group_of_basin[members] = next_id
            next_id += 1
            made += 1
            if next_id - first_id > 999:
                raise FlowDivideError("more than 999 island or seam groups; the numbering holds 999 per level-1 code")
        return made

    seam_groups = gather_around_seeds(np.nonzero(astride == 1)[0], 4, 100000 + level1 * 1000 + 1)
    island_groups = gather_around_seeds(np.nonzero((level3 == 0) & (astride == 0))[0], 3, 200000 + level1 * 1000 + 1)
    vote_columns = {"basin_id": basin_id, "level3_code": level3, "level3_from_neighbour": from_neighbour,
                    "level3_vote_grid_count": vote_pixels, "level3_coded_grid_count": coded_pixels}
    group_table = write_level3_groupings(grid, group_windows, group_columns, basins, group_of_basin, vote_columns, basin_table_path, tag)
    group_table_path = fd_tables.group_table_path(*fd_tables.root_and_run_of_basin_table(basin_table_path), "l3")
    report = {"groups": int(len(group_table)), "level3_groups": int((group_table["group_kind"] <= 2).sum()), "island_groups": island_groups, "seam_groups": seam_groups,
              "basins_from_neighbour": int(from_neighbour.sum()), "neighbour_rule": neighbour_report, "taken_from_the_nearest_outlet": taken_from_the_nearest,
              "seconds": round(time.time() - started, 1)}
    write_json(group_table_path + ".report.json", report)
    log(tag, "written %s: %d groups (%d Level-03, %d island, %d across the seam)" % (group_table_path, report["groups"], report["level3_groups"], island_groups, seam_groups))
    return report


# =============================================================================
#  [6b] FD1.4  the automatic groups: whole basins along a Hilbert curve (the appendix of the paper)
# =============================================================================
#
#  The groups of FD1.4 made from the basins alone, without the Level-03 units
#  them.  Everything works on the block cells (one degree) of the grid:
#    [0] where a basin's land is: the coarse basin view (400 cells to the degree) is read one block
#        row at a time and the land of every basin on every block cell is counted; a basin's cell is the
#        one holding most of its land, or the cell of its outlet when it is too small to show
#    [A] a basin is a group of its own when its own block window passes the capacity, or when it
#        drains at least MAJOR_RIVER_MIN_AREA_KM2 (the threshold read off the manual Level 03)
#    [B] the other basins are visited along a Hilbert curve over the block cells and cut into runs: a
#        run closes when the next cell would push its window past the capacity, or when that cell
#        does not touch the run (eight neighbours), because the curve orders the sea as well
#    [C] two groups are joined again, the cheapest pair first (the fewest extra pixels of window over
#        the two apart), while a join fits: either the two touch on the cells, or one lies inside the
#        other's window with nothing else around it; a group of [A] is never joined away
#    [D] block cells are moved from one group to a group beside them when the two windows together
#        get smaller (a batch move of everything inside the other's window, or one cell), the group
#        left behind staying in one piece; where the windows do not change, a cell goes to the group
#        it has more neighbours in
#    [C] and [D] run turn about, up to twelve rounds, until a round changes nothing; the groups are
#    numbered from 1 in the order they were made, kind 1 for one basin and 2 for several.

MAJOR_RIVER_MIN_AREA_KM2 = 250000.0    # [A]: a basin draining at least this much is a group of its own
HILBERT_CORRECTION_ROUNDS = 12         # [C] and [D] turn about, at most this many times


@njit(cache=True)
def _hilbert_index_of_cell(curve_order, degree_row, degree_col):
    """the place of one cell on a Hilbert curve of side 2^curve_order, by the usual halving"""
    curve_side = np.int64(1) << np.int64(curve_order)
    index = np.int64(0)
    folded_row = np.int64(degree_row)
    folded_col = np.int64(degree_col)
    side = curve_side // 2
    while side > 0:
        row_bit = 1 if (folded_row & side) > 0 else 0
        col_bit = 1 if (folded_col & side) > 0 else 0
        index += side * side * ((3 * col_bit) ^ row_bit)
        if row_bit == 0:
            if col_bit == 1:
                folded_row = curve_side - 1 - folded_row
                folded_col = curve_side - 1 - folded_col
            swapped = folded_row
            folded_row = folded_col
            folded_col = swapped
        side //= 2
    return index


@njit(cache=True)
def _hilbert_indices(curve_order, cell_row, cell_col):
    out = np.empty(cell_row.size, np.int64)
    for k in range(cell_row.size):
        out[k] = _hilbert_index_of_cell(curve_order, cell_row[k], cell_col[k])
    return out


@njit(cache=True)
def _window_blocks(rmin, rmax, cmin, cmax):
    if rmax <= rmin or cmax <= cmin:
        return np.int64(0)
    return np.int64(rmax - rmin) * np.int64(cmax - cmin)


@njit(cache=True)
def _land_per_cell_of_basins(entry_cell, entry_basin, entry_land, basin_count):
    """[0]: the cell holding most of each basin's land, -1 for a basin that shows nowhere"""
    cell_of_basin = np.full(basin_count, -1, np.int64)
    most_land = np.zeros(basin_count, np.int64)
    for k in range(entry_cell.size):
        basin = entry_basin[k]
        if entry_land[k] > most_land[basin]:
            most_land[basin] = entry_land[k]
            cell_of_basin[basin] = entry_cell[k]
    return cell_of_basin


@njit(cache=True)
def _touches_run(cell_in_run, deg_nrow, deg_ncol, degree_row, degree_col):
    for row_offset in range(-1, 2):
        for col_offset in range(-1, 2):
            n_row = degree_row + row_offset
            n_col = degree_col + col_offset
            if n_row < 0 or n_row >= deg_nrow or n_col < 0 or n_col >= deg_ncol:
                continue
            if cell_in_run[n_row * deg_ncol + n_col] != 0:
                return True
    return False


@njit(cache=True)
def _runs_along_the_curve(order, hilbert, cell_of_basin, rect, land, capacity_blocks, deg_nrow, deg_ncol, group_of_basin,
                          g_rect, g_count, g_land, g_kind, g_over, group_count):
    """[B]: the basins in curve order, cut into runs.  The basins of one cell go into one group together,
    unless they reach too far apart to share one; then the cell is added a basin at a time.  Returns
    (group_count, runs closed at a gap, cells that had to be split)."""
    cell_in_run = np.zeros(deg_nrow * deg_ncol, np.uint8)
    run_cells = np.empty(deg_nrow * deg_ncol, np.int64)
    run_cell_count = 0
    open_group = -1
    runs_closed_at_a_gap = 0
    cells_split = 0
    position = 0
    free_count = order.size
    while position < free_count:
        here = hilbert[order[position]]
        cell_end = position
        c_rmin = np.int64(1 << 60)
        c_rmax = np.int64(-1)
        c_cmin = np.int64(1 << 60)
        c_cmax = np.int64(-1)
        while cell_end < free_count and hilbert[order[cell_end]] == here:
            b = order[cell_end]
            c_rmin = min(c_rmin, rect[b, 0])
            c_rmax = max(c_rmax, rect[b, 1])
            c_cmin = min(c_cmin, rect[b, 2])
            c_cmax = max(c_cmax, rect[b, 3])
            cell_end += 1
        degree_row = cell_of_basin[order[position]] // deg_ncol
        degree_col = cell_of_basin[order[position]] % deg_ncol
        if _window_blocks(c_rmin, c_rmax, c_cmin, c_cmax) > capacity_blocks:
            # two basins can share a cell and still reach far apart: the cell is added a basin at a time
            cells_split += 1
            for member in range(position, cell_end):
                b = order[member]
                opens = open_group < 0
                if not opens:
                    if _window_blocks(min(g_rect[open_group, 0], rect[b, 0]), max(g_rect[open_group, 1], rect[b, 1]),
                                      min(g_rect[open_group, 2], rect[b, 2]), max(g_rect[open_group, 3], rect[b, 3])) > capacity_blocks:
                        opens = True
                    elif not _touches_run(cell_in_run, deg_nrow, deg_ncol, degree_row, degree_col):
                        opens = True
                        runs_closed_at_a_gap += 1
                if opens:
                    open_group = group_count
                    group_count += 1
                    g_rect[open_group, 0] = 1 << 60
                    g_rect[open_group, 1] = -1
                    g_rect[open_group, 2] = 1 << 60
                    g_rect[open_group, 3] = -1
                    g_kind[open_group] = 2
                    for k in range(run_cell_count):
                        cell_in_run[run_cells[k]] = 0
                    run_cell_count = 0
                g_rect[open_group, 0] = min(g_rect[open_group, 0], rect[b, 0])
                g_rect[open_group, 1] = max(g_rect[open_group, 1], rect[b, 1])
                g_rect[open_group, 2] = min(g_rect[open_group, 2], rect[b, 2])
                g_rect[open_group, 3] = max(g_rect[open_group, 3], rect[b, 3])
                group_of_basin[b] = open_group
                g_count[open_group] += 1
                g_land[open_group] += land[b]
                cell = degree_row * deg_ncol + degree_col
                if cell_in_run[cell] == 0:
                    cell_in_run[cell] = 1
                    run_cells[run_cell_count] = cell
                    run_cell_count += 1
            position = cell_end
            continue
        opens = open_group < 0
        if not opens:
            if _window_blocks(min(g_rect[open_group, 0], c_rmin), max(g_rect[open_group, 1], c_rmax),
                              min(g_rect[open_group, 2], c_cmin), max(g_rect[open_group, 3], c_cmax)) > capacity_blocks:
                opens = True
            elif not _touches_run(cell_in_run, deg_nrow, deg_ncol, degree_row, degree_col):
                # the curve has stepped over water or empty land to a cell the run does not reach
                opens = True
                runs_closed_at_a_gap += 1
        if opens:
            open_group = group_count
            group_count += 1
            g_rect[open_group, 0] = 1 << 60
            g_rect[open_group, 1] = -1
            g_rect[open_group, 2] = 1 << 60
            g_rect[open_group, 3] = -1
            g_kind[open_group] = 2
            for k in range(run_cell_count):
                cell_in_run[run_cells[k]] = 0
            run_cell_count = 0
        for member in range(position, cell_end):
            b = order[member]
            g_rect[open_group, 0] = min(g_rect[open_group, 0], rect[b, 0])
            g_rect[open_group, 1] = max(g_rect[open_group, 1], rect[b, 1])
            g_rect[open_group, 2] = min(g_rect[open_group, 2], rect[b, 2])
            g_rect[open_group, 3] = max(g_rect[open_group, 3], rect[b, 3])
            group_of_basin[b] = open_group
            g_count[open_group] += 1
            g_land[open_group] += land[b]
        cell = degree_row * deg_ncol + degree_col
        if cell_in_run[cell] == 0:
            cell_in_run[cell] = 1
            run_cells[run_cell_count] = cell
            run_cell_count += 1
        position = cell_end
    return group_count, runs_closed_at_a_gap, cells_split


@njit(cache=True)
def _group_adjacency(entry_cell, entry_basin, entry_land, group_of_basin, cell_of_basin, group_count, deg_nrow, deg_ncol):
    """which groups touch, on the cells themselves by the eight-neighbour test, and the group holding
    most of each cell (owner_of_cell, -1 for a cell with no land)"""
    cell_count = deg_nrow * deg_ncol
    cell_holds_group = np.zeros((cell_count, group_count), np.uint8)
    land_in_cell = np.zeros((cell_count, group_count), np.int64)
    for k in range(entry_cell.size):
        g = group_of_basin[entry_basin[k]]
        if g < 0:
            continue
        cell_holds_group[entry_cell[k], g] = 1
        land_in_cell[entry_cell[k], g] += entry_land[k]
    owner_of_cell = np.full(cell_count, -1, np.int64)
    for cell in range(cell_count):
        most = np.int64(0)
        for g in range(group_count):
            if land_in_cell[cell, g] > most:
                most = land_in_cell[cell, g]
                owner_of_cell[cell] = g
    # a basin too small to show at the coarse resolution still puts its group in its own cell
    for b in range(group_of_basin.size):
        g = group_of_basin[b]
        if g < 0:
            continue
        cell_holds_group[cell_of_basin[b], g] = 1
        if owner_of_cell[cell_of_basin[b]] < 0:
            owner_of_cell[cell_of_basin[b]] = g
    # the groups of every cell, laid out one cell after another; the groups that share a cell touch,
    # and every group of a cell touches every group of each of its eight neighbours
    list_start = np.zeros(cell_count + 1, np.int64)
    for cell in range(cell_count):
        list_start[cell + 1] = list_start[cell]
        for g in range(group_count):
            if cell_holds_group[cell, g] != 0:
                list_start[cell + 1] += 1
    cell_groups = np.empty(list_start[cell_count], np.int64)
    filled = 0
    for cell in range(cell_count):
        for g in range(group_count):
            if cell_holds_group[cell, g] != 0:
                cell_groups[filled] = g
                filled += 1
    adjacency = np.zeros((group_count, group_count), np.uint8)
    for degree_row in range(deg_nrow):
        for degree_col in range(deg_ncol):
            cell = degree_row * deg_ncol + degree_col
            for here in range(list_start[cell], list_start[cell + 1]):
                g = cell_groups[here]
                for other in range(here + 1, list_start[cell + 1]):
                    h = cell_groups[other]
                    adjacency[g, h] = 1
                    adjacency[h, g] = 1
                for row_offset in range(-1, 2):
                    for col_offset in range(-1, 2):
                        n_row = degree_row + row_offset
                        n_col = degree_col + col_offset
                        if n_row < 0 or n_row >= deg_nrow or n_col < 0 or n_col >= deg_ncol:
                            continue
                        n_cell = n_row * deg_ncol + n_col
                        for there in range(list_start[n_cell], list_start[n_cell + 1]):
                            h = cell_groups[there]
                            if h != g:
                                adjacency[g, h] = 1
                                adjacency[h, g] = 1
    return adjacency, owner_of_cell


@njit(cache=True)
def _lies_inside(owner_of_cell, group_inside, group_around, deg_nrow, deg_ncol):
    """is everything around this group's cells the one other group: no sea, no grid edge, no third group"""
    cells_inside = 0
    not_around = 0
    around = 0
    for degree_row in range(deg_nrow):
        for degree_col in range(deg_ncol):
            if owner_of_cell[degree_row * deg_ncol + degree_col] != group_inside:
                continue
            cells_inside += 1
            for row_offset in range(-1, 2):
                for col_offset in range(-1, 2):
                    n_row = degree_row + row_offset
                    n_col = degree_col + col_offset
                    if n_row < 0 or n_row >= deg_nrow or n_col < 0 or n_col >= deg_ncol:
                        not_around += 1
                        continue
                    owner = owner_of_cell[n_row * deg_ncol + n_col]
                    if owner == group_inside:
                        continue
                    if owner == group_around:
                        around += 1
                        continue
                    not_around += 1
    return cells_inside > 0 and not_around == 0 and around > 0


@njit(cache=True)
def _contains(outer, inner):
    return outer[0] <= inner[0] and inner[1] <= outer[1] and outer[2] <= inner[2] and inner[3] <= outer[3]


@njit(cache=True)
def _cheapest_join(g_rect, alive, adjacency, owner_of_cell, capacity_blocks, deg_nrow, deg_ncol):
    """[C]: the pair whose joined window costs the fewest extra blocks over the two apart; a pair may be
    joined when the two touch and the joined window fits, or when one holds the other with nothing else
    around it (then the capacity has nothing to say).  Returns (first, second, extra), first < 0 when
    no pair may be joined."""
    group_count = g_rect.shape[0]
    best_first = -1
    best_second = -1
    best_extra = np.int64(0)
    for first in range(group_count):
        if alive[first] == 0:
            continue
        first_blocks = _window_blocks(g_rect[first, 0], g_rect[first, 1], g_rect[first, 2], g_rect[first, 3])
        for second in range(first + 1, group_count):
            if alive[second] == 0:
                continue
            one_holds_the_other = (_contains(g_rect[first], g_rect[second]) and _lies_inside(owner_of_cell, second, first, deg_nrow, deg_ncol)) or \
                                  (_contains(g_rect[second], g_rect[first]) and _lies_inside(owner_of_cell, first, second, deg_nrow, deg_ncol))
            if not one_holds_the_other and adjacency[first, second] == 0:
                continue
            joined = _window_blocks(min(g_rect[first, 0], g_rect[second, 0]), max(g_rect[first, 1], g_rect[second, 1]),
                                    min(g_rect[first, 2], g_rect[second, 2]), max(g_rect[first, 3], g_rect[second, 3]))
            if not one_holds_the_other and joined > capacity_blocks:
                continue
            extra = joined - first_blocks - _window_blocks(g_rect[second, 0], g_rect[second, 1], g_rect[second, 2], g_rect[second, 3])
            if best_first < 0 or extra < best_extra:
                best_extra = extra
                best_first = first
                best_second = second
    return best_first, best_second, best_extra


@njit(cache=True)
def _stays_in_one_piece(piece_cell, group_pieces, first, last, piece_left_out, piece_is_left_out, use_left_out_flags, deg_nrow, deg_ncol, cell_mark, stack):
    """would the group's cells still be connected (eight neighbours) without the piece(s) taken away"""
    cell_count = 0
    for listed in range(first, last):
        p = group_pieces[listed]
        if listed == piece_left_out or (use_left_out_flags and piece_is_left_out[p] != 0):
            continue
        cell = piece_cell[p]
        if cell_mark[cell] == 0:
            cell_mark[cell] = 1
            cell_count += 1
    if cell_count == 0:
        return True
    if piece_left_out >= 0 and cell_mark[piece_cell[group_pieces[piece_left_out]]] != 0:
        # another piece of this group holds that cell too: taking this one moves no cell
        for listed in range(first, last):
            p = group_pieces[listed]
            if listed != piece_left_out and not (use_left_out_flags and piece_is_left_out[p] != 0):
                cell_mark[piece_cell[p]] = 0
        return True
    height = 0
    reached = 0
    for listed in range(first, last):
        p = group_pieces[listed]
        if listed != piece_left_out and not (use_left_out_flags and piece_is_left_out[p] != 0):
            stack[0] = piece_cell[p]
            height = 1
            cell_mark[stack[0]] = 2
            reached = 1
            break
    while height > 0:
        height -= 1
        cell = stack[height]
        degree_row = cell // deg_ncol
        degree_col = cell % deg_ncol
        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                n_row = degree_row + row_offset
                n_col = degree_col + col_offset
                if n_row < 0 or n_row >= deg_nrow or n_col < 0 or n_col >= deg_ncol:
                    continue
                n_cell = n_row * deg_ncol + n_col
                if cell_mark[n_cell] != 1:
                    continue
                cell_mark[n_cell] = 2
                reached += 1
                stack[height] = n_cell
                height += 1
    for listed in range(first, last):
        p = group_pieces[listed]
        if listed != piece_left_out and not (use_left_out_flags and piece_is_left_out[p] != 0):
            cell_mark[piece_cell[p]] = 0
    return reached == cell_count


@njit(cache=True)
def _csr_lists(key, key_count):
    """the items grouped by key: start[key] .. start[key + 1] index into items"""
    start = np.zeros(key_count + 1, np.int64)
    for k in range(key.size):
        start[key[k] + 1] += 1
    for k in range(key_count):
        start[k + 1] += start[k]
    fill = start[:-1].copy()
    items = np.empty(key.size, np.int64)
    for k in range(key.size):
        items[fill[key[k]]] = k
        fill[key[k]] += 1
    return start, items


@njit(cache=True)
def _move_cells_to_a_group_beside_them(piece_cell, piece_group, piece_rect, piece_land, piece_count_basins, g_rect, g_count, g_land, capacity_blocks, deg_nrow, deg_ncol):
    """[D]: the pieces (the basins of one cell that belong to one group) moved to a group beside them,
    a batch (everything a group has inside another group's window, each piece touching the taker)
    or one piece at a time, while the two windows together get smaller and the group left behind stays
    in one piece; at equal windows a piece goes where it has more neighbours.  Changes piece_group and
    the group arrays in place.  Returns the number of moves."""
    piece_count = piece_cell.size
    group_count = g_rect.shape[0]
    cell_count = deg_nrow * deg_ncol
    cell_mark = np.zeros(cell_count, np.uint8)
    stack = np.empty(piece_count + 1, np.int64)
    piece_is_taken = np.zeros(piece_count, np.uint8)
    piece_is_best_batch = np.zeros(piece_count, np.uint8)
    moves = 0
    for round_index in range(piece_count + 1):
        group_start, group_pieces = _csr_lists(piece_group, group_count)
        cell_start, cell_pieces = _csr_lists(piece_cell, cell_count)
        # first the batch move
        batch_from = -1
        batch_to = -1
        batch_gain = np.int64(0)
        batch_left = np.zeros(4, np.int64)
        for group_here in range(group_count):
            first = group_start[group_here]
            last = group_start[group_here + 1]
            if last - first < 2:
                continue
            for group_there in range(group_count):
                if group_there == group_here:
                    continue
                taken = 0
                left = np.array([1 << 60, -1, 1 << 60, -1], np.int64)
                for listed in range(group_start[group_there], group_start[group_there + 1]):
                    cell_mark[piece_cell[group_pieces[listed]]] = 3
                for pass_index in range(2):
                    for listed in range(first, last):
                        p = group_pieces[listed]
                        if piece_is_taken[p] != 0:
                            continue
                        if not _contains(g_rect[group_there], piece_rect[p]):
                            continue
                        p_row = piece_cell[p] // deg_ncol
                        p_col = piece_cell[p] % deg_ncol
                        touches = False
                        for row_offset in range(-1, 2):
                            for col_offset in range(-1, 2):
                                n_row = p_row + row_offset
                                n_col = p_col + col_offset
                                if n_row < 0 or n_row >= deg_nrow or n_col < 0 or n_col >= deg_ncol:
                                    continue
                                if cell_mark[n_row * deg_ncol + n_col] == 3:
                                    touches = True
                        if not touches:
                            continue
                        piece_is_taken[p] = 1
                        cell_mark[piece_cell[p]] = 3
                        taken += 1
                for listed in range(group_start[group_there], group_start[group_there + 1]):
                    cell_mark[piece_cell[group_pieces[listed]]] = 0
                for listed in range(first, last):
                    p = group_pieces[listed]
                    if piece_is_taken[p] != 0:
                        cell_mark[piece_cell[p]] = 0
                    else:
                        left[0] = min(left[0], piece_rect[p, 0])
                        left[1] = max(left[1], piece_rect[p, 1])
                        left[2] = min(left[2], piece_rect[p, 2])
                        left[3] = max(left[3], piece_rect[p, 3])
                gain = _window_blocks(g_rect[group_here, 0], g_rect[group_here, 1], g_rect[group_here, 2], g_rect[group_here, 3]) - _window_blocks(left[0], left[1], left[2], left[3])
                if taken > 0 and taken < last - first and gain > batch_gain and \
                        _stays_in_one_piece(piece_cell, group_pieces, first, last, -1, piece_is_taken, True, deg_nrow, deg_ncol, cell_mark, stack):
                    batch_gain = gain
                    batch_from = group_here
                    batch_to = group_there
                    batch_left[:] = left
                    for listed in range(first, last):
                        piece_is_best_batch[group_pieces[listed]] = piece_is_taken[group_pieces[listed]]
                for listed in range(first, last):
                    piece_is_taken[group_pieces[listed]] = 0
        if batch_from >= 0:
            for listed in range(group_start[batch_from], group_start[batch_from + 1]):
                p = group_pieces[listed]
                if piece_is_best_batch[p] == 0:
                    continue
                g_land[batch_from] -= piece_land[p]
                g_count[batch_from] -= piece_count_basins[p]
                g_land[batch_to] += piece_land[p]
                g_count[batch_to] += piece_count_basins[p]
                piece_group[p] = batch_to
                piece_is_best_batch[p] = 0
                moves += 1
            g_rect[batch_from, :] = batch_left
            continue
        # then the single-cell move
        best_piece = -1
        best_group = -1
        best_gain = np.int64(0)
        best_boundary_gain = np.int64(0)
        best_left = np.zeros(4, np.int64)
        best_taking = np.zeros(4, np.int64)
        for p in range(piece_count):
            group_here = piece_group[p]
            first = group_start[group_here]
            last = group_start[group_here + 1]
            if last - first < 2:
                continue                       # the group would be left with nothing
            piece_position = -1
            left = np.array([1 << 60, -1, 1 << 60, -1], np.int64)
            for listed in range(first, last):
                q = group_pieces[listed]
                if q == p:
                    piece_position = listed
                    continue
                left[0] = min(left[0], piece_rect[q, 0])
                left[1] = max(left[1], piece_rect[q, 1])
                left[2] = min(left[2], piece_rect[q, 2])
                left[3] = max(left[3], piece_rect[q, 3])
            blocks_here = _window_blocks(g_rect[group_here, 0], g_rect[group_here, 1], g_rect[group_here, 2], g_rect[group_here, 3])
            blocks_left = _window_blocks(left[0], left[1], left[2], left[3])
            degree_row = piece_cell[p] // deg_ncol
            degree_col = piece_cell[p] % deg_ncol
            tried = np.full(9, -1, np.int64)
            tried_count = 0
            for row_offset in range(-1, 2):
                for col_offset in range(-1, 2):
                    n_row = degree_row + row_offset
                    n_col = degree_col + col_offset
                    if n_row < 0 or n_row >= deg_nrow or n_col < 0 or n_col >= deg_ncol:
                        continue
                    n_cell = n_row * deg_ncol + n_col
                    for listed in range(cell_start[n_cell], cell_start[n_cell + 1]):
                        group_there = piece_group[cell_pieces[listed]]
                        if group_there == group_here:
                            continue
                        seen = False
                        for k in range(tried_count):
                            if tried[k] == group_there:
                                seen = True
                        if seen:
                            continue
                        if tried_count < 9:
                            tried[tried_count] = group_there
                            tried_count += 1
                        taking = np.array([min(g_rect[group_there, 0], piece_rect[p, 0]), max(g_rect[group_there, 1], piece_rect[p, 1]),
                                           min(g_rect[group_there, 2], piece_rect[p, 2]), max(g_rect[group_there, 3], piece_rect[p, 3])], np.int64)
                        blocks_there = _window_blocks(g_rect[group_there, 0], g_rect[group_there, 1], g_rect[group_there, 2], g_rect[group_there, 3])
                        blocks_taking = _window_blocks(taking[0], taking[1], taking[2], taking[3])
                        if blocks_taking > capacity_blocks and blocks_taking != blocks_there:
                            continue
                        gain = blocks_here + blocks_there - blocks_left - blocks_taking
                        if gain < 0:
                            continue
                        # where the windows come out the same, the cell goes to the group it has more neighbours in
                        neighbours_here = 0
                        neighbours_there = 0
                        for near_row_offset in range(-1, 2):
                            for near_col_offset in range(-1, 2):
                                if near_row_offset == 0 and near_col_offset == 0:
                                    continue
                                near_row = degree_row + near_row_offset
                                near_col = degree_col + near_col_offset
                                if near_row < 0 or near_row >= deg_nrow or near_col < 0 or near_col >= deg_ncol:
                                    continue
                                near_cell = near_row * deg_ncol + near_col
                                for near_listed in range(cell_start[near_cell], cell_start[near_cell + 1]):
                                    near_group = piece_group[cell_pieces[near_listed]]
                                    if near_group == group_here:
                                        neighbours_here += 1
                                    elif near_group == group_there:
                                        neighbours_there += 1
                        boundary_gain = np.int64(neighbours_there - neighbours_here)
                        if gain == 0 and boundary_gain <= 0:
                            continue
                        if gain < best_gain or (gain == best_gain and boundary_gain <= best_boundary_gain):
                            continue
                        if not _stays_in_one_piece(piece_cell, group_pieces, first, last, piece_position, piece_is_taken, False, deg_nrow, deg_ncol, cell_mark, stack):
                            continue
                        best_gain = gain
                        best_boundary_gain = boundary_gain
                        best_piece = p
                        best_group = group_there
                        best_left[:] = left
                        best_taking[:] = taking
        if best_piece < 0:
            break
        group_here = piece_group[best_piece]
        g_rect[group_here, :] = best_left
        g_land[group_here] -= piece_land[best_piece]
        g_count[group_here] -= piece_count_basins[best_piece]
        g_rect[best_group, :] = best_taking
        g_land[best_group] += piece_land[best_piece]
        g_count[best_group] += piece_count_basins[best_piece]
        piece_group[best_piece] = best_group
        moves += 1
    return moves


def fd1_4_group_by_hilbert_curve(basin_table_path, coarse_basin_view_path, grid, capacity_pixels, grouping="hilbert", tag="fd1.4"):
    """the automatic groups of whole basins ([0] to [D] above) at one capacity, written in two
    tables (group_hilbert_fine, basin_group_hilbert_fine): the group id stands in the
    level3_code column as well, and the vote columns are zero"""
    started = time.time()
    basins = fd_tables.read_basin_table(basin_table_path)
    basin_count = len(basins)
    basin_id = basins["basin_id"].to_numpy(np.int64)
    if not np.array_equal(basin_id, np.arange(1, basin_count + 1)):
        raise FlowDivideError("the basin table must list the basins 1 .. N in order")
    area = basins["basin_area_km2"].to_numpy(np.float64)
    if not np.isfinite(area).all() or (area <= 0).any():
        raise FlowDivideError("every basin needs a real drainage area: the area decides which basins stand alone")
    land = basins["basin_grid_count"].to_numpy(np.int64)
    block = grid.block_pixels
    deg_nrow = -(-grid.nrow // block)
    deg_ncol = -(-grid.ncol // block)
    if deg_nrow * deg_ncol > 50000000:
        raise FlowDivideError("%d x %d blocks is more than this step indexes; use a larger block" % (deg_nrow, deg_ncol))
    capacity_blocks = capacity_pixels // (block * block)
    rectangles = basin_rectangles(basins)
    if grid.periodic and (rectangles[:, 3] > grid.ncol).any():
        raise FlowDivideError("the automatic grouping works on a flat grid of blocks; %d basins lie across the antimeridian" % int((rectangles[:, 3] > grid.ncol).sum()))
    rect = rectangles // block                                       # every rectangle in blocks
    rect[:, 1] = (rectangles[:, 1] - 1) // block + 1                 # row_max, col_max are one past: the block past the last one
    rect[:, 3] = (rectangles[:, 3] - 1) // block + 1
    outlet_row = basins["outlet_row"].to_numpy(np.int64)
    outlet_col = basins["outlet_col"].to_numpy(np.int64)
    # [0] where every basin's land is, from the coarse basin view
    with rasterio.open(coarse_basin_view_path) as view:
        factor = int(round(view.transform.a / abs(grid.pixel_width)))            # fine pixels to a coarse cell
        if factor < 1 or block % factor != 0 or view.width != -(-grid.ncol // factor) or view.height != -(-grid.nrow // factor):
            raise FlowDivideError("the coarse basin view %s does not fold this grid by a whole number of cells to the block" % coarse_basin_view_path)
        # the view on the grid folded by that factor, origin, both cell sides, no rotation and the same CRS, and
        # unsigned (a view shifted by one cell or of another CRS would place the basins in other blocks)
        expected = (grid.transform.c, grid.pixel_width * factor, grid.transform.f, grid.pixel_height * factor)
        found = (view.transform.c, view.transform.a, view.transform.f, view.transform.e)
        tolerance = 1e-9 * max(1.0, abs(grid.pixel_width) * factor)
        if any(abs(a - b) > tolerance for a, b in zip(found, expected)) or view.transform.b != 0 or view.transform.d != 0 \
                or view.crs != grid.crs or view.dtypes[0] not in ("uint8", "uint16", "uint32"):
            raise FlowDivideError("the coarse basin view %s is not this grid folded by %d (origin, cell size, rotation, CRS "
                                  "or type differ)" % (coarse_basin_view_path, factor))
        coarse_per_block = block // factor
        entries_cell = []
        entries_basin = []
        entries_land = []
        for degree_row in range(deg_nrow):
            row0 = degree_row * coarse_per_block
            nrow = min(coarse_per_block, view.height - row0)
            if nrow <= 0:
                break
            strip = view.read(1, window=Window(0, row0, view.width, nrow))
            for degree_col in range(deg_ncol):
                block_cells = strip[:, degree_col * coarse_per_block:(degree_col + 1) * coarse_per_block]
                values = block_cells[block_cells != 0]
                if values.size == 0:
                    continue
                ids, counts = np.unique(values, return_counts=True)
                # an id past the basin table is an error, not a value to leave out
                if int(ids.max()) > basin_count:
                    raise FlowDivideError("the coarse basin view %s holds basin %d, past the %d basins of the table"
                                          % (coarse_basin_view_path, int(ids.max()), basin_count))
                entries_cell.append(np.full(ids.size, degree_row * deg_ncol + degree_col, np.int64))
                entries_basin.append(ids.astype(np.int64) - 1)
                entries_land.append(counts.astype(np.int64))
    entry_cell = np.concatenate(entries_cell) if entries_cell else np.zeros(0, np.int64)
    entry_basin = np.concatenate(entries_basin) if entries_basin else np.zeros(0, np.int64)
    entry_land = np.concatenate(entries_land) if entries_land else np.zeros(0, np.int64)
    cell_of_basin = _land_per_cell_of_basins(entry_cell, entry_basin, entry_land, basin_count)
    too_small = cell_of_basin < 0
    # a basin that keeps the cell of its outlet has its outlet on the grid (the column taken round a periodic
    # one), and every cell is one of the block grid, before the kernels index with it; an outlet at column 1000000 of a
    # grid of 2 would index past the arrays
    small_rows = outlet_row[too_small].astype(np.int64)
    small_cols = outlet_col[too_small].astype(np.int64)
    if grid.periodic:
        small_cols = small_cols % grid.ncol
    if small_rows.size and (small_rows.min() < 0 or small_rows.max() >= grid.nrow or small_cols.min() < 0 or small_cols.max() >= grid.ncol):
        raise FlowDivideError("a basin below one coarse cell has its outlet outside the grid of %d x %d pixels" % (grid.nrow, grid.ncol))
    cell_of_basin[too_small] = (small_rows // block) * deg_ncol + small_cols // block
    if cell_of_basin.size and (cell_of_basin.min() < 0 or cell_of_basin.max() >= deg_nrow * deg_ncol):
        raise FlowDivideError("a basin lies in a block cell outside the %d x %d block cells of the grid" % (deg_nrow, deg_ncol))
    log(tag, "the land of %d basins reaches %d block cells in all; %d basins are below one coarse cell and keep the cell of their outlet" % (basin_count, entry_cell.size, int(too_small.sum())))
    # [A] the basins that stand alone
    own_blocks = (rect[:, 1] - rect[:, 0]) * (rect[:, 3] - rect[:, 2])
    over_the_cap = own_blocks > capacity_blocks
    major_river = (area >= MAJOR_RIVER_MIN_AREA_KM2) & ~over_the_cap
    alone = np.nonzero(over_the_cap | major_river)[0]
    group_limit = basin_count + 1
    g_rect = np.zeros((group_limit, 4), np.int64)
    g_count = np.zeros(group_limit, np.int64)
    g_land = np.zeros(group_limit, np.int64)
    g_kind = np.zeros(group_limit, np.int64)
    g_over = np.zeros(group_limit, np.uint8)
    group_of_basin = np.full(basin_count, -1, np.int64)
    group_count = 0
    for b in alone:
        g_rect[group_count] = rect[b]
        g_count[group_count] = 1
        g_land[group_count] = land[b]
        g_kind[group_count] = 1
        g_over[group_count] = 1 if over_the_cap[b] else 0
        group_of_basin[b] = group_count
        log(tag, "  basin %d drains %.0f km2 and holds %d pixels in a window of %d blocks: a group of its own, %s" % (
            b + 1, area[b], land[b], own_blocks[b], "its window passes the capacity" if over_the_cap[b] else "a major river"))
        group_count += 1
    log(tag, "%d basins have a window larger than the capacity of %d blocks and %d more drain at least %.0f km2; all %d become groups of their own" % (
        int(over_the_cap.sum()), capacity_blocks, int(major_river.sum()), MAJOR_RIVER_MIN_AREA_KM2, alone.size))
    # [B] the rest along the curve
    curve_order = 1
    while (1 << curve_order) < deg_nrow or (1 << curve_order) < deg_ncol:
        curve_order += 1
    free = np.nonzero(group_of_basin < 0)[0]
    if free.size:
        hilbert = np.zeros(basin_count, np.int64)
        hilbert[free] = _hilbert_indices(curve_order, cell_of_basin[free] // deg_ncol, cell_of_basin[free] % deg_ncol)
        order = free[np.lexsort((free, hilbert[free]))]
        group_count, runs_closed_at_a_gap, cells_split = _runs_along_the_curve(order, hilbert, cell_of_basin, rect, land, capacity_blocks, deg_nrow, deg_ncol,
                                                                                group_of_basin, g_rect, g_count, g_land, g_kind, g_over, group_count)
        log(tag, "the curve has order %d over a grid of %d x %d blocks; %d basins follow it; it closed %d runs, %d of them at a gap; %d cells held basins too far apart to share one group" % (
            curve_order, deg_nrow, deg_ncol, free.size, group_count - alone.size, runs_closed_at_a_gap, cells_split))
    # [C] and [D] turn about
    for correction_round in range(HILBERT_CORRECTION_ROUNDS):
        groups_before = group_count
        adjacency, owner_of_cell = _group_adjacency(entry_cell, entry_basin, entry_land, group_of_basin, cell_of_basin, group_count, deg_nrow, deg_ncol)
        alive = np.ones(group_count, np.uint8)
        alias = np.arange(group_count)
        joins = 0
        blocks_saved = 0
        blocks_paid = 0
        while True:
            first, second, extra = _cheapest_join(g_rect[:group_count], alive, adjacency, owner_of_cell, capacity_blocks, deg_nrow, deg_ncol)
            if first < 0:
                break
            owner_of_cell[owner_of_cell == second] = first
            g_rect[first] = [min(g_rect[first, 0], g_rect[second, 0]), max(g_rect[first, 1], g_rect[second, 1]), min(g_rect[first, 2], g_rect[second, 2]), max(g_rect[first, 3], g_rect[second, 3])]
            g_count[first] += g_count[second]
            g_land[first] += g_land[second]
            g_over[first] |= g_over[second]
            alive[second] = 0
            alias[second] = first
            adjacency[first] |= adjacency[second]
            adjacency[:, first] |= adjacency[:, second]
            adjacency[first, first] = 0
            joins += 1
            if extra < 0:
                blocks_saved -= int(extra)
            else:
                blocks_paid += int(extra)
        # the groups that survive keep their order
        kept = np.nonzero(alive)[0]
        new_index = np.full(group_count, -1, np.int64)
        new_index[kept] = np.arange(kept.size)
        resolved = group_of_basin.copy()
        for _ in range(group_count):
            moved = alias[resolved] != resolved
            if not moved.any():
                break
            resolved[moved] = alias[resolved[moved]]
        group_of_basin = new_index[resolved]
        g_rect[:kept.size] = g_rect[kept]
        g_count[:kept.size] = g_count[kept]
        g_land[:kept.size] = g_land[kept]
        g_kind[:kept.size] = g_kind[kept]
        g_over[:kept.size] = g_over[kept]
        group_count = kept.size
        log(tag, "%d joins made, %d blocks of window saved and %d paid; %d groups remain, and no pair that may be joined fits the capacity together" % (joins, blocks_saved, blocks_paid, group_count))
        # [D]: one piece for every cell a group has basins in, numbered in the order the basins first show
        # them
        piece_key = cell_of_basin * group_count + group_of_basin
        unique_keys, first_seen, inverse = np.unique(piece_key, return_index=True, return_inverse=True)
        by_first_seen = np.argsort(first_seen, kind="stable")
        rank = np.empty(unique_keys.size, np.int64)
        rank[by_first_seen] = np.arange(unique_keys.size)
        unique_keys = unique_keys[by_first_seen]
        piece_of_basin = rank[inverse]
        piece_cell = unique_keys // group_count
        piece_group = (unique_keys % group_count).astype(np.int64)
        piece_count = unique_keys.size
        piece_rect = np.zeros((piece_count, 4), np.int64)
        piece_rect[:, 0] = np.iinfo(np.int64).max
        piece_rect[:, 2] = np.iinfo(np.int64).max
        piece_rect[:, 1] = -1
        piece_rect[:, 3] = -1
        np.minimum.at(piece_rect[:, 0], piece_of_basin, rect[:, 0])
        np.maximum.at(piece_rect[:, 1], piece_of_basin, rect[:, 1])
        np.minimum.at(piece_rect[:, 2], piece_of_basin, rect[:, 2])
        np.maximum.at(piece_rect[:, 3], piece_of_basin, rect[:, 3])
        piece_land = np.bincount(piece_of_basin, weights=land, minlength=piece_count).astype(np.int64)
        piece_basins = np.bincount(piece_of_basin, minlength=piece_count).astype(np.int64)
        moves = _move_cells_to_a_group_beside_them(piece_cell, piece_group, piece_rect, piece_land, piece_basins, g_rect[:group_count], g_count[:group_count], g_land[:group_count],
                                                   capacity_blocks, deg_nrow, deg_ncol)
        group_of_basin = piece_group[piece_of_basin]
        log(tag, "round %d: %d groups after the joins, %d block cells moved" % (correction_round + 1, group_count, moves))
        if group_count == groups_before and moves == 0:
            break
    # [E] the numbering, from 1 in the order the groups were made; the kind is settled now
    g_kind[:group_count] = np.where(g_count[:group_count] == 1, 1, 2)
    if group_count >= 1000:
        raise FlowDivideError("%d groups do not fit the codes 1 .. 999" % group_count)
    # the checks: the group records rebuilt from the membership must match; a group over
    # the capacity must hold a basin that is over on its own
    rebuilt_count = np.bincount(group_of_basin, minlength=group_count)
    rebuilt_land = np.bincount(group_of_basin, weights=land, minlength=group_count).astype(np.int64)
    if not np.array_equal(rebuilt_count, g_count[:group_count]) or not np.array_equal(rebuilt_land, g_land[:group_count]):
        raise FlowDivideError("a group's basin count or land does not agree with its basins")
    for g in range(group_count):
        members = np.nonzero(group_of_basin == g)[0]
        made = (int(rect[members, 0].min()), int(rect[members, 1].max()), int(rect[members, 2].min()), int(rect[members, 3].max()))
        if made != tuple(int(v) for v in g_rect[g]):
            raise FlowDivideError("group %d holds rows %d..%d columns %d..%d of blocks, but its basins make %s" % (g + 1, g_rect[g, 0], g_rect[g, 1], g_rect[g, 2], g_rect[g, 3], made))
        if (g_rect[g, 1] - g_rect[g, 0]) * (g_rect[g, 3] - g_rect[g, 2]) > capacity_blocks and g_over[g] == 0:
            raise FlowDivideError("group %d passes the capacity and holds no basin that does" % (g + 1))
    windows = np.column_stack([g_rect[:group_count, 0] * block, g_rect[:group_count, 1] * block, g_rect[:group_count, 2] * block, g_rect[:group_count, 3] * block])
    group_ids = np.arange(1, group_count + 1)
    group_columns = {"group_id": group_ids, "group_kind": g_kind[:group_count], "level_code": group_ids,
                     "level3_count": np.zeros(group_count, np.int64), "basin_count": g_count[:group_count],
                     "coded_basin_count": g_count[:group_count], "neighbour_basin_count": np.zeros(group_count, np.int64),
                     "land_grid_count": g_land[:group_count]}
    group_table = group_table_rows(grid, windows, group_columns, 0)
    check_groups_against_basins(group_table, group_of_basin + 1, land)
    # the two files of the grouping: both markers go before either file is replaced, then
    # the basin map, then the group table
    root, run = fd_tables.root_and_run_of_basin_table(basin_table_path)
    group_table_path = fd_tables.group_table_path(root, run, grouping)
    basin_map_path = fd_tables.group_basin_map_path(root, run, grouping)
    for path in (group_table_path, basin_map_path):
        if os.path.exists(path + ".done"):
            os.remove(path + ".done")
    fd_tables.write_basin_map(basin_map_path, np.concatenate([[0], group_of_basin + 1]), run, "group", grouping)
    fd_tables.write_group_table(group_table, group_table_path, "fd1.4 Hilbert: %d groups hold %d basins" % (group_count, basin_count))
    over = int((group_table["window_grid_count"] > capacity_pixels).sum())
    report = {"groups": group_count, "groups_of_one_basin": int((g_kind[:group_count] == 1).sum()), "over_the_capacity": over, "basins_alone": int(alone.size),
              "window_grid_count": int(group_table["window_grid_count"].sum()), "seconds": round(time.time() - started, 1)}
    write_json(group_table_path + ".report.json", report)
    log(tag, "written %s: %d groups hold %d basins; %d pass the capacity" % (group_table_path, group_count, basin_count, over))
    return report


# =============================================================================
#  [7] FD1.5  fitting to memory: the Pfafstetter cut and the final regions
# =============================================================================
#
#  A basin whose own window exceeds the capacity is cut at tributary outlets by the Pfafstetter
#  scheme: the main stem is traced upstream from the outlet along the inflow with the largest upstream
#  pixel count; the four largest tributaries become the even pieces 2, 4, 6, 8 (from the outlet
#  upstream) and the stretches of main stem between them the odd pieces 1, 3, 5, 7, 9; a piece that
#  still exceeds the capacity is cut again the same way.  The walk runs on the channel network of the
#  basin held in memory (the channel pixels of FD1.1, a few percent of the basin), the labelling of
#  every pixel of the basin with its deepest piece runs on the work tiles as FD1.3 does, with the
#  piece outlets as terminals.  The pieces are then merged into regions in code order while the merged
#  window fits, and the regions are numbered so that every flow between regions goes to a larger
#  number.  A group of whole basins over the capacity is divided along a block line; groups under it
#  may be merged.  The step ends with the region mask and the region and piece tables (no member mask is written).

class ChannelNetwork:
    """the channel pixels of one basin, sorted by their unrolled pixel key, with the downstream link
    and the inflow lists (CSR) that the Pfafstetter walks need"""

    def __init__(self, row, col, code, acc, grid):
        self.row = row
        self.col = col                       # unrolled columns: may run past ncol - 1 on a periodic grid
        self.code = code
        self.acc = acc
        self.grid = grid
        stride = 2 * grid.ncol + 2
        self.key = row.astype(np.int64) * stride + col.astype(np.int64)
        order = np.argsort(self.key, kind="stable")
        self.row = self.row[order]
        self.col = self.col[order]
        self.code = self.code[order]
        self.acc = self.acc[order]
        self.key = self.key[order]
        self.stride = stride
        self.down = _channel_downstream(self.row, self.col, self.code, self.key, stride)
        self.inflow_ptr, self.inflow_idx = _channel_inflows(self.down)

    def index_of(self, row, col):
        key = np.int64(row) * self.stride + np.int64(col)
        position = int(np.searchsorted(self.key, key))
        if position < self.key.size and self.key[position] == key:
            return position
        return -1


@njit(cache=True)
def _channel_downstream(row, col, code, key, stride):
    down = np.full(row.size, -1, np.int64)
    for i in range(row.size):
        drow = DROW[code[i]]
        dcol = DCOL[code[i]]
        if drow == 0 and dcol == 0:
            continue
        target = (row[i] + drow) * stride + (col[i] + dcol)
        position = np.searchsorted(key, target)
        if position < key.size and key[position] == target:
            down[i] = position
    return down


@njit(cache=True)
def _channel_inflows(down):
    count = np.zeros(down.size + 1, np.int64)
    for i in range(down.size):
        if down[i] >= 0:
            count[down[i] + 1] += 1
    ptr = np.cumsum(count)
    fill = ptr[:-1].copy()
    idx = np.empty(ptr[-1], np.int64)
    for i in range(down.size):
        if down[i] >= 0:
            idx[fill[down[i]]] = i
            fill[down[i]] += 1
    return ptr, idx


@njit(cache=True)
def _walk_mainstem(start, inflow_ptr, inflow_idx, acc, is_piece_outlet, mainstem, junction_position, junction_index, junction_acc):
    """One main stem from `start` upstream: at every pixel the inflow with the largest upstream count
    continues the main stem, every other inflow is a junction; an inflow that is the outlet of another
    piece belongs to that piece and is neither.  Returns (mainstem length, junction count); the main
    stem is written into `mainstem` and the junctions into the three junction arrays."""
    length = 0
    junction_count = 0
    current = start
    while True:
        if length >= mainstem.size:
            return -1, junction_count
        mainstem[length] = current
        length += 1
        best = -1
        best_acc = -1
        for k in range(inflow_ptr[current], inflow_ptr[current + 1]):
            candidate = inflow_idx[k]
            if acc[candidate] > best_acc:
                best_acc = acc[candidate]
                best = candidate
        for k in range(inflow_ptr[current], inflow_ptr[current + 1]):
            candidate = inflow_idx[k]
            if candidate == best or is_piece_outlet[candidate] != 0:
                continue
            if junction_count >= junction_position.size:
                return -2, junction_count
            junction_position[junction_count] = length - 1
            junction_index[junction_count] = candidate
            junction_acc[junction_count] = acc[candidate]
            junction_count += 1
        if best < 0 or is_piece_outlet[best] != 0:
            break
        current = best
    return length, junction_count


TRIBUTARIES_PER_LEVEL = 4
CHANNEL_BOUND_MARGIN = 1.01             # the bound on a tributary below the channel threshold, 1 % wide of
                                        # our smallest pixel, for a provider's pixel area smaller than ours


class Piece:
    """one Pfafstetter piece: its id (in the order the pieces were made),
    its code, its outlet pixel, the pixel its outlet flows into, the piece it flows into, the pieces that
    flow into it, and what the labelling pass found for it.  Columns are unrolled on a periodic grid."""

    def __init__(self, piece_id, code, level, parent, outlet_row, outlet_col, downstream_row, downstream_col):
        self.id = piece_id
        self.code = code
        self.level = level
        self.parent = parent                         # the piece this one is a part of (None at level 1)
        self.outlet_row = outlet_row
        self.outlet_col = outlet_col
        self.downstream_row = downstream_row         # -1 at the basin outlet
        self.downstream_col = downstream_col
        self.next_down = None                        # the piece its outlet flows into, as it stands (re-pointed when that piece is cut)
        self.next_down_at_creation = None            # as the walk defined it: what is written for a piece that is not used
        self.inflows = []                            # the pieces of the deepest level whose outlets flow into it
        self.acc_at_outlet = 0
        self.expected_pixels = 0                     # acc at the outlet less the acc at the outlets of its inflows
        self.pixels = 0
        self.rectangle = None                        # (row_min, row_max, col_min, col_max), unrolled
        self.used = False
        self.region_number = 0
        self.children = []

    @property
    def parent_code(self):
        return self.parent.code if self.parent is not None else 0


class PixelReader:
    """DIR and ACC pixels read on demand through a small cache of raster blocks, for the walks that
    leave the channel network: a stem with fewer than four channel tributaries is walked over every
    pixel, so that the tiny tributaries count too.  Columns may be
    unrolled on a periodic grid; a pixel outside the grid reads as nodata."""

    def __init__(self, dir_path, acc_path, grid, blocks_kept=256):
        from collections import OrderedDict
        self.dir_dataset = rasterio.open(dir_path)
        try:
            self.acc_dataset = rasterio.open(acc_path)
        except Exception:
            self.dir_dataset.close()                # not left open when the second open fails
            raise
        self.grid = grid
        self.blocks = OrderedDict()
        self.blocks_kept = blocks_kept

    def close(self):
        self.dir_dataset.close()
        self.acc_dataset.close()

    def _block(self, block_row, block_col):
        key = (block_row, block_col)
        if key in self.blocks:
            self.blocks.move_to_end(key)
            return self.blocks[key]
        window = Window(block_col * RASTER_BLOCK, block_row * RASTER_BLOCK, min(RASTER_BLOCK, self.grid.ncol - block_col * RASTER_BLOCK), min(RASTER_BLOCK, self.grid.nrow - block_row * RASTER_BLOCK))
        pair = (self.dir_dataset.read(1, window=window), self.acc_dataset.read(1, window=window))
        self.blocks[key] = pair
        if len(self.blocks) > self.blocks_kept:
            self.blocks.popitem(last=False)
        return pair

    def _locate(self, row, col):
        if row < 0 or row >= self.grid.nrow:
            return None
        if self.grid.periodic:
            col = col % self.grid.ncol
        elif col < 0 or col >= self.grid.ncol:
            return None
        return self._block(row // RASTER_BLOCK, col // RASTER_BLOCK), row % RASTER_BLOCK, col % RASTER_BLOCK

    def code(self, row, col):
        located = self._locate(row, col)
        if located is None:
            return MERIT_NODATA
        pair, local_row, local_col = located
        return int(pair[0][local_row, local_col])

    def acc(self, row, col):
        located = self._locate(row, col)
        if located is None:
            return 0
        pair, local_row, local_col = located
        return int(pair[1][local_row, local_col])

    def inflows(self, row, col):
        """the neighbours whose flow direction points at (row, col), in the row-major order of the
        3 x 3 neighbourhood, each with its upstream count: [(row, col, acc), ...]"""
        found = []
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                code = self.code(row + drow, col + dcol)
                if DROW[code] == -drow and DCOL[code] == -dcol and not (DROW[code] == 0 and DCOL[code] == 0):
                    found.append((row + drow, col + dcol, self.acc(row + drow, col + dcol)))
        return found


class BasinToCut:
    """what the walks of one basin share: the channel network, the reader for the pixels outside it, the
    outlets of the pieces made so far (a walk stops at another piece's outlet), the pieces in the order
    they were made, and the count above which a tributary cannot be below the channel threshold"""

    def __init__(self, network, reader, grid, channel_threshold_km2, rectangle):
        self.network = network
        self.reader = reader
        self.grid = grid
        self.is_piece_outlet = np.zeros(network.row.size, np.uint8)         # flags on the network pixels
        self.piece_outlet_keys = set()                                     # unrolled keys of every piece outlet
        self.pieces = []
        # a tributary of less than the threshold area holds fewer pixels than this, whatever its latitude.
        # the area the channel mask was made from may be a provider's, whose pixel is smaller than our
        # exact one (MERIT's rgetara, 2.3e-3 below the WGS84 zone), so a tributary just under the threshold could
        # hold a pixel or two more than the bound our pixel gives and outrank the fourth channel tributary unseen.
        # CHANNEL_BOUND_MARGIN makes the bound 1 % larger than that: the walk on the network
        # then gives the answer only where the walk over every pixel would give the same one, and otherwise
        # hands over to it, which costs time and never changes the cut
        smallest_pixel_m2 = float(grid.row_pixel_areas_m2(rectangle[0], rectangle[1] - rectangle[0]).min())
        self.channel_pixels_max = int(math.ceil(CHANNEL_BOUND_MARGIN * channel_threshold_km2 * 1e6 / smallest_pixel_m2))
        # the room of one walk, allocated once for every walk of the basin
        self.mainstem = np.empty(min(network.row.size + 1, 1 << 22), np.int64)
        self.junction_position = np.empty(1 << 20, np.int64)
        self.junction_index = np.empty(1 << 20, np.int64)
        self.junction_acc = np.empty(1 << 20, np.int64)

    def key(self, row, col):
        return int(row) * self.network.stride + int(col)

    def add_piece(self, piece):
        self.pieces.append(piece)
        self.piece_outlet_keys.add(self.key(piece.outlet_row, piece.outlet_col))
        index = self.network.index_of(piece.outlet_row, piece.outlet_col)
        if index >= 0:
            self.is_piece_outlet[index] = 1
        piece.acc_at_outlet = int(self.network.acc[index]) if index >= 0 else self.reader.acc(piece.outlet_row, piece.outlet_col)

    def walk_on_network(self, start_index):
        """the main stem and the junctions found on the channel network; None when the stem has fewer
        than four channel tributaries, or its fourth could be outranked by a tributary below the
        threshold, in which case the walk over every pixel gives the answer"""
        network = self.network
        mainstem = self.mainstem
        junction_position = self.junction_position
        junction_index = self.junction_index
        junction_acc = self.junction_acc
        while True:
            length, junction_count = _walk_mainstem(start_index, network.inflow_ptr, network.inflow_idx, network.acc, self.is_piece_outlet, mainstem, junction_position, junction_index, junction_acc)
            if length >= 0:
                break
            # out of room: the buffers are doubled and the walk run again
            if length == -1:
                self.mainstem = mainstem = np.empty(mainstem.size * 2, np.int64)
            else:
                self.junction_position = junction_position = np.empty(junction_position.size * 2, np.int64)
                self.junction_index = junction_index = np.empty(junction_index.size * 2, np.int64)
                self.junction_acc = junction_acc = np.empty(junction_acc.size * 2, np.int64)
        acc_sorted = np.sort(junction_acc[:junction_count])[::-1]
        if junction_count < TRIBUTARIES_PER_LEVEL or acc_sorted[TRIBUTARIES_PER_LEVEL - 1] < self.channel_pixels_max:
            return None
        stem = [(int(network.row[k]), int(network.col[k])) for k in mainstem[:length]]
        junctions = [(int(junction_position[k]), int(network.row[junction_index[k]]), int(network.col[junction_index[k]]), int(junction_acc[k])) for k in range(junction_count)]
        return stem, junctions

    def walk_on_disk(self, start_row, start_col):
        """the walk over every pixel: from the start upstream, at every pixel the inflow with the largest upstream
        count continues the stem (ties: the first in row-major order); every other inflow that is not
        another piece's outlet is a junction; the stem ends at a source or where it would enter another
        piece"""
        stem = []
        junctions = []
        row = start_row
        col = start_col
        start_key = self.key(start_row, start_col)
        limit = self.grid.nrow * self.grid.ncol
        while True:
            stem.append((row, col))
            if len(stem) > limit:
                raise FlowDivideError("the main stem walk does not end (a cycle?)")
            inflows = self.reader.inflows(row, col)
            best = -1
            for k, (n_row, n_col, n_acc) in enumerate(inflows):
                if best < 0 or n_acc > inflows[best][2]:
                    best = k
            for k, (n_row, n_col, n_acc) in enumerate(inflows):
                if k == best:
                    continue
                n_key = self.key(n_row, n_col)
                if n_key != start_key and n_key in self.piece_outlet_keys:
                    continue
                junctions.append((len(stem) - 1, n_row, n_col, n_acc))
            if best < 0:
                break
            best_key = self.key(inflows[best][0], inflows[best][1])
            if best_key != start_key and best_key in self.piece_outlet_keys:
                break
            row, col = inflows[best][0], inflows[best][1]
        return stem, junctions


def define_pieces_of_one_walk(basin, parent, level, start_row, start_col, downstream_row, downstream_col, no_junction="identity"):
    """The nine pieces (fewer when the walk has fewer junctions) of one walk: from the start upstream
    along the main stem, the four largest tributaries become the even pieces and the stretches between
    them the odd pieces.  `parent` is the piece being cut (None for the basin); the pieces that flowed
    into the parent flow into the stretch that holds their confluence from now on, and the expected
    pixel count of every new piece follows: the count at its outlet less the counts at the outlets of
    the pieces flowing into it.  A walk with no junction: no_junction "identity" makes one child that is
    the whole parent; "none" changes nothing and returns (None, None); "error"
    raises.  Returns (the pieces made, the main stem as a list of (row, col))."""
    network = basin.network
    start_index = network.index_of(start_row, start_col)
    walked = basin.walk_on_network(start_index) if start_index >= 0 else None
    if walked is None:
        walked = basin.walk_on_disk(start_row, start_col)
    stem, junctions = walked
    if parent is not None and not junctions:
        if no_junction == "error":
            raise FlowDivideError("piece %d cannot be cut: its main stem has no junction left, so a smaller capacity cannot be met on this basin" % parent.code)
        if no_junction == "none":
            return None, None
    # the four largest tributaries (ties: the one nearer the outlet, then the tributary pixel first in the scan of a
    # pixel's neighbours, row then column), then back in the order along the stem (ties: the larger, then the scan).
    junctions.sort(key=lambda j: (-j[3], j[0], j[1], j[2]))
    chosen = sorted(junctions[:TRIBUTARIES_PER_LEVEL], key=lambda j: (j[0], -j[3], j[1], j[2]))
    base = parent.code * 10 if parent is not None else 0
    made = []
    stretches = []                                   # (first stem position, odd piece)
    piece = Piece(len(basin.pieces) + 1, base + 1, level, parent, start_row, start_col, downstream_row, downstream_col)
    basin.add_piece(piece)
    made.append(piece)
    stretches.append((0, piece))
    current_odd = piece
    for k, (position, tributary_row, tributary_col, tributary_acc) in enumerate(chosen):
        even = Piece(len(basin.pieces) + 1, base + 2 * (k + 1), level, parent, tributary_row, tributary_col, stem[position][0], stem[position][1])
        even.next_down = current_odd
        even.next_down_at_creation = current_odd
        basin.add_piece(even)
        made.append(even)
        shares_confluence = (k + 1 < len(chosen) and chosen[k + 1][0] == position)
        if not shares_confluence and position + 1 < len(stem):
            odd = Piece(len(basin.pieces) + 1, base + 2 * (k + 1) + 1, level, parent, stem[position + 1][0], stem[position + 1][1], stem[position][0], stem[position][1])
            odd.next_down = current_odd
            odd.next_down_at_creation = current_odd
            basin.add_piece(odd)
            made.append(odd)
            stretches.append((position + 1, odd))
            current_odd = odd
    for piece in made:
        if piece.next_down is not None:
            piece.next_down.inflows.append(piece)
    # what flowed into the parent flows into the stretch holding its confluence pixel
    if parent is not None:
        position_of = {(r, c): k for k, (r, c) in enumerate(stem)}
        for inflow in parent.inflows:
            if (inflow.downstream_row, inflow.downstream_col) not in position_of:
                raise FlowDivideError("piece %d flows into piece %d but not onto its main stem" % (inflow.code, parent.code))
            position = position_of[(inflow.downstream_row, inflow.downstream_col)]
            target = stretches[0][1]
            for first_position, stretch in stretches:
                if first_position <= position:
                    target = stretch
            inflow.next_down = target
            target.inflows.append(inflow)
        parent.inflows = []
        # the parent's own outlet flows on where it did; the new piece 1 takes its place among the inflows
        made[0].next_down_at_creation = parent.next_down_at_creation
        if parent.next_down is not None:
            parent.next_down.inflows = [made[0] if p is parent else p for p in parent.next_down.inflows]
            made[0].next_down = parent.next_down
    for piece in made:
        piece.expected_pixels = piece.acc_at_outlet - sum(inflow.acc_at_outlet for inflow in piece.inflows)
    return made, stem


def _label_basin_pieces(dir_path, bsn_path, grid, basin_id, rectangle, piece_outlet_global_sorted, piece_code_sorted, out_piece_path, tile, tag):
    """every pixel of the basin labelled with the deepest piece whose outlet its path reaches first (the
    three tile passes of FD1.3 with the piece outlets as terminals, restricted to the basin's pixels);
    the piece raster (UInt16) over the basin's window; the pixel count and rectangle of every code"""
    row_min, row_max, col_min, col_max = rectangle
    tiles = tiles_of_grid(grid, tile, row_min, row_max, col_min, col_max)
    records = TilePassRecords()
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(bsn_path) as bsn_dataset:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            mask = read_block(bsn_dataset, row0, nrow, col0, ncol, grid.periodic, 0)
            order, land_count, taken = tile_topological_order(halo)
            label = np.zeros(nrow * ncol, np.uint32)
            max_edge = 2 * (nrow + ncol) + 8
            exit_global = np.empty(max_edge, np.int64)
            exit_destination = np.empty(max_edge, np.int64)
            status, exit_count = _label_tile_backwards(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, piece_outlet_global_sorted, piece_code_sorted,
                                                       mask, np.uint32(basin_id), label, exit_global, exit_destination, np.zeros(1, np.int64), 0)
            if status != 0:
                raise FlowDivideError("piece labelling pass A failed with status %d in the tile at row %d col %d of basin %d" % (status, row0, col0, basin_id))
            inlet_local = np.empty(max_edge, np.int32)
            inlet_global = np.empty(max_edge, np.int64)
            inlet_count = tile_inlets(halo, row0, col0, grid.nrow, grid.ncol, grid.periodic, inlet_local, inlet_global)
            inlet_label = np.empty(inlet_count, np.int64)
            inlet_exit_global = np.empty(inlet_count, np.int64)
            _inlet_labels(inlet_local, inlet_count, label, exit_global, inlet_label, inlet_exit_global)
            # an inlet of another basin carries label 0 and no exit; it is never asked for
            records.exit_global.append(exit_global[:exit_count].copy())
            records.exit_destination.append(exit_destination[:exit_count].copy())
            records.inlet_local.append(inlet_local[:inlet_count].copy())
            records.inlet_global.append(inlet_global[:inlet_count].copy())
            records.inlet_exit_global.append(inlet_exit_global)
            records.inlet_label.append(inlet_label)
            records.tile_exit_offsets.append(records.tile_exit_offsets[-1] + exit_count)
            records.tile_inlet_offsets.append(records.tile_inlet_offsets[-1] + inlet_count)
            del halo, mask, order, label
    records.concatenate()
    status, inlet_of_exit, next_exit = _link_exits_to_inlets(records.exit_global, records.exit_destination, records.inlet_global, records.inlet_exit_global)
    if status != 0:
        raise FlowDivideError("the exit graph of basin %d cannot be linked (status %d)" % (basin_id, status))
    status, label_of_exit = _resolve_labels_over_exit_graph(next_exit, inlet_of_exit, records.inlet_label, records.exit_global.size)
    if status != 0:
        raise FlowDivideError("the exit graph of basin %d cannot be resolved (status %d)" % (basin_id, status))
    code_limit = int(piece_code_sorted.max()) + 1 if piece_code_sorted.size else 1
    count = np.zeros(code_limit, np.int64)
    p_row_min = np.full(code_limit, np.iinfo(np.int32).max, np.int32)
    p_row_max = np.full(code_limit, -1, np.int32)
    p_col_min = np.full(code_limit, np.iinfo(np.int32).max, np.int32)
    p_col_max = np.full(code_limit, -1, np.int32)
    p_col_min_shift = np.full(code_limit, np.iinfo(np.int32).max, np.int32)
    p_col_max_shift = np.full(code_limit, -1, np.int32)
    window_nrow = row_max - row_min
    window_ncol = col_max - col_min
    piece_transform = grid.transform * rasterio.Affine.translation(col_min, row_min)
    profile = {"driver": "GTiff", "dtype": "uint16", "count": 1, "width": window_ncol, "height": window_nrow, "crs": grid.crs,
               "transform": piece_transform, "nodata": 0, "tiled": True, "blockxsize": RASTER_BLOCK, "blockysize": RASTER_BLOCK,
               "compress": "DEFLATE", "BIGTIFF": "YES", "NUM_THREADS": GDAL_WRITE_THREADS}
    temporary = out_piece_path + ".partial.tif"
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(bsn_path) as bsn_dataset, rasterio.open(temporary, "w", **profile) as piece_out:
        for tile_index, (row0, nrow, col0, ncol) in enumerate(tiles):
            halo = read_halo_tile(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            mask = read_block(bsn_dataset, row0, nrow, col0, ncol, grid.periodic, 0)
            order, land_count, taken = tile_topological_order(halo)
            label = np.zeros(nrow * ncol, np.uint32)
            first = records.tile_exit_offsets[tile_index]
            last = records.tile_exit_offsets[tile_index + 1]
            status, exit_count = _label_tile_backwards(halo, order, row0, col0, grid.nrow, grid.ncol, grid.periodic, piece_outlet_global_sorted, piece_code_sorted,
                                                       mask, np.uint32(basin_id), label, records.exit_global[first:last], np.zeros(1, np.int64), label_of_exit[first:last], 1)
            if status != 0 or exit_count != last - first:
                raise FlowDivideError("piece labelling pass C failed with status %d in the tile at row %d col %d of basin %d" % (status, row0, col0, basin_id))
            _collect_basin_statistics(label, nrow, ncol, row0, col0, grid.ncol, grid.periodic, count, p_row_min, p_row_max, p_col_min, p_col_max, p_col_min_shift, p_col_max_shift)
            piece_out.write(label.reshape(nrow, ncol).astype(np.uint16), 1, window=Window(col0 - col_min, row0 - row_min, ncol, nrow))
            del halo, mask, order, label
    # the raster stays under its partial name: the caller publishes it once its counts have passed
    # the tiles were laid out on the basin's window, which is already unrolled where the basin lies
    # across the seam, so the columns collected here are in that frame and need no unrolling
    return count, p_row_min.astype(np.int64), p_row_max.astype(np.int64), p_col_min.astype(np.int64), p_col_max.astype(np.int64)


CUT_PIECE_TABLE_COLUMNS = ["piece", "level", "sub_of", "flows_into", "used", "region", "outlet_row", "outlet_col", "parent_inlet_row", "parent_inlet_col", "acc_at_outlet",
                       "aca_at_outlet_km2", "expected_grid_count", "labelled_grid_count", "row_min", "row_max", "col_min", "col_max",
                       "deg_row_min", "deg_row_max", "deg_col_min", "deg_col_max", "window_grid_count", "window_over_int32", "minlon", "minlat", "maxlon", "maxlat"]


def fd1_5_pfafstetter_cut(dir_path, bsn_path, str_path, acc_path, basin_row, grid, capacity_pixels, piece_table_path, piece_raster_path, mainstem_path,
                          codes_path=None, region_code=0, levels=None, aca_path=None, aca_to_km2=1e-6, channel_threshold_km2=1.0, tile=WORK_TILE_DEFAULT, tag="fd1.5"):
    """One basin over the capacity cut into Pfafstetter pieces and its pieces merged into regions.
    levels: how deep the cut goes.  An integer 2 .. 4 cuts every piece of every level to that depth
    before anything is labelled (codes 1..9, 11..99, 111..999, 1111..9999; a piece
    whose stem has no junction gets one child that is the whole piece), and the pieces used are the
    shallowest that fit the capacity, a piece of the deepest level used whatever its window; None
    decides the depth here: a piece expected to hold more than half the capacity is cut before
    labelling, and what is still over the capacity after labelling is cut again, down to four levels.
    region_code: the code of the basin's group, so that the regions can be numbered code * 100 + n.
    Writes the piece table (the columns of pfafstetter_pieces,
    every piece, used or not, numbered in the order the pieces were made), the piece raster (the
    deepest piece of every pixel of the basin, UInt16, over the basin's window), the main stem of the
    first walk as pixel rows and columns, and, when codes_path is given, the Pfafstetter code of every
    piece id (for the figures; the piece table carries no codes).  Returns the piece table."""
    started = time.time()
    if levels is not None and not 2 <= int(levels) <= 4:
        raise FlowDivideError("levels must be 2, 3 or 4, or None for the depth to be decided here; got %r" % levels)
    basin_id = int(basin_row["basin_id"])
    rectangle = (int(basin_row["basin_row_min"]), int(basin_row["basin_row_max"]), int(basin_row["basin_col_min"]), int(basin_row["basin_col_max"]))
    window = grid.window_of_rectangle(*rectangle)
    log(tag, "basin %d: %d pixels, window %d x %d blocks, %.2f x the capacity; %s" % (
        basin_id, int(basin_row["basin_grid_count"]), (window[1] - window[0]) // grid.block_pixels, (window[3] - window[2]) // grid.block_pixels,
        grid.window_pixels(*rectangle) / capacity_pixels, "every piece cut to %d levels" % levels if levels else "the depth decided by the windows"))
    # [a] the channel network of the basin, from the work tiles of its window
    rows = []
    cols = []
    codes = []
    accs = []
    with rasterio.open(dir_path) as dir_dataset, rasterio.open(bsn_path) as bsn_dataset, rasterio.open(str_path) as str_dataset, rasterio.open(acc_path) as acc_dataset:
        # the flow directions too, on the grid and uint8, before any code of them is looked up: the network indexes
        # its direction tables with them (a UInt16 256 would read past a table of 256)
        for dataset, path in ((dir_dataset, dir_path), (bsn_dataset, bsn_path), (str_dataset, str_path), (acc_dataset, acc_path)):
            check_raster_on_the_grid(dataset, grid, path)
        if dir_dataset.dtypes[0] != "uint8":
            raise FlowDivideError("the flow directions %s are %s, not uint8" % (dir_path, dir_dataset.dtypes[0]))
        for row0, nrow, col0, ncol in tiles_of_grid(grid, tile, *window):
            bsn_tile = read_block(bsn_dataset, row0, nrow, col0, ncol, grid.periodic, 0)
            str_tile = read_block(str_dataset, row0, nrow, col0, ncol, grid.periodic, 0)
            selected = (bsn_tile == basin_id) & (str_tile != 0)
            del str_tile
            if not selected.any():
                continue
            local_row, local_col = np.nonzero(selected)
            dir_tile = read_block(dir_dataset, row0, nrow, col0, ncol, grid.periodic, MERIT_NODATA)
            check_merit_flow_directions(dir_tile, "%s rows %d .. %d" % (dir_path, row0, row0 + nrow - 1))
            acc_tile = read_block(acc_dataset, row0, nrow, col0, ncol, grid.periodic, 0)
            rows.append((local_row + row0).astype(np.int32))
            cols.append((local_col + col0).astype(np.int32))
            codes.append(dir_tile[local_row, local_col])
            accs.append(acc_tile[local_row, local_col].astype(np.int64))
            del bsn_tile, dir_tile, acc_tile, selected
    if rows:
        network = ChannelNetwork(np.concatenate(rows), np.concatenate(cols), np.concatenate(codes), np.concatenate(accs), grid)
    else:
        # a threshold above the whole basin's area: no channel pixel, every stem walked over the pixels
        network = ChannelNetwork(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.uint8), np.zeros(0, np.int64), grid)
    del rows, cols, codes, accs
    outlet_row = int(basin_row["outlet_row"])
    outlet_col = int(basin_row["outlet_col"]) if int(basin_row["outlet_col"]) >= rectangle[2] else int(basin_row["outlet_col"]) + grid.ncol
    if network.row.size:
        if network.index_of(outlet_row, outlet_col) < 0:
            raise FlowDivideError("the outlet of basin %d is not a channel pixel" % basin_id)
        if int((network.down < 0).sum()) != 1:
            raise FlowDivideError("the channel network of basin %d has %d ends; it should have exactly one, the outlet" % (basin_id, int((network.down < 0).sum())))
    log(tag, "basin %d: %d channel pixels in the network" % (basin_id, network.row.size))
    reader = PixelReader(dir_path, acc_path, grid)
    # closed whatever stops the cut, not on the path that succeeds only
    try:
        basin = BasinToCut(network, reader, grid, channel_threshold_km2, rectangle)
        # [b] the pieces: the first level, then deeper
        level1, mainstem_level1 = define_pieces_of_one_walk(basin, None, 1, outlet_row, outlet_col, -1, -1)

        def cut_deeper(piece, no_junction):
            children, _ = define_pieces_of_one_walk(basin, piece, piece.level + 1, piece.outlet_row, piece.outlet_col, piece.downstream_row, piece.downstream_col, no_junction)
            if children is not None:
                piece.children = children
            return children

        def label_and_check():
            """the labelling pass over the basin: every pixel gets its deepest piece; the counts and rectangles
            of every piece, checked against the upstream counts"""
            deepest = [piece for piece in basin.pieces if not piece.children]
            piece_global = np.asarray([p.outlet_row * grid.ncol + (p.outlet_col % grid.ncol) for p in deepest], np.int64)
            piece_label = np.asarray([p.id for p in deepest], np.int64)
            sort = np.argsort(piece_global)
            count, p_row_min, p_row_max, p_col_min, p_col_max = _label_basin_pieces(dir_path, bsn_path, grid, basin_id, window, piece_global[sort], piece_label[sort], piece_raster_path, tile, tag)
            for piece in deepest:
                piece.pixels = int(count[piece.id])
                piece.rectangle = (int(p_row_min[piece.id]), int(p_row_max[piece.id]), int(p_col_min[piece.id]), int(p_col_max[piece.id]))
            # the pixels of a parent are those of its children, and so is its expected count; its rectangle the union of theirs
            for level in (3, 2, 1):
                for piece in basin.pieces:
                    if piece.level == level and piece.children:
                        piece.pixels = sum(child.pixels for child in piece.children)
                        piece.expected_pixels = sum(child.expected_pixels for child in piece.children)
                        piece.rectangle = (min(c.rectangle[0] for c in piece.children), max(c.rectangle[1] for c in piece.children),
                                           min(c.rectangle[2] for c in piece.children), max(c.rectangle[3] for c in piece.children))
            labelled = sum(piece.pixels for piece in deepest)
            if labelled != int(basin_row["basin_grid_count"]):
                raise FlowDivideError("basin %d: %d pixels labelled with a piece, %d in the basin" % (basin_id, labelled, int(basin_row["basin_grid_count"])))
            # the count of every piece must equal what the upstream counts say: the count at its outlet less
            # the counts at the outlets of the pieces flowing into it
            for piece in basin.pieces:
                if piece.expected_pixels != piece.pixels:
                    raise FlowDivideError("basin %d piece %d (code %d): %d pixels labelled, %d expected from the upstream counts" % (basin_id, piece.id, piece.code, piece.pixels, piece.expected_pixels))
            return deepest

        if levels is not None:
            # a fixed depth: every piece of every level cut to the depth asked, one labelling pass
            for level in range(2, int(levels) + 1):
                for piece in [p for p in basin.pieces if p.level == level - 1]:
                    cut_deeper(piece, "identity")
            deepest = label_and_check()
            log(tag, "basin %d: %d pieces on %d levels" % (basin_id, len(basin.pieces), max(p.level for p in basin.pieces)))
        else:
            # a piece holding more than half the capacity in pixels is cut before any labelling: its window is
            # larger than its pixel count, so it is most unlikely to fit; the labelling pass then decides the rest
            pending = [piece for piece in level1 if piece.expected_pixels > capacity_pixels // 2]
            while pending:
                piece = pending.pop(0)
                if piece.level >= 4:
                    continue
                children = cut_deeper(piece, "none")
                if children is not None:
                    pending += [child for child in children if child.expected_pixels > capacity_pixels // 2]
            rounds = 0
            while True:
                rounds += 1
                deepest = label_and_check()
                still_over = [piece for piece in deepest if grid.window_pixels(*piece.rectangle) > capacity_pixels]
                log(tag, "basin %d round %d: %d pieces on %d levels, %d of the deepest still over the capacity" % (basin_id, rounds, len(basin.pieces), max(p.level for p in basin.pieces), len(still_over)))
                if not still_over:
                    break
                deepest_over = max(still_over, key=lambda p: p.level)
                if rounds >= 4 or deepest_over.level >= 4:
                    raise FlowDivideError("basin %d: piece %d of level %d still exceeds the capacity after the fourth level of the cut (the deepest this package labels, codes of four digits); the capacity is too small for this basin"
                                          % (basin_id, deepest_over.code, deepest_over.level))
                for piece in still_over:
                    cut_deeper(piece, "error")
    finally:
        reader.close()
    # the old markers go before the new raster is published: a rerun that fails after this leaves no marker on a
    # raster whose table is not written
    for path in (piece_table_path, piece_raster_path):
        if os.path.exists(path + ".done"):
            os.remove(path + ".done")
    publish(piece_raster_path + ".partial.tif", piece_raster_path)
    # [c] the used pieces: the shallowest that fit, in hierarchical code order (1, 2, 31, 32, .., 4, ..); a
    #     piece of the deepest level is used whatever its window, and counted
    sequence = []
    still_over = []

    def take(piece):
        if grid.window_pixels(*piece.rectangle) <= capacity_pixels or not piece.children:
            if grid.window_pixels(*piece.rectangle) > capacity_pixels:
                still_over.append(piece)
            piece.used = True
            sequence.append(piece)
            return
        for child in sorted(piece.children, key=lambda p: p.code):
            take(child)

    for piece in sorted(level1, key=lambda p: p.code):
        take(piece)
    # [d] every used piece flows into a used piece: the deepest piece its outlet flows into (a used piece's
    #     outlet is the outlet of its first sub-piece, whichever level that reaches), then the used ancestor
    used_next_down = {}
    for piece in sequence:
        deepest_piece = piece
        while deepest_piece.children:
            deepest_piece = min(deepest_piece.children, key=lambda p: p.code)
        target = deepest_piece.next_down
        if target is None:
            used_next_down[piece.id] = None
            continue
        while not target.used:
            target = target.parent
        used_next_down[piece.id] = target
    # [e] consecutive used pieces merged while the merged window fits; the region graph; the numbering
    regions = []
    for piece in sequence:
        if regions:
            current = regions[-1]
            merged = (min(current["rectangle"][0], piece.rectangle[0]), max(current["rectangle"][1], piece.rectangle[1]),
                      min(current["rectangle"][2], piece.rectangle[2]), max(current["rectangle"][3], piece.rectangle[3]))
            if grid.window_pixels(*merged) <= capacity_pixels:
                current["pieces"].append(piece)
                current["rectangle"] = merged
                current["pixels"] += piece.pixels
                continue
        regions.append({"pieces": [piece], "rectangle": piece.rectangle, "pixels": piece.pixels})
    region_of_id = {}
    for index, region in enumerate(regions):
        for piece in region["pieces"]:
            region_of_id[piece.id] = index
    edges = set()
    for piece in sequence:
        if used_next_down[piece.id] is not None:
            source = region_of_id[piece.id]
            target = region_of_id[used_next_down[piece.id].id]
            if source != target:
                edges.add((source, target))
    order_level = [0] * len(regions)
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        if rounds > len(regions) + 1:
            raise FlowDivideError("the region graph of basin %d has a cycle" % basin_id)
        for source, target in edges:
            if order_level[target] < order_level[source] + 1:
                order_level[target] = order_level[source] + 1
                changed = True
    numbered = sorted(range(len(regions)), key=lambda i: (order_level[i], i))
    for number, index in enumerate(numbered, start=1):
        for piece in regions[index]["pieces"]:
            piece.region_number = number
    # [f] the tables: the upstream area at every outlet comes from the area raster when it is still on
    #     disk, -1 otherwise
    aca_at_outlet = {}
    if aca_path is not None and os.path.exists(aca_path):
        with rasterio.open(aca_path) as aca_dataset:
            check_raster_on_the_grid(aca_dataset, grid, aca_path)
            for piece in basin.pieces:
                aca_at_outlet[piece.id] = float(aca_dataset.read(1, window=Window(piece.outlet_col % grid.ncol, piece.outlet_row, 1, 1))[0, 0]) * aca_to_km2
                # a piece outlet is a land pixel of the basin: its area is positive and finite
                if not (0.0 < aca_at_outlet[piece.id] < ACCUMULATION_MAX):
                    raise FlowDivideError("basin %d: the upstream area at the outlet of piece %d reads %g"
                                          % (basin_id, piece.id, aca_at_outlet[piece.id]))
    records = []
    for piece in basin.pieces:
        if piece.used:
            flows_into = used_next_down[piece.id]
        elif piece.level >= 2 and piece.code % 10 == 1 and piece.next_down is not None:
            # sub-piece 1 flows where its parent flows: the piece of its own level that holds the parent's
            # downstream pixel
            flows_into = piece.next_down
            while flows_into.level > piece.level:
                flows_into = flows_into.parent
        else:
            flows_into = piece.next_down_at_creation
        piece_window = grid.window_of_rectangle(*piece.rectangle)
        window_pixels = (piece_window[1] - piece_window[0]) * (piece_window[3] - piece_window[2])
        box = grid.pixel_box_lon_lat(*piece_window)
        records.append({"piece": piece.id, "level": piece.level, "sub_of": piece.parent.id if piece.parent is not None else 0,
                        "flows_into": flows_into.id if flows_into is not None else 0, "used": int(piece.used),
                        "region": region_code * 100 + piece.region_number if piece.used else 0,
                        "outlet_row": piece.outlet_row, "outlet_col": piece.outlet_col % grid.ncol,
                        "parent_inlet_row": piece.downstream_row, "parent_inlet_col": piece.downstream_col % grid.ncol if piece.downstream_col >= 0 else -1,
                        "acc_at_outlet": piece.acc_at_outlet, "aca_at_outlet_km2": aca_at_outlet.get(piece.id, -1.0),
                        "expected_grid_count": piece.expected_pixels, "labelled_grid_count": piece.pixels,
                        "row_min": piece.rectangle[0], "row_max": piece.rectangle[1], "col_min": piece.rectangle[2], "col_max": piece.rectangle[3],
                        "deg_row_min": piece_window[0] // grid.block_pixels, "deg_row_max": piece_window[1] // grid.block_pixels,
                        "deg_col_min": piece_window[2] // grid.block_pixels, "deg_col_max": piece_window[3] // grid.block_pixels,
                        "window_grid_count": window_pixels, "window_over_int32": int(window_pixels > capacity_pixels),
                        "minlon": box[0], "minlat": box[1], "maxlon": box[2], "maxlat": box[3]})
    table = pd.DataFrame(records)[CUT_PIECE_TABLE_COLUMNS]
    for column in ("minlon", "minlat", "maxlon", "maxlat"):
        table[column] = table[column].map(lambda value: "%.4f" % value)      # four decimals
    write_table(table, piece_table_path, float_format="%.6f")
    stem = pd.DataFrame({"row": [r for r, _ in mainstem_level1], "col": [c % grid.ncol for _, c in mainstem_level1]})
    stem["lon"], stem["lat"] = grid.pixel_centre_lon_lat(stem["row"].to_numpy(np.int64), stem["col"].to_numpy(np.int64))
    write_table(stem, mainstem_path, float_format="%.7f")
    if codes_path is not None:
        write_table(pd.DataFrame({"piece": [p.id for p in basin.pieces], "code": [p.code for p in basin.pieces], "level": [p.level for p in basin.pieces]}), codes_path)
    used = table[table["used"] == 1]
    log(tag, "basin %d: %d pieces defined, %d used, %d regions, %d pieces of the deepest level still over the capacity, %d seconds" % (
        basin_id, len(table), len(used), len(regions), len(still_over), int(time.time() - started)))
    if still_over:
        raise FlowDivideError("basin %d: %d pieces of the deepest level still exceed the capacity; a deeper cut is needed (levels=%s); the tables are written for inspection" % (basin_id, len(still_over), levels))
    if len(regions) > 99:
        raise FlowDivideError("basin %d: %d regions, but a Level-03 code numbers at most 99" % (basin_id, len(regions)))
    # the piece table and the piece raster are marked done only now, after every check: fd1.5.regions_final refuses
    # them without their markers
    summary = "basin %d: %d pieces, %d used, %d regions" % (basin_id, len(table), len(used), len(regions))
    fd_tables.write_done_marker(piece_table_path, summary)
    fd_tables.write_done_marker(piece_raster_path, summary)
    return table

def _joined_codes(kept, absorbed):
    """(level2_code, level1_code) of two joined regions: a code while both share it, -1 once two different ones met,
    and a Level-01 code 0 (not known yet) takes the other side's"""
    level2 = kept.level2_code if kept.level2_code == absorbed.level2_code else -1
    if kept.level1_code == -1 or absorbed.level1_code == -1:
        level1 = -1
    elif kept.level1_code == 0:
        level1 = absorbed.level1_code
    elif absorbed.level1_code != 0 and kept.level1_code != absorbed.level1_code:
        level1 = -1
    else:
        level1 = kept.level1_code
    return level2, level1


class RegionBuild:
    """one region while it is being built: whole basins, or the pieces of one cut basin"""

    def __init__(self, region_id, level3, kind):
        self.region_id = region_id
        self.level3 = level3
        self.level3_lead = level3           # the Level-03 code the build id was made from; never cleared by a join
        self.level3_members = [level3] if level3 > 0 else []   # every Level-03 unit in this region, in code order:
                                        # what the published four-digit id of a divided unit is read from, and a
                                        # column of the region table
        self.level2_code = level3 // 10     # the Level-02 unit, 0 when the region holds parts of more than one
        self.level1_code = level3 // 100    # the continent, 0 when the region holds parts of more than one
        self.kind = kind                    # 1 whole basins of a Level-03 unit, 2 a part of one, 3 island, 4 seam, 5 pieces of a cut basin
        self.basins = []                    # basin indices (0-based rows of the basin table)
        self.cut_basin_id = 0
        self.piece_codes = []
        self.rectangle = None               # (row_min, row_max, col_min, col_max) of its members, unrolled
        self.pixels = 0
        self.area_km2 = 0.0
        self.build_phase = 2                # where the list holds it (see _regions_of_the_level2_units):
                                            # 1 a unit or group that is one region as it is, made in the walk over the
                                            # groups; 2 a part of a divided one, or the whole basins of a unit that holds
                                            # a cut basin, made after that walk

    def window(self, grid):
        return grid.window_of_rectangle(*self.rectangle)

    def window_pixels(self, grid):
        return grid.window_pixels(*self.rectangle)


def _rectangle_union(first, second):
    return (min(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), max(first[3], second[3]))


def _windows_touch(grid, first, second):
    """do the two block windows share a block or lie against one another"""
    block = grid.block_pixels
    a = [v // block for v in first]
    b = [v // block for v in second]
    if a[0] > b[1] or b[0] > a[1]:                 # the block past the last one is where a neighbour starts
        return False
    if a[2] > b[3] or b[2] > a[3]:
        return False
    return True


def _window_contains(first, second):
    return first[0] <= second[0] and first[1] >= second[1] and first[2] <= second[2] and first[3] >= second[3]


def _rectangle_of_rows(rect_rows):
    """the rectangle that holds all the rectangles of an (n, 4) array"""
    return (int(rect_rows[:, 0].min()), int(rect_rows[:, 1].max()), int(rect_rows[:, 2].min()), int(rect_rows[:, 3].max()))


def _box_key(rectangle):
    """the order of two rectangles that are otherwise equally good: further north, then further west, then
    the smaller row_max, then the smaller col_max"""
    return (rectangle[0], rectangle[2], rectangle[1], rectangle[3])


def _split_basins_along_block_lines(grid, basins_rect, basins_land, member_indices, capacity_pixels, land_cut_above, land_merge_below, parts_available, what, tag):
    """A set of whole basins divided into parts along block lines until every part's window fits the
    capacity: the line, across either side of the window, is the one that makes the larger of the two
    resulting windows smallest (ties: the more even land); a basin goes to the side that holds the
    centre of its own rectangle.  When every window fits, a part holding more land than land_cut_above
    is cut again for the land, along the line that balances the land best among the lines that leave
    both sides within the capacity and above land_merge_below.  Afterwards a part with less land than
    land_merge_below is merged back into the touching part that gives the smallest window, while that
    window fits and the land stays under land_cut_above.  Returns a list of parts, each an array of
    member indices."""
    members = np.asarray(member_indices, np.int64)
    rect = basins_rect[members]
    land = basins_land[members]
    centre_col = (rect[:, 2] + rect[:, 3] - 1) // 2            # the centre pixel: col_max, row_max are one past
    centre_row = (rect[:, 0] + rect[:, 1] - 1) // 2
    if members.size < 1 or parts_available < 1:
        raise FlowDivideError("%s: %d basins and %d region numbers left" % (what, members.size, parts_available))
    part_of = np.zeros(members.size, np.int64)
    part_count = 1
    settled = set()
    block = grid.block_pixels
    for _round in range(2 * parts_available):
        part_land = np.zeros(part_count, np.int64)
        part_rect = []
        over = []
        for part in range(part_count):
            mask = part_of == part
            part_land[part] = land[mask].sum()
            part_rect.append(_rectangle_of_rows(rect[mask]))
            if grid.window_pixels(*part_rect[part]) > capacity_pixels:
                over.append(part)
        cutting_for_the_window = bool(over)
        if cutting_for_the_window:
            part = over[0]
        else:
            # the part with the most land above the limit, the first of them when two hold the same
            candidates = [(-int(part_land[p]), p) for p in range(part_count) if p not in settled and part_land[p] > land_cut_above]
            if not candidates:
                break
            part = min(candidates)[1]
        if part_count >= parts_available:
            if not cutting_for_the_window:
                break                                          # no numbers left to balance with; the windows are what matter
            raise FlowDivideError("%s needs more than %d parts; a Level-03 code numbers at most 99 regions" % (what, parts_available))
        mask = part_of == part
        part_rows = rect[mask]
        part_land_rows = land[mask]
        this_rect = part_rect[part]
        best = None
        for axis in (0, 1):
            # the members sorted by the centre along this axis: a line then splits the sorted order at one
            # position, and the rectangles and the land of the two sides come from prefix and suffix extrema
            centres = centre_col[mask] if axis == 0 else centre_row[mask]
            sorted_order = np.argsort(centres, kind="stable")
            sorted_centres = centres[sorted_order]
            sorted_rows = part_rows[sorted_order]
            sorted_land = part_land_rows[sorted_order]
            prefix_min = np.minimum.accumulate(sorted_rows[:, [0, 2]], axis=0)
            prefix_max = np.maximum.accumulate(sorted_rows[:, [1, 3]], axis=0)
            suffix_min = np.minimum.accumulate(sorted_rows[::-1, [0, 2]], axis=0)[::-1]
            suffix_max = np.maximum.accumulate(sorted_rows[::-1, [1, 3]], axis=0)[::-1]
            prefix_land = np.cumsum(sorted_land)
            total_land = int(prefix_land[-1])
            first_block = (this_rect[2] if axis == 0 else this_rect[0]) // block
            last_block = ((this_rect[3] if axis == 0 else this_rect[1]) - 1) // block     # the last block inside
            for line in range(first_block + 1, last_block + 1):
                boundary = line * block
                split = int(np.searchsorted(sorted_centres, boundary, side="left"))    # members [0, split) lie before the line
                if split == 0 or split == sorted_centres.size:
                    continue
                first_rect = (int(prefix_min[split - 1, 0]), int(prefix_max[split - 1, 0]), int(prefix_min[split - 1, 1]), int(prefix_max[split - 1, 1]))
                second_rect = (int(suffix_min[split, 0]), int(suffix_max[split, 0]), int(suffix_min[split, 1]), int(suffix_max[split, 1]))
                sizes = (grid.window_pixels(*first_rect), grid.window_pixels(*second_rect))
                first_land = int(prefix_land[split - 1])
                second_land = total_land - first_land
                imbalance = abs(first_land - second_land)
                if cutting_for_the_window:
                    key = (max(sizes), imbalance)
                else:
                    if sizes[0] > capacity_pixels or sizes[1] > capacity_pixels:
                        continue
                    if first_land < land_merge_below or second_land < land_merge_below:
                        continue
                    key = (imbalance, max(sizes))
                if best is None or key < best[0]:
                    best = (key, axis, boundary)
        if best is None:
            if not cutting_for_the_window:
                settled.add(part)
                continue
            raise FlowDivideError("%s: no block line has basins of part %d (%d pixels in its window) on both sides" % (what, part, grid.window_pixels(*this_rect)))
        _, axis, boundary = best
        centres = centre_col if axis == 0 else centre_row
        moving = mask & (centres >= boundary)
        part_of[moving] = part_count
        log(tag, "  %s: part %d cut for the %s along a %s line at block %d; %d basins move to part %d" % (what, part, "window" if cutting_for_the_window else "land", "column" if axis == 0 else "row", boundary // block, int(moving.sum()), part_count))
        part_count += 1
    # a part that holds too little land is merged back into a touching part
    if part_count > 1 and land_merge_below > 0:
        alive = {part: True for part in range(part_count)}
        part_rect = {}
        part_land = {}
        for part in range(part_count):
            mask = part_of == part
            part_rect[part] = _rectangle_of_rows(rect[mask])
            part_land[part] = int(land[mask].sum())
        joins = {}
        cannot = set()
        while sum(alive.values()) > 1:
            # the least land first; two parts of the same size are separated by where they lie
            small = [(part_land[p], _box_key(part_rect[p]), p) for p in alive if alive[p] and p not in cannot and part_land[p] < land_merge_below]
            if not small:
                break
            smallest = min(small)[2]
            host = None
            for p in alive:
                if not alive[p] or p == smallest or not _windows_touch(grid, grid.window_of_rectangle(*part_rect[p]), grid.window_of_rectangle(*part_rect[smallest])):
                    continue
                merged = _rectangle_union(part_rect[p], part_rect[smallest])
                pixels = grid.window_pixels(*merged)
                if pixels > capacity_pixels or part_land[p] + part_land[smallest] > land_cut_above:
                    continue
                key = (pixels, _box_key(part_rect[p]))
                if host is None or key < host[0]:
                    host = (key, p, merged)
            if host is None:
                cannot.add(smallest)
                continue
            _, p, merged = host
            log(tag, "  %s: part %d (%d pixels of land) joins part %d" % (what, smallest, part_land[smallest], p))
            part_rect[p] = merged
            part_land[p] += part_land[smallest]
            alive[smallest] = False
            joins[smallest] = p
            cannot = set()
        final = np.arange(part_count)
        for part in range(part_count):
            target = part
            while target in joins:
                target = joins[target]
            final[part] = target
        part_of = final[part_of]
    parts = []
    for part in sorted(set(part_of.tolist())):
        parts.append(members[part_of == part])
    return parts


# The rule the merge follows, in the step's arguments so that a change of the rule re-runs the step instead
# of reusing its marker.  Of the partners a region may take, the one that:
#   window-min    leaves the smallest joined window (the globe comes out as 107
#                 regions at 1400 square degrees, the coded ones 28.74% land in their windows)
#   window-max    fills the capacity furthest, "filling" read as the window (104
#                 regions and 27.73% land, so a fuller window is mostly emptier space)
#   land-max      brings the most land into the region, the window only the constraint
#   density-max   leaves the most land per pixel of the joined window
# The environment variable is for measuring the four on the real grid; the published runs take the default.
MERGE_RULE = os.environ.get("FLOWDIVIDE_MERGE_RULE", "land-max")
# Which joins may cross a Level-02 unit.  "leftovers-only" (the default): only in a second pass, and
# only for a region under a quarter of the capacity.  "from-the-start": any two touching coded regions may
# join as soon as the window fits.  Measured: the unit restriction refused 2,018,232 pairs where
# the capacity refused 94, so this is what decides how many regions there are.
MERGE_ACROSS = os.environ.get("FLOWDIVIDE_MERGE_ACROSS", "leftovers-only")
if MERGE_ACROSS not in ("leftovers-only", "from-the-start"):
    raise SystemExit("FLOWDIVIDE_MERGE_ACROSS is leftovers-only or from-the-start, not %r" % MERGE_ACROSS)
if MERGE_RULE not in ("window-min", "window-max", "land-max", "density-max"):
    raise SystemExit("FLOWDIVIDE_MERGE_RULE is window-min, window-max, land-max or density-max, not %r" % MERGE_RULE)
# How the regions are numbered in the tables and the masks.  "two-or-four-digits":
# a region that is a whole Level-02 unit takes the two digits of that unit; a part of a divided unit takes four
# digits -- the unit's two, then the last digit of the smallest and of the largest Level-03 code in it, so that
# the id names the run of units it holds (9112 is {911, 912}, 5699 is {569} alone); the groups astride the
# antimeridian become 10001, 10002 and an island group 20000 + its Level-01 region * 100 + n; the build ids
# (level3 * 100 + n) are kept when the partition has a divided Level-03 unit, a part whose codes are not a run,
# or a region of pieces.  "as-built": always the build ids.  The rule is in the step's arguments, so that
# changing it reruns the step instead of republishing the old numbering.
REGION_ID_RULE = os.environ.get("FLOWDIVIDE_REGION_ID_RULE", "two-or-four-digits")
if REGION_ID_RULE not in ("two-or-four-digits", "as-built"):
    raise SystemExit("FLOWDIVIDE_REGION_ID_RULE is two-or-four-digits or as-built, not %r" % REGION_ID_RULE)
SEAM_GROUP_ID_BASE = 100000             # fd1.4: a basin group astride the antimeridian
ISLAND_GROUP_ID_BASE = 200000           # fd1.4: an island basin group
SEAM_PUBLISHED_ID_BASE = 10000          # published: 10001, 10002
ISLAND_PUBLISHED_ID_BASE = 20000        # published: 20000 + the Level-01 region * 100 + the number within it
PUBLISHED_ID_LIMIT = 30000              # the largest published id an island group may take

# An island group carries no HydroBASINS code, so which of the nine Level-01 regions it is numbered under is
# ours to decide: the islands read like the 30 m products, where the continent is in
# the id.  The rule: the nearest Level-01 region measured as polygons on the sphere (every vertex of the nine
# HydroBASINS Level-03 layers dissolved by the first digit of PFAF_ID against every vertex of the group's own
# outline, measured once outside this package), and the three rows marked "by hand", where the measured
# answer is not what the geography says:
#   the Azores    nearest Africa 841 km (Madeira, the Canaries), Europe 1364 km -- the Azores are Portugal
#   Pitcairn      nearest North America 4237 km, South America 4347 km, Oceania 5002 km -- eastern Polynesia;
#                 the HydroBASINS Australia layer does not reach it, which is what the measurement is saying
#   St Helena     nearest South America 1020 km, Africa 1543 km -- 210 of the group's 220 km2 are St Helena and
#                 Ascension, counted with Africa; the measurement follows Trindade alone, 10 km2
# The box is the whole-degree window the group had when the table was made: a group whose window has moved by
# more than ISLAND_BOX_TOLERANCE_DEG is not that group any more, and then nothing is renumbered.
ISLAND_BOX_TOLERANCE_DEG = 1.0
ISLAND_GROUP_CONTINENTS = {
    # build id: (Level-01 region, the basins it held, the land they covered, minlon, minlat, maxlon, maxlat, what it is)
    200001: (5, 29529, 2159235, -174.0, 1.0, -154.0, 27.0, "Hawaii and the Line Islands"),
    200002: (5, 6709, 166162, -179.0, -45.0, -175.0, -29.0, "the Chatham and Kermadec Islands"),
    200003: (1, 52827, 1449375, 51.0, -54.0, 78.0, -37.0, "Kerguelen, Crozet, Amsterdam"),
    200004: (1, 17349, 607082, 55.0, -22.0, 73.0, -3.0, "Reunion, Mauritius, the Chagos Archipelago"),
    200005: (1, 14668, 499754, -26.0, 14.0, -22.0, 18.0, "Cape Verde"),
    200006: (6, 26581, 800259, -39.0, -60.0, -26.0, -53.0, "South Georgia and the South Sandwich Islands"),
    200007: (5, 40769, 575928, -180.0, -23.0, -154.0, 1.0, "Samoa, Tonga, the Cook Islands"),
    200008: (5, 101265, 545958, -154.0, -28.0, -134.0, -7.0, "the Marquesas, Tuamotu and Gambier Islands"),
    200009: (3, 9028, 208512, 76.0, 74.0, 83.0, 80.0, "the Russian Arctic islands"),
    200010: (2, 12444, 351105, -32.0, 36.0, -24.0, 40.0, "the Azores (by hand)"),
    200011: (5, 1004, 6901, -131.0, -26.0, -124.0, -23.0, "Pitcairn, Henderson, Ducie (by hand)"),
    200012: (5, 5168, 26752, 97.0, -1.0, 117.0, 10.0, "the islands of the South China Sea and west of Sumatra"),
    200013: (1, 2908, 71122, 37.0, -47.0, 51.0, -45.0, "the Prince Edward Islands"),
    200014: (5, 2153, 19374, 96.0, -13.0, 106.0, -10.0, "Christmas Island and the Cocos Islands"),
    200015: (6, 1065, 22107, -110.0, -28.0, -105.0, -26.0, "Easter Island"),
    200016: (1, 1733, 28621, -30.0, -21.0, -5.0, -7.0, "St Helena, Ascension, Trindade (by hand)"),
    200017: (7, 1964, 10431, -80.0, 15.0, -64.0, 33.0, "Bermuda"),
    200018: (1, 2326, 37900, -13.0, -55.0, 4.0, -37.0, "Tristan da Cunha, Gough, Bouvet"),
    200019: (5, 719, 4380, 162.0, -19.0, 165.0, -8.0, "the islands north of New Caledonia"),
    200020: (4, 24090, 41057, 71.0, -1.0, 74.0, 13.0, "the Maldives and Lakshadweep"),
    200021: (5, 433, 1146, -179.0, 27.0, -175.0, 29.0, "Midway and Kure"),
    200022: (3, 869, 8181, 156.0, 76.0, 159.0, 78.0, "the De Long Islands"),
    200023: (9, 246, 1176, -16.0, 79.0, -15.0, 80.0, "the islands off north-east Greenland"),
}


def island_group_continent_of(build_id, basin_count, land_grid_count, box):
    """the table's row for this island group, None when it is not in the table or does not match the row.  What is
    compared are the group's totals -- the basins, the land they cover, the four edges of its window -- not its
    members: a group that swapped one basin for another of the same size inside the same window would still match.
    It is a guard against the table being read on other ground, not a proof of identity; the
    comparison is written the way round that makes a NaN edge fail the test instead of passing it"""
    row = ISLAND_GROUP_CONTINENTS.get(int(build_id))
    if row is None or int(basin_count) != row[1] or int(land_grid_count) != row[2]:
        return None
    for measured, remembered in zip(box, row[3:7]):
        if not abs(float(measured) - remembered) <= ISLAND_BOX_TOLERANCE_DEG:
            return None
    return row


def region_row_order_key(region_id):
    """the order of the rows: the Level-02 unit first, then the id itself, so that the parts of a divided unit
    (3513, 3546) sit where the unit sits -- between 34 and 36 -- instead of after every unit.  The groups
    astride the antimeridian (10001, 10002) and the island groups (20000 +) keep their place at the end."""
    if region_id < 100:
        return (region_id, 0)                       # a whole Level-02 unit
    if region_id < 10000:
        return (region_id // 100, region_id)        # a part of that unit
    return (region_id, region_id)


def _renumber_the_regions_for_publication(regions, merges, grid, tag):
    """The numbering the tables and the masks carry: a region that is a whole
    Level-02 unit takes the two digits of that unit; a part of a divided unit takes four digits -- the two of
    the unit, then the last digit of the smallest and of the largest Level-03 code in it, so that the id
    names the run of units it holds (91 divided into {911, 912} and {913, 914} gives 9112 and 9134, and a
    part holding one unit repeats its digit: {569} gives 5699).  Every part is a run of consecutive codes,
    which _regions_of_the_level2_units makes it; a part that is not is refused.  The groups astride the
    antimeridian become 10001, 10002 in the order of their build ids, and an island group becomes
    20000 + its Level-01 region * 100 + its number within that region, the largest first (20501 is Hawaii);
    the region an island group belongs to is ISLAND_GROUP_CONTINENTS.  Nothing is renumbered when a
    region holds the pieces of a cut basin (their numbers carry the order the pieces are computed in) or when
    a Level-03 unit was divided (three digits cannot hold the part number), so the partitions at the finer
    capacities keep the build ids (61101, 62201, ..).  Changes region_id in place, and in the merge records
    the region that was kept; an absorbed id stays as it was built, that region no longer exists."""
    if REGION_ID_RULE == "as-built":
        return False
    regions_of_unit = {}
    for region in regions:
        if region.cut_basin_id != 0 or region.kind == 5:
            log(tag, "  the build ids are published as they are: region %d holds the pieces of cut basin %d, and their "
                "numbers carry the order they are computed in" % (region.region_id, region.cut_basin_id))
            return False
        if region.level3_lead <= 0:
            continue                                    # an island group or a group astride the antimeridian
        if region.region_id % 100 != 1:
            log(tag, "  the build ids are published as they are: region %d is part %d of the Level-03 unit %d, and the "
                "published numbering names whole Level-03 units only" % (region.region_id, region.region_id % 100, region.level3_lead))
            return False
        codes = sorted(region.level3_members)
        if not codes or codes != list(range(codes[0], codes[-1] + 1)):
            log(tag, "  the build ids are published as they are: region %d holds the Level-03 units %s, which are not a "
                "run of consecutive codes, so four digits cannot name them"
                % (region.region_id, " ".join(str(code) for code in codes)))
            return False
        unit = region.level3_lead // 10
        if unit < 10 or unit > 99:
            log(tag, "  the build ids are published as they are: region %d lies in the Level-02 unit %d, which is not a "
                "two-digit code" % (region.region_id, unit))
            return False
        if region.level2_code != unit:
            # a join across the units (merge=any, the second pass of merge=balanced) leaves level2_code at 0: the
            # region is no longer the unit its lead code names
            log(tag, "  the build ids are published as they are: region %d holds basins of more than one Level-02 unit"
                % region.region_id)
            return False
        regions_of_unit[unit] = regions_of_unit.get(unit, 0) + 1
    published_of_build = {}
    taken = set()
    seam_count = 0
    islands_to_number = []
    island_region_of_build = {}
    for region in sorted(regions, key=lambda r: r.region_id):   # the build ids in order, so the seam groups keep theirs
        build_id = region.region_id
        if region.level3_lead > 0:
            unit = region.level3_lead // 10
            codes = sorted(region.level3_members)
            published_id = unit if regions_of_unit[unit] == 1 else unit * 100 + (codes[0] % 10) * 10 + (codes[-1] % 10)
        elif SEAM_GROUP_ID_BASE <= build_id < ISLAND_GROUP_ID_BASE:
            seam_count += 1
            published_id = SEAM_PUBLISHED_ID_BASE + seam_count
        elif build_id >= ISLAND_GROUP_ID_BASE:
            # the whole-degree window of the group, the way the table carries it, so that a group that is not the
            # one the table meant is not numbered under its continent
            try:
                box = grid.pixel_box_lon_lat(*region.window(grid))
            except Exception as failure:                # a grid the window does not transform on
                log(tag, "  the build ids are published as they are: the window of island group %d does not transform to "
                    "longitude and latitude (%s)" % (build_id, failure))
                return False
            island_row = island_group_continent_of(build_id, int(region.basins.size), int(region.pixels), box)
            if island_row is None:
                log(tag, "  the build ids are published as they are: island group %d (%d basins, %d land pixels, %.1f .. "
                    "%.1f E, %.1f .. %.1f N) is not in the table of island groups, or holds other basins than the table "
                    "was made on" % (build_id, int(region.basins.size), int(region.pixels), box[0], box[2], box[1], box[3]))
                return False
            islands_to_number.append((island_row[0], -region.pixels, build_id))
            island_region_of_build[build_id] = region
            continue                                   # numbered below, once they are in order
        else:
            log(tag, "  the build ids are published as they are: region %d carries no Level-03 code and is neither an "
                "island group nor a group astride the antimeridian" % build_id)
            return False
        if published_id <= 0 or published_id >= PUBLISHED_ID_LIMIT:
            # not a failure: the short numbering simply does not reach this run
            log(tag, "  the build ids are published as they are: the short numbering of region %d would need the id %d, "
                "past the limit of %d" % (build_id, published_id, PUBLISHED_ID_LIMIT))
            return False
        if published_id in taken:
            raise FlowDivideError("the published id %d of region %d is already taken" % (published_id, build_id))
        taken.add(published_id)
        published_of_build[build_id] = published_id
    # the islands of one Level-01 region: the largest first, 20000 + region * 100 + the number within it
    number_within_level1 = {}
    for level1_code, negative_pixels, build_id in sorted(islands_to_number):
        number_within_level1[level1_code] = number_within_level1.get(level1_code, 0) + 1
        if number_within_level1[level1_code] > 99:
            log(tag, "  the build ids are published as they are: Level-01 region %d holds more than 99 island groups, "
                "and two digits cannot number them" % level1_code)
            return False
        published_id = ISLAND_PUBLISHED_ID_BASE + level1_code * 100 + number_within_level1[level1_code]
        if published_id >= PUBLISHED_ID_LIMIT:
            log(tag, "  the build ids are published as they are: the short numbering of island group %d would need the "
                "id %d, past the limit of %d" % (build_id, published_id, PUBLISHED_ID_LIMIT))
            return False
        if published_id in taken:
            raise FlowDivideError("the published id %d of island group %d is already taken" % (published_id, build_id))
        taken.add(published_id)
        published_of_build[build_id] = published_id
        log(tag, "  island group %d -> %d (Level-01 %d, %d land pixels)" % (build_id, published_id, level1_code, -negative_pixels))
    # from here nothing can refuse any more, so this is where the regions are changed.  The column says what the
    # id says: an island group carries no HydroBASINS code, and the Level-01 region it is numbered under is the one
    # the table gives it
    for level1_code, _, build_id in islands_to_number:
        island_region_of_build[build_id].level1_code = level1_code
    for region in regions:
        region.region_id = published_of_build[region.region_id]
    for record in merges:
        if record["kept_region_id"] in published_of_build:
            record["kept_region_id"] = published_of_build[record["kept_region_id"]]
    whole_units = sum(1 for region in regions if region.level3_lead > 0 and region.region_id < 100)
    parts_of_units = sum(1 for region in regions if region.level3_lead > 0 and 100 <= region.region_id < 10000)
    log(tag, "  published numbering: %d regions -- %d whole Level-02 units (two digits), %d parts of a divided unit "
        "(four digits), %d astride the antimeridian (%d ..), %d island groups numbered under their Level-01 region "
        "(%d + region * 100 + n)"
        % (len(regions), whole_units, parts_of_units, seam_count, SEAM_PUBLISHED_ID_BASE + 1, len(islands_to_number),
           ISLAND_PUBLISHED_ID_BASE))
    return True


def _regions_of_the_level2_units(grid, regions, capacity_pixels, tag):
    """A region is a HydroBASINS Level-02 unit when its whole-degree window fits the capacity; a unit that
    does not fit is divided along its own Level-03 units, the largest unit first and each one put in the
    first part that still holds it, so that a part is filled as far as it goes.  The groups that carry no
    code (the islands, and on a periodic grid the groups the antimeridian cuts) and the regions of the
    pieces of a cut basin are left as they are.

    Measured on the globe at 1400 square degrees: 57 of the 61 coded units fit as they are, and the four that do not
    (35 at 1.40 of the capacity, 56 at 1.10, 77 at 1.31, 91 at 1.12) come out as two regions each, so 65
    coded regions.  Building upward from the Level-03 units instead (291 of them, all fitting) and joining
    them two at a time stopped at 79, because a join had to stay inside one Level-02 unit."""
    merges = []
    alive = [True] * len(regions)
    windows = {}

    def window_of(indices):
        rectangle = regions[indices[0]].rectangle
        for index in indices[1:]:
            rectangle = _rectangle_union(rectangle, regions[index].rectangle)
        return rectangle, grid.window_pixels(*rectangle)

    planned_joins = []          # (the index kept, the index joined into it), all planned before the first join
    def plan_the_part(indices):
        """the part's regions to be joined into the one with the smallest id"""
        indices = sorted(indices, key=lambda index: regions[index].region_id)
        for index in indices[1:]:
            planned_joins.append((indices[0], index))
    def join_one(kept, index):
        """one region joined into its part's keeper, with its merge record"""
        before = regions[kept].window_pixels(grid)
        partner = regions[index].window_pixels(grid)
        regions[kept].rectangle = _rectangle_union(regions[kept].rectangle, regions[index].rectangle)
        regions[kept].basins = np.concatenate([regions[kept].basins, regions[index].basins])
        regions[kept].pixels += regions[index].pixels
        regions[kept].area_km2 += regions[index].area_km2
        regions[kept].level3_members = sorted(set(regions[kept].level3_members) | set(regions[index].level3_members))
        kept_level3_before = regions[kept].level3          # the merge record carries the code before the join
        if regions[index].level3 != regions[kept].level3:
            regions[kept].kind = 6
            # the region is the Level-02 unit now, not one Level-03 unit: saying level3 = the keeper's
            # code would name every basin it took in after the wrong unit
            regions[kept].level3 = 0
        # the Level-02 and the continent are kept as long as both sides share them, so that the
        # table still says which unit a joined region is (setting all three columns to 0 would lose it)
        # 0 none (island, seam), -1 parts of more than one unit (written as 0): "none" and
        # "mixed" must stay apart, or a third region of the first code would make a mixed region one unit again
        regions[kept].level2_code, regions[kept].level1_code = _joined_codes(regions[kept], regions[index])
        joined_pixels = regions[kept].window_pixels(grid)
        merges.append({"kept_region_id": regions[kept].region_id, "absorbed_region_id": regions[index].region_id,
                       "kept_level3": kept_level3_before, "absorbed_level3": regions[index].level3,
                       "joined_window_pixels": joined_pixels,
                       "extra_window_pixels": joined_pixels - before - partner,
                       "joined_land_pixels": regions[kept].pixels})
        alive[index] = False

    by_unit = {}
    for index, region in enumerate(regions):
        if region.cut_basin_id != 0 or region.level3 <= 0:
            continue                                        # a region of pieces, an island or a seam group
        by_unit.setdefault(region.level3 // 10, []).append(index)
    units_that_fit = 0
    units_divided = 0
    for unit in sorted(by_unit):
        indices = by_unit[unit]
        _, whole_unit_pixels = window_of(indices)
        if whole_unit_pixels <= capacity_pixels:
            plan_the_part(indices)
            units_that_fit += 1
            continue
        units_divided += 1
        # the unit does not fit: its Level-03 units in code order, a new part started as soon as the window
        # no longer fits, so that every part is a run of consecutive Level-03 codes and its published id can
        # name the run it holds.  Level-03 codes
        # run along the main stem, so consecutive codes are units that lie next to each other.  Measured on
        # the globe at 1400 square degrees: the four units that do not fit still come out as two parts each,
        # and their windows are more even than the packing gave (35: 1071 and 1035 blocks against 1395 and
        # 819; 91: 989 and 1116 against 1197 and 234).
        parts = []
        # the code, then the build id: a Level-03 unit that was itself divided has two regions of the same
        # code, and without the second key their order would not be decided
        for index in sorted(indices, key=lambda index: (regions[index].level3_lead, regions[index].region_id)):
            if parts:
                _, pixels = window_of(parts[-1] + [index])
                if pixels <= capacity_pixels:
                    parts[-1].append(index)
                    continue
            parts.append([index])
        log(tag, "  Level-02 unit %d does not fit (%.2f of the capacity): %d Level-03 units into %d regions, "
            "each a run of consecutive codes -- %s"
            % (unit, whole_unit_pixels / capacity_pixels, len(indices), len(parts),
               "; ".join("%d..%d (%d blocks)" % (regions[part[0]].level3_lead, regions[part[-1]].level3_lead,
                                                 window_of(part)[1] // (grid.block_pixels * grid.block_pixels))
                         for part in parts)))
        for part in parts:
            _, part_pixels = window_of(part)
            if part_pixels > capacity_pixels:
                raise FlowDivideError(
                    "the Level-02 unit %d was divided into a part of %d window pixels, past the capacity of %d: the "
                    "Level-03 unit %d does not fit on its own, so the unit cannot be divided along its Level-03 units"
                    % (unit, part_pixels, capacity_pixels, regions[part[0]].level3_lead))
            plan_the_part(part)
    # the joins in a fixed order, so that the merge records come out the same on every run (the windows and
    # the land they record depend on what the keeper had taken in before).  Every join is planned first, then
    # each round takes the first region of its list still to be joined (choose_planned_pair); its list holds the
    # regions made in the walk over the groups (a unit that is one region as it is) before those made after it (the
    # parts of a divided unit, the whole basins of a unit that holds a cut basin), each in the order of the groups --
    # which is (build_phase, the order this list was built in).  Joining unit by unit gives the same regions in the
    # end, but the records of MERIT 700 and 350 square degrees in another order.
    for kept, index in sorted(planned_joins, key=lambda pair: (regions[pair[1]].build_phase, pair[1])):
        join_one(kept, index)
    regions[:] = [region for region, keep in zip(regions, alive) if keep]
    log(tag, "  %d Level-02 units fit as they are, %d were divided; %d joins; %d regions now"
        % (units_that_fit, units_divided, len(merges), len(regions)))
    return merges


def _merge_regions_under_the_capacity(grid, regions, capacity_pixels, policy, land_cut_above, tag):
    """Groups under the capacity joined two at a time.  Policy "within_parent" and "any": the pair whose
    joined window costs the fewest pixels over the two windows added up, only windows that touch or
    contain one another, never a region of pieces (within_parent: only two regions of one Level-02
    unit).  Policy "balanced": the smallest region that has a partner joins the partner in its Level-02
    unit that fills the capacity furthest, a join adding no more empty window than the smaller of
    the two windows holds; when nothing can grow inside its unit any more, a region under a quarter of
    the capacity may join a touching coded region of another unit.  Returns the merge records.

    Filling comes first, evenness second. The
    filling is the partner chosen: of the partners a region may take, the one whose joined window comes
    closest to the capacity from below. The evenness is the order: the smallest region goes first, so the
    small ones are taken in rather than one region growing until it fills the capacity by itself. Choosing
    the partner with the SMALLEST joined window instead leaves the regions small and their number large
    (the globe at 1400 square degrees: 107 regions, 82 of them coded)."""
    merges = []
    refused = {"not the same unit": 0, "a region of pieces": 0, "the windows do not touch": 0,
               "the joined window passes the capacity": 0, "the joined land passes the cut": 0,
               "the join adds too much empty window": 0}
    if policy == "none":
        return merges
    if policy == "level2":
        return _regions_of_the_level2_units(grid, regions, capacity_pixels, tag)
    if policy not in ("within_parent", "any", "balanced"):
        raise FlowDivideError("unknown merge policy '%s'" % policy)
    alive = [True] * len(regions)

    def may_join(first, second, across):
        if first.cut_basin_id != 0 or second.cut_basin_id != 0:
            refused["a region of pieces"] += 1
            return False                                # a region of the pieces of one cut basin
        if policy in ("within_parent", "balanced"):
            # by the Level-02 code, which a joined region keeps while both sides share it: 0 an
            # island or seam region, never joined here; -1 coded regions of several units, which the pass across the
            # units may still join
            if first.level2_code == 0 or second.level2_code == 0:
                refused["not the same unit"] += 1
                return False                                # an island or seam region carries no code
            if not across and (first.level2_code < 0 or first.level2_code != second.level2_code):
                refused["not the same unit"] += 1
                return False
        first_window = first.window(grid)
        second_window = second.window(grid)
        touching = (_windows_touch(grid, first_window, second_window) or _window_contains(first_window, second_window)
                    or _window_contains(second_window, first_window))
        if not touching:
            refused["the windows do not touch"] += 1
        return touching

    def join(kept, absorbed, extra, joined_pixels):
        kept.rectangle = _rectangle_union(kept.rectangle, absorbed.rectangle)
        kept.basins = np.concatenate([kept.basins, absorbed.basins])
        kept.pixels += absorbed.pixels
        kept.area_km2 += absorbed.area_km2
        # every Level-03 unit of both sides, not only the keeper's
        kept.level3_members = sorted(set(kept.level3_members) | set(absorbed.level3_members))
        kept_level3_before = kept.level3
        if absorbed.level3 != kept.level3:
            kept.kind = 6
            kept.level3 = 0                 # the region is no longer one Level-03 unit
        kept.level2_code, kept.level1_code = _joined_codes(kept, absorbed)
        merges.append({"kept_region_id": kept.region_id, "absorbed_region_id": absorbed.region_id, "kept_level3": kept_level3_before,
                       "absorbed_level3": absorbed.level3, "joined_window_pixels": joined_pixels,
                       "extra_window_pixels": extra, "joined_land_pixels": kept.pixels})
        log(tag, "  merge: region %d takes in region %d; the window becomes %d pixels (%d extra)" % (kept.region_id, absorbed.region_id, joined_pixels, extra))

    if policy in ("within_parent", "any"):
        while True:
            best = None
            for i in range(len(regions)):
                if not alive[i]:
                    continue
                for j in range(i + 1, len(regions)):
                    if not alive[j] or not may_join(regions[i], regions[j], False):
                        continue
                    joined = _rectangle_union(regions[i].rectangle, regions[j].rectangle)
                    joined_pixels = grid.window_pixels(*joined)
                    if joined_pixels > capacity_pixels or regions[i].pixels + regions[j].pixels > land_cut_above:
                        continue
                    extra = joined_pixels - regions[i].window_pixels(grid) - regions[j].window_pixels(grid)
                    key = (extra, min(regions[i].region_id, regions[j].region_id), max(regions[i].region_id, regions[j].region_id))
                    if best is None or key < best[0]:
                        best = (key, i, j, joined_pixels)
            if best is None:
                break
            _, i, j, joined_pixels = best
            kept, absorbed = (i, j) if regions[i].region_id < regions[j].region_id else (j, i)
            join(regions[kept], regions[absorbed], best[0][0], joined_pixels)
            alive[absorbed] = False
    else:
        leftover = int(0.25 * capacity_pixels)
        for across in ((True,) if MERGE_ACROSS == "from-the-start" else (False, True)):
            while True:
                tried = set()
                found = False
                while True:
                    small_side_only = across and MERGE_ACROSS == "leftovers-only"
                    candidates = [(regions[i].window_pixels(grid), regions[i].region_id, i) for i in range(len(regions))
                                  if alive[i] and i not in tried and (not small_side_only or regions[i].window_pixels(grid) < leftover)]
                    if not candidates:
                        break
                    window_pixels, _, i = min(candidates)
                    tried.add(i)
                    best = None
                    for j in range(len(regions)):
                        if j == i or not alive[j] or not may_join(regions[i], regions[j], across):
                            continue
                        joined = _rectangle_union(regions[i].rectangle, regions[j].rectangle)
                        joined_pixels = grid.window_pixels(*joined)
                        if joined_pixels > capacity_pixels:
                            refused["the joined window passes the capacity"] += 1
                            continue
                        if regions[i].pixels + regions[j].pixels > land_cut_above:
                            refused["the joined land passes the cut"] += 1
                            continue
                        partner_pixels = regions[j].window_pixels(grid)
                        extra = joined_pixels - window_pixels - partner_pixels
                        if extra > min(window_pixels, partner_pixels):
                            refused["the join adds too much empty window"] += 1
                            continue
                        # the rule of MERGE_RULE, then the least empty window added, then the smaller id so
                        # that the run is reproducible
                        joined_land = regions[i].pixels + regions[j].pixels
                        if MERGE_RULE == "window-min":
                            first_key = joined_pixels
                        elif MERGE_RULE == "window-max":
                            first_key = -joined_pixels
                        elif MERGE_RULE == "land-max":
                            first_key = -joined_land
                        else:
                            first_key = -round(1e9 * joined_land / joined_pixels)
                        key = (first_key, extra, regions[j].region_id)
                        if best is None or key < best[0]:
                            best = (key, j, joined_pixels, extra)
                    if best is not None:
                        _, j, joined_pixels, extra = best
                        kept, absorbed = (i, j) if regions[i].region_id < regions[j].region_id else (j, i)
                        join(regions[kept], regions[absorbed], extra, joined_pixels)
                        alive[absorbed] = False
                        found = True
                        break
                if not found:
                    break
    regions[:] = [region for region, keep in zip(regions, alive) if keep]
    log(tag, "  %d merges under %s(%s); the pairs refused: %s" % (len(merges), policy, MERGE_RULE,
        ", ".join("%s %d" % (reason, count) for reason, count in refused.items() if count)))
    return merges


@njit(cache=True)
def _mask_strip(bsn_strip, region_id_of_basin, region_index_of_basin, member_of_basin, rgn_strip,
                region_count, region_row_min, region_row_max, region_col_min, region_col_max, region_col_min_shift, region_col_max_shift,
                member_count, row0, grid_ncol, periodic):
    """the region of every pixel of a strip from its basin, and the member counted (no member raster is
    written, nothing read it; the count stays for the checks); a cut basin (region 0 in
    the lookup) is left for the piece rasters.  The statistics are kept by region index."""
    nrow, ncol = bsn_strip.shape
    half = grid_ncol // 2
    for row in range(nrow):
        for col in range(ncol):
            basin = bsn_strip[row, col]
            if basin == 0:
                continue
            region_id = region_id_of_basin[basin]
            if region_id == 0:
                continue
            rgn_strip[row, col] = region_id
            index = region_index_of_basin[basin]
            region_count[index] += 1
            member_count[member_of_basin[basin]] += 1
            grid_row = row0 + row
            if grid_row < region_row_min[index]:
                region_row_min[index] = grid_row
            if grid_row >= region_row_max[index]:
                region_row_max[index] = grid_row + 1        # one past
            if col < region_col_min[index]:
                region_col_min[index] = col
            if col >= region_col_max[index]:
                region_col_max[index] = col + 1
            if periodic:
                shifted = (col + half) % grid_ncol
                if shifted < region_col_min_shift[index]:
                    region_col_min_shift[index] = shifted
                if shifted >= region_col_max_shift[index]:
                    region_col_max_shift[index] = shifted + 1


@njit(cache=True)
def _mask_strip_pieces(piece_strip, bsn_strip, cut_basin_id, region_id_of_code, region_index_of_code, member_of_code, rgn_strip, col_offset,
                       region_count, region_row_min, region_row_max, region_col_min, region_col_max, region_col_min_shift, region_col_max_shift,
                       member_count, row0, grid_ncol, periodic):
    """the pixels of one cut basin in a strip, from its piece raster (the deepest piece codes) placed at
    col_offset in the strip; the columns wrap on a periodic grid.  Every piece pixel must lie in the
    basin the raster is for and must not carry a region yet.  Returns 0, or 1 when a pixel is not
    of the basin, 2 when a pixel already has a region, 3 when a code has no region."""
    nrow, ncol = piece_strip.shape
    strip_ncol = rgn_strip.shape[1]
    half = grid_ncol // 2
    for row in range(nrow):
        for local_col in range(ncol):
            code = piece_strip[row, local_col]
            if code == 0:
                continue
            col = col_offset + local_col
            if periodic:
                col = col % grid_ncol
            if col < 0 or col >= strip_ncol:
                continue
            if bsn_strip[row, col] != cut_basin_id:
                return 1
            if rgn_strip[row, col] != 0:
                return 2
            if code < 0 or code >= region_id_of_code.size or region_id_of_code[code] == 0:     # a negative code too
                return 3
            rgn_strip[row, col] = region_id_of_code[code]
            index = region_index_of_code[code]
            region_count[index] += 1
            member_count[member_of_code[code]] += 1
            grid_row = row0 + row
            if grid_row < region_row_min[index]:
                region_row_min[index] = grid_row
            if grid_row >= region_row_max[index]:
                region_row_max[index] = grid_row + 1        # one past
            if col < region_col_min[index]:
                region_col_min[index] = col
            if col >= region_col_max[index]:
                region_col_max[index] = col + 1
            if periodic:
                shifted = (col + half) % grid_ncol
                if shifted < region_col_min_shift[index]:
                    region_col_min_shift[index] = shifted
                if shifted >= region_col_max_shift[index]:
                    region_col_max_shift[index] = shifted + 1
    return 0


PIECE_FINE_COLUMNS = fd_tables.PIECE_TABLE_COLUMNS


def fd1_5_regions_final(basin_table_path, grouping, bsn_path, cuts, grid, capacity_pixels, outputs, continent,
                        merge_policy="none", land_cut_above=None, land_merge_below=0, level1=None, strip_rows=None, tag="fd1.5"):
    """The final regions of the partition at one capacity, from the basin groups of FD1.4 and the cuts
    of fd1_5_pfafstetter_cut (cuts: {basin id: (its piece table, its piece raster)}).  A group whose
    window fits is one region; a group over the capacity is divided along block lines; a cut basin's
    regions are taken from the cut and the other basins of its group form regions of their own; groups
    under the capacity are merged under merge_policy (none, within_parent, any, balanced).  Writes the
    files named in `outputs`:
        regions       one row per region (region_fine): the codes, the window, the land, the counts,
                      cut_basin_id, the rectangle of its pixels, topological_level
        pieces        one row per piece of a cut basin (piece_fine): piece_id, its parent piece, its region,
                      outlet and parent inlet, counts and areas, rectangle, downstream_depth, crossing_length_m
        basin_region  the region of every basin (0 for a cut basin), in basin id order
        merges        the joins made under the merge policy
        rgn           the region of every land pixel (UInt32); no member raster (mbr) is written (nothing after
                      this step would read it), the members are the rows of the tables
    A region is built as level3 * 100 + n (n = 1 .. 99, the pieces of a cut basin first, in the order
    of the cut); an island or seam group that is one region keeps its group id, and one that is divided
    gives its parts the next free numbers of its range (200000 + level1 * 1000 + k, 100000 + ...),
    counted from the largest number fd1.4 gave; every flow between regions goes to a larger number.
    Those are the ids the step builds and checks with.  What the files carry is the short numbering of
    _renumber_the_regions_for_publication when the partition allows it: 11, 12, .. 91 for a whole
    Level-02 unit, 351, 352, .. for the parts of a divided one, 10001, 10002 astride the antimeridian and
    20501, 20101, .. for the islands (20000 + the Level-01 region * 100 + the number within it).
    level1: the level-1 number of the island and seam regions; the smallest code's first digit when None."""
    started = time.time()
    if land_cut_above is None:
        land_cut_above = capacity_pixels
    if strip_rows is None:
        strip_rows = strip_rows_for(grid.ncol)
    # the basin table of fd1.4, the grouping's group table and the group of every basin
    # (the level ids of the basin table for l3, the grouping's basin map for a Hilbert grouping)
    basins = fd_tables.read_basin_table(basin_table_path)
    root, run = fd_tables.root_and_run_of_basin_table(basin_table_path)
    groups = fd_tables.read_group_table(fd_tables.group_table_path(root, run, grouping), 0)
    basin_count = len(basins)
    basin_ids = basins["basin_id"].to_numpy(np.int64)
    basins_rect = basin_rectangles(basins).copy()          # a working copy: unrolled per group on a periodic grid
    basins_land = basins["basin_grid_count"].to_numpy(np.int64)
    basins_area = basins["basin_area_km2"].to_numpy(np.float64)
    group_of_basin = fd_tables.group_of_basin(root, run, grouping, basins)[1:].astype(np.int64)
    cut_set = sorted(int(b) for b in cuts)
    is_cut = np.zeros(basin_count, bool)
    is_cut[np.asarray(cut_set, np.int64) - 1] = True
    for cut_id in cut_set:
        for path in cuts[cut_id]:
            fd_tables.file_is_complete(path)             # fd1.5.pfafstetter_cut finished for this basin
    cut_tables = {cut_id: read_table(cuts[cut_id][0]) for cut_id in cut_set}
    level1_of_continent = level1 if level1 is not None else (int(groups.loc[groups["level_code"] > 0, "level_code"].min()) // 100 if (groups["level_code"] > 0).any() else 0)
    regions = []
    number_used = {}
    # the next free number of the island range and of the seam range, counted from the largest number fd1.4 gave
    next_free_number = {3: 1, 4: 1}
    base_of_range = {3: 200000 + level1_of_continent * 1000, 4: 100000 + level1_of_continent * 1000}
    for group_id, kind in zip(groups["group_id"].astype(int), groups["group_kind"].astype(int)):
        if kind in (3, 4):
            base_of_range[kind] = (group_id // 1000) * 1000
            next_free_number[kind] = max(next_free_number[kind], group_id % 1000 + 1)

    def next_number(level3):
        number_used[level3] = number_used.get(level3, 0) + 1
        if number_used[level3] > 99:
            raise FlowDivideError("more than 99 regions in Level-03 unit %d" % level3)
        return number_used[level3]

    def add_whole_basin_regions(member_indices, level3, kind, base_id, what, holds_a_cut_basin=False):
        member_indices = np.asarray(member_indices, np.int64)
        if grid.periodic and level3 > 0:
            # a unit on both sides of the antimeridian: when moving its western rectangles past the seam
            # gives a narrower span than the grid's own frame, they are moved, as in the group window of FD1.4
            # (an island or seam group keeps its rectangles as fd1.4 joined them).  The rule:
            # a rectangle already astride the seam or empty takes no part, a
            # rectangle is western when its centre is (not only when all of it is), the spans are right open, and a
            # unit that holds a cut basin is refused rather than moved
            rows = basins_rect[member_indices]
            taking_part = (rows[:, 3] > rows[:, 2]) & (rows[:, 3] <= grid.ncol)
            western = taking_part & ((rows[:, 2] + rows[:, 3] - 1) // 2 < grid.ncol // 2)
            eastern = taking_part & ~western
            if western.any() and eastern.any():
                span_as_it_is = int(rows[taking_part, 3].max() - rows[taking_part, 2].min())
                span_across_the_seam = int(rows[western, 3].max() + grid.ncol - rows[eastern, 2].min())
                if span_across_the_seam < span_as_it_is:
                    if holds_a_cut_basin:
                        raise FlowDivideError("Level-03 unit %d lies across the antimeridian and holds a cut basin, whose "
                                              "pieces could not follow the move" % level3)
                    basins_rect[member_indices[western], 2] += grid.ncol
                    basins_rect[member_indices[western], 3] += grid.ncol
        rectangle = _rectangle_of_rows(basins_rect[member_indices])
        land = int(basins_land[member_indices].sum())
        whole_as_it_is = grid.window_pixels(*rectangle) <= capacity_pixels and land <= land_cut_above
        if whole_as_it_is:
            parts = [member_indices]
        else:
            # the region numbers left: to this code, or in the range of the island or seam groups
            parts_available = 99 - number_used.get(level3, 0) if level3 > 0 else 999 - next_free_number[kind] + 1
            parts = _split_basins_along_block_lines(grid, basins_rect, basins_land, member_indices, capacity_pixels, land_cut_above, land_merge_below, parts_available, what, tag)
        for part_number, part in enumerate(parts, start=1):
            if level3 > 0:
                region_id = level3 * 100 + next_number(level3)
            elif len(parts) == 1:
                region_id = base_id                      # an island or seam group that is one region keeps its id
            else:
                if next_free_number[kind] > 999:
                    raise FlowDivideError("the numbers of the %s groups are used up" % ("island" if kind == 3 else "seam"))
                region_id = base_of_range[kind] + next_free_number[kind]      # a divided one: the next free numbers of its range
                next_free_number[kind] += 1
            region = RegionBuild(region_id, level3, kind if len(parts) == 1 else 2)
            region.basins = np.asarray(part, np.int64)
            region.rectangle = _rectangle_of_rows(basins_rect[region.basins])
            region.pixels = int(basins_land[region.basins].sum())
            region.area_km2 = float(basins_area[region.basins].sum())
            # a group that forms one region as it stands is made while the walk goes over the groups, and a divided
            # group, or the whole basins of a unit that holds a cut basin, after that walk (fd1.5 cases 1 and 4
            # against cases 2 and 3); the order of the joins of merge=level2 follows that list
            region.build_phase = 1 if (whole_as_it_is and not holds_a_cut_basin) else 2
            regions.append(region)

    # the basins of every group, the cut ones apart: a group is worked in the order of its id, and the
    # regions of a cut basin come first and keep the numbers of the cut
    order = np.argsort(group_of_basin, kind="stable")
    sorted_groups = group_of_basin[order]
    for _, group in groups.sort_values("group_id").iterrows():
        group_id = int(group["group_id"])
        kind = int(group["group_kind"])
        level3 = int(group["level_code"])
        first = int(np.searchsorted(sorted_groups, group_id, side="left"))
        last = int(np.searchsorted(sorted_groups, group_id, side="right"))
        members = order[first:last]
        if members.size == 0:
            continue
        for index in members[is_cut[members]]:
            cut_id = int(basin_ids[index])
            table = cut_tables[cut_id]
            used = table[table["used"] == 1]
            for region_number in sorted(int(n) for n in used["region"].unique()):
                rows = used[used["region"] == region_number]
                region_id = level3 * 100 + next_number(level3) if level3 > 0 else group_id * 100 + next_number(group_id)
                region = RegionBuild(region_id, level3, 5)
                region.cut_basin_id = cut_id
                region.piece_codes = [int(c) for c in rows["piece"]]
                region.rectangle = (int(rows["row_min"].min()), int(rows["row_max"].max()), int(rows["col_min"].min()), int(rows["col_max"].max()))
                region.pixels = int(rows["labelled_grid_count"].sum())
                region.basins = np.zeros(0, np.int64)
                regions.append(region)
        whole = members[~is_cut[members]]
        holds_a_cut_basin = bool(is_cut[members].any())
        if whole.size:
            if kind in (3, 4):
                add_whole_basin_regions(whole, 0, kind, group_id, "group %d" % group_id, holds_a_cut_basin)
            else:
                add_whole_basin_regions(whole, level3, 1, group_id, "unit %d" % level3, holds_a_cut_basin)
    log(tag, "%d regions before merging (%d of pieces)" % (len(regions), sum(1 for r in regions if r.kind == 5)))
    merges = _merge_regions_under_the_capacity(grid, regions, capacity_pixels, merge_policy, land_cut_above, tag)
    _renumber_the_regions_for_publication(regions, merges, grid, tag)
    for region in regions:
        if region.window_pixels(grid) > capacity_pixels:
            raise FlowDivideError("region %d has a window of %d pixels, over the capacity %d" % (region.region_id, region.window_pixels(grid), capacity_pixels))
    ids = [region.region_id for region in regions]
    if len(set(ids)) != len(ids):
        raise FlowDivideError("two regions share an id")
    # the members: a whole basin is a member with its own id, a piece a member numbered above the last basin id
    region_ids = np.asarray(sorted(ids), np.int64)
    region_index_of_id = {int(region_id): index + 1 for index, region_id in enumerate(region_ids)}
    region_id_of_basin = np.zeros(basin_count + 1, np.uint32)
    region_index_of_basin = np.zeros(basin_count + 1, np.uint32)
    member_of_basin = np.zeros(basin_count + 1, np.uint32)
    member_of_basin[1:] = np.arange(1, basin_count + 1, dtype=np.uint32)
    for region in regions:
        region_id_of_basin[region.basins + 1] = region.region_id
        region_index_of_basin[region.basins + 1] = region_index_of_id[region.region_id]
    next_member = basin_count + 1
    piece_rows = []
    piece_member_of_code = {}
    region_of_used_code = {}
    level3_of_region = {region.region_id: region.level3 for region in regions}
    for cut_id in cut_set:
        table = cut_tables[cut_id]
        used = table[table["used"] == 1].sort_values("piece")
        piece_member_of_code[cut_id] = {}
        for code in used["piece"].astype(int):
            piece_member_of_code[cut_id][code] = next_member
            next_member += 1
        region_of_used_code[cut_id] = {}
        for region in regions:
            if region.cut_basin_id == cut_id:
                for code in region.piece_codes:
                    region_of_used_code[cut_id][code] = region.region_id
        aca_of_code = dict(zip(used["piece"].astype(int), used["aca_at_outlet_km2"].astype(float)))
        for _, row in used.iterrows():
            code = int(row["piece"])
            parent = int(row["flows_into"])
            region_id = region_of_used_code[cut_id][code]
            # the piece's own area: the upstream area at its outlet less that at the outlets of the used
            # pieces flowing into it; -1 when the area raster was not there for the cut.
            # Why a difference of upstream areas and not a sum of pixel areas on the ellipsoid: the pieces
            # of a basin then add up to the basin's own upstream
            # area exactly, whichever raster that came from.  On the MERIT 90 m line the upstream area is
            # MERIT's own, read as it is, so the piece areas are on the same
            # footing as the column they are cut from; summing ellipsoid pixel areas instead would leave
            # the pieces not adding up to the basin.  The exact ellipsoid is used where this code
            # integrates pixel areas itself: the region windows, the land share, and the 30 m line's
            # own upstream area.
            children_aca = [aca_of_code[int(c)] for c, p in zip(used["piece"].astype(int), used["flows_into"].astype(int)) if p == code]
            own_area = -1.0 if aca_of_code[code] < 0 or any(a < 0 for a in children_aca) else aca_of_code[code] - sum(children_aca)
            # the piece table takes only a positive area: a piece without one stops here
            if not (0.0 < own_area < 1.0e30):
                raise FlowDivideError("piece %d of basin %d has an area of %g km2 (its outlet's upstream area less that of the pieces "
                                      "flowing into it); the area raster of the cut is missing or wrong" % (code, cut_id, own_area))
            piece_rows.append({"piece_id": piece_member_of_code[cut_id][code], "basin_id": cut_id, "level3_code": level3_of_region[region_id], "piece_kind": 1,
                               "parent_piece_id": piece_member_of_code[cut_id][parent] if parent else 0, "region_id": region_id,
                               "outlet_row": int(row["outlet_row"]), "outlet_col": int(row["outlet_col"]),
                               "outlet_lon": float(grid.pixel_centre_lon_lat(int(row["outlet_row"]), int(row["outlet_col"]))[0]),
                               "outlet_lat": float(grid.pixel_centre_lon_lat(int(row["outlet_row"]), int(row["outlet_col"]))[1]),
                               "parent_inlet_row": int(row["parent_inlet_row"]), "parent_inlet_col": int(row["parent_inlet_col"]),
                               "piece_grid_count": int(row["labelled_grid_count"]), "acc_at_outlet": int(row["acc_at_outlet"]),
                               "aca_at_outlet_km2": float(row["aca_at_outlet_km2"]), "piece_area_km2": own_area,
                               "row_min": int(row["row_min"]), "row_max": int(row["row_max"]), "col_min": int(row["col_min"]), "col_max": int(row["col_max"])})
    pieces = pd.DataFrame(piece_rows) if piece_rows else pd.DataFrame({column: [] for column in PIECE_FINE_COLUMNS})
    region_of_piece = {}
    if len(pieces):
        pieces["bbox_grid_count"] = (pieces["row_max"] - pieces["row_min"]) * (pieces["col_max"] - pieces["col_min"])
        boxes = [grid.pixel_box_lon_lat(int(a), int(b), int(c), int(d)) for a, b, c, d in zip(pieces["row_min"], pieces["row_max"], pieces["col_min"], pieces["col_max"])]
        pieces["minlon"] = [b[0] for b in boxes]
        pieces["minlat"] = [b[1] for b in boxes]
        pieces["maxlon"] = [b[2] for b in boxes]
        pieces["maxlat"] = [b[3] for b in boxes]
        parent_of = dict(zip(pieces["piece_id"].astype(int), pieces["parent_piece_id"].astype(int)))
        region_of_piece = dict(zip(pieces["piece_id"].astype(int), pieces["region_id"].astype(int)))
        depth = {}

        def depth_of(piece_id):
            if piece_id not in depth:
                parent = parent_of[piece_id]
                depth[piece_id] = 0 if parent == 0 else depth_of(parent) + 1
            return depth[piece_id]

        pieces["downstream_depth"] = [depth_of(int(p)) for p in pieces["piece_id"]]
        lengths = []
        for _, row in pieces.iterrows():
            if int(row["parent_piece_id"]) == 0:
                lengths.append(0.0)
            elif grid.geographic:
                lengths.append(earth_distance_m(grid.latitude_of_row(int(row["outlet_row"])), grid.longitude_of_col(int(row["outlet_col"])),
                                                grid.latitude_of_row(int(row["parent_inlet_row"])), grid.longitude_of_col(int(row["parent_inlet_col"]))))
            else:
                lengths.append(math.hypot((int(row["outlet_row"]) - int(row["parent_inlet_row"])) * grid.pixel_height,
                                          (int(row["outlet_col"]) - int(row["parent_inlet_col"])) * grid.pixel_width))
        pieces["crossing_length_m"] = lengths
        for _, row in pieces.iterrows():
            if int(row["parent_piece_id"]) != 0 and region_of_piece[int(row["parent_piece_id"])] < int(row["region_id"]):
                raise FlowDivideError("piece %d flows into a region with a smaller number than its own" % int(row["piece_id"]))
        pieces = pieces[PIECE_FINE_COLUMNS]
    # the mask pass: rgn strip by strip, with the statistics of every region and the pixels of every member
    region_count_limit = len(region_ids) + 1
    r_count = np.zeros(region_count_limit, np.int64)
    r_row_min = np.full(region_count_limit, np.iinfo(np.int32).max, np.int32)
    r_row_max = np.full(region_count_limit, -1, np.int32)
    r_col_min = np.full(region_count_limit, np.iinfo(np.int32).max, np.int32)
    r_col_max = np.full(region_count_limit, -1, np.int32)
    r_col_min_shift = np.full(region_count_limit, np.iinfo(np.int32).max, np.int32)
    r_col_max_shift = np.full(region_count_limit, -1, np.int32)
    member_count = np.zeros(next_member, np.int64)
    # the piece tables checked: a cut basin has one piece that flows into no other, its outlet is the basin's, and the outlet of
    # every other piece flows, by the flow directions, into the parent inlet its row gives
    offsets = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    with rasterio.open(grid.path) as dir_dataset:
        for cut_id in cut_set:
            table = cut_tables[cut_id]
            used_rows = {int(row["piece"]): row for _, row in table[table["used"] == 1].iterrows()}
            all_rows = {int(row["piece"]): row for _, row in table.iterrows()}
            roots = []
            for code, row in used_rows.items():
                down = int(row["flows_into"])
                while down > 0 and int(all_rows[down]["used"]) != 1:
                    down = int(all_rows[down]["sub_of"])
                if down == 0:
                    roots.append(code)
                    continue
                outlet_row = int(row["outlet_row"])
                outlet_col = int(row["outlet_col"]) % grid.ncol if grid.periodic else int(row["outlet_col"])
                if not (0 <= outlet_row < grid.nrow and 0 <= outlet_col < grid.ncol):
                    raise FlowDivideError("basin %d piece %d: its outlet (row %d, column %d) is outside the grid"
                                          % (cut_id, code, outlet_row, outlet_col))
                code_there = int(dir_dataset.read(1, window=Window(outlet_col, outlet_row, 1, 1))[0, 0])
                step = offsets.get(code_there)
                inlet_col = int(row["parent_inlet_col"]) % grid.ncol if grid.periodic else int(row["parent_inlet_col"])
                down_col = (outlet_col + step[1]) % grid.ncol if (step and grid.periodic) else (outlet_col + step[1] if step else -1)
                if step is None or outlet_row + step[0] != int(row["parent_inlet_row"]) or down_col != inlet_col:
                    raise FlowDivideError("basin %d piece %d: its outlet (row %d, column %d, code %d) does not flow into the "
                                          "parent inlet its row gives (row %d, column %d)" % (cut_id, code, outlet_row, outlet_col,
                                          code_there, int(row["parent_inlet_row"]), inlet_col))
            if len(roots) != 1:
                raise FlowDivideError("basin %d has %d pieces that flow into no other piece; it must have one" % (cut_id, len(roots)))
            root = used_rows[roots[0]]
            basin_row = basins.iloc[cut_id - 1]
            # the columns are taken round the grid only when it is periodic; on the 30 m grid a column
            # of ncol + 100 would pass as 100
            root_col = int(root["outlet_col"]) % grid.ncol if grid.periodic else int(root["outlet_col"])
            basin_col = int(basin_row["outlet_col"]) % grid.ncol if grid.periodic else int(basin_row["outlet_col"])
            if int(root["outlet_row"]) != int(basin_row["outlet_row"]) or root_col != basin_col:
                raise FlowDivideError("basin %d: the piece that flows into no other has its outlet at (%d, %d); the basin's is at "
                                      "(%d, %d)" % (cut_id, int(root["outlet_row"]), int(root["outlet_col"]),
                                                    int(basin_row["outlet_row"]), int(basin_row["outlet_col"])))
            # and its upstream area at that outlet is the basin's in the basin table (the same to
            # the last digit on the 91 real tables); -1 in the table has nothing to compare
            basin_area = float(basin_row["basin_area_km2"])
            root_area = float(root["aca_at_outlet_km2"])
            if basin_area >= 0.0 and not abs(root_area - basin_area) <= 1.0e-6 + 1.0e-9 * basin_area:
                raise FlowDivideError("basin %d: the piece that flows into no other has %.6f km2 at its outlet; the basin table "
                                      "has %.6f km2" % (cut_id, root_area, basin_area))
    piece_rasters = {}
    # the piece rasters are closed whatever stops this pass, their opening included (not left to garbage
    # collection when an error is raised)
    try:
        for cut_id in cut_set:
            table = cut_tables[cut_id]
            code_limit = int(table["piece"].max()) + 1
            region_id_of_code = np.zeros(code_limit, np.uint32)
            region_index_of_code = np.zeros(code_limit, np.uint32)
            member_of_code = np.zeros(code_limit, np.uint32)
            used_codes = set(int(c) for c in table.loc[table["used"] == 1, "piece"])
            parent_of_code = dict(zip(table["piece"].astype(int), table["sub_of"].astype(int)))
            cut_codes = set(parent_of_code.values())          # the pieces that were cut again never appear in the raster
            for code in table["piece"].astype(int):
                if code in cut_codes:
                    continue
                used_ancestor = code
                while used_ancestor not in used_codes:
                    used_ancestor = parent_of_code[used_ancestor]
                    if used_ancestor == 0:
                        raise FlowDivideError("piece %d of basin %d has no used ancestor" % (code, cut_id))
                region_id_of_code[code] = region_of_used_code[cut_id][used_ancestor]
                region_index_of_code[code] = region_index_of_id[region_of_used_code[cut_id][used_ancestor]]
                member_of_code[code] = piece_member_of_code[cut_id][used_ancestor]
            basin_row = basins.iloc[cut_id - 1]
            window = grid.window_of_rectangle(int(basin_row["basin_row_min"]), int(basin_row["basin_row_max"]), int(basin_row["basin_col_min"]), int(basin_row["basin_col_max"]))
            piece_dataset = rasterio.open(cuts[cut_id][1])
            piece_rasters[cut_id] = (piece_dataset, region_id_of_code, region_index_of_code, member_of_code, window)
            # a piece raster is unsigned and on the run's grid, its window placed where the basin's rectangle says (an
            # Int32 one would hand the kernel negative codes); the origin and size are checked too: the
            # read below takes the window from the basin's rectangle and never looks at the raster's own origin
            expected_transform = grid.transform * rasterio.Affine.translation(window[2], window[0])
            origin_tolerance = 1e-6 * abs(grid.transform.a)
            if piece_dataset.dtypes[0] not in ("uint8", "uint16", "uint32") or piece_dataset.crs != grid.crs or \
                    abs(piece_dataset.transform.a - grid.transform.a) > 1e-12 or abs(piece_dataset.transform.e - grid.transform.e) > 1e-12 or \
                    piece_dataset.transform.b != 0.0 or piece_dataset.transform.d != 0.0 or \
                    abs(piece_dataset.transform.c - expected_transform.c) > origin_tolerance or \
                    abs(piece_dataset.transform.f - expected_transform.f) > origin_tolerance or \
                    piece_dataset.width != window[3] - window[2] or piece_dataset.height != window[1] - window[0]:
                raise FlowDivideError("the piece raster %s is not an unsigned raster on the grid of the run, over the window of "
                                      "basin %d (rows %d .. %d, columns %d .. %d)"
                                      % (cuts[cut_id][1], cut_id, window[0], window[1], window[2], window[3]))
            # the outlet pixel of every used piece is its own in the piece raster, and the pixel it flows into is its
            # parent's: a flows_into that named a sibling, with the inlet left where it was, would pass
            # the flow-direction check above
            rows_by_code = {int(row["piece"]): row for _, row in table.iterrows()}

            def used_ancestor_of(code):
                while code > 0 and int(rows_by_code[code]["used"]) != 1:
                    code = int(rows_by_code[code]["sub_of"])
                return code

            def piece_code_at(mosaic_row, mosaic_col):
                piece_row = mosaic_row - window[0]
                piece_col = mosaic_col - window[2]
                if grid.periodic:
                    piece_col %= grid.ncol
                if not (0 <= piece_row < piece_dataset.height and 0 <= piece_col < piece_dataset.width):
                    return 0
                code = int(piece_dataset.read(1, window=Window(piece_col, piece_row, 1, 1))[0, 0])
                return code if code in rows_by_code else 0

            for code, row in rows_by_code.items():
                if int(row["used"]) != 1:
                    continue
                if used_ancestor_of(piece_code_at(int(row["outlet_row"]), int(row["outlet_col"]))) != code:
                    raise FlowDivideError("basin %d piece %d: its outlet (row %d, column %d) belongs to another piece in the "
                                          "piece raster" % (cut_id, code, int(row["outlet_row"]), int(row["outlet_col"])))
                parent = used_ancestor_of(int(row["flows_into"]))
                if parent and used_ancestor_of(piece_code_at(int(row["parent_inlet_row"]), int(row["parent_inlet_col"]))) != parent:
                    raise FlowDivideError("basin %d piece %d: the pixel its outlet flows into (row %d, column %d) is not its "
                                          "parent's in the piece raster" % (cut_id, code, int(row["parent_inlet_row"]),
                                                                            int(row["parent_inlet_col"])))
        rgn_temporary = outputs["rgn"] + ".partial.tif"
        with rasterio.open(bsn_path) as bsn_dataset, rasterio.open(rgn_temporary, "w", **raster_profile(grid, "uint32", 0)) as rgn_out:
            check_raster_on_the_grid(bsn_dataset, grid, bsn_path)
            if bsn_dataset.dtypes[0] != "uint32":         # no negative id can come out of it
                raise FlowDivideError("the basin mask %s is %s, not uint32" % (bsn_path, bsn_dataset.dtypes[0]))
            for row0 in range(0, grid.nrow, strip_rows):
                nrow = min(strip_rows, grid.nrow - row0)
                bsn_strip = bsn_dataset.read(1, window=Window(0, row0, grid.ncol, nrow))
                # a basin id past the basin table is refused before the kernel indexes with it (Numba does not
                # check bounds, so it would read and write past the lookup arrays)
                if bsn_strip.size and int(bsn_strip.max()) > basin_count:
                    raise FlowDivideError("the basin mask holds basin %d in rows %d .. %d, past the %d basins of the table"
                                          % (int(bsn_strip.max()), row0, row0 + nrow - 1, basin_count))
                rgn_strip = np.zeros((nrow, grid.ncol), np.uint32)
                _mask_strip(bsn_strip, region_id_of_basin, region_index_of_basin, member_of_basin, rgn_strip,
                            r_count, r_row_min, r_row_max, r_col_min, r_col_max, r_col_min_shift, r_col_max_shift, member_count, row0, grid.ncol, grid.periodic)
                for cut_id, (dataset, region_id_of_code, region_index_of_code, member_of_code, window) in piece_rasters.items():
                    if window[1] <= row0 or window[0] >= row0 + nrow:
                        continue
                    read_row0 = max(row0, window[0])
                    read_row1 = min(row0 + nrow - 1, window[1] - 1)          # the last row read, inside both
                    piece_strip = dataset.read(1, window=Window(0, read_row0 - window[0], window[3] - window[2], read_row1 - read_row0 + 1))
                    status = _mask_strip_pieces(piece_strip, bsn_strip[read_row0 - row0:read_row1 - row0 + 1], np.uint32(cut_id), region_id_of_code, region_index_of_code, member_of_code,
                                                rgn_strip[read_row0 - row0:read_row1 - row0 + 1], window[2],
                                                r_count, r_row_min, r_row_max, r_col_min, r_col_max, r_col_min_shift, r_col_max_shift, member_count, read_row0, grid.ncol, grid.periodic)
                    if status != 0:
                        raise FlowDivideError("the piece raster of basin %d does not agree with the basin mask in rows %d .. %d (status %d: 1 a pixel of another basin, 2 a pixel with a region already, 3 a code without a region)" % (cut_id, read_row0, read_row1, status))
                missing = int(((bsn_strip != 0) & (rgn_strip == 0)).sum())
                if missing:
                    raise FlowDivideError("%d land pixels in rows %d .. %d got no region" % (missing, row0, row0 + nrow - 1))
                rgn_out.write(rgn_strip, 1, window=Window(0, row0, grid.ncol, nrow))
                if (row0 // strip_rows) % 20 == 0:
                    log(tag, "mask rows %d .. %d of %d" % (row0, row0 + nrow - 1, grid.nrow))
    finally:
        for dataset, _, _, _, _ in piece_rasters.values():
            dataset.close()
    # the checks: every region and every piece holds in the mask what its members say
    for region in regions:
        index = region_index_of_id[region.region_id]
        if int(r_count[index]) != region.pixels:
            raise FlowDivideError("region %d: %d pixels in the mask, %d from its members" % (region.region_id, int(r_count[index]), region.pixels))
    for _, row in pieces.iterrows():
        if int(member_count[int(row["piece_id"])]) != int(row["piece_grid_count"]):
            raise FlowDivideError("piece %d: %d pixels in the mask, %d from the cut" % (int(row["piece_id"]), int(member_count[int(row["piece_id"])]), int(row["piece_grid_count"])))
    # and every whole basin with a region its table row's pixels (two basins wrong by +1 and -1
    # would pass the region's total)
    whole_ids = np.flatnonzero(region_id_of_basin[1:] != 0) + 1
    table_pixels = basins["basin_grid_count"].to_numpy(np.int64)
    disagreeing = whole_ids[member_count[whole_ids] != table_pixels[whole_ids - 1]]
    if disagreeing.size:
        first = int(disagreeing[0])
        raise FlowDivideError("%d whole basins hold another number of pixels in the mask than in the table; basin %d: %d in "
                              "the mask, %d in the table" % (disagreeing.size, first, int(member_count[first]), int(table_pixels[first - 1])))
    col_min_unrolled, col_max_unrolled, _ = unroll_rectangles(grid, r_col_min.astype(np.int64), r_col_max.astype(np.int64), r_col_min_shift, r_col_max_shift)
    if grid.periodic:
        # the box measured in the mask is put in the frame the region was planned in (the frame of its
        # members' rectangles, unrolled where they lie across the seam): the
        # frame in which the box lies inside the planned rectangle
        for region in regions:
            index = region_index_of_id[region.region_id]
            planned = region.rectangle
            for shift in (0, grid.ncol, -grid.ncol):
                if planned[2] <= col_min_unrolled[index] + shift and col_max_unrolled[index] + shift <= planned[3]:
                    col_min_unrolled[index] += shift
                    col_max_unrolled[index] += shift
                    break
            else:
                raise FlowDivideError("region %d: its pixels in the mask (columns %d .. %d) lie outside its planned rectangle (%d .. %d)" % (
                    region.region_id, col_min_unrolled[index], col_max_unrolled[index], planned[2], planned[3]))
    # the topological level of a region: 0 when it depends on no other, else one more than the highest it depends on
    depends = {}
    for _, row in pieces.iterrows():
        if int(row["parent_piece_id"]) != 0:
            source = int(row["region_id"])
            target = region_of_piece[int(row["parent_piece_id"])]
            if source != target:
                depends.setdefault(source, set()).add(target)
    level_of = {}

    def level(region_id):
        if region_id not in level_of:
            level_of[region_id] = 0 if region_id not in depends else 1 + max(level(t) for t in depends[region_id])
        return level_of[region_id]

    rows = []
    for region in sorted(regions, key=lambda r: region_row_order_key(r.region_id)):
        index = region_index_of_id[region.region_id]
        window = region.window(grid)
        window_pixels = (window[1] - window[0]) * (window[3] - window[2])
        box = grid.pixel_box_lon_lat(*window)
        # an island or seam region carries no code; its level1 is the continent's
        # an island group takes the Level-01 region it is numbered under; the rest of the groups without a
        # code take the continent of the run
        # which Level-03 units this region holds: the id of a divided unit's part only hints at the run
        # (9112), and this column is what says it.  The codes are joined by commas, not
        # spaces: the table itself is space separated, and "-" when the region holds no coded unit (an
        # island group, a region of pieces)
        rows.append({"region_id": region.region_id,
                     "level3_members": " ".join(str(code) for code in sorted(region.level3_members)),
                     "level1_code": max(region.level1_code, 0) if region.level3_lead > 0 else (max(region.level1_code, 0) or level1_of_continent),
                     "level2_code": max(region.level2_code, 0), "level3_code": region.level3,
                     "row_min": window[0], "row_max": window[1], "col_min": window[2], "col_max": window[3],
                     "nrow": window[1] - window[0], "ncol": window[3] - window[2], "region_grid_count": region.pixels,
                     "basin_count": int(region.basins.size) if region.cut_basin_id == 0 else 1, "piece_count": len(region.piece_codes), "window_grid_count": window_pixels,
                     "fill_percent": 100.0 * region.pixels / window_pixels, "minlon": box[0], "minlat": box[1], "maxlon": box[2], "maxlat": box[3],
                     "cut_basin_id": region.cut_basin_id, "bbox_row_min": int(r_row_min[index]), "bbox_row_max": int(r_row_max[index]),
                     "bbox_col_min": int(col_min_unrolled[index]), "bbox_col_max": int(col_max_unrolled[index]), "topological_level": level(region.region_id)})
    region_table = pd.DataFrame(rows)[fd_tables.REGION_TABLE_COLUMNS]
    # the four files of the partition: the tables of fd_tables, the region map with
    # its key (capacity, block, grouping), and a .done marker on each
    summary = ("%d regions at %d pixels, blocks of %d, groups %s; every basin in one region or cut into pieces"
               % (len(regions), capacity_pixels, grid.block_pixels, grouping))
    fd_tables.write_region_table(region_table, outputs["regions"], summary)
    fd_tables.write_piece_table(pieces[fd_tables.PIECE_TABLE_COLUMNS], outputs["pieces"], summary)
    fd_tables.write_basin_map(outputs["basin_region"], np.asarray(region_id_of_basin, np.int64), continent, "region",
                              fd_tables.region_map_key(capacity_pixels, grid.block_pixels, grouping))
    MERGE_COLUMNS = ["kept_region_id", "absorbed_region_id", "kept_level3", "absorbed_level3", "joined_window_pixels", "extra_window_pixels", "joined_land_pixels"]
    write_table(pd.DataFrame(merges) if merges else pd.DataFrame({column: [] for column in MERGE_COLUMNS}), outputs["merges"])
    publish(rgn_temporary, outputs["rgn"])
    fd_tables.write_done_marker(outputs["rgn"], summary)
    crossings = 0
    for _, row in pieces.iterrows():
        if int(row["parent_piece_id"]) != 0 and region_of_piece[int(row["parent_piece_id"])] != int(row["region_id"]):
            crossings += 1
    report = {"regions": len(regions), "regions_of_pieces": sum(1 for r in regions if r.kind == 5), "cut_basins": len(cut_set), "pieces": len(pieces),
              "merges": len(merges), "capacity_pixels": capacity_pixels, "merge_policy": merge_policy,
              "largest_window_grid_count": int(region_table["window_grid_count"].max()), "exit_pixels_between_regions": crossings,
              "index_bits": 32 if int(region_table["window_grid_count"].max()) <= 2 ** 31 - 1 else 64, "seconds": round(time.time() - started, 1)}
    write_json(outputs["regions"] + ".report.json", report)
    log(tag, "written %s: %d regions, %d cut basins in %d pieces, %d exit pixels between regions, largest window %.3f x the capacity (%d-bit indices)" % (
        outputs["regions"], len(regions), len(cut_set), len(pieces), crossings, report["largest_window_grid_count"] / capacity_pixels, report["index_bits"]))
    return report


# =============================================================================
#  [9] FD1.7  the global basin id (the 30 m grids, once every continent is done)
# =============================================================================

@njit(cache=True)
def _merge_the_runs(area, grid_count, run_start, run_count_of, global_id):
    """the global ids of the basins of several runs laid end to end (run r at run_start[r] .. +run_count_of[r]): the
    runs' own orders merged, the larger area first, then the larger pixel count, then the earlier run, as
    number_the_basins_globally orders them, one comparison at a time"""
    runs = run_start.size
    next_of = np.zeros(runs, np.int64)
    total = 0
    for r in range(runs):
        total += run_count_of[r]
    for g in range(1, total + 1):
        chosen = -1
        for r in range(runs):
            if next_of[r] >= run_count_of[r]:
                continue
            if chosen < 0:
                chosen = r
                continue
            i = run_start[r] + next_of[r]
            j = run_start[chosen] + next_of[chosen]
            if area[i] > area[j] or (area[i] == area[j] and grid_count[i] > grid_count[j]):
                chosen = r
        global_id[run_start[chosen] + next_of[chosen]] = g
        next_of[chosen] += 1


def fd1_7_global_basin_id(runs, global_table_path, tag="fd1.7"):
    """The global number of every basin over the runs named: runs is a list of
    (run name, Level-01 code, basin table path); the runs are taken in Level-01 code order.  Every run's table (written
    by fd1.6 or fd1.7) is written again with global_basin_id and the same rows; the global table, every basin after a
    "run" column in global_basin_id order, is written to global_table_path.  The ids hold only for the runs named."""
    started = time.time()
    runs = sorted(runs, key=lambda item: item[1])
    if len(set(code for _, code, _ in runs)) != len(runs):
        raise FlowDivideError("two runs share a Level-01 code")
    tables = []
    for run, _, path in runs:
        _, stage = fd_tables.basin_table_stage(path)
        if stage not in ("fd1.6", "fd1.7"):
            raise FlowDivideError("%s was last written by %s; the global ids are given after fd1.6" % (path, stage or "a step that did not say"))
        table = fd_tables.read_basin_table(path)
        if (table["basin_area_km2"] == fd_tables.NO_AREA).any():
            raise FlowDivideError("%s has basins without an area; the global order is by area" % path)
        tables.append(table)
    counts = np.array([len(t) for t in tables], np.int64)
    if counts.sum() > 2 ** 32 - 1:
        raise FlowDivideError("%d basins are more than a uint32 id can number" % counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    area = np.concatenate([t["basin_area_km2"].to_numpy(np.float64) for t in tables])
    grid_count = np.concatenate([t["basin_grid_count"].to_numpy(np.int64) for t in tables])
    global_id = np.zeros(area.size, np.int64)
    _merge_the_runs(area, grid_count, starts, counts, global_id)
    # the global table of an earlier run stops counting as done before any run's table changes
    if os.path.exists(global_table_path + ".done"):
        os.remove(global_table_path + ".done")
    written = []
    for (run, _, path), table, start, count in zip(runs, tables, starts, counts):
        table = table.copy()
        table["global_basin_id"] = global_id[start:start + count]
        fd_tables.write_basin_table(table, path, "fd1.7", "global_basin_id of %d basins over the runs named, largest area first" % area.size)
        written.append((run, table))
    ensure_directory(os.path.dirname(global_table_path))
    fd_tables.write_global_basin_table(written, global_table_path)
    log(tag, "%d basins over %d runs; the global table is %s (%.0f s)" % (area.size, len(runs), global_table_path, time.time() - started))
    return {"basins": int(area.size), "runs": [run for run, _, _ in runs], "seconds": round(time.time() - started, 1)}
