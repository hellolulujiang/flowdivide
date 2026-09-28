"""The default style of a GeoPackage view, checked on a made-up layer.

    python3 test_color_id_style.py

A GeoPackage view carries a QGIS categorized fill on color_id in its layer_styles table, marked as the default and
named in gpkg_contents, so that QGIS colours the layer when it is opened.
"""
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd2_views
from fd1_partition import FlowDivideError

FAILED = []


def check(condition, what):
    print("  %s %s" % ("ok  " if condition else "FAIL", what))
    if not condition:
        FAILED.append(what)


def made_up_view(path, largest_colour):
    """a GeoPackage of three squares with color_id 1 .. largest_colour, as fd2 writes a view"""
    import geopandas
    from shapely.geometry import MultiPolygon, box
    frame = geopandas.GeoDataFrame({"basin_id": [1, 2, 3], "color_id": [1, 2, largest_colour]},
                                   geometry=[MultiPolygon([box(index, 0, index + 1, 1)]) for index in range(3)], crs="EPSG:4326")
    frame.to_file(path, driver="GPKG", layer="basins", engine="pyogrio")


def the_style_is_written_as_the_default():
    print("the style is written as the default of the layer")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "view.gpkg")
        made_up_view(path, 3)
        fd2_views.embed_color_id_style(path, "basins", 3)
        fd2_views.check_color_id_style(path, "basins")
        connection = sqlite3.connect(path)
        rows = connection.execute("SELECT f_table_name, f_geometry_column, styleName, useAsDefault FROM layer_styles").fetchall()
        listed = connection.execute("SELECT data_type FROM gpkg_contents WHERE table_name = 'layer_styles'").fetchall()
        connection.close()
        check(rows == [("basins", "geom", "color_id", 1)], "one row in layer_styles, on the layer and its geometry column, the default")
        check(listed == [("attributes",)], "layer_styles is named in gpkg_contents")
        # written twice, still one style
        fd2_views.embed_color_id_style(path, "basins", 3)
        connection = sqlite3.connect(path)
        count = connection.execute("SELECT COUNT(*) FROM layer_styles").fetchone()[0]
        contents = connection.execute("SELECT COUNT(*) FROM gpkg_contents WHERE table_name = 'layer_styles'").fetchone()[0]
        connection.close()
        check(count == 1 and contents == 1, "a second write replaces the style instead of adding one")
        import geopandas
        read = geopandas.read_file(path, layer="basins", engine="pyogrio")
        check(list(read["color_id"]) == [1, 2, 3], "the layer reads as before")


def a_view_without_a_cell_is_styled():
    print("a view in which no object holds a cell is styled too")
    import geopandas
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "view.gpkg")
        frame = geopandas.GeoDataFrame({"basin_id": [1, 2], "color_id": [-9999, -9999]},
                                       geometry=geopandas.GeoSeries([None, None], crs="EPSG:4326"), crs="EPSG:4326")
        frame.to_file(path, driver="GPKG", layer="basins", engine="pyogrio", geometry_type="MultiPolygon")
        try:
            fd2_views.embed_color_id_style(path, "basins", 0)
            fd2_views.check_color_id_style(path, "basins")
            styled = True
        except FlowDivideError:
            styled = False
        check(styled, "the layer, declared MultiPolygon as fd2 writes it, takes the style")


def a_colour_past_the_palette_is_refused():
    print("a color_id past the palette is refused")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "view.gpkg")
        largest = len(fd2_views.COLOR_ID_STYLE_PALETTE) + 1
        made_up_view(path, largest)
        try:
            fd2_views.embed_color_id_style(path, "basins", largest)
            refused = False
        except FlowDivideError:
            refused = True
        check(refused, "color_id %d stops the step" % largest)


def the_palette_is_the_figures_okabe_ito():
    print("the palette of the style is the okabe-ito palette of PALETTES")
    check(fd2_views.PALETTES["okabe-ito"] == [hex_colour for hex_colour, _ in fd2_views.COLOR_ID_STYLE_PALETTE],
          "the same hues in the same order")


def main():
    the_style_is_written_as_the_default()
    a_view_without_a_cell_is_styled()
    a_colour_past_the_palette_is_refused()
    the_palette_is_the_figures_okabe_ito()
    print("")
    if FAILED:
        print("FAILED %d:" % len(FAILED))
        for line in FAILED:
            print("  " + line)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
