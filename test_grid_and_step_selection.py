"""Regression checks for FD3 grid alignment and --only, using temporary tiny files.

    python test_grid_and_step_selection.py

Synthetic inputs exercise validation only; their values are not research results.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import rasterio
from affine import Affine

import fd3_attributes as fd3
import flowdivide


class GridAlignment(unittest.TestCase):
    def setUp(self):
        self.reference = self.dataset(Affine(1 / 1200, 0, -180, 0, -1 / 1200, 90))

    @staticmethod
    def dataset(transform):
        return SimpleNamespace(width=432000, height=208800, crs="EPSG:4326", transform=transform)

    def check_grid(self, transform, channel=False):
        candidate = self.dataset(transform)
        fd3._check_on_the_grid_of_the_flow_directions(
            self.reference, "dir.tif", candidate if channel else None, None if channel else candidate)

    def test_matching_and_roundoff_grids(self):
        self.check_grid(self.reference.transform)
        self.check_grid(Affine(1 / 1200 + 1e-12, 0, -180, 0, -1 / 1200, 90))

    def test_accumulated_pixel_width_difference(self):
        with self.assertRaises(fd3.FlowDivideError):
            self.check_grid(Affine(1 / 1200 + 9e-10, 0, -180, 0, -1 / 1200, 90))

    def test_accumulated_pixel_height_difference(self):
        with self.assertRaises(fd3.FlowDivideError):
            self.check_grid(Affine(1 / 1200, 0, -180, 0, -1 / 1200 + 9e-10, 90))

    def test_accumulated_rotation_in_channel_mask(self):
        with self.assertRaises(fd3.FlowDivideError):
            self.check_grid(Affine(1 / 1200, 9e-10, -180, 0, -1 / 1200, 90), channel=True)

    def test_origin_shift(self):
        with self.assertRaises(fd3.FlowDivideError):
            self.check_grid(Affine(1 / 1200, 0, -180 + 0.02 / 1200, 0, -1 / 1200, 90))

    def test_nonfinite_transform(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(fd3.FlowDivideError):
                self.check_grid(Affine(value, 0, -180, 0, -1 / 1200, 90))

    def test_dimension_and_crs_mismatch(self):
        for field, value in (("width", 431999), ("crs", "EPSG:3857")):
            candidate = self.dataset(self.reference.transform)
            setattr(candidate, field, value)
            with self.subTest(field=field), self.assertRaises(fd3.FlowDivideError):
                fd3._check_on_the_grid_of_the_flow_directions(self.reference, "dir.tif", None, candidate)


class StepSelection(unittest.TestCase):
    def test_failed_preflight_preserves_chain_and_files(self):
        with tempfile.TemporaryDirectory(prefix="flowdivide_preflight_") as temporary:
            folder = Path(temporary)
            native = folder / "native.tif"
            with rasterio.open(native, "w", driver="GTiff", height=2, width=2, count=1,
                               dtype="uint8", crs="EPSG:3857", transform=Affine(1, 0, 0, 0, -1, 2),
                               nodata=247) as raster:
                raster.write(np.array([[4, 4], [0, 0]], dtype=np.uint8), 1)
            dataset = flowdivide.Dataset("probe", str(native), "merit", [(str(folder / "unused.geojson"), None)],
                                         block_pixels=1, capacities=[("2^31", 2100000000, 2)],
                                         hilbert=False, figures=True, views=[("10m", 10)])
            chain = flowdivide.Chain(dataset, str(folder / "output"), ["fd1", "fd2"], [], "2^31", [],
                                     32, 1, 1, False, separate_processes=False)
            chain.timing = False
            chain.only = {"fd1.0", "misspelled-step"}
            before_chain = dict(vars(chain))
            before_layout = dict(vars(chain.layout))
            before_dataset = dict(vars(dataset))
            with self.assertRaisesRegex(flowdivide.FlowDivideError, "--only names unknown steps"):
                chain.run()
            self.assertEqual(vars(chain), before_chain)
            self.assertEqual(vars(chain.layout), before_layout)
            self.assertEqual(vars(dataset), before_dataset)
            self.assertEqual(chain.held_back, [])
            self.assertEqual(chain.done, set())
            self.assertTrue(dataset.figures)  # the projected summary must not mutate the original dataset
            self.assertEqual([path for path in Path(chain.layout.root).rglob("*") if path.is_file()], [])
            chain.only = {"fd1.0"}
            with contextlib.redirect_stdout(io.StringIO()):
                chain.run()
            self.assertTrue(Path(chain.marker("fd1.0")).is_file())

    def test_cli_selections(self):
        with tempfile.TemporaryDirectory(prefix="flowdivide_selection_") as temporary:
            folder = Path(temporary)
            native = folder / "native.tif"
            groups = folder / "groups.geojson"
            output = folder / "output"
            with rasterio.open(native, "w", driver="GTiff", height=2, width=2, count=1,
                               dtype="uint8", crs="EPSG:4326", transform=Affine(1, 0, 0, 0, -1, 2),
                               nodata=247) as raster:
                raster.write(np.array([[4, 4], [0, 0]], dtype=np.uint8), 1)
            groups.write_text(json.dumps({
                "type": "FeatureCollection", "features": [{
                    "type": "Feature", "properties": {"PFAF_ID": 611},
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
                }]}))
            environment = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1", NUMBA_CACHE_DIR=str(folder / "numba_cache"))
            command = [sys.executable, str(Path(__file__).with_name("flowdivide.py")), "run", "probe",
                       "--dir", str(native), "--convention", "merit", "--out-root", str(output),
                       "--block", "1", "--capacities", "2^31:2", "--group-vectors", str(groups),
                       "--no-hilbert", "--steps", "fd1", "--in-process", "--no-open", "--timing", "off"]

            def run_only(selection):
                return subprocess.run(command + ["--only", selection], env=environment,
                                      capture_output=True, text=True, timeout=90)

            result = run_only("fd1.0,misspelled-step")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--only names unknown steps", result.stderr)
            self.assertEqual([path for path in output.rglob("*") if path.is_file()], [])
            # fd1.0 is planned separately. A valid later step must survive that initial plan.
            for selection in ("fd1.0", "fd1.4_level3"):
                result = run_only(selection)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((output / "probe" / "_logs" / (selection + ".ok")).is_file())
            with rasterio.open(output / "probe" / "global" / "fineresolution" / "l3" / "l3_probe_3600s.tif") as raster:
                np.testing.assert_array_equal(raster.read(1), np.full((2, 2), 611, dtype=np.uint16))
            for selection in ("misspelled-step", "fd3_shv_2^31"):
                with self.subTest(selection=selection):
                    result = run_only(selection)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("--only names unknown steps", result.stderr)
                    self.assertNotIn("==== finished", result.stdout)
            for selection in ("", " , "):
                with self.subTest(selection=selection):
                    result = run_only(selection)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("--only must name at least one step", result.stderr)
                    self.assertNotIn("==== finished", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
