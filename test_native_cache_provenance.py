"""Small marker tests: no production products or calculations are used."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from flowdivide import Chain, FlowDivideError, file_identity


class NativeCacheProvenance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flowdivide_native_")
        self.folder = Path(self.temporary.name)
        self.native = self.folder / "native input=.tif"
        self.native.write_bytes(b"original direction input")
        self.chain = object.__new__(Chain)
        self.chain.layout = SimpleNamespace(root=str(self.folder / "run"), external={})
        Path(self.chain.layout.root, "_logs").mkdir(parents=True)
        self.chain.dataset = SimpleNamespace(raw_dir=str(self.native))

    def tearDown(self):
        self.temporary.cleanup()

    def native_identity(self):
        native = Path(self.chain.dataset.raw_dir)
        paths = sorted(native.glob("*.tif")) if native.is_dir() else [native]
        return ";".join("%s:%s" % (path.name, file_identity(str(path))) for path in paths)

    def legacy_markers(self, identity=None):
        # Exact legacy argument format, including paths with whitespace/input= in their name.
        arguments = "recode input convention=merit input=" + (identity or self.native_identity())
        Path(self.chain.marker("fd1.0")).write_text("args: " + arguments + "\nsignature: recode_old\n")
        Path(self.chain.marker("fd1.1")).write_text("args: accumulation\nsignature: accumulation_old\n")

    def consumer(self):
        return Chain.Step("fd3_ldn", "distance", lambda: None, (), depends_on=["fd1.1"])

    def signature(self):
        step = self.consumer()
        return self.chain.signature(step, {step.label: step}, {})

    def test_unchanged_legacy_native_allows_cached_dependency(self):
        self.legacy_markers()
        self.assertIsNone(self.chain.marker_external_that_changed("fd1.1"))
        self.assertEqual(self.signature(), self.signature())

    def test_changed_legacy_native_refuses_reuse_of_indirect_dependency(self):
        self.legacy_markers()
        self.signature()
        self.native.write_bytes(b"replaced direction input of another size")
        with self.assertRaisesRegex(FlowDivideError, "changed since"):
            self.signature()

    def test_same_size_content_update_is_detected_by_mtime(self):
        self.legacy_markers()
        stat = self.native.stat()
        self.native.write_bytes(b"x" * stat.st_size)
        os.utime(self.native, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
        self.assertEqual(self.chain.marker_external_that_changed("fd1.1"), str(self.native))

    def test_missing_native_or_unrecorded_legacy_input_is_refused(self):
        self.legacy_markers()
        self.native.rename(self.folder / "moved_native")
        self.assertEqual(self.chain.marker_external_that_changed("fd1.1"), str(self.native))
        self.native.write_bytes(b"present again")
        Path(self.chain.marker("fd1.0")).write_text("args: legacy without input identity\n")
        self.assertEqual(self.chain.marker_external_that_changed("fd1.1"), str(self.native))

    def test_tile_addition_deletion_and_replacement_change_identity(self):
        tiles = self.folder / "tiles"
        tiles.mkdir()
        first, second = tiles / "a.tif", tiles / "b.tif"
        first.write_bytes(b"a")
        self.chain.dataset.raw_dir = str(tiles)
        self.legacy_markers()
        second.write_bytes(b"b")
        self.assertIsNotNone(self.chain.marker_external_that_changed("fd1.1"))
        self.legacy_markers()
        second.rename(tiles / "b.moved")
        self.assertIsNotNone(self.chain.marker_external_that_changed("fd1.1"))
        self.legacy_markers()
        first.write_bytes(b"replacement of a")
        self.assertIsNotNone(self.chain.marker_external_that_changed("fd1.1"))

    def test_new_markers_include_native_and_all_tiles_even_inside_run_root(self):
        tiles = Path(self.chain.layout.root) / "native"
        tiles.mkdir()
        first = tiles / "a.tif"
        first.write_bytes(b"a")
        self.chain.dataset.raw_dir = str(tiles)
        identities = self.chain.external_identities(self.consumer())
        self.assertEqual(len(identities), 2)
        self.assertTrue(any(str(first) in row for row in identities))
        self.assertTrue(any(str(tiles)+" " in row for row in identities))


if __name__ == "__main__":
    unittest.main()
