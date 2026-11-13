import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class VideoDecoderOptionalDependencyTests(unittest.TestCase):
    def test_module_imports_without_decord_and_constructor_is_actionable(self):
        module_path = Path(__file__).parents[1] / "grn" / "utils" / "video_decoder.py"
        spec = importlib.util.spec_from_file_location("video_decoder_without_decord", module_path)
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, {"decord": None}):
            spec.loader.exec_module(module)

        self.assertFalse(module._HAS_DECORD)
        with self.assertRaisesRegex(ImportError, "decord is required"):
            module.EncodedVideoDecord("missing.mp4")


if __name__ == "__main__":
    unittest.main()
