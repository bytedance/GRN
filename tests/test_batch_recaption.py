"""Regression tests for 'tools/batch_recaption.py' (issue #35).

The captioning tool depends on torch, transformers, cv2 and a 32B Qwen
checkpoint. None of those are available here, so the heavy dependencies are
replaced by stub modules: these tests exercise the import path, the explicit
model loader and the dataset adapter, and never download or run a model.
"""

import importlib
import os
import pathlib
import sys
import tempfile
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Modules replaced by stubs, and modules re-imported per test.
STUBBED = (
    "torch",
    "torch.utils",
    "torch.utils.data",
    "numpy",
    "cv2",
    "PIL",
    "PIL.Image",
    "tqdm",
    "transformers",
    "grn.dataset.dataset_joint_vi",
    "grn.utils.video_decoder",
    "tools.batch_recaption",
)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _Dataset:
    """Stand-in for 'torch.utils.data.Dataset'."""

    def __init__(self, *args, **kwargs):
        pass


class _VideoDecoder:
    """Stand-in for 'grn.utils.video_decoder.EncodedVideoDecord'."""

    raise_on_open = False

    def __init__(self, *args, **kwargs):
        if type(self).raise_on_open:
            raise RuntimeError("decoder is not available in this test")


def _install_stubs(model_loader):
    """Install stand-ins for the heavy dependencies of the captioning tool.

    'model_loader' is called whenever the tool asks transformers for the
    checkpoint, so a test can tell an import-time load from an explicit one.
    """

    class _ConditionalGeneration:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model_loader(*args, **kwargs)

    class _Processor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model_loader(*args, **kwargs)

    torch_utils_data = _module("torch.utils.data", Dataset=_Dataset, DataLoader=_Dataset)
    torch_utils = _module("torch.utils", data=torch_utils_data)
    sys.modules["torch"] = _module("torch", bfloat16=object(), utils=torch_utils)
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

    sys.modules["numpy"] = _module("numpy")
    sys.modules["cv2"] = _module("cv2")
    sys.modules["PIL"] = _module("PIL")
    sys.modules["PIL.Image"] = _module("PIL.Image", Image=_module("PIL.Image.Image"))
    sys.modules["tqdm"] = _module("tqdm", tqdm=object)
    sys.modules["transformers"] = _module(
        "transformers",
        Qwen3VLForConditionalGeneration=_ConditionalGeneration,
        AutoProcessor=_Processor,
    )

    # 'local_or_download' does not exist in the repository, so the stub that
    # stands in for the dataset module does not provide it either.
    sys.modules["grn.dataset.dataset_joint_vi"] = _module("grn.dataset.dataset_joint_vi")
    sys.modules["grn.utils.video_decoder"] = _module(
        "grn.utils.video_decoder", EncodedVideoDecord=_VideoDecoder
    )


def _import_tool(model_loader):
    sys.modules.pop("tools.batch_recaption", None)
    _install_stubs(model_loader)
    return importlib.import_module("tools.batch_recaption")


class BatchRecaptionTest(unittest.TestCase):
    def setUp(self):
        self._saved = {name: sys.modules[name] for name in STUBBED if name in sys.modules}
        for name in STUBBED:
            sys.modules.pop(name, None)
        _VideoDecoder.raise_on_open = False

    def tearDown(self):
        _VideoDecoder.raise_on_open = False
        for name in STUBBED:
            sys.modules.pop(name, None)
        sys.modules.update(self._saved)

    def test_import_does_not_load_the_captioning_model(self):
        loaded = []

        def record(*args, **kwargs):
            loaded.append(args[0] if args else kwargs.get("model_id"))
            return object()

        module = _import_tool(record)

        self.assertEqual(loaded, [], "importing the tool must not load the captioning model")
        self.assertTrue(callable(module.main))

    def test_import_survives_an_unavailable_model(self):
        def unavailable(*args, **kwargs):
            raise OSError("checkpoint is not available")

        module = _import_tool(unavailable)

        self.assertTrue(callable(module.load_model))

    def test_load_model_reports_an_actionable_error(self):
        def unavailable(*args, **kwargs):
            raise OSError("checkpoint is not available")

        module = _import_tool(unavailable)

        with self.assertRaises(RuntimeError) as raised:
            module.load_model()

        message = str(raised.exception)
        self.assertIn("Qwen3-VL-32B-Instruct", message)
        self.assertIn("checkpoint is not available", message)

    def test_processor_is_loaded_on_demand_and_reused(self):
        loaded = []

        def record(*args, **kwargs):
            loaded.append(kwargs)
            return object()

        module = _import_tool(record)
        self.assertEqual(loaded, [], "the processor must not be loaded at import time")

        first = module.load_processor()
        second = module.load_processor()

        self.assertIs(first, second)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].get("padding_side"), "left")

    def test_missing_local_video_is_skipped_without_a_download_helper(self):
        module = _import_tool(lambda *args, **kwargs: object())
        meta = {"video_path": "/nonexistent/clip.mp4"}

        sample = module.VideoDataset([meta], 1)[0]

        self.assertIsNone(sample["frames"])
        self.assertEqual(sample["meta"]["video_path"], "/nonexistent/clip.mp4")

    def test_source_video_is_not_deleted_when_decoding_fails(self):
        module = _import_tool(lambda *args, **kwargs: object())
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"not a real video")
            _VideoDecoder.raise_on_open = True

            sample = module.VideoDataset([{"video_path": str(video_path)}], 1)[0]

            self.assertIsNone(sample["frames"])
            self.assertTrue(video_path.exists(), "the caller-owned source video must be kept")


if __name__ == "__main__":
    unittest.main()
