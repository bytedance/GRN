"""Offline checkpoint routing tests; no model weights or GPU are required."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class PipelineCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = Mock(side_effect=lambda **kw: kw["filename"])
        hub.snapshot_download = Mock(return_value=self.directory.name)
        self.hub = hub
        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        torch.device = lambda device: device
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        infer = types.ModuleType("grn.utils_t2iv.infer")
        infer.load_tokenizer = Mock(return_value=(object(), object()))
        infer.load_transformer = Mock(return_value=object())
        infer.gen_one_example = Mock()
        infer.images2video = Mock()
        loader = types.ModuleType("grn.utils_t2iv.load")
        loader.load_visual_tokenizer = Mock(return_value=object())
        schedule = types.ModuleType("grn.schedules")
        schedule.get_encode_decode_func = Mock(return_value=(None,) * 4)
        dynamic = types.ModuleType("grn.schedules.dynamic_resolution")
        dynamic.get_dynamic_resolution_meta = Mock()
        dynamic.get_first_full_spatial_size_scale_index = Mock()
        pil = types.ModuleType("PIL")
        pil.Image = Mock()
        modules = {
            "huggingface_hub": hub, "torch": torch, "numpy": Mock(),
            "PIL": pil, "grn.utils_t2iv.infer": infer,
            "grn.utils_t2iv.load": loader, "grn.schedules": schedule,
            "grn.schedules.dynamic_resolution": dynamic,
        }
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, modules).start()
        path = Path(__file__).resolve().parents[1] / "tools" / "grn_pipeline.py"
        spec = importlib.util.spec_from_file_location("checkpoint_pipeline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pipeline = module.GRNPipeline

    def test_supported_checkpoints(self):
        cases = (
            ("T2I", "GRN2b", "GRN_T2I_2B_FSA_251600.pth"),
            ("T2I", "GRN8b", "GRN_T2I_titan_FSA_256x256.pth"),
            ("T2V", "GRN2b", "GRN_T2V_2B_FSA_sft147400_ema.pth"),
            ("T2V", "GRN8b", "GRN_T2V_TITAN_FSA_sft10600_ema.pth"),
        )
        for task, model, filename in cases:
            with self.subTest(task=task, model=model):
                self.hub.hf_hub_download.reset_mock()
                pipeline = self.pipeline.from_pretrained(
                    hf_repo_id="bytedance-research/GRN", task=task, model=model,
                    device="cpu",
                )
                first_call = self.hub.hf_hub_download.call_args_list[0]
                self.assertEqual(first_call.kwargs["filename"], filename)
                self.assertEqual(pipeline.args.model_path, filename)
                self.assertEqual(pipeline.args.model, model)

    def test_unknown_model_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "GRNinvalid"):
            self.pipeline.from_pretrained(
                hf_repo_id="bytedance-research/GRN", task="T2I", model="GRNinvalid",
            )
        self.hub.hf_hub_download.assert_not_called()

    def test_local_checkpoint_path_is_preserved(self):
        pipeline = self.pipeline.from_pretrained(
            model_path="custom.pth", task="T2I", model="GRN8b", device="cpu",
        )
        self.assertEqual(pipeline.args.model_path, "custom.pth")
        self.hub.hf_hub_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
