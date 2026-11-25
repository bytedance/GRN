"""Exercise the Gradio button callback without weights or a GPU."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import gradio as gr
from PIL import Image


class DemoGenerationTest(unittest.IsolatedAsyncioTestCase):
    args = ["A test prompt", "image", 3.0, 1.0, 42, 1024, 1024]

    def setUp(self):
        gpu = SimpleNamespace(GPU=lambda **_: lambda function: function)
        pipeline_module = SimpleNamespace(GRNPipeline=Mock())
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        path = Path(__file__).resolve().parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("app", path)
        self.app = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {"spaces": gpu, "torch": torch, "tools.grn_pipeline": pipeline_module},
        ), patch.object(sys, "path", sys.path.copy()):
            spec.loader.exec_module(self.app)
        self.demo = self.app.create_demo()

    async def click_generate(self):
        event = next(
            event for event in self.demo.config["dependencies"]
            if event["api_name"] == "generate_and_display"
        )
        return await self.demo.process_api(event["id"], self.args)

    async def test_load_failure_is_reported_as_a_ui_error(self):
        self.app.load_pipeline = Mock(side_effect=RuntimeError("failed in model loader"))
        with self.assertRaises(gr.Error) as error:
            await self.click_generate()
        self.assertEqual(error.exception.message, "Could not load the GRN model. Please try again.")

    async def test_generation_failure_is_reported_as_a_ui_error(self):
        self.app.pipe = Mock(side_effect=RuntimeError("failed in inference"))
        with self.assertRaises(gr.Error) as error:
            await self.click_generate()
        self.assertEqual(error.exception.message, "Could not generate the requested content. Please try again.")

    async def test_empty_generation_is_reported_as_a_ui_error(self):
        self.app.pipe = Mock(return_value=SimpleNamespace(images=[]))
        with self.assertRaises(gr.Error) as error:
            await self.click_generate()
        self.assertEqual(error.exception.message, "The GRN model did not produce any content.")

    def test_failed_second_request_returns_no_gallery_update(self):
        self.app.pipe = Mock(side_effect=[
            SimpleNamespace(images=[Image.new("RGB", (4, 4), "red")]),
            RuntimeError("synthetic inference detail"),
        ])
        url = self.demo.config.get("api_prefix", "") + "/api/generate_and_display"
        with TestClient(self.demo.app) as client:
            response = client.post(url, json={"data": self.args})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["data"][0]), 1)
            response = client.post(url, json={"data": self.args})
        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertIn("Could not generate the requested content.", body["error"])
        self.assertNotIn("synthetic inference detail", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn("data", body)

    async def test_image_generation_still_populates_gallery(self):
        self.app.pipe = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (4, 4), "red")]))
        result = await self.click_generate()
        self.assertEqual(len(result["data"][0]), 1)
        self.assertTrue(Path(result["data"][0][0]["image"]["path"]).is_file())
        self.app.pipe.assert_called_once()
        self.assertEqual(self.app.pipe.call_args.kwargs["prompt"], "<T2I>A test prompt")
