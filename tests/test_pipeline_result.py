import tempfile
import unittest
from pathlib import Path

from tools.pipeline_result import PipelineResult


class PipelineResultTests(unittest.TestCase):
    def test_cleanup_removes_returned_video_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.mp4"
            output.write_bytes(b"video")
            result = PipelineResult(videos=[str(output)], cleanup_paths=[str(output)])

            result.cleanup()

            self.assertFalse(output.exists())
            result.cleanup()


if __name__ == "__main__":
    unittest.main()
