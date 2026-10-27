import re
import unittest
from pathlib import Path

import cv2


class ImageToVideoExampleTest(unittest.TestCase):
    root = Path(__file__).parents[1]

    def test_default_first_frame_is_checked_in_and_decodable(self):
        script = (self.root / 'tools/i2v_infer.py').read_text(encoding='utf-8')
        match = re.search(r"^first_frame_path='([^']+)'$", script, re.MULTILINE)

        self.assertIsNotNone(match)
        relative_path = match.group(1).removeprefix('./')
        first_frame = self.root / relative_path
        self.assertTrue(first_frame.is_file())
        self.assertIsNotNone(cv2.imread(str(first_frame)))

        readme = (self.root / 'README.md').read_text(encoding='utf-8')
        self.assertIn(f"first_frame_path='{match.group(1)}'", readme)


if __name__ == '__main__':
    unittest.main()
