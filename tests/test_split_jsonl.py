"""CPU regressions for the JSONL splitting CLI."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'split_jsonl.py'


class SplitJsonlTest(unittest.TestCase):
    def assert_split(self, inputs, unterminated=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.mkdir()
            rows = []
            for filename, ids in inputs.items():
                file_rows = [json.dumps({'id': i}) + '\n' for i in ids]
                rows.extend(file_rows)
                text = ''.join(file_rows)
                if filename in unterminated:
                    text = text.rstrip('\n')
                (source / filename).write_text(text)
            output = root / 'output'
            argv = [str(SCRIPT), '--jsonl_folder_list', str(source),
                    '--save_dir', str(output), '--chunk_size', '2',
                    '--bucket_size', '1', '--worker', '1']
            # The tool only uses threads. Replace its Manager lock to avoid
            # starting an unrelated IPC server in CPU-only test sandboxes.
            with patch('multiprocessing.Manager') as manager, patch.object(sys, 'argv', argv):
                manager.return_value.Lock.return_value = threading.Lock()
                runpy.run_path(str(SCRIPT), run_name='__main__')
            parts = sorted(output.glob('*/*.jsonl'))
            actual = [line for part in parts for line in part.read_text().splitlines(keepends=True)]
            self.assertCountEqual(actual, rows)
            self.assertEqual(len(parts), (len(rows) + 1) // 2)
            for part in parts:
                self.assertLessEqual(len(part.read_text().splitlines()), 2)
                for line in part.read_text().splitlines():
                    self.assertIsInstance(json.loads(line), dict)

    def test_unterminated_records_from_multiple_files(self):
        inputs = {f'part_{i}.jsonl': [i] for i in range(4)}
        self.assert_split(inputs, unterminated=inputs)

    def test_mixed_terminated_and_unterminated_files(self):
        inputs = {f'part_{i}.jsonl': [i] for i in range(4)}
        self.assert_split(inputs, unterminated={'part_0.jsonl', 'part_2.jsonl'})

    def test_plain_filename(self):
        self.assert_split({'metadata.jsonl': range(3)})

    def test_underscored_filename(self):
        self.assert_split({'train_data.jsonl': range(5)})

    def test_existing_shard_filename(self):
        self.assert_split({'000001_000001_000000003.jsonl': range(3)})

    def test_empty_file(self):
        self.assert_split({'empty.jsonl': []})

    def test_multiple_files(self):
        self.assert_split({'train.jsonl': range(3), 'validation.jsonl': range(3, 6)})


if __name__ == '__main__':
    unittest.main()
