"""CPU-only checks for the JSONL splitter's destination handling."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'split_jsonl.py'


class SplitJsonlDestinationTest(unittest.TestCase):
    def run_split(self, source, extra_args):
        argv = [
            str(SCRIPT), '--jsonl_folder_list', str(source),
            '--chunk_size', '1', '--bucket_size', '2', '--worker', '1',
            *extra_args,
        ]
        # Only threads use the lock; the multiprocessing manager is not relevant.
        errors = io.StringIO()
        with patch('multiprocessing.Manager') as manager, patch.object(sys, 'argv', argv):
            manager.return_value.Lock.return_value = threading.Lock()
            with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
                try:
                    runpy.run_path(str(SCRIPT), run_name='__main__')
                except SystemExit as exc:
                    return exc.code, errors.getvalue()
        return 0, errors.getvalue()

    def test_rerun_cannot_overwrite_shards_or_sibling_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            rows = ['{"id": 1}', '{"id": 2}']
            (source / 'metadata_2.jsonl').write_text('\n'.join(rows) + '\n')
            for precreate in (False, True):
                with self.subTest(precreate=precreate):
                    dest = root / f'dest-{precreate}'
                    if precreate:
                        dest.mkdir()
                    args = ['--save_dir', str(dest)]
                    code, errors = self.run_split(source, args)
                    self.assertEqual(code, 0, errors)
                    def snapshot():
                        return {path.relative_to(dest): path.read_bytes()
                                for path in dest.rglob('*') if path.is_file()}
                    self.assertCountEqual(
                        [json.loads(value) for value in snapshot().values()],
                        [json.loads(row) for row in rows],
                    )
                    marker = dest / '.keep'
                    marker.write_bytes(b'keep this unrelated file')
                    original = snapshot()
                    (source / 'metadata_2.jsonl').write_text('{"id": 9}\n')
                    code, errors = self.run_split(source, args)
                    self.assertEqual(code, 2, errors)
                    self.assertIn('--save_dir must be empty', errors)
                    self.assertEqual(snapshot(), original)
                    (source / 'metadata_2.jsonl').write_text('\n'.join(rows) + '\n')

    def test_requires_explicit_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / 'metadata_1.jsonl').write_text('{"id": 1}\n')
            for extra_args in ([], ['--save_dir', '']):
                with self.subTest(extra_args=extra_args):
                    code, errors = self.run_split(source, extra_args)
                    self.assertEqual(code, 2, errors)
                    self.assertIn('--save_dir', errors)

    def test_does_not_replace_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / 'existing-file'
            dest.write_bytes(b'data')
            code, errors = self.run_split(root, ['--save_dir', str(dest)])
            self.assertEqual(code, 2, errors)
            self.assertIn('--save_dir must be a directory', errors)
            self.assertEqual(dest.read_bytes(), b'data')
