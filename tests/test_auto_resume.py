import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

arg_util = types.ModuleType('grn.utils_t2iv.arg_util')
arg_util.Args = object
sys.modules.setdefault('grn.utils_t2iv.arg_util', arg_util)

from grn.utils_t2iv import save_and_load


class AutoResumeTest(unittest.TestCase):
    checkpoint = {
        'epoch': 3,
        'iter': 17,
        'g_it': 42,
        'trainer': {'weight': 'value'},
        'args': {'model': 'GRN_L'},
    }

    def test_copies_and_loads_checkpoint_from_bed(self):
        with tempfile.TemporaryDirectory() as local_out, tempfile.TemporaryDirectory() as bed:
            source = os.path.join(bed, 'ckpt-global_step_42.pth')
            with open(source, 'wb') as checkpoint_file:
                checkpoint_file.write(b'checkpoint')
            args = SimpleNamespace(auto_resume=True, local_out_path=local_out, bed=bed)

            with (
                mock.patch.object(save_and_load, 'glob_with_global_step', side_effect=[[], [source]]),
                mock.patch.object(save_and_load.torch, 'load', return_value=self.checkpoint) as load,
                mock.patch.object(save_and_load.dist, 'barrier'),
            ):
                result = save_and_load.auto_resume(args)

            target = os.path.join(local_out, os.path.basename(source))
            with open(target, 'rb') as checkpoint_file:
                self.assertEqual(b'checkpoint', checkpoint_file.read())
            load.assert_called_once_with(target, map_location='cpu')
            self.assertEqual((3, 17), result[1:3])

    def test_loads_checkpoint_already_in_local_output(self):
        with tempfile.TemporaryDirectory() as local_out, tempfile.TemporaryDirectory() as bed:
            source = os.path.join(local_out, 'ckpt-global_step_42.pth')
            with open(source, 'wb') as checkpoint_file:
                checkpoint_file.write(b'checkpoint')
            args = SimpleNamespace(auto_resume=True, local_out_path=local_out, bed=bed)

            with (
                mock.patch.object(save_and_load, 'glob_with_global_step', return_value=[source]),
                mock.patch.object(save_and_load.shutil, 'copyfile') as copyfile,
                mock.patch.object(save_and_load.torch, 'load', return_value=self.checkpoint) as load,
                mock.patch.object(save_and_load.dist, 'barrier'),
            ):
                result = save_and_load.auto_resume(args)

            copyfile.assert_not_called()
            load.assert_called_once_with(source, map_location='cpu')
            self.assertEqual((3, 17), result[1:3])


if __name__ == '__main__':
    unittest.main()
