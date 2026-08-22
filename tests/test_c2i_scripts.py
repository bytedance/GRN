import re
import unittest
from pathlib import Path


class ClassToImageScriptTest(unittest.TestCase):
    root = Path(__file__).parents[1]

    def test_grn_l_scripts_select_the_l_configuration(self):
        for relative_path in (
            'scripts/c2i/train_GRN_ind_L.sh',
            'scripts/c2i/eval_GRN_ind_L.sh',
        ):
            with self.subTest(script=relative_path):
                script = (self.root / relative_path).read_text(encoding='utf-8')
                experiment = re.search(r'^exp_name=(\S+)$', script, re.MULTILINE)
                model = re.search(r'^\s*--model\s+(\S+)\s*\\?$', script, re.MULTILINE)

                self.assertIsNotNone(experiment)
                self.assertIsNotNone(model)
                self.assertEqual('GRN_ind_L', experiment.group(1))
                self.assertEqual('GRN_L', model.group(1))


if __name__ == '__main__':
    unittest.main()
