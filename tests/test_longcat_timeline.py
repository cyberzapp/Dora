import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('longcat_timeline', Path(__file__).resolve().parents[1]/'backend/longcat_timeline.py')
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)


class TimelineTests(unittest.TestCase):
    def test_full_voice_and_boundary_lengths_have_no_gaps(self):
        for samples in [16000, 18560, 77440, 77441, 849749, 300*16000]:
            total, windows = module.plan(samples)
            indices = []
            for i, window in enumerate(windows):
                self.assertEqual(window.model_frames % 4, 1)
                self.assertLessEqual(window.model_frames, 121)
                self.assertEqual(window.source_start + window.carry_frames, window.output_start)
                self.assertEqual(window.carry_frames, 0 if i == 0 else 13)
                indices.extend(range(window.output_start, window.output_start + window.append_frames))
            self.assertEqual(indices, list(range(total)))
            self.assertEqual(total, (samples*25+15999)//16000)

    def test_preserves_fractional_final_audio_frame(self):
        total, windows = module.plan(849749)
        self.assertEqual(total, 1328)
        self.assertEqual(len(windows), 13)
        self.assertGreater(windows[-1].padded_feature_frames, 0)

    def test_invalid_geometry_rejected(self):
        for kwargs in [dict(source_samples=0), dict(source_samples=16000, context=121),
                       dict(source_samples=16000, max_frames=120)]:
            with self.assertRaises(ValueError): module.plan(**kwargs)


if __name__ == '__main__': unittest.main()
