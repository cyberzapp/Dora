"""Check reference conditioning and schedule, without claiming video quality."""
import json
import unittest
from pathlib import Path
import mlx.core as mx
import numpy as np
from longcat_sampler import dmd_schedule, sample


class SamplerTests(unittest.TestCase):
    def test_continuation_preserves_anchor_and_context(self):
        reference = mx.full((1, 4, 2, 2, 2), 0.125)
        def dit(x, t, *args, **kwargs):
            self.assertEqual(x.shape[2], 4)
            self.assertTrue(mx.all(t[:, :2] == 0).item())
            self.assertEqual(kwargs['num_cond_latents'], 2)
            self.assertEqual(kwargs['num_ref_latents'], 1)
            return mx.ones_like(x)
        output = sample(dit, reference, mx.zeros((1,9,5,5,32)),
                        mx.zeros((1,1,4,32)), mx.ones((1,4)),
                        mx.zeros((1,1,4,32)), mx.ones((1,4)), frames=9, num_ref_latents=1)
        self.assertTrue(mx.array_equal(output[:, :, :2], reference).item())

    def test_interrupted_resume_matches_uninterrupted(self):
        reference = mx.full((1, 4, 1, 2, 2), 0.125)
        arguments = (lambda x,t,*a,**k: x*0.1+t[:,None,:,None,None]*0.0001,
                     reference, mx.zeros((1,9,5,5,32)),
                     mx.zeros((1,1,4,32)), mx.ones((1,4)),
                     mx.zeros((1,1,4,32)), mx.ones((1,4)))
        saved = []
        expected = sample(*arguments, frames=9, callback=lambda s,x: saved.append(x))
        actual = sample(*arguments, frames=9, resume_latents=saved[3], start_step=4)
        self.assertTrue(mx.array_equal(expected, actual).item())
        with self.assertRaises(ValueError):
            sample(*arguments, frames=9, resume_latents=saved[3], start_step=9)

    def test_reference_schedule(self):
        reference = Path(__file__).resolve().parents[1] / '.local/longcat-scheduler-reference.json'
        if not reference.exists():
            self.skipTest('Generate the independent Diffusers CPU reference first')
        expected = json.loads(reference.read_text())
        np.testing.assert_allclose(dmd_schedule(), expected['sigmas'], atol=1e-7, rtol=1e-6)

    def test_reference_fixed_and_zero_timestep(self):
        reference = mx.full((1, 4, 1, 2, 2), 0.125)
        calls = []
        def dit(latents, timestep, text, audio, **kwargs):
            self.assertEqual(timestep.shape, (1, 3))
            self.assertEqual(float(timestep[0, 0]), 0)
            self.assertTrue(mx.array_equal(latents[:, :, :1], reference).item())
            calls.append(1)
            return mx.ones_like(latents)
        output = sample(dit, reference, mx.zeros((1, 9, 5, 5, 32)),
                        mx.zeros((1, 1, 4, 32)), mx.ones((1, 4)),
                        mx.zeros((1, 1, 4, 32)), mx.ones((1, 4)), frames=9)
        self.assertEqual(len(calls), 8)
        calls.clear()
        sample(dit, reference, mx.zeros((1, 9, 5, 5, 32)),
               mx.zeros((1, 1, 4, 32)), mx.ones((1, 4)),
               mx.zeros((1, 1, 4, 32)), mx.ones((1, 4)), frames=9, text_scale=4, audio_scale=4)
        self.assertEqual(len(calls), 24)
        self.assertEqual(output.shape, (1, 4, 3, 2, 2))
        self.assertTrue(mx.array_equal(output[:, :, :1], reference).item())


if __name__ == '__main__':
    unittest.main()
