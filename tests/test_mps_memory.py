"""Numerical safeguards for inference memory optimizations (run on real MPS)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import torch
from diffusers.models.attention import FeedForward
from mps_attention import bounded_attention, bounded_feed_forward


@unittest.skipUnless(torch.backends.mps.is_available(), 'Requires actual Metal access')
class MemoryTests(unittest.TestCase):
    def test_feed_forward_equivalence_and_exception_cleanup(self):
        torch.manual_seed(1247)
        model = FeedForward(320, activation_fn='geglu').half().to('mps').eval()
        inputs = torch.randn(2, 1025, 320, device='mps', dtype=torch.float16)
        with torch.inference_mode():
            expected = model(inputs)
            with self.assertRaisesRegex(RuntimeError, 'cleanup'):
                with bounded_feed_forward(model):
                    actual = model(inputs)
                    torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
                    raise RuntimeError('cleanup')
        self.assertNotIn('forward', model.__dict__)

    def test_attention_mask_and_remainder(self):
        import torch.nn.functional as functional
        original = functional.scaled_dot_product_attention
        torch.manual_seed(1247)
        q = torch.randn(1, 2, 513, 32, device='mps', dtype=torch.float16)
        k = torch.randn(1, 2, 129, 32, device='mps', dtype=torch.float16)
        v = torch.randn_like(k)
        mask = torch.zeros(513, 129, device='mps', dtype=torch.float16)
        mask[:, -5:] = float('-inf')
        with torch.inference_mode():
            expected = original(q, k, v, attn_mask=mask)
            with bounded_attention():
                actual = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)
        self.assertIs(functional.scaled_dot_product_attention, original)


if __name__ == '__main__':
    unittest.main()
