"""Reference-graph numerical checks, not generated-media or quality evidence."""
import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
from longcat_attention import ContinuationAttention
from longcat_video_avatar.models.avatar.attention import Attention


class Triple(nn.Module):
    def __call__(self, x): return mx.concatenate([x, x, x], axis=-1)


def dense(q, k, v, scale):
    scores = np.asarray(q) @ np.asarray(k).swapaxes(-1, -2) * scale
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (exp / exp.sum(axis=-1, keepdims=True)) @ np.asarray(v)


class ContinuationAttentionTests(unittest.TestCase):
    def test_reference_and_context_have_independent_attention(self):
        for frames, conditioned in [(5, 3), (3, 3)]:
            module = ContinuationAttention(dim=8, num_heads=1)
            module.qkv = Triple(); module.q_norm = nn.Identity(); module.k_norm = nn.Identity()
            module.proj = nn.Identity()
            module.rope_3d = lambda q, k, *args, **kwargs: (q, k)
            x = mx.array(np.random.default_rng(7).normal(size=(1, frames*2, 8)).astype(np.float32))
            # Meituan's graph: ref→ref; context→context; noise→all.
            q = np.asarray(x)[:, None]; ref_end = 2; cond_end = conditioned*2
            parts = [dense(q[:, :, :ref_end], q[:, :, :ref_end], q[:, :, :ref_end], module.scale),
                     dense(q[:, :, ref_end:cond_end], q[:, :, ref_end:cond_end], q[:, :, ref_end:cond_end], module.scale)]
            if conditioned < frames: parts.append(dense(q[:, :, cond_end:], q, q, module.scale))
            expected = np.concatenate(parts, axis=2).reshape(x.shape)
            actual, cache, routing = module(x, (frames, 1, 2), num_cond_latents=conditioned,
                                            num_ref_latents=1, ref_img_index=10, mask_frame_range=0, return_kv=True)
            # Metal SDPA uses a different accumulation path from NumPy CPU;
            # compare numerics at 3e-3 and separately enforce exact isolation.
            np.testing.assert_allclose(np.asarray(actual), expected, atol=3e-3, rtol=3e-3)
            changed = mx.concatenate([x[:, :2] + 10, x[:, 2:]], axis=1)
            perturbed, _ = module(changed, (frames, 1, 2), num_cond_latents=conditioned,
                                  num_ref_latents=1, ref_img_index=10, mask_frame_range=0)
            self.assertTrue(mx.array_equal(actual[:, 2:cond_end], perturbed[:, 2:cond_end]).item(),
                            'Portrait tokens leaked into carried-context attention')
            self.assertIsNone(routing); self.assertEqual(cache[0].shape, (1, 1, frames*2, 8))
            merged = dense(q[:, :, :cond_end], q[:, :, :cond_end], q[:, :, :cond_end], module.scale)
            self.assertGreater(float(np.max(np.abs(merged[:, :, :2] - parts[0]))), 0.01)

    def test_image_to_video_path_and_parameter_schema_are_unchanged(self):
        original = Attention(dim=64, num_heads=4)
        corrected = ContinuationAttention(dim=64, num_heads=4)
        corrected.load_weights(tree_flatten(original.parameters()), strict=True)
        x = mx.random.normal((1, 12, 64))
        before, _ = original(x, (3, 2, 2), num_cond_latents=1)
        after, _ = corrected(x, (3, 2, 2), num_cond_latents=1)
        self.assertTrue(mx.array_equal(before, after).item())


if __name__ == '__main__': unittest.main()
