"""Numerical infrastructure checks; synthetic tensors are NOT generated media."""
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from longcat_streaming import StreamedBlock, TensorStore, build_streamed_dit, build_streamed_text_encoder


class StreamingTests(unittest.TestCase):
    def test_cancel_before_allocating_block(self):
        def cancel():
            raise InterruptedError('cancelled')
        def factory():
            self.fail('Cancelled render allocated another block')
        block = StreamedBlock(None, 'blocks.0.', factory, cancel)
        with self.assertRaises(InterruptedError):
            block()

    def test_actual_text_encoder_stream_matches_resident(self):
        from longcat_video_avatar.models.umt5 import UMT5EncoderModel
        config = dict(vocab_size=32, d_model=32, d_kv=8, num_heads=4,
                      d_ff=64, num_layers=2, relative_attention_num_buckets=16)
        resident = UMT5EncoderModel.from_config(config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'config.json').write_text(json.dumps(config))
            mx.save_safetensors(str(path / 'model.safetensors'), dict(tree_flatten(resident.parameters())))
            streamed = build_streamed_text_encoder(path)
            ids = mx.array([[1, 2, 3, 0]]); mask = mx.array([[1, 1, 1, 0]])
            error = mx.max(mx.abs(resident(ids, mask) - streamed(ids, mask)))
            self.assertLess(float(error), 1e-5)

    def test_preserves_tensor_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            expected = {"bf": mx.array([1.125, -0.25], dtype=mx.bfloat16),
                        "fp": mx.array([1.123456, -123.5]),
                        "packed": mx.array([0, 4294967295], dtype=mx.uint32)}
            mx.save_safetensors(str(Path(folder) / "model.safetensors"), expected)
            store = TensorStore(Path(folder))
            for name, value in expected.items():
                actual = store.read(name)
                self.assertEqual(actual.dtype, value.dtype)
                self.assertTrue(mx.array_equal(actual, value).item())

    def test_quantized_stream_matches_resident(self):
        def factory():
            module = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
            nn.quantize(module, bits=4, group_size=64)
            return module

        with tempfile.TemporaryDirectory() as folder:
            resident = factory()
            weights = {"blocks.0." + k: v for k, v in tree_flatten(resident.parameters())}
            mx.save_safetensors(str(Path(folder) / "model.safetensors"), weights)
            streamed = StreamedBlock(TensorStore(Path(folder)), "blocks.0.", factory)
            x = mx.random.normal((1, 8, 64))
            expected, actual = resident(x), streamed(x)
            self.assertTrue(mx.array_equal(expected, actual).item())
            self.assertTrue(mx.array_equal(actual, streamed(x)).item())

    def test_incomplete_shards_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            mx.save_safetensors(str(path / "one.safetensors"), {"x": mx.ones((2,))})
            (path / "model.index.json").write_text(json.dumps({"weight_map": {
                "x": "one.safetensors", "missing": "two.safetensors"}}))
            with self.assertRaises(ValueError):
                TensorStore(path)

    def test_actual_dit_stream_matches_resident(self):
        from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
        config = dict(in_channels=4, out_channels=4, hidden_size=64, depth=2,
                      num_heads=4, caption_channels=32, mlp_ratio=4, adaln_tembed_dim=32,
                      frequency_embedding_size=32, patch_size=[1, 2, 2],
                      text_tokens_zero_pad=False, audio_window=5, audio_block=5,
                      audio_channel=32, intermediate_dim=32, output_dim=32,
                      context_tokens=8, vae_scale=4, audio_prenorm=False,
                      class_range=24, class_interval=4)
        resident = LongCatVideoAvatarTransformer3DModel.from_config(config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps(config))
            mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(resident.parameters())))
            completed_blocks = []
            streamed = build_streamed_dit(path, on_block_done=lambda index, count: completed_blocks.append((index, count)))
            args = (mx.random.normal((1, 4, 2, 4, 4)), mx.array([0.5]),
                    mx.random.normal((1, 1, 6, 32)), mx.random.normal((1, 5, 5, 5, 32)))
            mask = mx.ones((1, 1, 1, 6))
            expected = resident(*args, encoder_attention_mask=mask)
            actual = streamed(*args, encoder_attention_mask=mask)
            mx.eval(expected, actual)
            self.assertEqual(completed_blocks, [(0, 2), (1, 2)])
            self.assertLess(float(mx.max(mx.abs(expected - actual))), 1e-5)
            # The same actual architecture must also accept a portrait anchor
            # plus carried video context for longer voice-overs.
            continuation = (mx.random.normal((1, 4, 4, 4, 4)), mx.ones((1, 4)),
                            args[2], mx.random.normal((1, 9, 5, 5, 32)))
            options = dict(encoder_attention_mask=mask, num_cond_latents=2,
                           num_ref_latents=1, ref_img_index=10, mask_frame_range=0)
            from longcat_attention import ContinuationAttention
            for block in resident.blocks:
                corrected = ContinuationAttention(dim=config['hidden_size'], num_heads=config['num_heads'])
                corrected.load_weights(tree_flatten(block.attn.parameters()), strict=True)
                block.attn = corrected
            expected = resident(*continuation, **options)
            actual = streamed(*continuation, **options)
            mx.eval(expected, actual)
            self.assertLess(float(mx.max(mx.abs(expected - actual))), 1e-5)


if __name__ == "__main__":
    unittest.main()
