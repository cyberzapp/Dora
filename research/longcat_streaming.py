"""Bounded tensor loading for the published LongCat MLX checkpoint.

This is infrastructure, not a validated video-generation pipeline. Never read
unfinished .partial downloads. Published quantization and BF16 bits are retained.
"""
from __future__ import annotations

import gc
import json
import math
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np


class TensorStore:
    """Read only selected tensors, without materializing a whole 5 GB shard."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.entries = {}
        for path in sorted(self.folder.glob("*.safetensors")):
            with path.open("rb") as source:
                raw = source.read(8)
                if len(raw) != 8:
                    raise ValueError(f"Truncated header: {path}")
                length = struct.unpack("<Q", raw)[0]
                if length > 64 * 1024 * 1024:
                    raise ValueError(f"Oversized header: {path}")
                header = json.loads(source.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if name in self.entries:
                    raise ValueError(f"Duplicate tensor: {name}")
                self.entries[name] = (path, 8 + length, entry)
        if not self.entries:
            raise FileNotFoundError(f"No completed safetensors in {folder}")
        indexes = list(self.folder.glob("*.index.json"))
        for index in indexes:
            expected = json.loads(index.read_text())["weight_map"]
            if set(expected) != set(self.entries):
                raise ValueError("Checkpoint shards are incomplete or contain unexpected tensors")
            for name, filename in expected.items():
                if self.entries[name][0].name != filename:
                    raise ValueError(f"Wrong shard for {name}")

    def read(self, name: str):
        path, base, entry = self.entries[name]
        dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "U32": "<u4",
                 "I32": "<i4", "I64": "<i8"}.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {entry['dtype']}")
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if start < 0 or end - start != size or base + end > path.stat().st_size:
            raise ValueError(f"Invalid tensor extent: {name}")
        mapping = np.memmap(path, mode="r", dtype=dtype, offset=base + start, shape=shape)
        result = mx.array(mapping)
        if entry["dtype"] == "BF16":
            result = result.view(mx.bfloat16)
        mx.eval(result)  # Complete the copy before releasing the file mapping.
        del mapping
        return result

    def weights(self, prefix: str):
        return [(name[len(prefix):], self.read(name)) for name in self.entries
                if name.startswith(prefix)]


class StreamedBlock:
    """Instantiate, strictly load, execute, and release one genuine model block."""

    def __init__(self, store: TensorStore, prefix: str, factory, check_cancel=None, on_complete=None):
        self.store, self.prefix, self.factory = store, prefix, factory
        self.check_cancel = check_cancel
        self.on_complete = on_complete

    def __call__(self, *args, **kwargs):
        if self.check_cancel:
            self.check_cancel()
        block = self.factory()
        try:
            block.load_weights(self.store.weights(self.prefix), strict=True)
            result = block(*args, **kwargs)
            mx.eval(result)
            if self.on_complete:
                self.on_complete()
            return result
        finally:
            del block
            gc.collect()
            mx.clear_cache()


def build_streamed_dit(folder: Path, check_cancel=None, on_block_done=None):
    """Build the real port with resident embeddings and streamed blocks.

    Call only after all checkpoint files have passed download verification.
    No model weights are requantized: quantize creates the checkpoint schema,
    then strict loading replaces every parameter with published values.
    """
    import mlx.nn as nn
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatAvatarSingleStreamBlock, LongCatVideoAvatarTransformer3DModel,
    )

    config = json.loads((folder / "config.json").read_text())
    store = TensorStore(folder)
    quant = config.get("quantization")

    def prepare(module):
        if quant:
            patterns = quant["skip_patterns"]
            nn.quantize(module, bits=quant["bits"], group_size=quant["group_size"],
                        class_predicate=lambda path, layer: isinstance(layer, nn.Linear)
                        and not any(pattern in path for pattern in patterns))
        return module

    def factory():
        from longcat_attention import ContinuationAttention
        keys = ("hidden_size", "num_heads", "mlp_ratio", "adaln_tembed_dim",
                "output_dim", "audio_prenorm", "class_range", "class_interval")
        block = LongCatAvatarSingleStreamBlock(**{key: config[key] for key in keys})
        block.attn = ContinuationAttention(dim=config['hidden_size'], num_heads=config['num_heads'])
        return prepare(block)

    model = prepare(LongCatVideoAvatarTransformer3DModel.from_config({**config, "depth": 0}))
    resident = [(name, store.read(name)) for name in store.entries if not name.startswith("blocks.")]
    model.load_weights(resident, strict=True)
    del resident
    model.depth = config["depth"]
    model.blocks = [StreamedBlock(store, f"blocks.{index}.", factory, check_cancel,
                                 (lambda index=index: on_block_done(index, config['depth'])) if on_block_done else None)
                    for index in range(config["depth"])]
    return model


def build_streamed_text_encoder(folder: Path):
    from longcat_video_avatar.models.umt5 import UMT5EncoderModel, T5SelfAttentionBlock
    config = json.loads((folder / 'config.json').read_text())
    store = TensorStore(folder)
    model = UMT5EncoderModel.from_config({**config, 'num_layers': 0})
    model.load_weights([(name, store.read(name)) for name in store.entries
                        if not name.startswith('blocks.')], strict=True)
    def factory():
        return T5SelfAttentionBlock(config['d_model'], config['d_kv'] * config['num_heads'],
                                    config['d_ff'], config['num_heads'],
                                    config['relative_attention_num_buckets'], False)
    model.blocks = [StreamedBlock(store, f'blocks.{index}.', factory)
                    for index in range(config['num_layers'])]
    return model
