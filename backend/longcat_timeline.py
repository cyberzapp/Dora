"""Exact frame windows for reference-anchored local LongCat continuation.

Planning only: callers must actually generate and verify every listed window.
The source audio is remuxed once, unchanged, after assembling video frames.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    source_start: int
    model_frames: int
    carry_frames: int
    output_start: int
    append_frames: int
    padded_feature_frames: int


def plan(source_samples: int, sample_rate=16000, fps=25, max_frames=121, context=13):
    if any(type(value) is not int or value <= 0 for value in (source_samples, sample_rate, fps)):
        raise ValueError('Positive integer sample count and timebase required')
    if max_frames < 5 or (max_frames - 1) % 4 or context < 1 or (context - 1) % 4 or context >= max_frames:
        raise ValueError('Model and context lengths must be 4n+1, with room for new frames')
    target = (source_samples * fps + sample_rate - 1) // sample_rate
    available_features = source_samples * fps // sample_rate
    windows = []
    written = 0
    while written < target:
        carry = min(context, written) if windows else 0
        start = written - carry
        needed = min(max_frames, target - start)
        model_frames = max(5, 1 + ((needed - 1 + 3) // 4) * 4)
        append = min(model_frames - carry, target - written)
        windows.append(Window(start, model_frames, carry, written, append,
                              max(0, start + model_frames - available_features)))
        written += append
    return target, windows
