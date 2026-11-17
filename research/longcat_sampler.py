"""Bounded AI2V sampling following Meituan's reference-conditioned loop.

Unlike the community demo, the reference occupies the first existing latent,
gets timestep zero, remains unchanged, and is included in VAE decoding.
"""
import mlx.core as mx
import numpy as np


def dmd_schedule(shift=7.0):
    indices = (1000 - np.arange(1, 9) * 125)[::-1]
    raw = np.linspace(0, 1, 1000, dtype=np.float32)[::-1][indices]
    sigmas = shift * raw / (1 + (shift - 1) * raw)
    return np.concatenate([sigmas, np.array([0], dtype=np.float32)])


def sample(dit, reference, audio, positive, positive_mask, negative, negative_mask,
           frames=29, seed=42, shift=7.0, text_scale=1.0, audio_scale=1.0, callback=None,
           resume_latents=None, start_step=0, check_cancel=None,
           num_ref_latents=0, ref_img_index=10, mask_frame_range=0):
    if frames < 5 or (frames - 1) % 4:
        raise ValueError('Frame count must be 4n+1, at least 5')
    if audio.shape[1] < frames:
        raise ValueError('Insufficient audio features; do not silently repeat speech')
    mx.random.seed(seed)
    if num_ref_latents not in (0, 1):
        raise ValueError('Only one optional portrait anchor is supported')
    num_cond_latents = reference.shape[2]
    shape = list(reference.shape); shape[2] = (frames - 1) // 4 + 1 + num_ref_latents
    if not num_ref_latents < num_cond_latents < shape[2]:
        raise ValueError('Conditioning must include video context and leave new frames to generate')
    if not isinstance(start_step, int) or not 0 <= start_step <= 8:
        raise ValueError('Invalid resume step')
    if (resume_latents is None) != (start_step == 0):
        raise ValueError('Resume requires both latents and a completed step')
    noise = mx.random.normal(shape, dtype=mx.float32)
    latents = mx.concatenate([reference.astype(mx.float32), noise[:, :, num_cond_latents:]], axis=2)
    if resume_latents is not None:
        if list(resume_latents.shape) != shape or not bool(mx.all(mx.isfinite(resume_latents))):
            raise ValueError('Invalid resume latents')
        if not bool(mx.array_equal(resume_latents[:, :, :num_cond_latents], reference.astype(mx.float32))):
            raise ValueError('Resume reference does not match the portrait')
        latents = resume_latents.astype(mx.float32)
    audio = audio[:, :frames]
    silent = mx.zeros_like(audio)
    sigmas = dmd_schedule(shift)
    for step in range(start_step, 8):
        timestep = mx.full((1, shape[2]), float(sigmas[step] * 1000), dtype=mx.float32)
        timestep[:, :num_cond_latents] = 0
        predictions = []
        # Meituan's v1.5 DMD demo forces both scales to 1 and uses one pass.
        guided = text_scale != 1.0 or audio_scale != 1.0
        passes = ((negative, negative_mask, audio), (positive, positive_mask, audio),
                  (negative, negative_mask, silent)) if guided else ((positive, positive_mask, audio),)
        for text, mask, sound in passes:
            if check_cancel:
                check_cancel()
            extra = dict(num_ref_latents=1, ref_img_index=ref_img_index, mask_frame_range=mask_frame_range) if num_ref_latents else {}
            value = dit(latents.astype(mx.bfloat16), timestep, text, sound,
                        encoder_attention_mask=mask, num_cond_latents=num_cond_latents, **extra)
            mx.eval(value)
            predictions.append(value.astype(mx.float32))
        if guided:
            uncond_text, cond, uncond = predictions
            velocity = -(uncond + text_scale * (cond - uncond_text) + audio_scale * (uncond_text - uncond))
        else:
            velocity = -predictions[0]
        generated = latents[:, :, num_cond_latents:] + float(sigmas[step + 1] - sigmas[step]) * velocity[:, :, num_cond_latents:]
        latents = mx.concatenate([reference.astype(mx.float32), generated], axis=2)
        mx.eval(latents)
        if not bool(mx.all(mx.isfinite(latents))):
            raise RuntimeError(f'Nonfinite latents at step {step + 1}')
        if callback:
            callback(step + 1, latents)
    return latents
