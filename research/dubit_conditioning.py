"""Scoped local Dub-It conditioning adapters; no training or quality claim.

Freeze the provided target speech instead of synthesizing replacement audio, and
reuse actual pre-encoded camera latents to avoid the video-encoder memory peak
coinciding with the streamed transformer and its adapter.
"""
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx


@contextmanager
def frozen_target_audio(target):
    import ltx_pipelines_mlx.lipdub as module
    if target.ndim!=3 or target.shape[0]!=1 or target.shape[2]!=128 or not bool(mx.all(mx.isfinite(target)).item()):
        raise ValueError('Invalid real target-audio tokens')
    original=module.create_noised_state
    evidence=dict(audio_states_frozen=0)
    def build(**kwargs):
        positions=kwargs['positions']
        if positions.shape[-1]==1:
            if tuple(kwargs['base_shape'])!=tuple(target.shape):raise ValueError('Target audio does not cover the dubbing window')
            kwargs=dict(kwargs,initial_latent=target,sigma=0.0,frozen=True)
            evidence['audio_states_frozen']+=1
        return original(**kwargs)
    try:
        module.create_noised_state=build
        yield evidence
    finally:module.create_noised_state=original


@contextmanager
def cached_camera_conditioning(video,latents,frames=49,fps=25):
    import ltx_pipelines_mlx.lipdub as module
    from ltx_pipelines_mlx.iclora_utils import reference_conditioning_from_latent
    video=Path(video).resolve();original=module.append_ic_lora_reference_video_conditionings
    evidence=dict(cached_video_stages=0)
    def append(conditionings,references,*,height,width,num_frames,video_encoder,
               reference_downscale_factor,frame_rate,conditioning_attention_strength=1.0,
               conditioning_attention_mask=None):
        if (num_frames!=frames or frame_rate!=fps or reference_downscale_factor!=1 or height!=width
                or len(references)!=1 or Path(references[0][0]).resolve()!=video
                or conditioning_attention_mask is not None or conditioning_attention_strength!=1.0):
            raise ValueError('Cached camera conditioning differs from this explicit probe')
        if height not in latents:raise ValueError('Missing actual camera latent for this stage')
        latent=latents[height]
        expected=(1,128,(frames-1)//8+1,height//32,width//32)
        if tuple(latent.shape)!=expected or not bool(mx.all(mx.isfinite(latent)).item()):
            raise ValueError('Invalid cached camera video latent')
        conditionings.append(reference_conditioning_from_latent(latent,frame_rate=fps,
            downscale_factor=1,strength=references[0][1]))
        evidence['cached_video_stages']+=1
    try:
        module.append_ic_lora_reference_video_conditionings=append
        yield evidence
    finally:module.append_ic_lora_reference_video_conditionings=original
