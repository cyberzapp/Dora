"""Actual base-model reference-token memory probe, without the Dub-It adapter.

No avatar video is produced and no Dub-It compatibility/quality claim follows.
This checks the larger reference-conditioned attention geometry, not trained
dubbing behavior or the additional adapter memory. Run with OS network denied.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research')]
from download_model_manifest import digest
from inference_lease import GPUInferenceLease
from ltx_audio_preview import validate_model,MODEL


def main():
    output=ROOT/'.local/dubit-streaming-base-probe'
    if output.exists():raise ValueError('Retain previous probe evidence')
    model=ROOT/'.local/ltx-2.3-q8-weights'
    manifest_hash=validate_model(model,MODEL)
    reference=ROOT/'.local/dubit-camera-reference-two'
    cache=ROOT/'.local/dubit-camera-encoder-probe'
    info=json.loads((reference/'reference.json').read_text())
    cache_record=json.loads((cache/'report.json').read_text())
    if digest(reference/'camera-motion-reference.mkv')!=cache_record['reference_sha256']:
        raise ValueError('Camera reference changed')
    if digest(Path(info['audio']))!=info['source_audio_sha256']:
        raise ValueError('Original voice-over changed')
    for phase in cache_record['phases']:
        if digest(cache/f'camera-{phase["size"]}-latent.safetensors')!=phase['latent_sha256']:
            raise ValueError('Camera latent changed')
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active inference',flush=True)):
        import imageio_ffmpeg
        import mlx.core as mx
        from ltx_core_mlx.model.transformer.model import X0Model
        from ltx_core_mlx.utils.positions import compute_audio_positions,compute_video_positions
        from ltx_core_mlx.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
        from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
        from ltx_pipelines_mlx.lipdub import patchify_lipdub_audio_reference_latent
        from ltx_pipelines_mlx.a2vid_two_stage import encode_source_audio
        from ltx_pipelines_mlx.iclora_utils import reference_conditioning_from_latent
        from ltx_pipelines_mlx.utils.helpers import create_noised_state
        from ltx_pipelines_mlx.utils.samplers import denoise_loop
        from ltx_pipelines_mlx.utils.blocks import aggressive_cleanup
        output.mkdir();started=time.monotonic()
        bin_dir=output/'bin';bin_dir.mkdir();(bin_dir/'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ['PATH']=f'{bin_dir}:/usr/bin:/bin'
        mx.set_default_device(mx.gpu);mx.set_memory_limit(11*1024**3);mx.set_cache_limit(0)
        try:
            pipe=ICLoraPipeline(str(model),low_memory=True,low_ram_streaming=True)
            target=encode_source_audio(pipe,Path(info['audio']),num_frames=49,frame_rate=25,start_time=0,max_duration=1.96)
            embeddings_path=ROOT/'.local/ltx-real-text-probe/prompt-embeddings.safetensors'
            embeddings=mx.load(str(embeddings_path))
            pipe.load();x0_model=X0Model(pipe.dit)
            ref_tokens,ref_positions=patchify_lipdub_audio_reference_latent(pipe.audio_patchifier.unpatchify(target),pipe.audio_patchifier)
            phases=[]
            for size,frozen in [(256,False),(256,True),(512,True)]:
                begin=time.monotonic();dims=(7,size//32,size//32);count=7*(size//32)**2
                print(json.dumps(dict(stage='Actual streamed base forward',size=size,audio_frozen=frozen)),flush=True)
                latent=mx.load(str(cache/f'camera-{size}-latent.safetensors'))['video']
                video=create_noised_state(base_shape=(1,count,128),conditionings=[reference_conditioning_from_latent(
                    latent,frame_rate=25,downscale_factor=1,strength=1)],spatial_dims=dims,
                    positions=compute_video_positions(*dims,frame_rate=25),seed=42,sigma=1)
                audio=create_noised_state(base_shape=target.shape,conditionings=[],spatial_dims=dims,
                    positions=compute_audio_positions(49),seed=43,sigma=0 if frozen else 1,
                    initial_latent=target if frozen else None,frozen=frozen)
                audio=AudioConditionByReferenceLatent(patchified=ref_tokens,positions=ref_positions,strength=1).apply(audio,num_noisy_tokens=49)
                pipe._pre_denoise_flush(video,audio)
                result=denoise_loop(model=x0_model,video_state=video,audio_state=audio,
                    video_text_embeds=embeddings['video'],audio_text_embeds=embeddings['audio'],sigmas=[1.,0.])
                mx.eval(result.video_latent,result.audio_latent)
                if not all(bool(mx.all(mx.isfinite(x)).item()) for x in (result.video_latent,result.audio_latent)):
                    raise ValueError('Nonfinite reference-conditioned model output')
                if frozen and not bool(mx.all(result.audio_latent[:,:49]==target).item()):
                    raise ValueError('Frozen target speech changed in an actual model step')
                phases.append(dict(size=size,audio_frozen=frozen,actual_base_model_steps=1,
                    video_tokens=int(video.latent.shape[1]),audio_tokens=int(audio.latent.shape[1]),
                    finite_output=True,frozen_target_preserved=True if frozen else None,
                    seconds=time.monotonic()-begin,observed_mlx_peak_bytes=mx.get_peak_memory()))
                (output/'progress.json').write_text(json.dumps(phases,indent=2))
                del result,video,audio,latent;aggressive_cleanup()
            report=dict(actual_pretrained_base_forward=True,adapter_loaded=False,adapter_inference_performed=False,
                generated_video=False,quality_passed=False,training_performed=False,
                model_manifest_sha256=manifest_hash,source_audio_sha256=info['source_audio_sha256'],
                text_embeddings_sha256=digest(embeddings_path),camera_cache=cache_record,
                code_sha256=digest(Path(__file__)),phases=phases,elapsed_seconds=time.monotonic()-started,
                limitation='Excludes adapter weights and full denoising schedules; not a dubbing result')
            (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        except Exception as error:
            (output/'failure.json').write_text(json.dumps(dict(error=str(error),generated_video=False,quality_passed=False),indent=2))
            raise


if __name__=='__main__':
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    main()
