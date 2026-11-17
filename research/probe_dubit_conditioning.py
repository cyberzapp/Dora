"""Check real-speech freezing and cached real-camera conditioning states.

No adapter weights, dubbing generation, or perceptual quality is tested here.
The actual audio VAE is used; control-state checks do not stand in for inference.
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


def main():
    output=ROOT/'.local/dubit-conditioning-probe'
    if output.exists():raise ValueError('Retain previous conditioning evidence')
    camera=ROOT/'.local/dubit-camera-reference-two/camera-motion-reference.mkv'
    cache=ROOT/'.local/dubit-camera-encoder-probe'
    camera_report=json.loads((cache/'report.json').read_text())
    if camera_report['reference_sha256']!=digest(camera):raise ValueError('Camera input differs from its encoded cache')
    with GPUInferenceLease():
        import mlx.core as mx
        import imageio_ffmpeg
        import ltx_pipelines_mlx.lipdub as lipdub
        from ltx_core_mlx.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
        from ltx_core_mlx.utils.positions import compute_audio_positions,compute_video_positions
        from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
        from ltx_pipelines_mlx.a2vid_two_stage import encode_source_audio
        from ltx_pipelines_mlx.utils import blocks
        from dubit_conditioning import frozen_target_audio,cached_camera_conditioning
        output.mkdir();start=time.monotonic()
        bin_dir=output/'bin';bin_dir.mkdir()
        (bin_dir/'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ['PATH']=f'{bin_dir}:/usr/bin:/bin'
        mx.set_default_device(mx.gpu);mx.set_memory_limit(11*1024**3);mx.set_cache_limit(0)
        pipe=A2VidDistilledPipeline(str(ROOT/'.local/ltx-2.3-q8-weights'),gemma_model_id=str(ROOT/'.local/ltx-gemma-weights'),low_memory=True,low_ram_streaming=True)
        voice=ROOT/'.local/dubit-camera-reference-two/reference.json'
        voice_record=json.loads(voice.read_text());audio=Path(voice_record['audio'])
        if digest(audio)!=voice_record['source_audio_sha256']:raise ValueError('Original voice changed')
        target=encode_source_audio(pipe,audio,num_frames=49,frame_rate=25,start_time=0,max_duration=1.96)
        mx.eval(target);pipe.audio_conditioner.free();blocks.aggressive_cleanup()
        latents={}
        for phase in camera_report['phases']:
            path=cache/f'camera-{phase["size"]}-latent.safetensors'
            if digest(path)!=phase['latent_sha256']:raise ValueError('Camera cache changed')
            latents[phase['size']]=mx.load(str(path))['video']
        raw_audio=pipe.audio_patchifier.unpatchify(target)
        ref_tokens,ref_positions=lipdub.patchify_lipdub_audio_reference_latent(raw_audio,pipe.audio_patchifier,negative_positions=True)
        original_builder=lipdub.create_noised_state;original_append=lipdub.append_ic_lora_reference_video_conditionings
        with frozen_target_audio(target) as audio_evidence,cached_camera_conditioning(camera,latents) as video_evidence:
            for stage,size in enumerate((256,512),1):
                dims=(7,size//32,size//32);positions=compute_audio_positions(49)
                state=lipdub.create_noised_state(base_shape=target.shape,conditionings=[],spatial_dims=dims,
                    positions=positions,seed=42+stage,sigma=1.0)
                state=AudioConditionByReferenceLatent(patchified=ref_tokens,positions=ref_positions,strength=1).apply(state,num_noisy_tokens=49)
                mx.eval(state.latent,state.denoise_mask)
                if not state.frozen or not bool(mx.all(state.denoise_mask==0).item()) or not bool(mx.all(state.latent[:,:49]==target).item()):
                    raise ValueError('The real target audio was not preserved in its frozen state')
                conditionings=[]
                lipdub.append_ic_lora_reference_video_conditionings(conditionings,[(str(camera),1)],height=size,width=size,
                    num_frames=49,video_encoder=None,reference_downscale_factor=1,frame_rate=25,
                    conditioning_attention_strength=1,conditioning_attention_mask=None)
                count=7*(size//32)**2
                video_state=lipdub.create_noised_state(base_shape=(1,count,128),conditionings=conditionings,
                    spatial_dims=dims,positions=compute_video_positions(*dims,frame_rate=25),seed=42,sigma=1)
                if video_state.frozen or video_state.latent.shape[1]!=2*count:
                    raise ValueError('Camera conditioning replaced or froze generated video tokens')
            if audio_evidence['audio_states_frozen']!=2 or video_evidence['cached_video_stages']!=2:
                raise ValueError('Both conditioning stages were not exercised')
        if lipdub.create_noised_state is not original_builder or lipdub.append_ic_lora_reference_video_conditionings is not original_append:
            raise ValueError('Scoped adapter did not restore the port functions')
        result=dict(actual_source_audio_encoded=True,actual_camera_latents_used=True,
            generated_video=False,adapter_inference_performed=False,quality_passed=False,training_performed=False,
            source_audio_sha256=digest(audio),reference_video_sha256=digest(camera),
            audio_states_frozen=audio_evidence['audio_states_frozen'],cached_video_stages=video_evidence['cached_video_stages'],
            target_audio_preserved_in_both_states=True,video_generation_tokens_not_frozen=True,
            functions_restored=True,elapsed_seconds=time.monotonic()-start,max_observed_mlx_peak_bytes=mx.get_peak_memory())
        (output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
