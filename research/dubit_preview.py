"""Offline, short Dub-It compatibility experiment using actual camera motion.

Not a production backend or a claim of perceptual acceptance. Requires the
user-downloaded official adapter. Run inside the OS network-denied sandbox.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'research')]
from download_model_manifest import digest
from inference_lease import GPUInferenceLease
from ltx_audio_preview import validate_model, MODEL, GEMMA, SOURCE_REVISION

CODE_HASH = digest(Path(__file__))
PROMPT = ('The same real man speaks naturally to camera. Preserve the reference '
          'performance, head motion, gaze, eyebrows, lighting and facial identity. '
          'His mouth articulates the supplied speech accurately.')


def validate_inputs(args):
    """Fail before loading MLX or taking the GPU lease if inputs are incomplete."""
    lock = json.loads((ROOT / 'research/dubit-source-lock.json').read_text())
    item = lock['files'][0]
    if not args.lora.is_file():
        raise ValueError('The official Dub-It adapter is missing; provide the downloaded local file')
    if args.lora.stat().st_size != item['bytes'] or digest(args.lora) != item['sha256']:
        raise ValueError('Dub-It adapter differs from the pinned official file; inference was not started')
    reference_path = args.reference / 'reference.json'
    reference = json.loads(reference_path.read_text())
    camera = args.reference / 'camera-motion-reference.mkv'
    cache = json.loads((args.cache / 'report.json').read_text())
    if reference['frames'] not in (49, 121) or (reference['fps'], reference['output_size']) != (25, [512, 512]):
        raise ValueError('Expected a prepared 49- or 121-frame reference at 512 square and 25 fps')
    if not reference['reference_only'] or reference['generated_video']:
        raise ValueError('Expected a prepared reference-only input')
    kind = reference.get('conditioning_kind', 'real-camera')
    if kind not in ('real-camera', 'kj-photo-motion') or cache.get('conditioning_kind', 'real-camera') != kind:
        raise ValueError('Conditioning kind differs from its encoded cache')
    if kind == 'kj-photo-motion':
        derived = reference['derived_conditioning']
        if digest(Path(derived['photo'])) != derived['photo_sha256']:
            raise ValueError('The real photo used for KJ conditioning has changed')
    if not (digest(camera) == reference['output_sha256'] == cache['reference_sha256']):
        raise ValueError('Camera conditioning changed after encoding')
    if digest(reference_path) != cache['reference_provenance_sha256']:
        raise ValueError('Camera preparation provenance changed')
    for key, hash_key in [('video', 'source_video_sha256'), ('audio', 'source_audio_sha256')]:
        if digest(Path(reference[key])) != reference[hash_key]:
            raise ValueError('An original input changed after reference preparation')
    if sorted(p['size'] for p in cache['phases']) != [256, 512]:
        raise ValueError('Both actual camera encodings are required')
    for phase in cache['phases']:
        if digest(args.cache / f'camera-{phase["size"]}-latent.safetensors') != phase['latent_sha256']:
            raise ValueError('Cached camera latent failed integrity verification')
    return lock, reference, camera, cache


def run(args, inputs):
    import imageio_ffmpeg
    import mlx.core as mx
    from ltx_pipelines_mlx.lipdub import LipDubPipeline
    from ltx_pipelines_mlx.a2vid_two_stage import encode_source_audio
    from ltx_pipelines_mlx.utils.blocks import aggressive_cleanup
    from ltx_staged_text import StagedPromptEncoder
    from ltx_reference_video_io import bundled_video_probe
    from dubit_conditioning import frozen_target_audio, cached_camera_conditioning
    from ltx_video_export import export_original_audio
    from dubit_adapter_evidence import streamed_adapter_evidence
    from PIL import Image, ImageOps

    lock, reference, camera, cache = inputs
    frames = reference['frames']
    prompt = (PROMPT if args.photo is None else
        'The same man from the reference photograph speaks naturally to camera. '
        'Preserve his photographic facial identity, hair, black shirt and appearance. '
        'Use restrained natural head and eye movements from the recorded reference performance. '
        'His mouth articulates the supplied speech accurately.')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    peak = 0
    def progress(stage):
        nonlocal peak
        peak = max(peak, mx.get_peak_memory())
        record = dict(stage=stage, elapsed_seconds=time.monotonic()-start,
                      max_observed_mlx_peak_bytes=peak, quality_passed=False)
        temporary = out / 'progress.tmp'
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(out / 'progress.json')
        print(json.dumps(record), flush=True)

    provenance = dict(generated_video=False, quality_passed=False, training_performed=False,
        source_revision=SOURCE_REVISION, adapter=lock, code_sha256=CODE_HASH,
        helper_sha256={name: digest(ROOT/'research'/name) for name in
            ('dubit_conditioning.py', 'ltx_reference_video_io.py', 'ltx_staged_text.py',
             'ltx_video_export.py', 'dubit_adapter_evidence.py')},
        reference=reference, camera_cache=cache, prompt=prompt, seed=42,
        frames=frames, fps=25, output_geometry=[512,512],
        appearance_photo_conditioned=args.photo is not None or reference.get('conditioning_kind')=='kj-photo-motion',
        camera_performance_conditioned=True,
        conditioning_kind=reference.get('conditioning_kind','real-camera'),
        audio_conditioning_mode=args.audio_conditioning,
        original_target_audio_frozen=args.audio_conditioning=='frozen-target',
        lora_mode='unfused', transformer_streaming=True)
    try:
        if shutil.disk_usage(out).free < 10*1024**3:
            raise ValueError('Leave 10 GiB of free space for rendering and system swap')
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(11*1024**3)
        mx.set_cache_limit(0)
        progress('Verifying complete pinned base and text weights')
        provenance['base_manifest_sha256'] = validate_model(args.model, MODEL)
        provenance['text_manifest_sha256'] = validate_model(args.gemma, GEMMA)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = out/'bin'; bin_dir.mkdir()
        (bin_dir/'ffmpeg').symlink_to(ffmpeg)
        os.environ['PATH'] = f'{bin_dir}:/usr/bin:/bin'
        images = None
        if args.photo is not None:
            provenance.update(photo=str(args.photo), photo_sha256=digest(args.photo), crop=args.crop,
                              photo_anchor_frame=0, photo_anchor_strength=1.0)
            source=ImageOps.exif_transpose(Image.open(args.photo)).convert('RGB')
            left,top,right,bottom=args.crop
            if not (0<=left<right<=source.width and 0<=top<bottom<=source.height and right-left==bottom-top):
                raise ValueError('Use a square crop inside the unchanged real photograph')
            crop=out/'source-photo-crop.png'
            source.crop(args.crop).resize((512,512),Image.Resampling.LANCZOS).save(crop)
            provenance['source_photo_crop_sha256']=digest(crop)
            images=[(str(crop),0,1.0)]
        (out/'request.json').write_text(json.dumps(provenance, indent=2))
        pipe = LipDubPipeline(str(args.model), lora_paths=[(str(args.lora),1.0)],
            gemma_model_id=str(args.gemma), low_memory=True, low_ram_streaming=True)
        pipe.prompt_encoder = StagedPromptEncoder(args.model, str(args.gemma), progress)
        pipe.verbose = True
        adapter_stages = {}
        # Report real denoising steps without loading a preview decoder alongside
        # the transformer. The upstream hook is called after each actual step.
        def step_hook(*geometry, stage=None):
            def on_step(index, total, video_x0, sigma):
                if index == 0:
                    adapter_stages[stage] = streamed_adapter_evidence(pipe.dit, args.lora)
                progress(f'Denoising stage {stage}: step {index+1} of {total}')
            return on_step
        pipe._stepwise_hook = step_hook
        progress('Encoding original target voice-over')
        target = encode_source_audio(pipe, Path(reference['audio']), num_frames=frames,
            frame_rate=25, start_time=0, max_duration=frames/25)
        mx.eval(target)
        pipe.audio_conditioner.free()
        aggressive_cleanup()
        reference_calls = 0
        def reference_audio(path):
            nonlocal reference_calls
            if Path(path).resolve() != camera.resolve():
                raise ValueError('Unexpected reference audio input')
            reference_calls += 1
            return pipe.audio_patchifier.unpatchify(target)
        pipe._encode_reference_audio_vae_latent = reference_audio
        latents = {p['size']: mx.load(str(args.cache/f'camera-{p["size"]}-latent.safetensors'))['video']
                   for p in cache['phases']}
        progress('Running actual Dub-It adapter with cached camera motion')
        audio_context = (frozen_target_audio(target) if args.audio_conditioning=='frozen-target'
                         else nullcontext(dict(audio_states_frozen=0)))
        with bundled_video_probe(), audio_context as audio_evidence, \
                cached_camera_conditioning(camera, latents, frames=frames) as video_evidence:
            video, returned_audio, fps = pipe.generate_lipdub(prompt=prompt,
                reference_video_path=str(camera), height=512, width=512,
                reference_strength=1.0, seed=42, images=images)
            mx.eval(video, returned_audio)
        if tuple(video.shape) != (1,128,(frames-1)//8+1,16,16) or not bool(mx.all(mx.isfinite(video)).item()):
            raise ValueError('Generated video latent has invalid geometry or nonfinite values')
        if fps != 25 or reference_calls != 1 or video_evidence['cached_video_stages'] != 2:
            raise ValueError('Generation did not exercise the actual audio reference and both camera stages')
        if set(adapter_stages) != {1,2}:
            raise ValueError('Dub-It adapter execution was not observed in both denoising stages')
        expected_audio = pipe.audio_patchifier.unpatchify(target)
        if returned_audio.shape != expected_audio.shape or not bool(mx.all(mx.isfinite(returned_audio)).item()):
            raise ValueError('Generated conditioning audio has invalid geometry or values')
        if args.audio_conditioning=='frozen-target' and (
                audio_evidence['audio_states_frozen'] != 2 or
                not bool(mx.all(returned_audio == expected_audio).item())):
            raise ValueError('Frozen-target comparison did not preserve the original audio tokens')
        checkpoint = out/'video-latent.partial.safetensors'
        mx.save_safetensors(str(checkpoint), {'video':video,'conditioning_audio':returned_audio})
        checkpoint.replace(out/'video-latent.safetensors')
        provenance.update(adapter_inference_performed=True,
            adapter_binding_evidence=adapter_stages,
            latent_sha256=digest(out/'video-latent.safetensors'),
            audio_conditioning_verified=True, conditioning_stages=2,
            target_audio_preserved_in_both_states=args.audio_conditioning=='frozen-target')
        (out/'checkpoint.json').write_text(json.dumps(provenance, indent=2))
        progress('Saved generated latent; releasing generation weights')
        pipe._detach_lora_adapters()
        pipe.dit = None
        pipe.prompt_encoder.free()
        pipe.image_conditioner.free()
        pipe.upsampler = None
        aggressive_cleanup()
        progress('Decoding actual generated video')
        silent = out/'generated-silent.mp4'
        pipe.video_decoder_block.decode_and_stream(video, str(silent), frame_rate=25)
        progress('Checking every frame, cadence and original audio')
        for key, hash_key in [('video','source_video_sha256'),('audio','source_audio_sha256')]:
            if digest(Path(reference[key])) != reference[hash_key]:
                raise ValueError('An original input changed during inference')
        if args.photo is not None and digest(args.photo)!=provenance['photo_sha256']:
            raise ValueError('Original reference photograph changed during inference')
        export_original_audio(silent, reference['audio'], out/'review.mov', frames=frames, fps=25)
        provenance.update(generated_video=True, output=str(out/'review.mov'),
            output_sha256=digest(out/'review.mov'), original_decoded_audio_verified=True,
            elapsed_seconds=time.monotonic()-start, max_observed_mlx_peak_bytes=peak)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        progress('Completed compatibility experiment; visual acceptance required')
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=str(error),generated_video=False,
            quality_passed=False, latent_checkpoint_available=(out/'checkpoint.json').is_file()),indent=2))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lora', required=True, type=lambda p: Path(p).resolve())
    for name, default in [('reference','dubit-camera-reference-two'),('cache','dubit-camera-encoder-probe'),
                          ('model','ltx-2.3-q8-weights'),('gemma','ltx-gemma-weights')]:
        parser.add_argument('--'+name,type=lambda p:Path(p).resolve(),default=ROOT/'.local'/default)
    parser.add_argument('--output',required=True,type=lambda p:Path(p).resolve())
    parser.add_argument('--consent',action='store_true')
    parser.add_argument('--photo',type=lambda p:Path(p).resolve(),help='Optional real-photo frame-zero appearance-anchor experiment')
    parser.add_argument('--crop',nargs=4,type=int)
    parser.add_argument('--audio-conditioning',choices=('reference','frozen-target'),default='reference',
        help='reference follows upstream joint denoising; frozen-target is an explicit experimental comparison')
    args = parser.parse_args()
    if not args.consent: parser.error('Explicit likeness consent is required')
    if (args.photo is None)!=(args.crop is None): parser.error('Supply both --photo and --crop for the appearance-anchor experiment')
    if args.photo is not None and not args.photo.is_file(): parser.error('The real source photo must exist')
    if args.output.exists(): parser.error('Use a fresh output directory')
    inputs = validate_inputs(args)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',LTX2_LORA_MODE='unfused')
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active local inference',flush=True)):
        run(args, inputs)


if __name__ == '__main__': main()
