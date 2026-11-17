"""Experimental full-length local renderer. Visual acceptance remains required."""
import argparse
import hashlib
import json
import signal
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(ROOT / 'vendor/longcat-avatar-mlx'))
from worker import Job, Cancelled, atomic_json, validate_request
from inference_lease import GPUInferenceLease
from longcat_timeline import plan
from head_composite import source_face_bounds, head_mask, composite_head
from render_storage import require_render_space
from longcat_benchmark import load_vae, decode_frames, release, sha
from longcat_sampler import sample
from longcat_streaming import build_streamed_dit, build_streamed_text_encoder
import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from transformers import T5TokenizerFast
from safetensors import safe_open


def save_tensors(path, values, metadata):
    temporary = path.with_name(path.stem + '.tmp.safetensors')
    mx.save_safetensors(str(temporary), values, metadata=metadata)
    temporary.replace(path)


def recent_frames(previous, current):
    if len(current) >= 13:
        return current[-13:].copy()
    return np.concatenate([previous, current])[-13:].copy() if previous is not None else current.copy()


def run(request, job):
    validate_request(request)
    if request.get('motion') or request.get('source_video'):
        raise ValueError('LongCat currently uses a reference photo and voice-over. Reference-motion integration is not validated yet.')
    folder = ROOT / '.local/longcat-models/q4'
    if not (folder / 'download-complete.json').exists():
        raise RuntimeError('Complete the LongCat model installation before rendering')
    manifest = json.loads((ROOT / 'backend/model_manifests/longcat-q4.json').read_text())
    job.status(.01, 'Verifying local model files')
    for entry in manifest['files']:
        job.check(); path = folder / entry['path']
        if path.stat().st_size != entry['bytes'] or (entry.get('sha256') and sha(path) != entry['sha256']):
            raise ValueError('Model verification failed: ' + entry['path'])
    width, height = request.get('width', 768), request.get('height', 512)
    if any(type(x) is not int or x < 128 or x > 1024 or x % 16 for x in (width, height)):
        raise ValueError('Model dimensions must be multiples of 16 between 128 and 1024')
    mx.set_memory_limit(8 * 1024**3); mx.set_cache_limit(256 * 1024**2)
    preserve_body = request.get('preserve_source_body', True)
    if type(preserve_body) is not bool: raise ValueError('preserve_source_body must be Boolean')
    face_bounds = source_face_bounds(request) if preserve_body else None
    fingerprint_data = dict(photo_sha256=sha(Path(request['photo'])), audio_sha256=sha(Path(request['audio'])),
                            width=width, height=height, model=manifest['revision'], recipe='dmd8-guidance1-avc13-v1')
    prompt = request.get('prompt') or 'The person in the reference photo speaks naturally toward the camera with subtle head movements. Preserve the reference appearance, hairstyle, clothing, composition and background. Stable camera.'
    fingerprint_data['prompt'] = prompt
    fingerprint_data['preserve_source_body'] = preserve_body
    fingerprint_data['face_bounds'] = face_bounds
    fingerprint_data['implementation'] = {name: sha(ROOT / name) for name in
        ['backend/longcat_render.py', 'backend/longcat_timeline.py', 'backend/head_composite.py', 'research/longcat_sampler.py',
         'research/longcat_streaming.py', 'research/longcat_attention.py', 'research/longcat_audio_probe.py', 'research/longcat_benchmark.py']}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    identity_path = job.directory / 'longcat-request.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != fingerprint_data:
        raise ValueError('Cached render inputs or settings changed. Create a new job.')
    atomic_json(identity_path, fingerprint_data)
    job.status(.02, 'Preparing the complete voice-over locally')
    audio_dir = job.directory / 'audio-features'
    # Recompute analysis on recovery to verify the actual current recording.
    job.run([sys.executable, ROOT / 'research/longcat_audio_probe.py', '--audio', request['audio'],
             '--weights', folder, '--full-audio', '--output', audio_dir])
    audio_report = json.loads((audio_dir / 'report.json').read_text())
    if audio_report['source_sha256'] != fingerprint_data['audio_sha256']:
        raise ValueError('Audio changed during preparation')
    audio = mx.load(str(audio_dir / 'audio.safetensors'))['audio']
    total_frames, windows = plan(audio_report['analysis_samples'])
    require_render_space(job.directory, total_frames, width, height)
    atomic_json(job.directory / 'timeline.json', dict(total_frames=total_frames, fps=25, windows=[asdict(w) for w in windows]))
    job.status(.04, 'Preparing the reference description')
    tokenizer = T5TokenizerFast.from_pretrained(str(folder / 'tokenizer'), local_files_only=True)
    tokens = tokenizer(prompt, return_tensors='np', padding='max_length', max_length=512, truncation=True)
    mask = mx.array(tokens.attention_mask)
    encoder = build_streamed_text_encoder(folder / 'text_encoder')
    text = encoder(mx.array(tokens.input_ids), mask)[:, None].astype(mx.bfloat16); mx.eval(text)
    del encoder, tokenizer; release(); job.check()
    photo = ImageOps.exif_transpose(Image.open(request['photo'])).convert('RGB')
    animation_mask = head_mask(photo, (width, height), face_bounds) if preserve_body else None
    photo = ImageOps.pad(photo, (width, height), method=Image.Resampling.LANCZOS)
    source_pixels = np.asarray(photo)
    pixels = np.asarray(photo, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    vae = load_vae(folder / 'vae')
    anchor = vae.normalize_latents(vae.encode(mx.array(pixels[None, :, None]).astype(mx.bfloat16))); mx.eval(anchor)
    del vae; release()
    shift = json.loads((folder / 'scheduler/scheduler_config.json').read_text())['shift']
    cache = job.directory / 'windows'; cache.mkdir(exist_ok=True)
    tail = None
    paths = []
    for index, window in enumerate(windows):
        job.check(); path = cache / f'{index:04d}.npy'; metadata_path = path.with_suffix('.json')
        cache_identity = dict(request=fingerprint, window=asdict(window))
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('identity') != cache_identity or metadata.get('sha256') != sha(path):
                raise ValueError('Generated window cache failed verification')
            frames = np.load(path, mmap_mode='r', allow_pickle=False)
            if frames.shape != (window.append_frames, height, width, 3) or frames.dtype != np.uint8:
                raise ValueError('Generated window geometry mismatch')
            tail = recent_frames(tail, frames)
            paths.append(path); del frames
            continue
        if window.carry_frames:
            if tail is None or len(tail) != 13: raise ValueError('Missing continuation context')
            vae = load_vae(folder / 'vae')
            carry = mx.array((tail.astype(np.float32).transpose(3, 0, 1, 2) / 127.5 - 1)[None]).astype(mx.bfloat16)
            condition = vae.normalize_latents(vae.encode(carry)); mx.eval(condition)
            condition = mx.concatenate([anchor, condition], axis=2)
            del vae, carry; release()
        else: condition = anchor
        sounds = audio[:, window.source_start:window.source_start + window.model_frames]
        if sounds.shape[1] < window.model_frames:
            # Only model context is padded at the final fractional frame. Source audio is untouched.
            sounds = mx.concatenate([sounds, mx.repeat(audio[:, -1:], window.model_frames - sounds.shape[1], axis=1)], axis=1)
        checkpoint = cache / f'{index:04d}-latents.safetensors'; resumed = None; completed = 0
        checkpoint_identity = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        if checkpoint.exists():
            with safe_open(str(checkpoint), framework='np') as stream: metadata = stream.metadata() or {}
            if metadata.get('identity') != checkpoint_identity: raise ValueError('Latent checkpoint provenance mismatch')
            completed = int(metadata['step']); resumed = mx.load(str(checkpoint))['latents']
        def step_done(step, latents):
            nonlocal completed
            save_tensors(checkpoint, {'latents': latents}, {'identity': checkpoint_identity, 'step': str(step)})
            completed = step
            job.status(.05 + .85 * (index + step/8) / len(windows), f'Generating segment {index+1} of {len(windows)} • step {step} of 8')
        def block_done(block, count):
            fraction = (completed + (block + 1) / count) / 8
            job.status(.05 + .85 * (index + fraction) / len(windows),
                       f'Generating segment {index+1} of {len(windows)} • step {completed+1} of 8 • block {block+1} of {count}')
        model = build_streamed_dit(folder / 'dit', job.check, block_done)
        latents = sample(model, condition, sounds, text, mask, text, mask,
                         frames=window.model_frames, seed=42+index, shift=shift,
                         resume_latents=resumed, start_step=completed, callback=step_done,
                         check_cancel=job.check, num_ref_latents=1 if window.carry_frames else 0)
        del model, condition, sounds, resumed; release()
        vae = load_vae(folder / 'vae')
        frames = decode_frames(vae, latents[:, :, 1:] if window.carry_frames else latents,
                               window.model_frames, job.check)[window.carry_frames:window.carry_frames+window.append_frames]
        del vae, latents; release()
        if preserve_body:
            for frame_index in range(len(frames)):
                job.check()
                frames[frame_index] = composite_head(source_pixels, frames[frame_index], animation_mask)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.save(stream, frames, allow_pickle=False)
        temporary.replace(path); atomic_json(metadata_path, dict(identity=cache_identity, sha256=sha(path)))
        tail = recent_frames(tail, frames)
        paths.append(path); del frames
    if sha(Path(request['audio'])) != fingerprint_data['audio_sha256'] or sha(Path(request['photo'])) != fingerprint_data['photo_sha256']:
        raise ValueError('Source media changed while rendering')
    job.status(.92, 'Encoding the complete video with original audio')
    import imageio_ffmpeg
    output = job.directory / 'generated.mov'
    raw = job.directory / 'assembled.rgb'
    # Disk streaming avoids accumulating a whole lesson in unified memory.
    with raw.open('wb') as stream:
        for path in paths:
            job.check(); frames = np.load(path, mmap_mode='r', allow_pickle=False)
            for frame in frames: stream.write(frame.tobytes())
            del frames
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    job.run([ff, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
             '-r', '25', '-i', raw, '-i', request['audio'], '-map', '0:v', '-map', '1:a',
             '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_f32le', output])
    job.status(.98, 'Verifying original audio and every exported frame')
    job.run([sys.executable, ROOT / 'scripts/verify_export.py', request['audio'], output,
             '--expected-frames', str(total_frames), '--fps', '25'])
    raw.unlink()
    atomic_json(job.directory / 'provenance.json', dict(backend='LongCat-Video-Avatar-1.5 MLX local',
                request=fingerprint_data, windows=len(windows), frames=total_frames,
                peak_mlx_bytes=mx.get_peak_memory(), quality_passed=False, visual_review_required=True))
    job.status(1, 'Video generated locally • visual review required', 'Completed', output)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('request', type=Path); args = parser.parse_args()
    request = json.loads(args.request.read_text()); job = Job(request['directory'], attempt=request.get('attempt'))
    signal.signal(signal.SIGTERM, lambda *_: setattr(job, 'cancelled', True))
    signal.signal(signal.SIGINT, lambda *_: setattr(job, 'cancelled', True))
    try:
        # Invalid or unconsented media must fail immediately, even when another
        # render owns Metal. Validate again in run() to catch later file changes.
        validate_request(request)
        with GPUInferenceLease(job.check,
                            lambda: job.status(0, 'Waiting for the active local render to release memory')):
            run(request, job)
    except (Cancelled, InterruptedError): job.status(0, 'Cancelled • cached progress retained', 'Cancelled')
    except Exception as error:
        job.status(0, str(error), 'Failed'); raise


if __name__ == '__main__': main()
