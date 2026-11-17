"""Experimental staged, local LongCat AI2V benchmark. Not the app renderer."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/longcat-avatar-mlx'))
sys.path.insert(0, str(ROOT / 'backend'))
from inference_lease import GPUInferenceLease
import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from longcat_streaming import TensorStore, build_streamed_dit, build_streamed_text_encoder
from longcat_sampler import sample


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def release():
    gc.collect()
    mx.clear_cache()


def load_vae(folder):
    from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
    model = AutoencoderKLWan.from_config(json.loads((folder / 'config.json').read_text()))
    store = TensorStore(folder)
    weights = store.weights('')
    # These are configuration-derived normalization constants, not trained weights.
    for name in ('mean', 'std', 'inv_std'):
        if name not in store.entries:
            weights.append((name, getattr(model, name)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def decode_frames(vae, normalized_latents, expected_frames, check_cancel=None):
    """Eager temporal chunks prevent a whole movie's decode graph accumulating."""
    z = vae.post_quant_conv(vae.denormalize_latents(normalized_latents).astype(mx.bfloat16))
    cache = [None] * vae._count_decoder_cache_slots()
    chunks = []
    for index in range(z.shape[2]):
        if check_cancel:
            check_cancel()
        chunk = vae.decoder(z[:, :, index:index+1], feat_cache=cache, feat_idx=[0])
        mx.eval(chunk)
        if not bool(mx.all(mx.isfinite(chunk))):
            raise ValueError(f'Nonfinite decoded video chunk {index}')
        chunks.append(np.asarray(mx.clip((chunk[0].transpose(1, 2, 3, 0) + 1) * 127.5, 0, 255).astype(mx.uint8)))
    frames = np.concatenate(chunks)
    if len(frames) != expected_frames:
        raise ValueError(f'Decoder frame mismatch: {len(frames)}')
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / '.local/longcat-models/q4')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=29)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prompt', default='The person in the reference photo speaks naturally toward the camera with subtle head movements. Preserve the reference appearance, hairstyle, clothing, seated composition and background. Stable camera.')
    args = parser.parse_args()
    if args.frames < 5 or args.frames > 125 or (args.frames - 1) % 4:
        raise ValueError('Frames must be 4n+1 in 5–125')
    if any(n < 128 or n > 1024 or n % 16 for n in (args.width, args.height)):
        raise ValueError('Dimensions must be multiples of 16 in 128–1024')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'request.json').exists() and not args.resume:
        raise ValueError('Use a new output directory; do not overwrite a previous benchmark')
    started = time.monotonic()
    lease = None
    compute_started = None
    def status(stage, **extra):
        result = dict(stage=stage, elapsed_seconds=time.monotonic()-started,
                      render_elapsed_seconds=time.monotonic()-compute_started if compute_started is not None else 0,
                      generated_video=stage == 'generated_awaiting_review', quality_passed=False, **extra)
        temporary = args.output / 'progress.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.output / 'progress.json')
        print(json.dumps(result), flush=True)
    try:
        status('waiting_for_verified_weights')
        deadline = time.monotonic() + args.wait_seconds
        while not (args.weights / 'download-complete.json').exists():
            if time.monotonic() >= deadline:
                raise FileNotFoundError('Complete verified model download required')
            time.sleep(5)
        manifest = json.loads((args.weights / 'download-manifest.json').read_text())
        status('verifying_weights')
        for item in manifest['files']:
            path = args.weights / item['path']
            if path.stat().st_size != item['bytes'] or (item.get('sha256') and sha(path) != item['sha256']):
                raise ValueError('Checkpoint verification failed: ' + item['path'])
        request = dict(image=str(args.image.resolve()), image_sha256=sha(args.image),
                       audio=str(args.audio.resolve()), audio_sha256=sha(args.audio),
                       frames=args.frames, width=args.width, height=args.height,
                       prompt=args.prompt, model_revision=manifest['revision'], seed=42,
                       text_guidance_scale=1.0, audio_guidance_scale=1.0,
                       sampler_revision='reference-ai2v-dmd-v2',
                       compatibility_test=True, original_audio_preserved=True)
        request_path = args.output / 'request.json'
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise ValueError('Resume inputs, model revision or settings have changed')
        request_path.write_text(json.dumps(request, indent=2))
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        def check_cancel():
            if (args.output / 'cancel').exists():
                raise InterruptedError('Render cancelled; completed steps retained for recovery')
        check_cancel()
        lease = GPUInferenceLease(check_cancel,
                               lambda: status('waiting_for_render_memory')).acquire()
        compute_started = time.monotonic()
        # Request an allocator budget; MLX may temporarily exceed it. Record peaks
        # instead of representing this setting as a guaranteed hard memory cap.
        mx.set_memory_limit(8 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)
        status('audio_encoding')
        subprocess.run([sys.executable, str(ROOT / 'research/longcat_audio_probe.py'),
                        '--weights', str(args.weights), '--audio', str(args.audio),
                        '--seconds', str(max(1, args.frames / 25)),
                        '--output', str(args.output / 'audio')], check=True)
        audio = mx.load(str(args.output / 'audio/audio.safetensors'))['audio']

        status('text_encoding')
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(str(args.weights / 'tokenizer'), local_files_only=True)
        encoder = build_streamed_text_encoder(args.weights / 'text_encoder')
        texts, masks = [], []
        for prompt in (args.prompt,):
            tokens = tokenizer(prompt, return_tensors='np', padding='max_length', max_length=512, truncation=True)
            mask = mx.array(tokens.attention_mask)
            value = encoder(mx.array(tokens.input_ids), mask)
            mx.eval(value)
            texts.append(value[:, None].astype(mx.bfloat16)); masks.append(mask)
        # No unconditional prediction is used by the distilled reference recipe.
        texts.append(texts[0]); masks.append(masks[0])
        del encoder, tokenizer
        release()

        status('portrait_encoding')
        vae = load_vae(args.weights / 'vae')
        portrait = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
        portrait = ImageOps.pad(portrait, (args.width, args.height), method=Image.Resampling.LANCZOS)
        pixels = np.asarray(portrait, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
        reference = vae.normalize_latents(vae.encode(mx.array(pixels[None, :, None]).astype(mx.bfloat16)))
        mx.eval(reference)
        del vae
        release()
        status('loading_streamed_transformer')
        dit = build_streamed_dit(args.weights / 'dit', check_cancel)
        shift = json.loads((args.weights / 'scheduler/scheduler_config.json').read_text())['shift']
        checkpoint = args.output / 'latest-latents.safetensors'
        resumed, start_step = None, 0
        if args.resume and checkpoint.exists():
            from safetensors import safe_open
            with safe_open(str(checkpoint), framework='np') as source:
                metadata = source.metadata() or {}
            if metadata.get('request_sha256') != request_hash:
                raise ValueError('Resume checkpoint provenance mismatch')
            start_step = int(metadata['completed_step'])
            resumed = mx.load(str(checkpoint))['latents']
        def step_done(step, latents):
            temporary = args.output / 'latents.tmp.safetensors'
            mx.save_safetensors(str(temporary), {'latents': latents},
                                metadata={'request_sha256': request_hash, 'completed_step': str(step)})
            temporary.replace(checkpoint)
            status('denoising', step=step, total_steps=8, peak_mlx_bytes=mx.get_peak_memory())
        latents = sample(dit, reference, audio, texts[0], masks[0], texts[1], masks[1],
                         frames=args.frames, shift=shift, callback=step_done,
                         resume_latents=resumed, start_step=start_step, check_cancel=check_cancel)
        del dit, texts, masks, audio
        release()
        status('video_decoding')
        vae = load_vae(args.weights / 'vae')
        frames = decode_frames(vae, latents, args.frames, check_cancel)
        del vae
        release()
        status('encoding_original_audio_export')
        if sha(args.audio) != request['audio_sha256'] or sha(args.image) != request['image_sha256']:
            raise ValueError('Source media changed during rendering. Restore the original inputs before resuming.')
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        temporary = args.output / 'video-only.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', f'{args.width}x{args.height}', '-r', '25', '-i', '-',
                        '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', str(temporary)],
                       input=frames.tobytes(), check=True)
        output = args.output / 'longcat-compatibility.mov'
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(temporary), '-i', str(args.audio),
                        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'pcm_f32le',
                        '-t', str(args.frames/25), str(output)], check=True)
        # Decode every exported frame and compare decoded PCM with the original excerpt.
        decoded = subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(output),
                                           '-map', '0:v', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
        if len(decoded) != args.frames * args.width * args.height * 3:
            raise ValueError('Exported video failed complete frame decode')
        def pcm(path):
            return subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(path),
                                            '-map', '0:a', '-t', str(args.frames/25),
                                            '-f', 'f32le', '-c:a', 'pcm_f32le', '-'])
        original_pcm, output_pcm = pcm(args.audio), pcm(output)
        if not original_pcm or output_pcm != original_pcm:
            raise ValueError('Original audio verification failed')
        report = dict(status='generated_awaiting_review', generated_video=True, quality_passed=False,
                      elapsed_seconds=time.monotonic()-compute_started, total_elapsed_seconds=time.monotonic()-started,
                      peak_mlx_bytes=mx.get_peak_memory(),
                      output=str(output.resolve()), frames=len(frames), request=request,
                      decoded_audio_sha256=hashlib.sha256(output_pcm).hexdigest(),
                      full_frame_decode_passed=True)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        status('generated_awaiting_review', output=str(output.resolve()))
    except InterruptedError as error:
        status('cancelled', error=str(error))
        raise
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    main()
