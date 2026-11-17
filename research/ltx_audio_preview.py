"""Short local LTX 2.3 audio-conditioned comparison. No production quality claim.

Use the pinned LTX runtime in a process with OS networking denied. This adapter
stages the two large text components and muxes original audio after video decode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from download_model_manifest import digest, matches
from inference_lease import GPUInferenceLease

# Capture the loaded script before a potentially long download wait. Hashing
# __file__ only when inference starts could describe an edit made while waiting.
SCRIPT_SHA256 = digest(Path(__file__))

SOURCE_REVISION = '747b30ef10e92a011e85b2aa0f6c4ee86ced5218'
MODEL = ('dgrauet/ltx-2.3-mlx-q8', '69f8f11996b7b2ee54e82d0db7a5acad14b8685d')
GEMMA = ('mlx-community/gemma-3-12b-it-4bit', '86cc6a8dedbc456dd0e4af01a9d09f396f77e558')
FPS = 25
PROMPT = ('A locked camera studio shot of the same man in the reference photo, seated and speaking '
          'clearly to the camera in a calm instructional tone. Natural restrained head movements, '
          'occasional blinking, precise mouth articulation synchronized to the recorded speech. '
          'Keep his face, hair, black shirt, lighting and background consistent.')


def validate_model(folder, expected):
    manifest_path = folder / 'download-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((folder / 'download-complete.json').read_text())
    if (manifest.get('repo'), manifest.get('revision')) != expected or complete != dict(
            repo=expected[0], revision=expected[1], verified=True):
        raise ValueError(f'Expected the verified pinned model: {expected[0]}')
    lock = json.loads(Path(__file__).with_name('ltx-source-lock.json').read_text())
    if manifest not in lock['models']:
        raise ValueError('Model inventory differs from the checked-in public weight lock')
    for item in manifest['files']:
        name = Path(item['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe model path')
        if not (item.get('sha256') or item.get('git_blob_sha1')) or not matches(
                folder / name, item['bytes'], item.get('sha256'), item.get('git_blob_sha1')):
            raise ValueError(f'Model file failed integrity verification: {name}')
    return digest(manifest_path)


def run(args):
    import importlib.metadata
    import shutil
    import imageio_ffmpeg
    import mlx.core as mx
    from PIL import Image, ImageOps
    from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
    from ltx_pipelines_mlx.a2vid_two_stage import encode_source_audio
    from ltx_pipelines_mlx.utils import blocks
    from ltx_staged_text import StagedPromptEncoder

    start = time.monotonic()
    prompt = args.prompt_text
    provenance = dict(generated_video=False, quality_passed=False, training_performed=False,
                      source_revision=SOURCE_REVISION, runtime_versions={name: importlib.metadata.version(name)
                        for name in ('ltx-core-mlx', 'ltx-pipelines-mlx', 'mlx', 'mlx-lm', 'transformers')},
                      model_revision=MODEL[1], gemma_revision=GEMMA[1], photo_sha256=digest(args.photo),
                      audio_sha256=digest(args.audio), code_sha256=SCRIPT_SHA256,
                      staged_text_code_sha256=digest(ROOT/'research/ltx_staged_text.py'),crop=args.crop,
                      frames=args.frames, fps=FPS, seconds=args.frames / FPS, seed=42, prompt=prompt,
                      prompt_file_sha256=args.prompt_file_sha256,
                      transformer_streaming=True, output_geometry=[512, 512])
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    observed_peak = 0

    def progress(stage):
        nonlocal observed_peak
        observed_peak = max(observed_peak, mx.get_peak_memory())
        state = dict(stage=stage, elapsed_seconds=time.monotonic() - start,
                     max_observed_mlx_peak_bytes=observed_peak, quality_passed=False)
        temp = out / 'progress.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / 'progress.json')
        print(json.dumps(state), flush=True)

    try:
        mx.set_default_device(mx.gpu)
        # Guideline only: MLX's allocator limit is not a hard system RAM cap.
        mx.set_memory_limit(11 * 1024**3)
        mx.set_cache_limit(0)
        progress('Verifying pinned local weights')
        provenance['model_manifest_sha256'] = validate_model(args.model, MODEL)
        provenance['text_manifest_sha256'] = validate_model(args.gemma, GEMMA)
        if shutil.disk_usage(out).free < 10 * 1024**3:
            raise ValueError('Leave 10 GiB of free space for the test and system swap')
        source = ImageOps.exif_transpose(Image.open(args.photo)).convert('RGB')
        left, top, right, bottom = args.crop
        if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height and right-left == bottom-top):
            raise ValueError('Use a square crop inside the original photo')
        crop = out / 'source-crop.png'
        source.crop(args.crop).resize((512, 512), Image.Resampling.LANCZOS).save(crop)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = out / 'bin'
        bin_dir.mkdir()
        (bin_dir / 'ffmpeg').symlink_to(ffmpeg)
        os.environ['PATH'] = f'{bin_dir}:/usr/bin:/bin'
        (out / 'request.json').write_text(json.dumps(provenance, indent=2))

        pipe = A2VidDistilledPipeline(str(args.model), gemma_model_id=str(args.gemma),
                                     low_memory=True, low_ram_streaming=True)
        pipe.prompt_encoder = StagedPromptEncoder(args.model, str(args.gemma),progress)
        pipe.verbose = True
        progress('Encoding actual voice-over into conditioning')
        pipe._source_audio_tokens = encode_source_audio(
            pipe, args.audio, num_frames=args.frames, frame_rate=FPS, start_time=0,
            max_duration=args.frames / FPS)
        try:
            video, _ = pipe.generate_two_stage(prompt=prompt, height=512, width=512,
                                               num_frames=args.frames, frame_rate=FPS, seed=42, image=str(crop))
            mx.eval(video)
        finally:
            pipe._source_audio_tokens = None
        pipe.dit = None
        pipe.prompt_encoder.free()
        pipe.image_conditioner.free()
        pipe.upsampler = None
        blocks.aggressive_cleanup()
        progress('Decoding generated video')
        silent = out / 'generated-silent.mp4'
        # Video decode only. The generated/reconstructed audio path is never used.
        pipe.video_decoder_block.decode_and_stream(video, str(silent), frame_rate=FPS)
        partial = out / 'partial.mov'
        subprocess.run([ffmpeg, '-v', 'error', '-i', str(silent), '-i', str(args.audio),
                        '-map', '0:v:0', '-map', '1:a:0', '-t', str(args.frames/FPS),
                        '-c:v', 'copy', '-c:a', 'pcm_f32le', str(partial)], check=True)
        progress('Verifying frames and original decoded audio')
        subprocess.run([sys.executable, str(ROOT / 'scripts/verify_export.py'), str(args.audio), str(partial),
                        '--seconds', str(args.frames/FPS), '--expected-frames', str(args.frames),
                        '--fps', str(FPS)], check=True)
        output = out / 'review.mov'
        partial.replace(output)
        provenance.update(generated_video=True, output=str(output), output_sha256=digest(output),
                          original_decoded_audio_verified=True, elapsed_seconds=time.monotonic()-start,
                          max_observed_mlx_peak_bytes=observed_peak)
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        progress('Completed local comparison; visual review required')
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps(dict(generated_video=False, quality_passed=False,
            error=str(error), elapsed_seconds=time.monotonic()-start), indent=2))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('photo', 'audio', 'model', 'gemma', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda s: Path(s).resolve())
    parser.add_argument('--crop', required=True, nargs=4, type=int)
    parser.add_argument('--frames', type=int, choices=(49, 121), default=49)
    parser.add_argument('--consent', action='store_true')
    parser.add_argument('--prompt-file',type=Path,help='Optional local prompt for a controlled motion comparison')
    parser.add_argument('--wait-for-weights', action='store_true',
                        help='Wait up to 12 hours for the separate download to finish; hold no GPU lease while waiting')
    args = parser.parse_args()
    args.prompt_file_sha256 = None
    args.prompt_text = PROMPT
    if args.prompt_file:
        prompt_bytes = args.prompt_file.read_bytes()
        args.prompt_text = prompt_bytes.decode('utf-8').strip()
        args.prompt_file_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        if not 20 <= len(args.prompt_text) <= 3000:
            parser.error('Use a local prompt between 20 and 3000 characters')
    if not args.consent:
        parser.error('Likeness consent is required')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    if not args.photo.is_file() or not args.audio.is_file():
        parser.error('The source photo and voice-over must exist')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    if args.wait_for_weights:
        markers = [folder / 'download-complete.json' for folder in (args.model, args.gemma)]
        deadline = time.monotonic() + 12 * 3600
        if not all(marker.is_file() for marker in markers):
            print('Waiting for verified public weights; no inference is running yet.', flush=True)
        while not all(marker.is_file() for marker in markers):
            if time.monotonic() >= deadline:
                raise TimeoutError('Model download did not finish in 12 hours')
            time.sleep(20)
    with GPUInferenceLease(on_wait=lambda: print('Waiting for the active local render', flush=True)):
        run(args)


if __name__ == '__main__':
    main()
