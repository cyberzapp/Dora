"""Test the pinned LTX photo codec on the actual reference; not avatar generation.

Run with .venv-ltx and OS networking denied. The full video transformer is not
loaded. This isolates detail loss in the reference-image codec from motion errors.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'research')]
from download_model_manifest import digest, matches
from inference_lease import GPUInferenceLease


def main():
    model = ROOT / '.local/ltx-2.3-q8-weights'
    output = ROOT / '.local/ltx-real-photo-probe'
    photo = Path('/path/to/your/local/media/Portrait_BlackT.png')
    expected_photo = '001684f6f48d608dcb34903147cdcd983e4c1ae0ee99a699b702af8ab0e11499'
    if digest(photo) != expected_photo:
        raise ValueError('The consenting reference photo changed')
    if output.exists():
        raise ValueError('Retain the previous photo-codec probe')
    lock = json.loads((ROOT / 'research/ltx-source-lock.json').read_text())
    inventory = lock['models'][0]
    names = ('vae_encoder.safetensors', 'vae_decoder.safetensors')
    records = [next(item for item in inventory['files'] if item['path'] == name) for name in names]
    deadline = time.monotonic() + 8 * 3600
    print('Waiting for verified photo-codec weights; no inference yet', flush=True)
    while not all((model / name).is_file() for name in names):
        if time.monotonic() >= deadline:
            raise TimeoutError('Photo-codec weights did not complete in eight hours')
        time.sleep(20)
    for item in records:
        if not matches(model / item['path'], item['bytes'], item['sha256']):
            raise ValueError('Photo-codec integrity verification failed')
    with GPUInferenceLease(on_wait=lambda: print('Waiting for the active local renderer', flush=True)):
        import mlx.core as mx
        import numpy as np
        from PIL import Image, ImageOps
        from ltx_pipelines_mlx.utils import blocks
        output.mkdir()
        started = time.monotonic()
        try:
            mx.set_default_device(mx.gpu)
            mx.set_memory_limit(11 * 1024**3)
            mx.set_cache_limit(0)
            source = ImageOps.exif_transpose(Image.open(photo)).convert('RGB')
            source = source.crop((445, 92, 969, 616)).resize((512, 512), Image.Resampling.LANCZOS)
            source.save(output / 'reference-crop.png')
            pixels = np.asarray(source).astype(np.float32) / 127.5 - 1
            pixels = mx.array(pixels.transpose(2, 0, 1)[None, :, None])
            print('Encoding the actual reference photograph', flush=True)
            conditioner = blocks.ImageConditioner(model)
            encoder = conditioner.load()
            latent = encoder.encode(pixels)
            mx.eval(latent)
            if latent.shape != (1, 128, 1, 16, 16) or not bool(mx.all(mx.isfinite(latent)).item()):
                raise ValueError('Invalid encoded photograph')
            encode_peak = mx.get_peak_memory()
            del encoder, pixels
            conditioner.free()
            blocks.aggressive_cleanup()
            mx.save_safetensors(str(output / 'reference-latent.safetensors'), {'photo': latent})
            print('Reconstructing the photograph with the actual video decoder', flush=True)
            decoder = blocks.VideoDecoder(model)
            restored = decoder.decode_single_frame(latent)
            mx.eval(restored)
            if restored.shape != (1, 512, 512, 3) or not bool(mx.all(mx.isfinite(restored)).item()):
                raise ValueError('Invalid reconstructed photograph')
            restored = np.asarray(restored[0])
            image = Image.fromarray(np.rint(restored * 255).clip(0, 255).astype(np.uint8))
            image.save(output / 'codec-reconstruction.png')
            mse = float(np.mean((np.asarray(source).astype(np.float32) / 255 - restored)**2))
            report = dict(actual_photo_encoder=True, actual_video_decoder=True,
                generated_video=False, quality_passed=False, training_performed=False,
                source_sha256=expected_photo, source_crop=[445, 92, 969, 616],
                model_revision=inventory['revision'], verified_codec_files=records,
                latent_shape=list(latent.shape), reconstructed_shape=list(restored.shape),
                reference_reconstruction_psnr_db=float(-10 * np.log10(max(mse, 1e-12))),
                elapsed_seconds=time.monotonic()-started,
                max_observed_mlx_peak_bytes=max(encode_peak, mx.get_peak_memory()))
            (output / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        except Exception as error:
            (output / 'failure.json').write_text(json.dumps(dict(error=str(error),
                generated_video=False, quality_passed=False), indent=2))
            raise


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    main()
