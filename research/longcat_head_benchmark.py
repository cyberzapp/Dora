"""Controlled head-crop evaluation using real LongCat generation and source pixels.

This is a research comparison, not a quality-approved app renderer. Cropping is
ordinary image processing; no replacement portrait or body is synthesized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from head_composite import head_mask, composite_head, crop_window, crop_box


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--photo', type=Path, required=True)
    parser.add_argument('--face-metadata', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=29)
    parser.add_argument('--crop-factor', type=float, default=2.7)
    parser.add_argument('--consent', action='store_true')
    args = parser.parse_args()
    if not args.consent: raise ValueError('Explicit likeness consent required')
    if args.output.exists(): raise ValueError('Use a new evaluation directory')
    args.output.mkdir(parents=True)
    photo_hash, audio_hash = digest(args.photo), digest(args.audio)
    photo = ImageOps.exif_transpose(Image.open(args.photo)).convert('RGB')
    bounds = json.loads(args.face_metadata.read_text())['face']
    box = crop_box(photo, bounds, args.crop_factor)
    source = args.output / 'source-crop.png'
    photo.crop(box).resize((512, 512), Image.Resampling.LANCZOS).save(source)
    preparation = dict(source_photo_sha256=photo_hash, source_audio_sha256=audio_hash,
                       crop_box=box, crop_factor=args.crop_factor, crop_edge_feather=True, model_size=[512, 512], generated_video=False,
                       quality_passed=False, purpose='Actual head-crop generation comparison')
    (args.output / 'preparation.json').write_text(json.dumps(preparation, indent=2))
    candidate = args.output / 'generation'
    subprocess.run([sys.executable, ROOT / 'research/longcat_benchmark.py', '--image', source,
                    '--audio', args.audio, '--output', candidate, '--frames', str(args.frames),
                    '--width', '512', '--height', '512', '--prompt',
                    'The reference person speaks naturally toward the camera with restrained head movement and natural blinking. Preserve the exact face, hairstyle, black T-shirt, lighting and background. Locked camera.'], check=True)
    report = json.loads((candidate / 'report.json').read_text())
    if report.get('generated_video') is not True: raise RuntimeError('No actual generated candidate')
    if digest(args.photo) != photo_hash or digest(args.audio) != audio_hash:
        raise ValueError('Source media changed during generation')
    reader = imageio_ffmpeg.read_frames(report['output'], pix_fmt='rgb24')
    metadata = next(reader); model_width, model_height = metadata['size']; fps = metadata['fps']
    pixels = np.asarray(photo)
    mask = head_mask(photo, photo.size, bounds)
    left, top, right, bottom = box
    # Only the generated crop can contribute; the rest remains the uploaded photo.
    mask *= crop_window(photo.size, box)
    width, height = photo.size; padded_width = width + width % 2; padded_height = height + height % 2
    raw = args.output / 'reprojected.rgb'; count = 0
    try:
        with raw.open('wb') as stream:
            for data in reader:
                frame = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(model_height, model_width, 3))
                projected = pixels.copy()
                projected[top:bottom, left:right] = np.asarray(frame.resize((right-left, bottom-top), Image.Resampling.LANCZOS))
                result = composite_head(pixels, projected, mask)
                if not np.array_equal(result[mask == 0], pixels[mask == 0]):
                    raise RuntimeError('Source appearance changed outside the animated head')
                padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
                padded[:height, :width] = result
                stream.write(padded.tobytes()); count += 1
    finally: reader.close()
    if count != args.frames: raise ValueError('Candidate frame count mismatch')
    output = args.output / 'head-comparison.mov'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24', '-s', f'{padded_width}x{padded_height}', '-r', str(fps),
                    '-i', raw, '-i', args.audio, '-map', '0:v', '-map', '1:a', '-t', str(count/fps),
                    '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_f32le', output], check=True)
    subprocess.run([sys.executable, ROOT / 'scripts/verify_export.py', args.audio, output,
                    '--seconds', str(count/fps), '--expected-frames', str(count), '--fps', str(fps)], check=True)
    raw.unlink()
    preparation.update(generated_video=True, output=str(output.resolve()), frames=count,
                       source_pixels_preserved_outside_head_before_encoding=True,
                       visual_review_required=True, candidate_report=report)
    (args.output / 'report.json').write_text(json.dumps(preparation, indent=2))
    print(json.dumps(preparation, indent=2))


if __name__ == '__main__': main()
