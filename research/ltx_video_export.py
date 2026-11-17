"""Publish a local generated-video export only after complete verification."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def export_original_audio(silent, audio, output, *, frames, fps):
    """Keep a failed partial for diagnosis; safely replace it on a later retry.

    Callers supply an actual generated silent movie. This helper verifies media
    integrity, not the origin of the frames or their perceptual quality.
    """
    import imageio_ffmpeg
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an already published video')
    if frames < 1 or fps <= 0:
        raise ValueError('Invalid expected output duration')
    partial = output.with_name(output.stem + '.partial' + output.suffix)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'error',
        '-i', str(silent), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
        '-t', str(frames/fps), '-c:v', 'copy', '-c:a', 'pcm_f32le', str(partial)], check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/verify_export.py'),
        str(audio), str(partial), '--seconds', str(frames/fps),
        '--expected-frames', str(frames), '--fps', str(fps)], check=True)
    partial.replace(output)
    return output
