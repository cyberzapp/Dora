"""Confine lip-driven deformation using two matched local Ditto renders."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from ltx_video_export import export_original_audio
from mouth_interior_review import digest


def midface_mask(width, height):
    y, x = np.ogrid[:height, :width]
    distance = np.maximum(abs((x / width - .4705) / .0235),
                          abs((y / height - .336) / .026))
    fade = np.clip((1.65 - distance) / .65, 0, 1)
    return (fade * fade * (3 - 2 * fade)).astype(np.float32)


def compare_tracks(first, second):
    for filename in ("ditto-base-motion.npz", "ditto-filtered-pose.npz"):
        with np.load(first / filename, allow_pickle=False) as a, np.load(second / filename, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError("Both renders must share the same original and filtered motion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talking", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_tracks(args.talking, args.control)
    a, b = [json.loads((p / "provenance.json").read_text()) for p in (args.talking, args.control)]
    if a["photo_sha256"] != b["photo_sha256"] or a["voice_sha256"] != b["voice_sha256"] or a["voice_sha256"] != digest(args.audio):
        raise ValueError("The paired renders must use the same original photo and voice-over")
    clips = [p / "ditto-blackt-trial.mov" for p in (args.talking, args.control)]
    captures = [cv2.VideoCapture(str(clip)) for clip in clips]
    try:
        properties = [tuple(round(c.get(key)) for key in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS))
            for c in captures]
        width, height, count, fps = properties[0]
        if not all(c.isOpened() for c in captures) or properties[0] != properties[1] or fps != 25 or count != a["video_frames"]:
            raise ValueError("Paired clips do not have the same framing and timeline")
        args.output.mkdir(parents=True, exist_ok=False)
        mask = midface_mask(width, height)
        active = mask > 0
        alpha = mask[active, None]
        silent = args.output / "silent.mov"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(silent)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
            try:
                for index in range(count):
                    (ok, speaking), (ok_control, resting) = [c.read() for c in captures]
                    if not ok or not ok_control:
                        raise ValueError(f"Missing paired frame {index}")
                    speaking[active] = np.rint((1 - alpha) * speaking[active] + alpha * resting[active]).astype(np.uint8)
                    process.stdin.write(speaking.tobytes())
            finally:
                process.stdin.close()
            if process.wait():
                raise RuntimeError("Midface test encoding failed")
    finally:
        for capture in captures:
            capture.release()
    export_original_audio(silent, args.audio, args.output / "ditto-blackt-trial.mov", frames=count, fps=fps)
    (args.output / "provenance.json").write_text(json.dumps({
        "talking_render_sha256": digest(clips[0]), "fixed_lip_control_sha256": digest(clips[1]),
        "photo_sha256": a["photo_sha256"], "voice_sha256": a["voice_sha256"],
        "mask_pixels": int(active.sum()), "quality_accepted": False,
        "method": "Fixed feathered midface region from the matched-pose render with held articulation",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
