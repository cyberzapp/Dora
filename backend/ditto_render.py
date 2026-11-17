"""Adapt the local Ditto trial runner to Avatar Studio's render job protocol."""
import json
import math
from pathlib import Path
import sys
import uuid

from worker import ROOT, ffmpeg, validate_request

TRIAL_ROOT = ROOT / ".local/ditto-experiment"
AUDIT_ROOT = ROOT / ".local/audio-motion-source-audit-2026-11-11"


def blink_centers(audio):
    """Choose isolated quiet moments on the model's 25 fps timeline."""
    import numpy as np
    import soundfile

    samples, rate = soundfile.read(audio, dtype="float32", always_2d=True)
    frame_count = len(samples) * 25 // rate
    if frame_count < 75:
        return []
    boundaries = np.arange(frame_count + 1, dtype=np.int64) * rate // 25
    energy = np.array([
        np.sqrt(np.mean(samples[start:end] ** 2))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ])
    speech_level = np.percentile(energy, 85)
    if not np.isfinite(speech_level) or speech_level < 0.002:
        return []
    quiet = energy < speech_level * 0.12
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    centers = []
    for start, end in zip(edges[::2], edges[1::2]):
        center = int((start + end - 1) // 2)
        if end - start < 5 or center < 35 or center > frame_count - 15:
            continue
        if centers and center - centers[-1] < 70:
            continue
        centers.append(center)
    return [center / 25 for center in centers]


def doctor():
    import torch

    report = {"metal_available": torch.backends.mps.is_available(), "missing": []}
    try:
        metadata = json.loads((AUDIT_ROOT / "digital-avatar--ditto-talkinghead-metadata.json").read_text())
        for item in metadata["siblings"]:
            name = item["rfilename"]
            if name.startswith("ditto_pytorch/"):
                path = TRIAL_ROOT / "checkpoints" / name
                if not path.is_file() or path.stat().st_size != item["size"]:
                    report["missing"].append(name)
        for path in (AUDIT_ROOT / "ditto-config.pkl", TRIAL_ROOT / "upstream/stream_pipeline_offline.py"):
            if not path.is_file():
                report["missing"].append(str(path))
    except (OSError, ValueError, KeyError) as error:
        report["error"] = str(error)
    print(json.dumps(report, indent=2))
    return int(not report["metal_available"] or bool(report["missing"]) or "error" in report)


def render_ditto(request, job):
    import soundfile

    validate_request(request)
    if request.get("source_video") or request.get("motion"):
        raise ValueError("Ditto evaluation uses the outfit photograph and voice-over.")
    audio = Path(request["audio"]).resolve()
    photo = Path(request["photo"]).resolve()
    info = soundfile.info(audio)
    seconds = info.frames / info.samplerate
    if not 0 < seconds <= 15:
        raise ValueError("Ditto evaluation supports clips up to 15 seconds.")
    frames = math.ceil(info.frames * 25 / info.samplerate)
    dimensions = (720, 1280) if request.get("portrait") else (1280, 720)
    attempt = job.directory / "ditto-attempts" / str(uuid.uuid4())
    attempt.mkdir(parents=True)
    progress_path = attempt / "progress.json"
    previous = None

    def publish_progress():
        nonlocal previous
        if not progress_path.is_file():
            return
        state = json.loads(progress_path.read_text())
        stage = state["stage"]
        written = state["written_frames"]
        if (stage, written) == previous:
            return
        previous = (stage, written)
        if stage == "writing_frames":
            job.status(.1 + .8 * min(written, frames) / frames,
                       f"Ditto rendered {written} of {frames} frames")
        elif stage == "muxing":
            job.status(.92, "Adding original voice-over")
        elif stage in ("loading_models", "analyzing_audio"):
            job.status(.05, "Preparing local Ditto models and speech")

    job.status(.01, "Checking Ditto model files and inputs")
    command = [sys.executable, ROOT / "scripts/run_ditto_quality_trial.py",
             "--trial-root", TRIAL_ROOT, "--audit-root", AUDIT_ROOT,
             "--photo", photo, "--audio", audio, "--seconds", repr(seconds),
             "--keep-render-features-on-metal", "--output-dir", attempt,
             "--progress-json", progress_path]
    for center in blink_centers(audio):
        command.extend(["--blink-at", str(center)])
    job.run(command, on_poll=publish_progress)
    clip = attempt / "ditto-blackt-trial.mov"
    width, height = dimensions
    fitted = attempt / "fitted.mov"
    job.status(.92, "Fitting the photo to the export canvas")
    job.run([ffmpeg(), "-v", "error", "-i", clip,
             "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
             "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy", fitted])
    job.status(.95, "Checking decoded frames and original audio")
    job.run([sys.executable, ROOT / "scripts/verify_export.py", audio, fitted,
             "--expected-frames", frames, "--fps", "25"])
    job.check()
    output = job.directory / "ditto.mov"
    fitted.replace(output)
    job.status(1, f"Completed • {frames} generated frames • original audio",
               "Completed", output)


if __name__ == "__main__":
    sys.exit(doctor())
