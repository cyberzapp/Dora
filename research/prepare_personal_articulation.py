"""Prepare reference-conditioned lip motion locally; no personalized training."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_stage0_expression(path):
    """Read the pinned upstream literal without importing its CUDA pipeline."""
    if digest(path) != "aea95e0e3e2f58dc1c94d1b8186e445c159d9fbbada4cabe29ff533c3712a3b0":
        raise ValueError("Unexpected TalkLikeYou neutralization source")
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NEUTRAL_LIP_EXPRESSION"
            for target in node.targets
        ):
            return ast.literal_eval(node.value.args[0])
    raise ValueError("Pinned TalkLikeYou source has no stage-zero expression")


def blend_motion_windows(engine, features, initial, reference_motion, seed):
    import numpy as np
    import torch
    from talklikeyou.smoothing import kalman_smooth

    length = engine.config["sequence_length"]
    overlap = engine.config["overlap"]
    stride = length - overlap
    device = engine.device
    one_hot = torch.zeros(1, engine.config["person_num"], device=device)
    habit = engine._habit_embedding({"motion": reference_motion})
    generator = torch.Generator(device=device).manual_seed(seed)
    motion = np.empty((len(features), len(initial)), dtype=np.float32)
    covered = 0
    for start in range(0, len(features), stride):
        end = min(start + length, len(features))
        condition = initial if start == 0 else motion[start].copy()
        noise = torch.randn((1, end - start, len(initial)), device=device, generator=generator)
        prediction = engine.motion_generator.sample(
            torch.from_numpy(condition).unsqueeze(0).to(device),
            torch.from_numpy(features[start:end]).unsqueeze(0).to(device),
            one_hot, noise=noise, habit_emb=habit,
        ).squeeze(0).cpu().numpy()
        shared = max(0, covered - start)
        if shared:
            weight = np.linspace(0, 1, shared + 2, dtype=np.float32)[1:-1, None]
            motion[start:covered] = (1 - weight) * motion[start:covered] + weight * prediction[:shared]
        motion[covered:end] = prediction[shared:]
        covered = end
        if end == len(features):
            break
    return kalman_smooth(motion, device).reshape(-1, 6, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ditto-root", type=Path, required=True)
    parser.add_argument("--articulation-root", type=Path, required=True)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.84)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--blend-windows", action="store_true")
    parser.add_argument("--stage0-source", type=Path,
                        help="Use the pinned TalkLikeYou neutral expression to initialize lip motion")
    args = parser.parse_args()
    if not 0 < args.seconds <= 15:
        parser.error("--seconds must be greater than zero and at most 15")
    ditto, root = args.ditto_root.resolve(), args.articulation_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "provenance.json").exists():
        parser.error("Choose a fresh output directory to retain the previous trajectory")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["MPLCONFIGDIR"] = str(ditto / "mpl-cache")
    sys.path[:0] = [str(root / "source"), str(ditto / "python-deps"), str(ditto / "upstream")]
    import cv2
    import numpy as np
    import torch
    import soundfile as sf
    import imageio_ffmpeg
    from core.atomic_components.cfg import parse_cfg
    from core.atomic_components.source2info import Source2Info
    from core.atomic_components.loader import load_image
    from talklikeyou.motion_generation import MotionGenerationEngine

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    metadata = json.loads((root / "model-metadata.json").read_text())
    for name in ["checkpoints/audio_encoder.pth", "checkpoints/habit_encoder.pt", "checkpoints/motion_generator.pt"]:
        item = next(f for f in metadata["siblings"] if f["rfilename"] == name)
        if digest(root / name) != item["lfs"]["sha256"]:
            raise ValueError(f"Checkpoint mismatch: {name}")
    manifest = json.loads((root / "source-manifest.json").read_text())
    for item in manifest["files"]:
        if digest(root / "source" / item["path"]) != item["sha256"]:
            raise ValueError(f"Source mismatch: {item['path']}")
    torch.manual_seed(1024)
    cfg = parse_cfg(str(ditto / "output/mps-explicit-controls.pkl"), str(ditto / "checkpoints/ditto_pytorch"))
    extractor = Source2Info(**cfg[0])
    reference_file = out / "reference-100-frames.mkv"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(args.reference),
        "-an", "-vf", "fps=25,scale=1280:1280:force_original_aspect_ratio=decrease",
        "-frames:v", "100", "-c:v", "ffv1", str(reference_file)], check=True)
    capture = cv2.VideoCapture(str(reference_file))
    reference_motion = []
    last_landmarks = None
    try:
        for index in range(100):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError("Reference has fewer than 100 frames at 25 fps")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cropped, _, last_landmarks = extractor._crop(rgb, last_landmarks)
            motion = extractor._get_kp_info(extractor._img_crop_to_bchw256(cropped))
            reference_motion.append(motion)
            if (index + 1) % 25 == 0:
                print(f"Extracted {index + 1}/100 actual reference frames", flush=True)
    finally:
        capture.release()
    source = load_image(str(args.photo), max_dim=1920)
    cropped, _, _ = extractor._crop(source)
    source_motion = extractor._get_kp_info(extractor._img_crop_to_bchw256(cropped))
    lip_indices = [6, 12, 14, 17, 19, 20]
    if args.stage0_source is not None:
        neutral = np.asarray(read_stage0_expression(args.stage0_source), dtype=np.float32)
        if neutral.shape != (6, 3) or not np.isfinite(neutral).all():
            raise ValueError("Invalid TalkLikeYou neutral expression")
        expression = source_motion["exp"].copy()
        expression.reshape(21, 3)[lip_indices] = neutral
        source_motion["exp"] = expression
    # The audio model uses 16 mel frames (200 ms) of right context. Supply real
    # future speech from the longer recording; do not drop the final video frames.
    sample_rate = sf.info(args.voice).samplerate
    export_samples = round(args.seconds * sample_rate)
    frame_count = math.ceil(export_samples * 25 / sample_rate)
    analysis_seconds = frame_count / 25 + 0.2
    count = math.ceil(analysis_seconds * sample_rate)
    audio, _ = sf.read(args.voice, frames=count, dtype="float32", always_2d=True)
    if len(audio) != count:
        raise ValueError(f"This trial needs {analysis_seconds:.2f} seconds of real speech context")
    analysis_audio = out / "voice-with-real-right-context.wav"
    sf.write(analysis_audio, audio, sample_rate, subtype="FLOAT")
    engine = MotionGenerationEngine(root / "checkpoints/motion_generator.pt",
        root / "checkpoints/audio_encoder.pth", root / "checkpoints/habit_encoder.pt",
        device=torch.device("mps"), sampling_steps=1, guidance_scale=1.3)
    features = engine.extract_audio_features(analysis_audio)
    if len(features) < frame_count:
        raise ValueError(f"Insufficient real-context audio features: {len(features)}")
    if args.blend_windows:
        with torch.inference_mode():
            initial = source_motion["exp"].reshape(21, 3)[lip_indices].reshape(-1)
            lips = blend_motion_windows(engine, features[:frame_count], initial, reference_motion, seed=1024)
    else:
        template = {"n_frames": frame_count, "motion": [source_motion.copy() for _ in range(frame_count)]}
        generated = engine.generate(template_data=template, audio_features=features[:frame_count],
            reference_data={"motion": reference_motion}, seed=1024)
        lips = np.stack([m["exp"].reshape(21, 3)[lip_indices] for m in generated["motion"]])
    if lips.shape != (frame_count, 6, 3) or not np.isfinite(lips).all():
        raise ValueError("Invalid generated lip trajectory")
    track = out / "personal-lip-track.npy"
    np.save(track, lips.astype(np.float32))
    report = {"method": "TalkLikeYou pretrained reference-conditioned articulation; no personal training",
        "source_commit": manifest["commit"], "model_revision": metadata["sha"],
        "photo_sha256": digest(args.photo), "voice_sha256": digest(args.voice),
        "reference_sha256": digest(args.reference), "reference_frames": 100, "reference_fps": 25,
        "output_frames": frame_count, "export_samples": export_samples, "sample_rate": sample_rate,
        "audio_features": len(features), "real_right_context_seconds": count / sample_rate - export_samples / sample_rate,
        "track_sha256": digest(track), "device": "mps", "fallback": "0", "quality_accepted": False,
        "window_policy": "aligned start and blended overlap" if args.blend_windows else "upstream"}
    if args.stage0_source is not None:
        report["stage0_source_sha256"] = digest(args.stage0_source)
        report["stage0_scope"] = "Neutral lip initialization only; original appearance features"
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
