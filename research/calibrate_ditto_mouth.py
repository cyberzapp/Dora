"""Constrain Ditto lip shape using geometry measured in a real camera take."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

LIP_POINTS = (6, 12, 14, 17, 19, 20)


def phone_weights(events, frame_count):
    """Use nearby acoustic peaks as soft evidence, with closure taking priority."""
    times = np.arange(frame_count) / 25
    unrounded = np.zeros(frame_count)
    protect = np.zeros(frame_count)
    for event in events:
        phone = event["phone"]
        confidence = event["peak_probability"]
        if confidence < .35:
            continue
        offset = times - event["peak_seconds"]
        if phone.rstrip("ː") in {"e", "ɛ", "i", "ɪ", "a", "ɑ", "æ", "ə", "ʌ"}:
            unrounded = np.maximum(unrounded, np.clip(1 - abs(offset) / .10, 0, 1))
        elif phone.rstrip("ː") in {"o", "ɔ", "u", "ʊ", "y", "ø", "m", "b", "p", "f", "v", "ʋ", "w", "bʰ", "pʰ"}:
            lead = .20 if phone.rstrip("ː") in {"o", "ɔ", "u", "ʊ", "y", "ø", "w"} else .12
            radius = np.where(offset < 0, lead, .12)
            protect = np.maximum(protect, np.clip(2 - 2 * abs(offset) / radius, 0, 1))
    return unrounded * (1 - protect)


def adjust(track, expressions, measurements, phone_gate=None):
    """Retain the audio trajectory's projection on the estimated aperture axis."""
    if (track.ndim != 3 or track.shape[1:] != (6, 3) or not len(track)
        or expressions.shape != (len(measurements), 18)
        or measurements.ndim != 2 or measurements.shape[1] != 3
        or not all(np.isfinite(values).all() for values in (track, expressions, measurements))):
        raise ValueError("Invalid mouth trajectory or camera geometry")
    center = np.median(expressions, axis=0)
    residual = expressions - center
    geometry = measurements - np.mean(measurements, axis=0)
    scale = np.std(geometry, axis=0)
    if np.any(scale < 1e-5):
        raise ValueError("Camera excerpt lacks sufficient variation in mouth shape")
    geometry /= scale
    coefficients, _, rank, _ = np.linalg.lstsq(geometry, residual, rcond=None)
    aperture_norm = np.linalg.norm(coefficients[0])
    if rank != 3 or aperture_norm < 1e-5:
        raise ValueError("Camera excerpt does not identify an aperture direction")
    aperture = coefficients[0] / aperture_norm
    shape = coefficients[1:] - np.outer(coefficients[1:] @ aperture, aperture)
    _, singular, basis = np.linalg.svd(shape, full_matrices=False)
    if singular[-1] < 1e-5:
        raise ValueError("Camera excerpt does not identify independent lip shape directions")
    generated = track.reshape(-1, 18) - center
    camera_aperture = residual @ aperture
    # Estimate typical camera shape at each degree of opening. Smooth weights
    # avoid switching abruptly between reference frames during a syllable.
    bandwidth = np.std(camera_aperture) / 2
    if bandwidth < 1e-5:
        raise ValueError("Camera aperture does not vary enough for calibration")
    targets = np.clip(generated @ aperture, *np.percentile(camera_aperture, [5, 95]))
    weights = np.exp(-0.5 * ((targets[:, None] - camera_aperture) / bandwidth) ** 2)
    expected = weights @ (residual @ basis.T) / weights.sum(axis=1, keepdims=True)
    offsets = 0.65 * (expected - generated @ basis.T) @ basis
    if phone_gate is not None:
        # Correct only unusually narrow corners. Separate width from aperture
        # and leave the camera model's upper-lip direction alone.
        width_axis = shape[0] / np.linalg.norm(shape[0])
        reference_width = residual @ width_axis
        typical_width = weights @ reference_width / weights.sum(axis=1)
        deficit = np.clip(typical_width - generated @ width_axis, 0, np.std(reference_width))
        offsets = .65 * (phone_gate * deficit)[:, None] * width_axis
    if not np.isfinite(offsets).all() or np.max(np.abs(offsets @ aperture)) > 1e-6:
        raise ValueError("Invalid correction or aperture projection changed")
    return (track.reshape(-1, 18) + offsets).reshape(track.shape).astype(np.float32)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ditto-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phone-evidence", type=Path)
    parser.add_argument("--evidence-audio", type=Path)
    parser.add_argument("--driving-audio", type=Path)
    args = parser.parse_args()
    root = args.ditto_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "personal-lip-track.npy"
    if output.exists():
        parser.error("Choose a fresh output directory")
    provenance = json.loads(args.track.with_name("provenance.json").read_text())
    if provenance["track_sha256"] != digest(args.track):
        raise ValueError("Unexpected articulation track digest")
    if any((args.phone_evidence, args.evidence_audio, args.driving_audio)) and not all((args.phone_evidence, args.evidence_audio, args.driving_audio)):
        parser.error("Supply --phone-evidence, --evidence-audio, and --driving-audio together")
    evidence = None
    if args.phone_evidence:
        evidence = json.loads(args.phone_evidence.read_text())
        if evidence["audio_sha256"] != digest(args.evidence_audio):
            raise ValueError("Phoneme analysis uses different audio")
        import soundfile as sf
        source, rate = sf.read(args.driving_audio, dtype="float32", always_2d=True)
        excerpt, excerpt_rate = sf.read(args.evidence_audio, dtype="float32", always_2d=True)
        if (digest(args.driving_audio) != provenance["voice_sha256"]
            or rate != excerpt_rate or len(excerpt) != provenance["export_samples"]
            or not np.array_equal(source[:len(excerpt)], excerpt)):
            raise ValueError("Phoneme audio and driving voice samples differ")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["MPLCONFIGDIR"] = str(root / "mpl-cache")
    sys.path[:0] = [str(root / "python-deps"), str(root / "upstream")]
    import cv2
    import torch
    from core.atomic_components.cfg import parse_cfg
    from core.atomic_components.source2info import Source2Info

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    cfg = parse_cfg(str(root / "output/mps-explicit-controls.pkl"), str(root / "checkpoints/ditto_pytorch"))
    extractor = Source2Info(**cfg[0])
    capture = cv2.VideoCapture(str(args.reference))
    if not np.isclose(capture.get(cv2.CAP_PROP_FPS), 25):
        capture.release()
        raise ValueError("Use the normalized 25 fps camera excerpt")
    expressions, measurements = [], []
    landmarks = None
    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                crop, _, landmarks = extractor._crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks)
                motion = extractor._get_kp_info(extractor._img_crop_to_bchw256(crop))
                expressions.append(motion["exp"].reshape(21, 3)[list(LIP_POINTS)].reshape(-1))
                # MediaPipe lip landmarks in the roll-normalized Ditto crop.
                mesh = extractor.landmark478(crop)
                if mesh is None or np.size(mesh) != 478 * 3:
                    raise ValueError(f"Camera face tracking failed at frame {len(expressions) - 1}")
                mesh = mesh.reshape(478, 3)
                width = np.linalg.norm(mesh[291, :2] - mesh[61, :2])
                face_width = np.linalg.norm(mesh[454, :2] - mesh[234, :2])
                if min(width, face_width) < 1e-5:
                    raise ValueError("Invalid camera face geometry")
                corner_y = (mesh[61, 1] + mesh[291, 1]) / 2
                measurements.append([(mesh[14, 1] - mesh[13, 1]) / face_width,
                                     width / face_width, (corner_y - mesh[13, 1]) / face_width])
    finally:
        capture.release()
    if len(expressions) < 75:
        raise ValueError("Camera excerpt is too short for calibration")
    track = np.load(args.track, allow_pickle=False)
    weights = phone_weights(evidence["events"], len(track)) if evidence else None
    adjusted = adjust(track, np.asarray(expressions), np.asarray(measurements), weights)
    np.save(output, adjusted)
    provenance["camera_shape_calibration"] = {
        "reference_excerpt_sha256": digest(args.reference), "frames": len(expressions),
        "parent_track_sha256": provenance["track_sha256"], "blend": 0.65,
        "method": "Width and upper-lip regression projected orthogonal to estimated aperture",
        "correction_rms": float(np.sqrt(np.mean((adjusted - track) ** 2))),
    }
    provenance["track_sha256"] = digest(output)
    if evidence:
        provenance["camera_shape_calibration"].update(
            method="Phone-weighted correction of narrow mouth corners at matched aperture",
            phone_evidence_sha256=digest(args.phone_evidence),
            changed_frames=int(np.any(adjusted != track, axis=(1, 2)).sum()),
        )
    provenance["quality_accepted"] = False
    output.with_name("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["camera_shape_calibration"], indent=2))


if __name__ == "__main__":
    main()
