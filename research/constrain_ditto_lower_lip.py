"""Retest an actual Ditto clip with a bounded lower-lip-only correction."""
import argparse
import json
from pathlib import Path

import numpy as np

from mouth_interior_review import digest

OPENING_BINS = (.14, .22, .30, float("inf"))
MAX_SHIFT = .009
# This is the lower-lip opening control used by the pinned upstream Stage0 code.
# Ditto's personal articulation order is [6, 12, 14, 17, 19, 20].
LOWER_LIP_INDEX = 4


def correction(rows, reference, count):
    excess = np.zeros(count, dtype=np.float32)
    key = "lower_bright_neutral_area_fraction"
    for low, high in zip(OPENING_BINS, OPENING_BINS[1:]):
        camera = [r[key] for r in reference if low <= r["opening_over_width"] < high]
        if len(camera) < 20:
            raise ValueError("Too few real-camera samples in an opening range")
        ceiling = np.percentile(camera, 90)
        for row in rows:
            if low <= row["opening_over_width"] < high:
                excess[row["frame"]] = max(0, row[key] - ceiling)
    # Four frames of support on either side at 25 fps. Keep the correction
    # bounded and avoid single-frame changes from the brightness proxy.
    window = np.arange(-4, 5, dtype=np.float32)
    weights = np.exp(-0.5 * (window / 1.6) ** 2)
    smoothed = np.convolve(excess, weights / weights.sum(), mode="same")
    return MAX_SHIFT * np.minimum(smoothed / .14, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trial = json.loads((args.baseline_dir / "provenance.json").read_text())
    measurements = json.loads(args.measurements.read_text())
    provenance = json.loads(args.track.with_name("provenance.json").read_text())
    track_digest = digest(args.track)
    video_digest = digest(args.baseline_dir / "ditto-blackt-trial.mov")
    if (provenance["track_sha256"] != track_digest or trial["articulation"]["track_sha256"] != track_digest
        or measurements["generated"]["sha256"] != video_digest):
        raise ValueError("Measurements, rendered clip and trajectory must describe the same trial")
    track = np.load(args.track, allow_pickle=False)
    count = measurements["generated"]["total_frames"]
    if count < 9 or track.shape != (count, 6, 3) or not np.isfinite(track).all():
        raise ValueError("Invalid trajectory")
    shift = correction(measurements["generated"]["open_frames"], measurements["camera"]["open_frames"], count)
    if not np.any(shift):
        raise ValueError("No excess lower-region brightness was detected")
    adjusted = track.copy()
    adjusted[:, LOWER_LIP_INDEX, 1] -= shift
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "personal-lip-track.npy"
    np.save(output, adjusted)
    provenance["lower_lip_correction"] = {
        "measurements_sha256": digest(args.measurements), "baseline_video_sha256": video_digest,
        "parent_track_sha256": track_digest, "max_shift": MAX_SHIFT,
        "actual_max_shift": float(np.max(shift)), "affected_frames": int(np.count_nonzero(shift)),
    }
    provenance["track_sha256"] = digest(output)
    provenance["quality_accepted"] = False
    output.with_name("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["lower_lip_correction"], indent=2))


if __name__ == "__main__":
    main()
