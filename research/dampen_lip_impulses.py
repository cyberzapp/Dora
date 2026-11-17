"""Make an experimental TalkLikeYou track with bounded impulse correction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def dampen(track):
    if track.ndim != 3 or track.shape[1:] != (6, 3) or len(track) < 3:
        raise ValueError("Expected at least three frames of six 3D mouth points")
    if track.dtype != np.float32 or not np.isfinite(track).all():
        raise ValueError("Expected finite float32 mouth points")
    result = track.copy()
    residual = track[1:-1] - (track[:-2] + track[2:]) / 2
    # Preserve vertical articulation and only damp isolated horizontal/depth
    # deviations from the adjacent frames.
    residual[:, :, 1] = 0
    magnitude = np.linalg.norm(residual.reshape(len(residual), -1), axis=1)
    threshold = float(np.percentile(magnitude, 80))
    gain = .65 * np.clip(1 - threshold / np.maximum(magnitude, 1e-8), 0, 1)
    result[1:-1] -= gain[:, None, None] * residual
    return result, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    provenance = json.loads(args.track.with_name("provenance.json").read_text())
    parent_digest = hashlib.sha256(args.track.read_bytes()).hexdigest()
    if provenance["track_sha256"] != parent_digest:
        raise ValueError("Input does not match its provenance")
    original = np.load(args.track, allow_pickle=False)
    track, threshold = dampen(original)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "personal-lip-track.npy"
    np.save(output, track)
    provenance["track_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance["impulse_correction"] = {
        "parent_track_sha256": parent_digest,
        "threshold": threshold,
        "changed_frames": int(np.any(track != original, axis=(1, 2)).sum()),
    }
    provenance["quality_accepted"] = False
    output.with_name("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(provenance["impulse_correction"])


if __name__ == "__main__":
    main()
