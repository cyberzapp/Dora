"""Apply a renderer-probed lip depth direction near unrounded acoustic cues."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_ditto_mouth import digest, phone_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("track", "directions", "phones", "audio", "driving_audio", "output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    import soundfile as sf

    provenance = json.loads(args.track.with_name("provenance.json").read_text())
    phones = json.loads(args.phones.read_text())
    if (digest(args.track) != provenance["track_sha256"]
        or digest(args.audio) != phones["audio_sha256"]
        or digest(args.driving_audio) != provenance["voice_sha256"]):
        raise ValueError("Mismatched track or audio")
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    driving, source_rate = sf.read(args.driving_audio, frames=len(audio), dtype="float32", always_2d=True)
    if rate != source_rate or not np.array_equal(audio, driving):
        raise ValueError("Acoustic evidence and driving audio differ")
    source = np.load(args.track, allow_pickle=False)
    with np.load(args.directions, allow_pickle=False) as probes:
        direction = probes["depth"]
    if direction.shape != (6, 3) or not np.isfinite(direction).all() or np.max(np.abs(direction)) > .004001:
        raise ValueError("Invalid calibrated direction")
    gate = phone_weights(phones["events"], len(source)).astype(np.float32)
    candidate = source + gate[:, None, None] * direction
    args.output.mkdir(parents=True, exist_ok=False)
    track = args.output / "personal-lip-track.npy"
    np.save(track, candidate)
    provenance["depth_calibration"] = {
        "parent_track_sha256": provenance["track_sha256"],
        "directions_sha256": digest(args.directions),
        "phones_sha256": digest(args.phones),
        "frames_changed": int(np.any(candidate != source, axis=(1, 2)).sum()),
    }
    provenance["track_sha256"] = digest(track)
    track.with_name("provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    main()
