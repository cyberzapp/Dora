"""Fit a bounded pose-conditioned geometry intervention from local render probes."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_ditto_mouth import digest, phone_weights

LIPS = [6, 12, 14, 17, 19, 20]
MEASURES = ("gap", "width", "depth_proxy", "chin_y")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("track", "response", "phones", "audio", "driving_audio", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    import soundfile as sf

    record = json.loads(args.track.with_name("provenance.json").read_text())
    phones = json.loads(args.phones.read_text())
    if (digest(args.track) != record["track_sha256"]
        or digest(args.audio) != phones["audio_sha256"]
        or digest(args.driving_audio) != record["voice_sha256"]):
        raise ValueError("Track or audio differs from its recorded input")
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    driving, driving_rate = sf.read(args.driving_audio, frames=len(audio), dtype="float32", always_2d=True)
    if rate != driving_rate or not np.array_equal(driving, audio):
        raise ValueError("Acoustic cues and driving audio differ")
    track = np.load(args.track, allow_pickle=False)
    probes = json.loads(args.response.read_text())
    centers, directions = [], []
    target = [0, .026, .010, 0]
    for index, rows in probes.items():
        derivative = np.array([
            [(rows[f"{point}-{axis}-+1"][measure] - rows[f"{point}-{axis}--1"][measure]) / .008
             for point in LIPS for axis in range(3)]
            for measure in MEASURES
        ])
        # Give aperture preservation more weight than the two correction targets.
        weight = np.array([3, 1, 1, 2])
        direction = np.linalg.lstsq(derivative * weight[:, None], np.array(target)*weight, rcond=.05)[0]
        direction *= min(1, .004 / max(np.max(np.abs(direction)), 1e-8))
        centers.append(track[int(index)])
        directions.append(direction)
    centers = np.stack(centers)
    scale = np.linalg.norm(centers[0] - centers[1])
    if scale <= 0 or not np.isfinite(directions).all():
        raise ValueError("Probe poses must be distinct and finite")
    distances = np.linalg.norm((track[:, None]-centers).reshape(len(track), len(centers), -1), axis=-1)
    proximity = np.exp(-.5 * (distances / scale)**2)
    local = proximity @ np.stack(directions) / np.maximum(proximity.sum(axis=1, keepdims=True), 1e-12)
    gate = phone_weights(phones["events"], len(track)) * proximity.max(axis=1)
    corrected = track + (gate[:, None] * local).reshape(track.shape).astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "personal-lip-track.npy"
    np.save(output, corrected)
    record["geometry_probe"] = {"response_sha256": digest(args.response), "target": dict(zip(MEASURES, target)),
                                "audio_evidence_sha256": digest(args.phones),
                                "parent_track_sha256": record["track_sha256"],
                                "corrected_frames": int(np.any(corrected != track, axis=(1, 2)).sum())}
    record["track_sha256"] = digest(output)
    output.with_name("provenance.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
