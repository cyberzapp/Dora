"""Test a camera-derived lower mouth interior on a locally generated avatar."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

from ltx_video_export import export_original_audio
from mouth_interior_review import INNER, digest


def read_frames(path, count):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 25) > .01:
            raise ValueError(f"Expected a 25 fps clip: {path}")
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as tracker:
            for index in range(count):
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"Decode stopped at {index}: {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = tracker.process(rgb)
                if not result.multi_face_landmarks:
                    raise ValueError(f"Face track missing at {index}: {path}")
                height, width = rgb.shape[:2]
                points = np.array([(p.x * width, p.y * height)
                                   for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)
                yield rgb, points
    finally:
        capture.release()


def mouth_geometry(points):
    left, right, top, bottom = points[[78, 308, 13, 14]]
    width = np.linalg.norm(right - left)
    if width < 12:
        return None
    normal = np.array([left[1] - right[1], right[0] - left[0]]) / width
    opening = (bottom - top) @ normal
    if opening < 3:
        return None
    center = (top + bottom) / 2
    transform_points = np.float32([center, center + (right-left)/2, center + normal*opening/2])
    return transform_points, float(opening / width)


def mask_interior(points, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points[INNER]).astype(np.int32)], 1)
    return cv2.distanceTransform(mask, cv2.DIST_L2, 5)


def select_sequence(target_openings, donors):
    """Prefer physically adjacent donor frames within each target speech span."""
    indices = np.asarray([d["index"] for d in donors])
    ratio = np.asarray([d["ratio"] for d in donors])
    step = indices[None, :] - indices[:, None]
    transition = np.where((step >= 0) & (step <= 3), .06 * np.abs(step - 1), np.inf)
    feature_cost = 5 * np.abs(np.asarray(target_openings)[:, None] - ratio)
    costs = feature_cost[0]
    history = []
    for observation in feature_cost[1:]:
        alternatives = costs[:, None] + transition
        predecessor = alternatives.argmin(axis=0)
        costs = observation + alternatives[predecessor, np.arange(len(donors))]
        history.append(predecessor)
    path = [int(costs.argmin())]
    for predecessor in reversed(history):
        path.append(int(predecessor[path[-1]]))
    return [donors[index] for index in reversed(path)]


def prepare_donors(path, count):
    donors = []
    for index, (rgb, points) in enumerate(read_frames(path, count)):
        geometry = mouth_geometry(points)
        if geometry is None:
            continue
        controls, ratio = geometry
        if not .12 <= ratio <= .65:
            continue
        x0, y0 = np.floor(points[INNER].min(axis=0) - 3).astype(int)
        x1, y1 = np.ceil(points[INNER].max(axis=0) + 4).astype(int)
        if x0 < 0 or y0 < 0 or x1 > rgb.shape[1] or y1 > rgb.shape[0]:
            continue
        offset = np.float32([x0, y0])
        pixels = rgb[y0:y1, x0:x1].copy()
        distance = mask_interior(points - offset, pixels.shape[:2])
        interior = pixels[distance >= 2.5]
        if len(interior) < 20:
            continue
        donors.append(dict(index=index, ratio=ratio, controls=controls-offset, pixels=pixels,
                           distance=distance, interior=np.median(interior, axis=0)))
    if len(donors) < 20:
        raise ValueError("Too few camera mouth samples")
    return donors


def composite(rgb, points, donor, strength):
    controls, ratio = mouth_geometry(points)
    if abs(donor["ratio"] - ratio) > .12:
        return rgb, 0
    x0, y0 = np.floor(points[INNER].min(axis=0) - 3).astype(int)
    x1, y1 = np.ceil(points[INNER].max(axis=0) + 4).astype(int)
    if x0 < 0 or y0 < 0 or x1 > rgb.shape[1] or y1 > rgb.shape[0]:
        return rgb, 0
    region = rgb[y0:y1, x0:x1]
    offset = np.float32([x0, y0])
    transform = cv2.getAffineTransform(donor["controls"], controls-offset)
    size = (region.shape[1], region.shape[0])
    warped = cv2.warpAffine(donor["pixels"], transform, size)
    donor_distance = cv2.warpAffine(donor["distance"], transform, size)
    distance = mask_interior(points-offset, region.shape[:2])
    opening = np.linalg.norm(controls[2] - controls[0])
    normal = (controls[2] - controls[0]) / opening
    ys, xs = np.indices(region.shape[:2])
    lower = (xs + x0 - controls[0,0]) * normal[0] + (ys + y0 - controls[0,1]) * normal[1]
    alpha = strength * np.clip((distance-1)/3, 0, 1) * np.clip((donor_distance-1)/2, 0, 1)
    alpha *= np.clip(lower / max(opening * .35, 1), 0, 1)
    active = alpha > 0
    if not active.any():
        return rgb, 0
    # Match only a small median RGB offset. Large offsets would remove the
    # reference's real local contrast and can produce flashing colors.
    target = region[distance >= 2.5]
    if not len(target):
        return rgb, 0
    correction = np.clip(np.median(target, axis=0) - donor["interior"], -18, 18)
    mixed = rgb.copy()
    dest = mixed[y0:y1, x0:x1]
    a = alpha[active, None]
    dest[active] = np.rint(np.clip(region[active]*(1-a)+(warped[active]+correction)*a, 0, 255)).astype(np.uint8)
    if np.any(dest[~active] != region[~active]):
        raise RuntimeError("Composite exceeded the eroded mouth-interior mask")
    return mixed, int(active.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurements = json.loads(args.measurements.read_text())
    if (digest(args.candidate) != measurements["generated"]["sha256"]
        or digest(args.reference) != measurements["camera"]["sha256"]):
        raise ValueError("Candidate or camera recording differs from the measurements")
    count = measurements["generated"]["total_frames"]
    strength = np.zeros(count, dtype=np.float32)
    for row in measurements["generated"]["open_frames"]:
        if row["opening_over_width"] >= .14:
            strength[row["frame"]] = np.clip((row["lower_bright_neutral_area_fraction"]-.11)/.12, 0, 1)
    if count < 5:
        raise ValueError("At least five frames are required for a transition")
    strength = .85 * np.convolve(strength, [.1, .2, .4, .2, .1], mode="same")
    args.output.mkdir(parents=True, exist_ok=False)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    reference = args.output / "camera-25fps.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-i", str(args.reference), "-t", "15", "-an",
                    "-vf", "fps=25,scale=1280:1280:force_original_aspect_ratio=decrease",
                    "-c:v", "ffv1", str(reference)], check=True)
    donors = prepare_donors(reference, 375)
    # The target is short, so keep only mouth landmarks and reopen for streaming
    # output instead of retaining all full-resolution frames in memory.
    points = [landmarks for _, landmarks in read_frames(args.candidate, count)]
    geometry = [mouth_geometry(landmarks) for landmarks in points]
    active = np.asarray([g is not None for g in geometry]) & (strength > .03)
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    selected = {}
    for first, end in zip(edges[::2], edges[1::2]):
        matches = select_sequence([g[1] for g in geometry[first:end]], donors)
        selected.update(zip(range(first, end), matches))
    if not selected:
        raise ValueError("No target frames need a mouth-interior test")
    source = cv2.VideoCapture(str(args.candidate))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent = args.output / "silent.mov"
    command = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(silent)]
    audit = []
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for index, landmarks in enumerate(points):
                    ok, bgr = source.read()
                    if not ok:
                        raise ValueError(f"Missing rendered frame {index}")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    donor = selected.get(index)
                    if donor is not None:
                        rgb, changed = composite(rgb, landmarks, donor, strength[index])
                        if changed:
                            audit.append(dict(frame=index, reference_frame=donor["index"], mask_pixels=changed))
                    encoder.stdin.write(rgb.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("Mouth-interior encoding failed")
    finally:
        source.release()
    if not audit:
        raise ValueError("No donor regions passed the composite constraints")
    export_original_audio(silent, args.audio, args.output / "ditto-blackt-trial.mov", frames=count, fps=25)
    (args.output / "provenance.json").write_text(json.dumps({
        "candidate_sha256": digest(args.candidate), "reference_sha256": digest(args.reference),
        "measurements_sha256": digest(args.measurements), "audio_sha256": digest(args.audio),
        "method": "Real-camera lower mouth pixels aligned to the generated inner-lip contour",
        "max_donor_step_in_continuous_span": 3,
        "frames": audit, "quality_accepted": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
