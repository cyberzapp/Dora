"""Compare visible mouth-interior geometry and brightness in local footage."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

# MediaPipe's two inner-lip contours in continuous polygon order.
INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
         324, 318, 402, 317, 14, 87, 178, 88, 95]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(image, landmarks):
    left, right = landmarks[[78, 308]]
    direction = right - left
    width = float(np.linalg.norm(direction))
    if width < 12:
        return None
    normal = np.array([-direction[1], direction[0]]) / width
    opening = float((landmarks[14] - landmarks[13]) @ normal / width)
    if opening < .06:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    polygon = np.rint(landmarks[INNER]).astype(np.int32)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    core = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
    if core.sum() < 20:
        return None
    skin = np.zeros_like(mask)
    for index in (205, 425):
        cv2.circle(skin, tuple(np.rint(landmarks[index]).astype(int)),
                   max(2, round(width * .07)), 1, -1)
    lightness, a, b = lab[core].astype(np.float32).T
    skin_pixels = lab[:, :, 0][skin.astype(bool)]
    if not len(skin_pixels):
        return None
    skin_lightness = np.median(skin_pixels)
    # A low-red, bright subset is a tooth candidate, not a semantic segmentation.
    neutral = (a < 150) & (b < 175) & (lightness > max(90, skin_lightness - 45))
    ratio = float(neutral.sum() / len(lightness))
    ys, xs = np.nonzero(core)
    center = (landmarks[13] + landmarks[14]) / 2
    lower = (np.column_stack((xs, ys)) - center) @ normal > 0
    return dict(opening_over_width=opening, bright_neutral_area_fraction=ratio,
                lower_bright_neutral_area_fraction=float((neutral & lower).sum() / len(lightness)),
                upper_bright_neutral_area_fraction=float((neutral & ~lower).sum() / len(lightness)),
                bright_neutral_lightness_above_skin=(float(np.median(lightness[neutral]) - skin_lightness)
                                                     if neutral.any() else None))


def inspect(video, seconds, temporary):
    normalized = temporary / "normalized.mkv"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-t", str(seconds),
                    "-an", "-vf", "fps=25,scale=1280:1280:force_original_aspect_ratio=decrease",
                    "-c:v", "ffv1", str(normalized)], check=True)
    cap = cv2.VideoCapture(str(normalized))
    frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or frames != round(seconds * 25):
        cap.release()
        raise ValueError(f"Expected {round(seconds * 25)} decoded frames from {video}, found {frames}")
    rows, crops = [], []
    try:
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as tracker:
            for i in range(frames):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Decode failed at frame {i}")
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = tracker.process(image)
                if not result.multi_face_landmarks:
                    raise ValueError(f"Face tracking failed at frame {i}")
                height, width = image.shape[:2]
                points = np.array([(p.x * width, p.y * height) for p in result.multi_face_landmarks[0].landmark])
                row = measure(image, points)
                if row is not None:
                    center = (points[78] + points[308]) / 2
                    span = np.linalg.norm(points[78] - points[308])
                    crop = cv2.getRectSubPix(image, (round(span * 1.8), round(span)), tuple(center))
                    crops.append(Image.fromarray(crop).resize((320, 180), Image.Resampling.LANCZOS))
                    rows.append(dict(frame=i, **row))
    finally:
        cap.release()
    return rows, crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.84)
    parser.add_argument("--reference-seconds", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (320 * 4, 204 * 4), "#181818")
    draw = ImageDraw.Draw(sheet)
    report = {}
    with tempfile.TemporaryDirectory(prefix="avatar-mouth-review-") as directory:
        for name, video, duration in (("camera", args.reference, args.reference_seconds),
                                      ("generated", args.candidate, args.seconds)):
            rows, crops = inspect(video, duration, Path(directory))
            if not rows:
                raise ValueError(f"No measurable open mouth in {video}")
            report[name] = {"path": str(video.resolve()), "sha256": digest(video),
                            "total_frames": round(duration * 25), "open_frames": rows}
            matches = [i for i, row in enumerate(rows) if .14 <= row["opening_over_width"] < .30]
            if len(matches) < 8:
                raise ValueError(f"Too few moderate openings in {video}: {len(matches)}")
            ranked = sorted(matches, key=lambda i: rows[i]["bright_neutral_area_fraction"])
            selected = np.rint(np.linspace(0, len(ranked)-1, 8)).astype(int)
            for column, position in enumerate(selected):
                index = ranked[position]
                y = (2 * (name == "generated") + column // 4) * 204
                x = column % 4 * 320
                draw.text((x + 6, y + 5), f"{name}  {rows[index]['frame']/25:.2f}s", fill="white")
                sheet.paste(crops[index], (x, y + 24))
    sheet.save(args.output / "matched-opening-crops.png")
    report["limitations"] = ("Descriptive comparison of different spoken performances. "
                             "Bright neutral pixels may include saliva, highlights or lip edges. "
                             "Separate capture and color pipeline may change relative luminance.")
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
