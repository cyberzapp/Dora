"""Measure adjacent-frame optical flow in stable portrait and background regions."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

REGIONS = {
    "background": ((.05, .12, .22, .44), (.82, .31, .97, .65)),
    "hair": ((.34, .14, .51, .23), (.50, .17, .60, .27)),
    "nose_bridge": ((.447, .31, .494, .362),),
    "below_nose": ((.447, .365, .494, .410),),
    "cheeks": ((.377, .36, .426, .43), (.51, .36, .556, .44)),
}


def region_mask(shape, boxes):
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    for left, top, right, bottom in boxes:
        mask[round(top*height):round(bottom*height), round(left*width):round(right*width)] = 255
    return mask


def adjacent_flow(before, after, mask):
    points = cv2.goodFeaturesToTrack(before, maxCorners=80, qualityLevel=.012, minDistance=5, mask=mask)
    if points is None or len(points) < 5:
        return None
    forward, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, winSize=(21, 21), maxLevel=2)
    if forward is None or status is None:
        return None
    reverse, returned, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, winSize=(21, 21), maxLevel=2)
    if reverse is None or returned is None:
        return None
    eligible = ((status[:, 0] == 1) & (returned[:, 0] == 1)
                & (np.linalg.norm((reverse - points)[:, 0, :], axis=1) < .3))
    displacement = (forward - points)[eligible, 0]
    if len(displacement) < 5:
        return None
    translation = np.median(displacement, axis=0)
    return dict(features=len(displacement), dx=float(translation[0]), dy=float(translation[1]),
                spread=float(np.median(np.linalg.norm(displacement - translation, axis=1))))


def inspect(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS)-25) > .01:
            raise ValueError(f"Expected a readable 25 fps clip: {path}")
        expected = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected < 2:
            raise ValueError(f"At least two frames are required: {path}")
        masks, previous, rows = None, None, []
        for index in range(expected):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Frame {index} could not be decoded")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is None:
                masks = {name: region_mask(gray.shape, boxes) for name, boxes in REGIONS.items()}
            else:
                flow = {name: adjacent_flow(previous, gray, mask) for name, mask in masks.items()}
                rows.append(dict(frame=index, seconds=index/25, flow=flow))
            previous = gray
    finally:
        capture.release()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(video=str(path.resolve()), sha256=digest, frames=expected,
                dimensions=list(gray.shape[::-1]), steps=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path")
    result = dict(regions=REGIONS, reports=[inspect(path) for path in args.videos],
                  limitations="Image-plane motion of tracked texture. Local facial deformation, lighting and codec effects may affect estimates. No player timing is measured.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
