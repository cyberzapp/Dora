"""Local estimated pose/expression diagnostics; never a perceptual quality score.

Processes prerecorded files only. FFmpeg applies display rotation before tracking.
Different performances are not word-aligned references or motion targets.
"""
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
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'vendor/ComfyUI-LivePortraitKJ/media_pipe/mp_models/face_landmarker_v2_with_blendshapes.task'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rotation(matrix):
    # Remove the estimated matrix's scale/shear before comparing rotations.
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64)[:3, :3])
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return Rotation.from_matrix(u @ correction @ vt)


def closure_runs(samples, threshold=0.6):
    """Consecutive bilateral estimates above a diagnostic threshold, not blink labels."""
    runs = []
    for sample in samples:
        scores = sample['blendshapes']
        if min(scores['eyeBlinkLeft'], scores['eyeBlinkRight']) < threshold:
            continue
        index = sample['frame']
        if runs and runs[-1]['last_frame'] + 1 == index:
            runs[-1]['last_frame'] = index
        else:
            runs.append(dict(first_frame=index, last_frame=index))
    for run in runs:
        run.update(start_seconds=run['first_frame']/25,
                   duration_seconds=(run['last_frame']-run['first_frame']+1)/25)
    return runs


def inspect(path, model, max_frames=None):
    path = Path(path).resolve()
    samples, missing, rotations = [], [], []
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    with tempfile.TemporaryDirectory(prefix='avatar-expression-') as temporary:
        normalized = Path(temporary) / 'normalized.mkv'
        options_ffmpeg = ['-frames:v', str(max_frames)] if max_frames is not None else []
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
                        '-map', '0:v:0', '-an', '-vf', 'fps=25,scale=1280:1280:force_original_aspect_ratio=decrease',
                        *options_ffmpeg, '-c:v', 'ffv1', str(normalized)], check=True)
        capture = cv2.VideoCapture(str(normalized))
        if not capture.isOpened():
            raise ValueError(f'Cannot decode {path}')
        expected = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = 0
        try:
            with vision.FaceLandmarker.create_from_options(options) as tracker:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = tracker.detect_for_video(image, index * 40)
                    if len(result.facial_transformation_matrixes) != 1 or len(result.face_blendshapes) != 1:
                        missing.append(index)
                    else:
                        pose = rotation(result.facial_transformation_matrixes[0])
                        if not rotations:
                            anchor = pose
                        relative = (anchor.inv() * pose).as_euler('xyz', degrees=True)
                        coefficients = {item.category_name: float(item.score) for item in result.face_blendshapes[0]}
                        if not np.isfinite(relative).all() or not np.isfinite(list(coefficients.values())).all():
                            raise ValueError('Non-finite tracker output')
                        samples.append(dict(frame=index, seconds=index / 25,
                                            relative_xyz_degrees=relative.tolist(), blendshapes=coefficients))
                        rotations.append(pose)
                    index += 1
        finally:
            capture.release()
    if not samples or index != expected or (max_frames is not None and index != max_frames):
        raise ValueError(f'Incomplete decode or no tracked face: {index}/{expected}')
    changes = []
    for previous, current, r0, r1 in zip(samples, samples[1:], rotations, rotations[1:]):
        if current['frame'] != previous['frame'] + 1:
            continue  # Never bridge missing detections or fabricate motion.
        changes.append(dict(frame=current['frame'], seconds=current['seconds'],
                            estimated_rotation_change_degrees=float(np.rad2deg((r0.inv() * r1).magnitude())),
                            largest_blendshape_change=max(abs(current['blendshapes'][key] - previous['blendshapes'][key])
                                                          for key in current['blendshapes'])))
    angles = np.asarray([sample['relative_xyz_degrees'] for sample in samples])
    keys = samples[0]['blendshapes'].keys()
    return dict(path=str(path), sha256=digest(path), frames_at_25fps=index, missing_frames=missing,
                bilateral_closure_threshold=0.6, bilateral_closure_runs=closure_runs(samples),
                relative_xyz_range_degrees=np.ptp(angles, axis=0).tolist(),
                blendshape_ranges={key: [min(s['blendshapes'][key] for s in samples),
                                         max(s['blendshapes'][key] for s in samples)] for key in keys},
                largest_adjacent_pose_changes=sorted(changes, key=lambda s: s['estimated_rotation_change_degrees'], reverse=True)[:12],
                largest_adjacent_expression_changes=sorted(changes, key=lambda s: s['largest_blendshape_change'], reverse=True)[:12],
                samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, help='Compare the same opening number of normalized frames in every clip')
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be positive')
    if args.output.exists():
        raise ValueError('Retain existing diagnostic evidence; choose a new output path')
    reports = [inspect(path, args.model.resolve(), args.frames) for path in args.videos]
    report = dict(method='MediaPipe VIDEO-mode CPU FaceLandmarker, 25fps rotation-normalized decode',
                  limitations='Estimated relative pose and expression coefficients, not calibrated physical measurements. '
                              'Large changes may be normal articulation, blinking or tracking noise. Different performances '
                              'are not synchronized targets. Bilateral closure spans use an uncalibrated threshold, not '
                              'confirmed blink labels. This cannot establish realism, identity or lip-sync quality.',
                  task_model=str(args.model.resolve()), task_model_sha256=digest(args.model),
                  quality_passed=False, camera_or_microphone_activated=False, videos=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps([dict(path=r['path'], frames=r['frames_at_25fps'], missing=len(r['missing_frames']),
                           relative_xyz_range_degrees=r['relative_xyz_range_degrees'],
                           largest_pose_changes=r['largest_adjacent_pose_changes'][:3]) for r in reports], indent=2))


if __name__ == '__main__':
    main()
