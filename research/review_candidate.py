"""Create labeled inspection frames and independently validate a candidate export.

No perceptual pass is inferred from these structural checks. Originals are read
only; the contact sheet is a diagnostic artifact, never a generated video.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def audio_hash(path, seconds):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
               '-map', '0:a:0', '-t', str(seconds), '-f', 'f32le', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    while data := process.stdout.read(1024 * 1024):
        digest.update(data)
    if process.wait():
        raise RuntimeError('Could not decode audio')
    return digest.hexdigest()


def frame_at(path, seconds):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Could not read {path} at {seconds}s')
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--motion', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate-label', default='Candidate output')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.candidate)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    expected = round(args.seconds * fps)
    if frames != expected:
        raise RuntimeError(f'Expected {expected} decoded frames, got {frames}')
    original_hash = audio_hash(args.audio, args.seconds)
    exported_hash = audio_hash(args.candidate, args.seconds)
    if original_hash != exported_hash:
        raise RuntimeError('Decoded audio differs from the source excerpt')
    rows = []
    face_rows = []
    import mediapipe as mp
    detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=.5)
    for time in np.linspace(0, max(0, args.seconds - .08), 4):
        panels = []
        face_panels = []
        box = None
        for label, path in [('KJ motion input', args.motion), (args.candidate_label, args.candidate)]:
            frame = frame_at(path, float(time))
            h, w = frame.shape[:2]
            if box is None:
                detection = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not detection.detections:
                    raise RuntimeError('No face found for diagnostic close-up')
                bounds = detection.detections[0].location_data.relative_bounding_box
                cx, cy = bounds.xmin+bounds.width/2, bounds.ymin+bounds.height/2
                size = max(bounds.width*w, bounds.height*h)*1.4
                box = (max(0,int(cx*w-size/2)), max(0,int(cy*h-size/2)),
                       min(w,int(cx*w+size/2)), min(h,int(cy*h+size/2)))
            x1,y1,x2,y2 = box
            face = frame[y1:y2,x1:x2]
            fh,fw = face.shape[:2]
            face = cv2.resize(face,(round(fw*384/max(fh,fw)),round(fh*384/max(fh,fw))))
            face_panel = np.zeros((424,384,3),dtype=np.uint8)
            face_panel[40:40+face.shape[0],:face.shape[1]]=face
            cv2.putText(face_panel, f'{label} | {time:.2f}s', (5,25),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
            face_panels.append(face_panel)
            resized = cv2.resize(frame, (640, round(h * 640 / w)))
            panel = np.zeros((480, 640, 3), dtype=np.uint8)
            if resized.shape[0] > 440:
                resized = cv2.resize(frame, (round(w * 440 / h), 440))
            ph, pw = resized.shape[:2]
            panel[40:40+ph, (640-pw)//2:(640-pw)//2+pw] = resized
            cv2.putText(panel, f'{label} | {time:.2f}s', (12, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
        face_rows.append(np.concatenate(face_panels,axis=1))
    detector.close()
    cv2.imwrite(str(output / 'comparison.jpg'), np.concatenate(rows, axis=0))
    cv2.imwrite(str(output / 'faces.jpg'), np.concatenate(face_rows,axis=0))
    report = {'decoded_frames': frames, 'fps': fps, 'seconds': args.seconds,
              'decoded_audio_sha256': exported_hash, 'audio_equal': True,
              'perceptual_quality': 'requires playback and human review',
              'candidate': str(Path(args.candidate).resolve())}
    (output / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
