"""Use the verified native job's static framing for an actual research render."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps

from flashhead_cache import sha
from flashhead_layout import Layout
from head_stabilization import transforms

ROOT = Path(__file__).resolve().parents[1]


def match_native(directory, report, photo):
    directory = Path(directory).resolve()
    request = json.loads((directory/'request.json').read_text())
    provenance = directory/'provenance.json'
    baseline = json.loads(provenance.read_text())
    if (report.get('model') != 'FlashHead Pro' or baseline.get('engine') != 'FlashHead Pro'
            or baseline.get('generated_video') is not True or baseline.get('all_frames_verified') is not True
            or baseline.get('original_audio_verified') is not True):
        raise ValueError('Matching needs an independently verified native FlashHead job')
    if (baseline['fps'] != 25 or baseline['frames'] != report['frames']
            or baseline['source_crop'] != report['crop']):
        raise ValueError('Native reference differs in timebase, frame count or inference crop')
    source = Path(request['photo'])
    original = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
    if sha(source) != baseline['source_photo_sha256'] or not np.array_equal(np.asarray(original), np.asarray(photo)):
        raise ValueError('Native reference differs in decoded source photograph')
    output = directory/'output.mov'
    subprocess.run([sys.executable, str(ROOT/'scripts/verify_export.py'), report['source_audio'], str(output),
                    '--seconds', str(report['seconds']), '--expected-frames', str(report['frames']), '--fps','25'],check=True)
    bounds = request['face_bounds']
    layout = Layout(original, tuple(baseline['output_size']), bounds, baseline['source_crop'])
    if layout.mode != baseline['framing']:
        raise ValueError('Native framing no longer matches the recorded reference')
    return layout, bounds, baseline['stabilize_head'], dict(path=str(output), sha256=sha(output), provenance_sha256=sha(provenance))


def track_raw(frames, count, transform_function=transforms):
    """Same raw eye-corner measurement and filter as the native worker."""
    import mediapipe as mp
    samples = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=.5) as mesh:
        for pixels in frames:
            result = mesh.process(np.array(pixels,copy=True))
            if not result.multi_face_landmarks:
                raise ValueError('Head stabilization could not track every generated RGB frame')
            points = result.multi_face_landmarks[0].landmark
            left = np.array([points[33].x,points[33].y]); right = np.array([points[263].x,points[263].y])
            delta = right-left; center = (left+right)/2
            samples.append([len(samples),float(np.rad2deg(np.arctan2(delta[1],delta[0]))),*center])
    if len(samples) != count:
        raise ValueError('Tracking did not cover the requested timeline')
    matrices, report = transform_function(samples, (512,512))
    return matrices, dict(source='generated RGB cache',**report)
