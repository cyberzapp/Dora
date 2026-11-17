"""Isolated CPU/MediaPipe post-processing of committed generated RGB caches.

Keep native tracking outside the process that owns PyTorch/Metal inference.
Tracking reads writable RGB copies; a native-library failure is reported by the
parent worker as a failed stage, preserving all committed neural output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from flashhead_cache import sha,atomic_json
from head_stabilization import transforms


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--frames',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    identity=json.loads((args.cache/'identity.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
    import mediapipe as mp
    samples=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=.5) as mesh:
        for path in sorted(args.cache.glob('commit-*.json')):
            record=json.loads(path.read_text());index=record['index'];frames=args.cache/f'chunk-{index:04d}.npy'
            if record['request']!=fingerprint or sha(frames)!=record['frames_sha256']:
                raise ValueError('Generated RGB cache failed verification')
            values=np.load(frames,mmap_mode='r',allow_pickle=False)
            if values.shape!=(28,512,512,3) or values.dtype!=np.uint8:raise ValueError('Invalid generated RGB geometry')
            for pixels in values[:record['append_frames']]:
                # Do not hand a read-only memmap into the native image binding.
                result=mesh.process(np.array(pixels,copy=True))
                if not result.multi_face_landmarks:raise ValueError('Head stabilization could not track every frame. Retry with stabilization disabled.')
                points=result.multi_face_landmarks[0].landmark
                left=np.array([points[33].x,points[33].y]);right=np.array([points[263].x,points[263].y])
                delta=right-left;center=(left+right)/2
                samples.append([len(samples),float(np.rad2deg(np.arctan2(delta[1],delta[0]))),*center])
    if len(samples)!=args.frames:raise ValueError('Motion tracking does not cover the requested timeline')
    corrections,report=transforms(samples,(512,512))
    temporary=args.output.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez(stream,transforms=corrections,samples=np.asarray(samples))
    temporary.replace(args.output)
    atomic_json(args.output.with_suffix('.json'),dict(settings=report,frames=len(samples),quality_passed=False))


if __name__=='__main__':main()
