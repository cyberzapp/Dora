"""Measure observable 2D eye-line motion, not 3D pose or identity quality."""
import argparse,json
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


def inspect(path):
    capture=cv2.VideoCapture(str(path));samples=[];missing=[];index=0
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=.5) as mesh:
        while True:
            ok,frame=capture.read()
            if not ok:break
            h,w=frame.shape[:2]
            result=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                missing.append(index)
            else:
                points=result.multi_face_landmarks[0].landmark
                left=np.array([points[33].x*w,points[33].y*h])
                right=np.array([points[263].x*w,points[263].y*h])
                delta=right-left;center=(left+right)/2
                samples.append([index,float(np.degrees(np.arctan2(delta[1],delta[0]))),float(center[0]/w),float(center[1]/h)])
            index+=1
    capture.release()
    if not samples:raise RuntimeError(f'No face tracked: {path}')
    values=np.asarray(samples)
    return {'path':str(Path(path).resolve()),'frames':index,'missing_frames':missing,
            'eye_line_roll_range_degrees':float(np.ptp(values[:,1])),
            'eye_line_roll_std_degrees':float(np.std(values[:,1])),
            'eye_center_horizontal_range_fraction':float(np.ptp(values[:,2])),
            'eye_center_vertical_range_fraction':float(np.ptp(values[:,3])),
            'samples':samples}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('videos',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    reports=[inspect(video) for video in args.videos]
    Path(args.output).write_text(json.dumps({'method':'2D MediaPipe eye-corner tracking; not perceptual acceptance or 3D pose','videos':reports},indent=2))
    print(json.dumps([{key:value for key,value in report.items() if key!='samples'} for report in reports],indent=2))
