"""Inspect every temporal chunk boundary; statistics are diagnostic, not a score."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--chunk-frames',type=int,default=16)
    args=parser.parse_args()
    if args.chunk_frames<2:parser.error('Chunk length must be at least 2')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(args.video)
    fps=capture.get(cv2.CAP_PROP_FPS)
    ok,first=capture.read()
    if not ok:raise RuntimeError('Video has no readable frames')
    import mediapipe as mp
    with mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=.5) as detector:
        result=detector.process(cv2.cvtColor(first,cv2.COLOR_BGR2RGB))
    if not result.detections:raise RuntimeError('No face detected for boundary review')
    box=result.detections[0].location_data.relative_bounding_box
    h,w=first.shape[:2];cx=(box.xmin+box.width/2)*w;cy=(box.ymin+box.height/2)*h
    size=max(box.width*w,box.height*h)*1.35
    x1,y1=max(0,int(cx-size/2)),max(0,int(cy-size/2))
    x2,y2=min(w,int(cx+size/2)),min(h,int(cy+size/2))
    faces=[first[y1:y2,x1:x2]]
    while True:
        ok,frame=capture.read()
        if not ok:break
        faces.append(frame[y1:y2,x1:x2])
    capture.release()
    changes=[float(np.abs(faces[i].astype(np.float32)-faces[i-1].astype(np.float32)).mean()) for i in range(1,len(faces))]
    rows=[];boundaries=[]
    for boundary in range(args.chunk_frames,len(faces)-1,args.chunk_frames):
        panels=[]
        for index in [boundary-1,boundary,boundary+1]:
            face=faces[index];fh,fw=face.shape[:2]
            face=cv2.resize(face,(round(fw*320/max(fh,fw)),round(fh*320/max(fh,fw))))
            panel=np.zeros((360,320,3),dtype=np.uint8)
            panel[40:40+face.shape[0],:face.shape[1]]=face
            label=f'Frame {index} | {index/fps:.2f}s'
            cv2.putText(panel,label,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels,axis=1))
        difference=changes[boundary-1]
        boundaries.append({'frame':boundary,'seconds':boundary/fps,'mean_absolute_pixel_change':difference,
                           'percentile_among_all_frame_changes':float(np.mean(np.asarray(changes)<=difference)*100)})
    if rows:cv2.imwrite(str(out/'boundaries.jpg'),np.concatenate(rows,axis=0))
    report={'decoded_frames':len(faces),'fps':fps,'chunk_frames':args.chunk_frames,'boundaries':boundaries,
            'all_frame_changes':changes,'interpretation':'Review aids only. Mouth articulation can produce large changes; playback is required to judge flicker and sync.'}
    (out/'temporal.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!='all_frame_changes'},indent=2))


if __name__=='__main__':main()
