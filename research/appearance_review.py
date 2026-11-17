"""Create labeled, source-resolution-aware appearance sheets from actual media.

No video synthesis, enhancement, automatic quality pass or cross-performance
lip-sync score. Static photographs and external references are explicitly labeled.
"""
import argparse
import json
from pathlib import Path
import textwrap

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont


def video_frame(path, index):
    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened(): raise ValueError('Cannot decode video: '+str(path))
        fps = reader.get(cv2.CAP_PROP_FPS); count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0: raise ValueError('Invalid video timebase')
        reader.set(cv2.CAP_PROP_POS_FRAMES, min(count-1, index))
        ok, frame = reader.read()
        if not ok: raise ValueError(f'Cannot decode requested frame {index}: {path}')
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)), fps, count
    finally: reader.release()


def face_region(photo, detector):
    result = detector.process(np.asarray(photo.convert('RGB')))
    if not result.detections: return None, 0
    box = max((d.location_data.relative_bounding_box for d in result.detections),key=lambda b:b.width*b.height)
    width,height=photo.size; side=max(box.width*width,box.height*height)*1.8
    cx=(box.xmin+box.width/2)*width;cy=(box.ymin+box.height/2)*height
    return (max(0,round(cx-side/2)),max(0,round(cy-side/2)),
            min(width,round(cx+side/2)),min(height,round(cy+side/2))),round(box.width*width)


def panel(photo, label, detail, size, font):
    width,height=size; result=Image.new('RGB',size,(22,23,27)); draw=ImageDraw.Draw(result)
    lines=textwrap.wrap(label,40)[:2]+textwrap.wrap(detail,44)[:2]
    for index,line in enumerate(lines):draw.text((10,8+index*22),line,font=font,fill='white')
    fitted=ImageOps.contain(photo,(width-20,height-110),Image.Resampling.LANCZOS)
    result.paste(fitted,((width-fitted.width)//2,104+(height-110-fitted.height)//2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--photo',type=Path,required=True)
    parser.add_argument('--candidate',action='append',required=True,help='Label=/absolute/path/to/actual-generated-video')
    parser.add_argument('--candidate-frames',type=int,help='Compare an equal opening frame count across generated candidates')
    parser.add_argument('--reference',type=Path,help='External quality reference, not an app-generated result')
    parser.add_argument('--reference-end-seconds',type=float,help='Explicit reference sampling end before an outro; generated candidates always include their final frame')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows',type=int,default=4)
    args=parser.parse_args()
    if not 2<=args.rows<=8:raise ValueError('Use 2–8 review rows')
    if args.candidate_frames is not None and args.candidate_frames<2:raise ValueError('Use at least two candidate frames')
    if args.output.exists():raise ValueError('Use a new review output directory')
    args.output.mkdir(parents=True)
    source=ImageOps.exif_transpose(Image.open(args.photo)).convert('RGB')
    cases=[dict(label='Original appearance • STILL PHOTO',path=args.photo,kind='source-photo',first=source,fps=0,count=1)]
    for item in args.candidate:
        label,path=item.split('=',1);path=Path(path).resolve()
        first,fps,count=video_frame(path,0)
        cases.append(dict(label=label,path=path,kind='generated-candidate',first=first,fps=fps,count=count))
        if args.candidate_frames is not None:
            if count<args.candidate_frames:raise ValueError(f'Not enough frames to compare {label}')
            cases[-1]['sample_end']=args.candidate_frames-1
    if args.reference:
        first,fps,count=video_frame(args.reference,0)
        cases.append(dict(label='HeyGen • EXTERNAL REFERENCE',path=args.reference,kind='external-reference',first=first,fps=fps,count=count))
        if args.reference_end_seconds is not None:
            if args.reference_end_seconds <= 0:raise ValueError('Reference sampling end must be positive')
            cases[-1]['sample_end']=min(count-1,round(args.reference_end_seconds*fps))
    font_path=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    full_rows=[];face_rows=[];summary=[]
    with mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=.5) as detector:
        for case in cases:
            case['box'],case['face_width']=face_region(case['first'],detector)
        for row in range(args.rows):
            full=[];faces=[]
            for case in cases:
                index=round(case.get('sample_end',case['count']-1)*row/(args.rows-1))
                image=case['first'] if index==0 else video_frame(case['path'],index)[0]
                detail=f"{image.width}×{image.height} source • face ≈{case['face_width']} px"
                if case['kind']=='source-photo':detail+=' • still'
                else:detail+=f" • f{index} / {index/case['fps']:.2f}s"
                full.append(panel(image,case['label'],detail,(480,440),font))
                crop=image.crop(case['box']) if case['box'] else image
                crop_detail=detail if case['box'] else 'No face detected • full frame shown'
                faces.append(panel(crop,case['label'],crop_detail,(400,500),font))
            full_rows.append(full);face_rows.append(faces)
    for name,rows,size in [('framing.jpg',full_rows,(480,440)),('faces.jpg',face_rows,(400,500))]:
        canvas=Image.new('RGB',(size[0]*len(cases),size[1]*args.rows))
        for row,items in enumerate(rows):
            for column,item in enumerate(items):canvas.paste(item,(column*size[0],row*size[1]))
        canvas.save(args.output/name,quality=94)
    for case in cases:
        summary.append(dict(label=case['label'],kind=case['kind'],path=str(case['path'].resolve()),
                            size=list(case['first'].size),fps=case['fps'],frames=case['count'],
                            last_sample_frame=case.get('sample_end',case['count']-1),
                            fixed_face_crop=case['box'],first_frame_detected_face_width=case['face_width']))
    report=dict(media=summary,quality_passed=False,playback_review_required=True,
                generated_video_created=False,automatic_quality_score=None,
                interpretation=('Generated candidates use equal opening frame counts; compare timestamps to verify alignment. ' if args.candidate_frames is not None else 'Rows sample each clip across its own duration. ')+
                'External references may have different performances. Fixed per-clip crops retain head movement. Display resizing is ordinary interpolation, not enhancement. Source pixel counts are reported to avoid hiding resolution differences.')
    (args.output/'appearance-review.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
