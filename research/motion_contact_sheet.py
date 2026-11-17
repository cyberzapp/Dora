"""Matched neighboring frames around the largest estimated motion review points."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-motion',type=Path,required=True)
    parser.add_argument('--appearance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a fresh review sheet path')
    report=json.loads(args.joint_motion.read_text()); clips=report['videos']
    if len(clips)!=2 or not report['shared_audio']:raise ValueError('A matched two-render comparison is required')
    records={Path(m['path']).resolve():m for m in json.loads(args.appearance.read_text())['media'] if m['kind']=='generated-candidate'}
    reasons={'head_reversals':'Estimated head reversal',
             'gaze_changes_with_eyes_open':'Estimated gaze change',
             'eyelid_asymmetry_changes':'Estimated eyelid asymmetry change'}
    events={}
    for clip_index,clip in enumerate(clips):
        for kind,label in reasons.items():
            if clip['review_points'][kind]:
                frame=clip['review_points'][kind][0]['frame']
                events.setdefault(frame,[]).append(f'{label} ({"earlier" if clip_index==0 else "shifted"})')
    if not events:raise ValueError('No tracked review points')
    count=report['frames']
    spans={n:(max(0,n-2),max(0,n-1),n,min(count-1,n+1)) for n in events}
    needed={n for span in spans.values() for n in span}
    paths=[Path(clip['video']).resolve() for clip in clips]
    pictures=[]
    for path in paths:
        crop=records[path]['fixed_face_crop']
        if not crop:raise ValueError('A fixed face crop is required for each clip')
        capture=cv2.VideoCapture(str(path)); frames={}
        try:
            if abs(capture.get(cv2.CAP_PROP_FPS)-25)>.01:raise ValueError('Expected an exact 25 fps review clip')
            for n in range(max(needed)+1):
                ok,pixels=capture.read()
                if not ok:raise ValueError('Incomplete review clip')
                if n in needed:
                    frame=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)).crop(crop)
                    frames[n]=ImageOps.contain(frame,(172,172),Image.Resampling.LANCZOS)
        finally:capture.release()
        pictures.append(frames)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    cell_w,cell_h=180,194
    height=40+len(events)*(32+2*cell_h)
    sheet=Image.new('RGB',(4*cell_w,height),(22,23,27)); draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'Matched 25 fps frames: earlier render above, shifted render below',font=font,fill='white')
    for row,(n,labels) in enumerate(sorted(events.items())):
        y=40+row*(32+2*cell_h)
        draw.text((8,y),f'{n/25:.2f}s: '+', '.join(labels),font=font,fill='white')
        for clip_index,frames in enumerate(pictures):
            for col,frame in enumerate(spans[n]):
                x=col*cell_w; top=y+32+clip_index*cell_h
                draw.text((x+5,top),f'{"Earlier" if clip_index==0 else "Shifted"} | {frame/25:.2f}s',font=font,fill='white')
                image=frames[frame]
                sheet.paste(image,(x+(cell_w-image.width)//2,top+20))
    sheet.save(args.output,quality=94)


if __name__=='__main__':main()
