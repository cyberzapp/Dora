"""Measure decoded regional motion; distinguish canvas movement from the head patch."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def inspect(video, regions):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-25)>.01:
        raise ValueError('Expected a readable 25 fps export')
    count=round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reference={};previous={};metrics={name:[] for name in regions};n=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);h,w=gray.shape
            for name,(x0,y0,x1,y1) in regions.items():
                patch=np.ascontiguousarray(gray[round(y0*h):round(y1*h),round(x0*w):round(x1*w)])
                if min(patch.shape)<12 or patch.std()<1:
                    raise ValueError(f'{name} lacks texture for registration')
                if n==0:reference[name]=patch.copy();previous[name]=patch.copy();continue
                matrix=np.eye(2,3,dtype=np.float32)
                try:
                    confidence,matrix=cv2.findTransformECC(reference[name],patch,matrix,cv2.MOTION_TRANSLATION,
                         (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,24,1e-5),None,5)
                    shift=matrix[:,2].tolist()
                    if not np.isfinite([confidence,*shift]).all():shift=None;confidence=None
                except cv2.error:shift=None;confidence=None
                difference=float(cv2.absdiff(previous[name],patch).mean())
                metrics[name].append(dict(frame=n,seconds=n/25,shift_xy_pixels=shift,
                                          correlation=confidence,mean_luma_change=difference))
                previous[name]=patch
            n+=1
    finally:cap.release()
    if n!=count or n<2:raise ValueError('Incomplete decoded timeline')
    summary={}
    for name,rows in metrics.items():
        valid=[r for r in rows if r['shift_xy_pixels'] is not None]
        distance=[float(np.linalg.norm(r['shift_xy_pixels'])) for r in valid]
        summary[name]=dict(bounds=regions[name],registration_failures=len(rows)-len(valid),
            p95_shift_pixels=float(np.percentile(distance,95)) if distance else None,
            maximum_shift_pixels=max(distance,default=None),
            p95_mean_luma_change=float(np.percentile([r['mean_luma_change'] for r in rows],95)),samples=rows)
    with video.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(video=str(video.resolve()),sha256=digest,frames=n,width=w,height=h,regions=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',nargs='+',type=Path)
    p.add_argument('--region',action='append',required=True,help='Label=x0,y0,x1,y1 (fractions of the full canvas)')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh report path')
    regions={}
    for specification in args.region:
        name,coordinates=specification.split('=',1);x0,y0,x1,y1=map(float,coordinates.split(','))
        if not (name and 0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Invalid normalized region')
        regions[name]=(x0,y0,x1,y1)
    reports=[inspect(path,regions) for path in args.video]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(videos=reports,quality_passed=False,
        interpretation='Translation registration to the first decoded frame. Facial deformation, matte changes, '
                       'and changing compression can affect estimates. This does not test player display judder.'),indent=2,allow_nan=False))
    print(json.dumps([{**r,'regions':{k:{a:b for a,b in v.items() if a!='samples'} for k,v in r['regions'].items()}} for r in reports],indent=2))


if __name__=='__main__':main()
