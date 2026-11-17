"""Inspect actual generated RGB frames on both sides of temporal chunk joins."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);args=p.parse_args()
    folder=args.benchmark.resolve()
    worker=not(folder/'report.json').exists()
    report=json.loads((folder/('provenance.json' if worker else 'report.json')).read_text())
    if report.get('generated_video') is not True:raise ValueError('Actual completed generation required')
    if worker:
        request=json.loads((folder/'request.json').read_text())
        source=ImageOps.exif_transpose(Image.open(request['photo']))
        x,y,fw,fh=request['face_bounds'];left,top,right,bottom=report['source_crop']
        cx=((x+fw/2)*source.width-left)*512/(right-left)
        cy=((1-y-fh/2)*source.height-top)*512/(bottom-top)
        side=max(fw*source.width,fh*source.height)*1.8*512/(right-left)
        box=(max(0,round(cx-side/2)),max(0,round(cy-side/2)),min(512,round(cx+side/2)),min(512,round(cy+side/2)))
    else:
        appearance=json.loads((folder/'appearance/appearance-review.json').read_text())
        box=next(item['fixed_face_crop'] for item in appearance['media'] if item['kind']=='generated-candidate')
    if not box:raise ValueError('Face crop unavailable')
    interval=28 if worker else report['output_frames_per_chunk'];count=report['frames']
    lead=0 if worker else report.get('analysis_lead_frames',0)
    if type(lead) is not int or not 0<=lead<interval:raise ValueError('Invalid analysis lead-in')
    boundaries=list(range(interval-lead,count,interval))
    if not worker and report.get('output_chunk_join_frames',boundaries)!=boundaries:
        raise ValueError('Generated chunk join metadata disagrees with the cached timeline')
    if not boundaries:raise ValueError('This clip does not contain a chunk join')
    def frame(index):
        index+=lead
        cache=folder/'flashhead-cache' if worker else folder
        path=cache/f'chunk-{index//interval:04d}.npy'
        if worker:
            import hashlib
            metadata=json.loads((cache/f'commit-{index//interval:04d}.json').read_text())
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=metadata['frames_sha256']:raise ValueError('Generated RGB cache failed verification')
        data=np.load(path,mmap_mode='r',allow_pickle=False)
        if data.dtype!=np.uint8 or data.ndim!=4 or data.shape[3]!=3:raise ValueError('Invalid cached RGB frames')
        return Image.fromarray(data[index%interval]).crop(box)
    w,h=box[2]-box[0],box[3]-box[1]
    sheet=Image.new('RGB',(w*4,(h+48)*len(boundaries)),(22,23,27));draw=ImageDraw.Draw(sheet);stats=[]
    for row,boundary in enumerate(boundaries):
        indices=[max(0,boundary-2),boundary-1,boundary,min(boundary+1,count-1)]
        for column,index in enumerate(indices):
            x,y=column*w,row*(h+48)
            draw.text((x+8,y+8),f'f{index} / {index/25:.2f}s',fill='white')
            draw.text((x+8,y+25),'NEW CHUNK' if index==boundary else 'Actual generated RGB cache',fill='white')
            sheet.paste(frame(index),(x,y+48))
        delta=np.abs(np.asarray(frame(boundary),dtype=np.float32)-np.asarray(frame(boundary-1),dtype=np.float32)).mean()
        stats.append(dict(first_frame_of_new_chunk=boundary,time_seconds=boundary/25,
                          face_mean_absolute_rgb_change=float(delta)))
    sheet.save(folder/'chunk-boundaries.jpg',quality=95)
    # Keep long-clip evidence legible rather than displaying a huge resized sheet.
    for start in range(0,len(boundaries),4):
        end=min(start+4,len(boundaries))
        sheet.crop((0,start*(h+48),w*4,end*(h+48))).save(folder/f'chunk-boundaries-page-{start//4+1:02d}.jpg',quality=95)
    (folder/'chunk-boundaries.json').write_text(json.dumps(dict(boundaries=stats,quality_passed=False,
        interpretation='Same fixed face crop in adjacent genuinely generated frames. Pixel differences include normal motion and articulation; they are not a perceptual-quality score.'),indent=2))


if __name__=='__main__':main()
