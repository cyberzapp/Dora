"""Research-only fixed subject matte around actual generated head pixels.

The union of tracked person masks across the complete clip avoids per-frame
matte changes. This is segmentation/compositing, not new portrait synthesis.
"""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research')]
from flashhead_cache import sha
from native_match_projection import match_native,track_raw


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--native-job',type=Path,required=True)
    p.add_argument('--output-name',default='static-subject-blend.mov')
    p.add_argument('--slow-head-path',action='store_true',help='Research-only stronger zero-phase rigid motion correction')
    args=p.parse_args();folder=args.benchmark.resolve()
    if Path(args.output_name).name!=args.output_name or not args.output_name.endswith('.mov'):
        raise ValueError('Choose a local MOV filename')
    output=folder/args.output_name
    if output.exists():raise ValueError('Retain previous output; use a fresh experiment folder')
    report=json.loads((folder/'report.json').read_text())
    photo=ImageOps.exif_transpose(Image.open(report['source_photo'])).convert('RGB')
    if sha(Path(report['source_photo']))!=report['source_sha256']['photo'] or sha(Path(report['source_audio']))!=report['source_sha256']['audio']:
        raise ValueError('Source media changed')
    layout,bounds,stabilize,reference=match_native(args.native_job,report,photo)
    lead=report.get('analysis_lead_frames',0);count=report['frames'];hashes={}
    def frames():
        for index in range((lead+count+27)//28):
            path=folder/f'chunk-{index:04d}.npy';hashes[path.name]=sha(path)
            data=np.load(path,mmap_mode='r',allow_pickle=False)
            if data.shape!=(28,512,512,3) or data.dtype!=np.uint8:raise ValueError('Invalid generated RGB cache')
            start=index*28
            yield from data[max(0,lead-start):min(28,lead+count-start)]
    if args.slow_head_path:
        from slow_head_path import transforms
        corrections,tracking=track_raw(frames(),count,transforms)
    else:
        corrections,tracking=track_raw(frames(),count) if stabilize else (None,None)
    import mediapipe as mp
    union=np.zeros((512,512),dtype=np.float32)
    source=np.asarray(photo.crop(report['crop']).resize((512,512),Image.Resampling.LANCZOS))
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        for index,pixels in enumerate([source]):
            result=segmenter.process(np.array(pixels,copy=True));union=np.maximum(union,result.segmentation_mask)
        for index,pixels in enumerate(frames()):
            result=segmenter.process(np.array(pixels,copy=True));mask=result.segmentation_mask
            if mask.shape!=(512,512) or not np.isfinite(mask).all():raise ValueError('Invalid segmentation')
            if corrections is not None:
                mask=cv2.warpAffine(mask,corrections[index],(512,512),flags=cv2.INTER_LINEAR)
            union=np.maximum(union,mask)
    # One conservative fixed support mask for the whole video. Never animate
    # its threshold, geometry, or alpha between frames.
    support=(union>.2).astype(np.float32)
    components,labels,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8),8)
    if components<2:raise ValueError('Person segmentation is empty')
    support=(labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.float32)
    support=cv2.dilate(support,np.ones((7,7),np.uint8))
    support=cv2.GaussianBlur(support,(9,9),1.6)
    left,top,right,bottom=report['crop'];sx,sy=(right-left)/512,(bottom-top)/512
    projection=np.array([[sx,0,left+(sx-1)/2],[0,sy,top+(sy-1)/2],[0,0,1.]])
    matrix=layout.source_transform@projection
    fixed=cv2.warpAffine(support,matrix[:2],layout.size,flags=cv2.INTER_LINEAR)
    old_mask=layout.mask.copy();layout.mask=np.minimum(layout.mask,fixed)
    removed=(old_mask>.01)&(layout.mask<.001)
    Image.fromarray((layout.mask*255).astype(np.uint8)).save(output.with_suffix('.alpha.png'))
    Image.fromarray((removed*255).astype(np.uint8)).save(output.with_suffix('.restored-background.png'))
    partial=output.with_suffix('.partial.mov');w,h=layout.size
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{w}x{h}','-r','25','-i','-','-i',report['source_audio'],'-map','0:v','-map','1:a',
        '-t',str(report['seconds']),'-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],stdin=subprocess.PIPE)
    emitted=0
    try:
        for index,pixels in enumerate(frames()):
            composed=layout.frame(pixels,None if corrections is None else corrections[index])
            if not np.array_equal(composed[removed],layout.source[removed]):raise ValueError('Static region changed before encoding')
            process.stdin.write(composed.tobytes());emitted+=1
    finally:process.stdin.close();code=process.wait()
    if code or emitted!=count:raise ValueError('Incomplete experiment export')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),report['source_audio'],str(partial),
        '--seconds',str(report['seconds']),'--expected-frames',str(count),'--fps','25'],check=True)
    partial.replace(output)
    result=dict(output=str(output),sha256=sha(output),new_inference=False,generated_rgb_sha256=hashes,
        reference=reference,frames=count,restored_static_pixels=int(removed.sum()),tracking=tracking,
        original_animated_support_pixels=int((old_mask>.01).sum()),quality_passed=False,
        mask='Single fixed union of source and all generated person segmentations; threshold .2, largest component, dilate7, blur9 sigma1.6',
        limitation='May reveal original head edges or retain background within motion envelope. Does not correct face deformation or eye/lip synthesis.')
    output.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
