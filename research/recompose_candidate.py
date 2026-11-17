"""Compare paste-back alone using cached, genuinely generated aligned faces."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from face_composite import composite_face


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--partial',action='store_true',help='Inspect completed chunks while inference continues')
    parser.add_argument('--appearance-video',help='Explicit same-frame, same-geometry source for compositor-only comparison')
    parser.add_argument('--method',choices=['masked','upstream'],default='masked')
    parser.add_argument('--audit',action='store_true',help='Check unchanged source pixels before encoding and save same-frame comparisons')
    args=parser.parse_args()
    if args.partial and args.method=='upstream':parser.error('Upstream comparison requires the completed surrounding-pixel cache')
    if args.audit and args.method=='upstream':parser.error('Pixel-preservation audit applies to the masked method only')
    folder=Path(args.benchmark).resolve()
    if args.partial:
        report=json.loads((folder/'request.json').read_text())
        cache=np.load(folder/'alignment.npz')
        chunks=sorted((folder/'chunks').glob('*.npz'))
        if not chunks:raise RuntimeError('No completed chunks are available yet')
        faces=np.concatenate([np.load(chunk)['faces'] for chunk in chunks])
        report['seconds']=len(faces)/25
        output_name='partial-composite.mov'
    else:
        report=json.loads((folder/'report.json').read_text())
        cache=np.load(folder/'restoration.npz')
        faces=cache['faces']
        output_name='masked-composite.mov'
    restorer=None
    if args.method=='upstream':
        import torch
        import torchvision.transforms.functional as vision
        from torchvision.transforms import InterpolationMode
        if not torch.backends.mps.is_available():raise RuntimeError('Original upstream comparison requires MPS')
        spec=importlib.util.spec_from_file_location('original_affine_restore',ROOT/'vendor/LatentSync/latentsync/utils/affine_transform.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        restorer=module.AlignRestore(resolution=report['resolution'],device='mps')
        output_name='upstream-single-encode.mov'
    mask=cv2.imread(str(ROOT/'vendor/LatentSync/latentsync/utils/mask.png'))
    mask=1-mask.astype(np.float32)/255
    prepared=folder/'source.mkv'
    if not prepared.exists():prepared=folder/'source.mp4'
    capture=cv2.VideoCapture(str(prepared))
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.appearance_video:
        capture.release();capture=cv2.VideoCapture(str(Path(args.appearance_video).resolve()))
        actual=(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual!=(width,height) or abs(capture.get(cv2.CAP_PROP_FPS)-25)>.001:
            raise ValueError('Appearance video must have identical geometry and a 25 fps timebase')
        output_name=output_name.replace('.mov','-original-source.mov')
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    command=[ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','25','-i','-',
             '-i',report['source_audio'],'-map','0:v','-map','1:a','-t',str(report['seconds']),
             '-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(folder/output_name)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    audit=[];panels=[]
    sample_indices={0,len(faces)//2,len(faces)-1}
    try:
        for index,face in enumerate(faces):
            ok,frame=capture.read()
            if not ok: raise RuntimeError('Missing source frame; refusing to loop motion')
            original=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            x1,y1,x2,y2=cache['boxes'][index]
            if restorer is not None:
                with torch.inference_mode():
                    tensor=torch.from_numpy(face).to('mps')
                    tensor=vision.resize(tensor,[int(y2-y1),int(x2-x1)],interpolation=InterpolationMode.BICUBIC,antialias=True)
                    composed=restorer.restore_img(original,tensor,cache['affine'][index])
            else:
                face=np.transpose(face,(1,2,0)).astype(np.float32)
                face=np.clip((face/2+.5)*255,0,255)
                composed=composite_face(original,face,cache['affine'][index],mask,(int(x2-x1),int(y2-y1)))
            if args.audit:
                # Independent, conservative support: the full unfeathered mask.
                # Compare actual RGB results BEFORE lossy video encoding.
                local_mask=cv2.resize(mask[...,0],(int(x2-x1),int(y2-y1)),interpolation=cv2.INTER_LINEAR)
                inverse=cv2.invertAffineTransform(cache['affine'][index].reshape(2,3).astype(np.float32))
                support=cv2.warpAffine(local_mask,inverse,(width,height),flags=cv2.INTER_LINEAR)>0
                difference=np.abs(composed.astype(np.int16)-original.astype(np.int16))
                outside_max=int(difference[~support].max()) if (~support).any() else 0
                if outside_max:
                    raise RuntimeError(f'Original pixels changed outside the synthesis region in frame {index}')
                audit.append(dict(frame=index,outside_mask_max_rgb_change=outside_max,
                                  modified_pixels=int((difference.max(axis=2)>0).sum()),
                                  synthesis_support_pixels=int(support.sum())))
                if index in sample_indices:
                    # Identical source-frame geometry, not different performances.
                    images=[]
                    for pixels in (original,composed):
                        aligned=cv2.warpAffine(pixels,cache['affine'][index].reshape(2,3).astype(np.float32),
                                               (int(x2-x1),int(y2-y1)),flags=cv2.INTER_LINEAR)
                        images.append(Image.fromarray(aligned).resize((420,560),Image.Resampling.LANCZOS))
                    row=Image.new('RGB',(840,610),(22,23,27));draw=ImageDraw.Draw(row)
                    draw.text((12,10),f'Unmodified camera frame {index} ({index/25:.2f}s)',fill='white')
                    draw.text((432,10),'AI mouth composite - UNAPPROVED',fill='white')
                    row.paste(images[0],(0,50));row.paste(images[1],(420,50));panels.append(row)
            process.stdin.write(composed.tobytes())
    finally:
        capture.release();process.stdin.close()
        # Reap the encoder on errors as well; leave no background writer behind.
        return_code=process.wait()
    if return_code: raise RuntimeError('Compositor export failed')
    subprocess.run([ff,'-v','error','-i',str(folder/output_name),'-f','null','-'],check=True)
    (folder/(output_name+'.json')).write_text(json.dumps({
        'method':args.method,
        'description':'Original MPS half-precision full-face restoration' if restorer else 'CPU float32 inverse affine; original pixels outside model synthesis mask before encoding',
        'generated_faces':'completed chunks' if args.partial else 'restoration.npz',
        'baseline':'review.mov','candidate':output_name,'frames':len(faces),'seconds':report['seconds'],
        'appearance_video':str(Path(args.appearance_video).resolve()) if args.appearance_video else str(prepared),
        'partial':args.partial,'quality_passed':False},indent=2))
    if args.audit:
        sheet=Image.new('RGB',(840,610*len(panels)))
        for row,panel in enumerate(panels):sheet.paste(panel,(0,row*610))
        sheet.save(folder/(output_name+'.comparison.jpg'),quality=95)
        (folder/(output_name+'.pixel-audit.json')).write_text(json.dumps(dict(
            interpretation='Exact RGB comparison before lossy encoding; proves spatial preservation only, not natural mouth shapes or lip sync.',
            frames=audit,quality_passed=False),indent=2))


if __name__=='__main__':main()
