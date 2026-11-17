"""Validate another export framing from verified actual generated RGB caches.

No new inference or substituted result. Original speech and cache provenance are
checked; framing preserves source subject pixels outside the head mask.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from flashhead_cache import ChunkCache,sha
from flashhead_layout import Layout
from flashhead_render import raw_frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--portrait',action='store_true');args=p.parse_args()
    folder=args.job.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Use a fresh output path')
    request=json.loads((folder/'request.json').read_text());report=json.loads((folder/'provenance.json').read_text())
    if report.get('generated_video') is not True:raise ValueError('Actual completed generation required')
    photo=Path(request['photo']);audio=Path(request['audio'])
    if sha(photo)!=report['source_photo_sha256'] or sha(audio)!=report['source_audio_sha256']:
        raise ValueError('Source media changed')
    identity=json.loads((folder/'flashhead-cache/identity.json').read_text())
    size=(1080,1920) if args.portrait else (1920,1080)
    layout=Layout(ImageOps.exif_transpose(Image.open(photo)).convert('RGB'),size,request['face_bounds'],report['source_crop'])
    corrections=None
    if report['stabilize_head']:
        corrections=np.load(folder/'head-stabilization.npz',allow_pickle=False)['transforms']
        if corrections.shape!=(report['frames'],2,3) or not np.isfinite(corrections).all():raise ValueError('Invalid stabilization cache')
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.stem+'.partial.mov')
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','25','-i','-','-i',str(audio),
        '-map','0:v','-map','1:a','-t',str(report['frames']/25),'-c:v','libx264','-crf','16',
        '-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],stdin=subprocess.PIPE)
    try:
        with ChunkCache(folder/'flashhead-cache',identity['request']) as cache:
            for index,frame in enumerate(raw_frames(cache,report['frames'])):
                process.stdin.write(layout.frame(frame,None if corrections is None else corrections[index]).tobytes())
    finally:process.stdin.close();code=process.wait()
    if code:raise RuntimeError('Export encoding failed')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),str(audio),str(partial),
                    '--expected-frames',str(report['frames']),'--fps','25'],check=True)
    partial.replace(output)
    output.with_suffix('.json').write_text(json.dumps(dict(source_job=str(folder),source_provenance_sha256=sha(folder/'provenance.json'),
        output=str(output),output_size=size,framing=layout.mode,frames=report['frames'],
        generated_cache_reused=True,new_inference_performed=False,source_body_preserved_before_encoding=True,
        original_audio_verified=True,quality_passed=False),indent=2))


if __name__=='__main__':main()
