"""Postprocess a genuine generated clip, retaining source body/background pixels."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from head_composite import head_mask,composite_head
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--photo',type=Path,required=True);p.add_argument('--face-metadata',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Use a new output path')
reader=imageio_ffmpeg.read_frames(str(a.candidate),pix_fmt='rgb24');metadata=next(reader)
w,h=metadata['size'];fps=metadata['fps']
photo=ImageOps.exif_transpose(Image.open(a.photo)).convert('RGB');mask=head_mask(photo,(w,h),json.loads(a.face_metadata.read_text())['face'])
source=np.asarray(ImageOps.pad(photo,(w,h),method=Image.Resampling.LANCZOS))
silent=a.output.with_name(a.output.stem+'-video.mp4');a.output.parent.mkdir(parents=True,exist_ok=True)
ff=imageio_ffmpeg.get_ffmpeg_exe()
encoder=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
count=0
try:
 for data in reader:
  frame=np.frombuffer(data,dtype=np.uint8).reshape(h,w,3)
  mixed=composite_head(source,frame,mask)
  if not np.array_equal(mixed[mask==0],source[mask==0]):raise RuntimeError('Source appearance preservation failed')
  encoder.stdin.write(mixed.tobytes());count+=1
 encoder.stdin.close()
 if encoder.wait():raise RuntimeError('Video encoder failed')
finally:
 reader.close()
 if encoder.poll() is None:encoder.terminate();encoder.wait()
if count==0:raise RuntimeError('No generated frames decoded')
subprocess.run([ff,'-v','error','-y','-i',str(silent),'-i',str(a.audio),'-map','0:v','-map','1:a','-c:v','copy','-c:a','pcm_f32le','-t',str(count/fps),str(a.output)],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),str(a.audio),str(a.output),'--seconds',str(count/fps),'--expected-frames',str(count),'--fps',str(fps)],check=True)
report=dict(operation='Appearance-preserving post-composite of actual generated clip',frames=count,fps=fps,source_body_background_exact_before_encoding=True,quality_passed=False,review_required=True,candidate=str(a.candidate.resolve()),photo=str(a.photo.resolve()),face_metadata=str(a.face_metadata.resolve()),note='Candidate decoded and encoded again; final renderer should composite raw generated frames before its single encode')
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
