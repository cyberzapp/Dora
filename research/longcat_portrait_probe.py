"""Real portrait VAE reconstruction diagnostic; not a talking-head render."""
import json,time,sys,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
from PIL import Image,ImageOps,ImageDraw
from longcat_benchmark import load_vae,sha,ROOT
parser=argparse.ArgumentParser();parser.add_argument('--width',type=int,default=384);parser.add_argument('--height',type=int,default=256);args=parser.parse_args()
folder=ROOT/'.local/longcat-vae-probe-model'
source=Path('/path/to/your/local/media/Portrait_BlackT.png')
output=ROOT/f'.local/longcat-portrait-probe-{args.height}';output.mkdir(exist_ok=True)
deadline=time.monotonic()+1800
while not (folder/'download-complete.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('VAE download did not finish')
 time.sleep(5)
manifest=json.loads((folder/'download-manifest.json').read_text())
for entry in manifest['files']:
 path=folder/entry['path']
 if entry.get('sha256') and sha(path)!=entry['sha256']:raise ValueError('VAE hash mismatch')
started=time.monotonic();mx.set_memory_limit(8*1024**3);mx.set_cache_limit(256*1024**2)
model=load_vae(folder/'vae')
portrait=ImageOps.pad(ImageOps.exif_transpose(Image.open(source)).convert('RGB'),(args.width,args.height),method=Image.Resampling.LANCZOS)
x=mx.array((np.asarray(portrait,dtype=np.float32)/127.5-1).transpose(2,0,1)[None,:,None]).astype(mx.bfloat16)
latent=model.encode(x);mx.eval(latent)
reconstruction=model.decode(latent);mx.eval(reconstruction)
assert bool(mx.all(mx.isfinite(reconstruction)))
arr=np.asarray(mx.clip((reconstruction[0,:,0].transpose(1,2,0)+1)*127.5,0,255).astype(mx.uint8))
image=Image.fromarray(arr);image.save(output/'reconstruction.png');portrait.save(output/'input.png')
sheet=Image.new('RGB',(args.width*2,args.height+24),'#202020');sheet.paste(portrait,(0,24));sheet.paste(image,(args.width,24));draw=ImageDraw.Draw(sheet);draw.text((8,6),'Real source resized',fill='white');draw.text((args.width+8,6),'LongCat VAE reconstruction — no animation',fill='white');sheet.save(output/'comparison.png')
report=dict(status='portrait_vae_passed',generated_video=False,quality_passed=False,elapsed_seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory(),source_sha256=sha(source),latent_shape=list(latent.shape),reconstruction_shape=list(reconstruction.shape),mean_absolute_pixel_error=float(np.abs(np.asarray(portrait,dtype=np.float32)-arr.astype(np.float32)).mean()))
(output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
