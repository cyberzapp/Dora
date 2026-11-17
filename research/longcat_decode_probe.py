"""Trained VAE temporal-cache and memory test, no generated-media export."""
import json,time,argparse
from pathlib import Path
from PIL import Image,ImageOps
import numpy as np
import mlx.core as mx
from longcat_benchmark import ROOT,load_vae,decode_frames,sha
p=argparse.ArgumentParser();p.add_argument('--height',type=int,default=256);p.add_argument('--width',type=int,default=384);a=p.parse_args()
folder=ROOT/'.local/longcat-vae-probe-model'
manifest=json.loads((folder/'download-manifest.json').read_text())
for item in manifest['files']:
 if item.get('sha256') and sha(folder/item['path'])!=item['sha256']:raise ValueError('VAE integrity mismatch')
mx.set_memory_limit(8*1024**3);mx.set_cache_limit(256*1024**2)
start=time.monotonic();vae=load_vae(folder/'vae')
photo=ImageOps.pad(Image.open('/path/to/your/local/media/Portrait_BlackT.png').convert('RGB'),(a.width,a.height))
x=mx.array((np.asarray(photo,dtype=np.float32).transpose(2,0,1)/127.5-1)[None,:,None]).astype(mx.bfloat16)
z=vae.normalize_latents(vae.encode(x));mx.eval(z)
# Repeated image latents exercise temporal decoding only. They are never exported
# or presented as audio-driven generation.
z=mx.repeat(z,8,axis=2)
frames=decode_frames(vae,z,29)
report=dict(status='temporal_decoder_passed',frames=len(frames),width=a.width,height=a.height,elapsed_seconds=time.monotonic()-start,peak_mlx_bytes=mx.get_peak_memory(),generated_video=False,quality_passed=False,input='repeated encoded portrait latents for decoder compatibility only')
(ROOT/f'.local/longcat-decode-{a.height}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
