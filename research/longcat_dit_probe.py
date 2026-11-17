"""Complete trained DiT forward compatibility. No video output or quality claim."""
import json,time
from pathlib import Path
import mlx.core as mx
from PIL import Image,ImageOps
import numpy as np
from longcat_benchmark import ROOT,load_vae,sha,release
from longcat_streaming import build_streamed_dit
folder=ROOT/'.local/longcat-models/q4'
manifest=json.loads((ROOT/'backend/model_manifests/longcat-q4.json').read_text())
entries=[x for x in manifest['files'] if x['path'].startswith('dit/')]
deadline=time.monotonic()+1800
while not all((folder/e['path']).exists() for e in entries):
 if time.monotonic()>deadline:raise TimeoutError('Transformer download pending')
 time.sleep(5)
for item in entries:
 path=folder/item['path']
 if path.stat().st_size!=item['bytes'] or (item.get('sha256') and sha(path)!=item['sha256']):raise ValueError('Transformer integrity mismatch')
progress=json.loads((ROOT/'.local/longcat-first-render/progress.json').read_text())
if progress['stage']!='waiting_for_verified_weights':raise RuntimeError('Render has advanced; defer standalone GPU probe')
mx.set_memory_limit(8*1024**3);mx.set_cache_limit(256*1024**2);mx.random.seed(42)
started=time.monotonic()
vae=load_vae(folder/'vae')
photo=ImageOps.pad(Image.open('/path/to/your/local/media/Portrait_BlackT.png').convert('RGB'),(384,256))
pixels=np.asarray(photo,dtype=np.float32).transpose(2,0,1)/127.5-1
reference=vae.normalize_latents(vae.encode(mx.array(pixels[None,:,None]).astype(mx.bfloat16)));mx.eval(reference)
del vae;release()
model=build_streamed_dit(folder/'dit')
x=mx.concatenate([reference,mx.random.normal((1,16,7,32,48))],axis=2)
t=mx.full((1,8),1000.0);t[:,0]=0
audio=mx.load(str(ROOT/'.local/longcat-audio-ffmpeg/audio.safetensors'))['audio']
# Zero text tests complete wiring only while the trained text encoder downloads.
text=mx.zeros((1,1,512,4096),dtype=mx.bfloat16);mask=mx.ones((1,512),dtype=mx.int32)
output=model(x,t,text,audio,encoder_attention_mask=mask,num_cond_latents=1);mx.eval(output)
if output.shape!=x.shape or not bool(mx.all(mx.isfinite(output))):raise RuntimeError('Invalid full-transformer output')
report=dict(status='complete_trained_dit_forward_passed',blocks=48,shape=list(output.shape),elapsed_seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory(),generated_video=False,quality_passed=False,conditioning='real portrait/audio; zero diagnostic text embeddings; one forward pass only')
(ROOT/'.local/longcat-dit-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
