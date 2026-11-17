"""Full-size pretrained transformer block compatibility; not generated media."""
import json,sys,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/longcat-avatar-mlx'))
import mlx.core as mx
import mlx.nn as nn
from longcat_streaming import TensorStore,StreamedBlock
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatAvatarSingleStreamBlock
parser=argparse.ArgumentParser();parser.add_argument('--render-shape',action='store_true');parser.add_argument('--quality-shape',action='store_true');parser.add_argument('--presenter-shape',action='store_true');args=parser.parse_args()
folder=ROOT/'.local/longcat-models/q4/dit'
config=json.loads((folder/'config.json').read_text())
manifest=json.loads((folder.parent/'download-manifest.json').read_text())
entry=next(x for x in manifest['files'] if x['path']=='dit/diffusion_pytorch_model-00001-of-00003.safetensors')
path=folder/Path(entry['path']).name
digest=hashlib.sha256()
with path.open('rb') as source:
 while chunk:=source.read(8*1024*1024):digest.update(chunk)
assert digest.hexdigest()==entry['sha256'],'Checkpoint integrity failure'
mx.set_memory_limit(8*1024**3);mx.set_cache_limit(256*1024**2)
q=config['quantization']
def factory():
 keys=('hidden_size','num_heads','mlp_ratio','adaln_tembed_dim','output_dim','audio_prenorm','class_range','class_interval')
 block=LongCatAvatarSingleStreamBlock(**{k:config[k] for k in keys})
 nn.quantize(block,bits=q['bits'],group_size=q['group_size'],class_predicate=lambda p,l:isinstance(l,nn.Linear) and not any(s in p for s in q['skip_patterns']))
 return block
started=time.monotonic()
block=StreamedBlock(TensorStore(folder),'blocks.0.',factory)
mx.random.seed(42)
shape=(31,32,48) if args.presenter_shape else ((8,32,48) if args.quality_shape else ((8,16,24) if args.render_shape else (2,2,2)))
x=mx.random.normal((1,shape[0]*shape[1]*shape[2],4096)).astype(mx.bfloat16)
y=mx.random.normal((1,6,4096)).astype(mx.bfloat16)
t=mx.random.normal((1,shape[0],512))
audio=mx.random.normal((shape[0],32,768)).astype(mx.bfloat16)
out=block(x,y,t,[6],shape,audio,num_cond_latents=1)
mx.eval(out)
assert bool(mx.all(mx.isfinite(out))),'Nonfinite output'
report=dict(status='pretrained_block_passed',shape=list(out.shape),elapsed_seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory(),checkpoint_sha256=digest.hexdigest(),generated_video=False,quality_passed=False)
(ROOT/('.local/longcat-block-presenter-shape-report.json' if args.presenter_shape else ('.local/longcat-block-quality-shape-report.json' if args.quality_shape else ('.local/longcat-block-render-shape-report.json' if args.render_shape else '.local/longcat-block-report.json')))).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
