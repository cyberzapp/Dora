"""Trained UMT5 block memory/operation check, not text or video generation."""
import json,time,sys
from pathlib import Path
import mlx.core as mx
from longcat_benchmark import ROOT,sha
from longcat_streaming import TensorStore,StreamedBlock
from longcat_video_avatar.models.umt5 import T5SelfAttentionBlock
folder=ROOT/'.local/longcat-models/q4/text_encoder'
manifest=json.loads((ROOT/'backend/model_manifests/longcat-q4.json').read_text())
entry=next(x for x in manifest['files'] if x['path']=='text_encoder/model-00003-of-00003.safetensors')
path=folder/Path(entry['path']).name
if sha(path)!=entry['sha256']:raise ValueError('Checkpoint integrity mismatch')
config=json.loads((ROOT/'.local/longcat-metadata/text_encoder/config.json').read_text())
mx.set_memory_limit(8*1024**3);mx.set_cache_limit(256*1024**2);mx.random.seed(42)
start=time.monotonic()
def factory():
 return T5SelfAttentionBlock(config['d_model'],config['d_kv']*config['num_heads'],config['d_ff'],config['num_heads'],config['relative_attention_num_buckets'],False)
block=StreamedBlock(TensorStore(folder),'blocks.23.',factory)
x=mx.random.normal((1,512,config['d_model'])).astype(mx.bfloat16)
output=block(x,mask=mx.ones((1,512),dtype=mx.int32));mx.eval(output)
if not bool(mx.all(mx.isfinite(output))):raise RuntimeError('Nonfinite trained text block output')
report=dict(status='trained_text_block_passed',shape=list(output.shape),elapsed_seconds=time.monotonic()-start,peak_mlx_bytes=mx.get_peak_memory(),checkpoint_sha256=entry['sha256'],generated_video=False,quality_passed=False,input='synthetic hidden-state compatibility test with real pretrained block weights')
(ROOT/'.local/longcat-text-block-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
