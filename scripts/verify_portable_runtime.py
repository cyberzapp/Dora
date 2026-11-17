#!/usr/bin/env python3
"""Probe a relocated runtime, never treating tensor/codec tests as avatar output."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

PROBE = r'''
import ctypes, io, json, os, pathlib, socket, subprocess, sys
root=pathlib.Path(sys.argv[1]).resolve()
def blocked(*args, **kwargs): raise RuntimeError('Network access is disabled for the runtime probe')
socket.socket.connect=blocked
assert sys.flags.no_user_site == 1
for path in sys.path:
    if path: assert pathlib.Path(path).resolve().is_relative_to(root), ('external sys.path',path)
import ssl, sqlite3, bz2, lzma, hashlib
import numpy as np
import scipy, soundfile, pyloudnorm
from PIL import Image
from transformers import T5TokenizerFast, WhisperFeatureExtractor
import imageio_ffmpeg
import mlx.core as mx
# Actual Metal computation, not a model-generation result.
x=mx.ones((8,8)); y=x@x; mx.eval(y); assert float(y[0,0])==8
photo=Image.new('RGB',(32,32),(10,20,30)); encoded=io.BytesIO(); photo.save(encoded,format='PNG')
assert Image.open(io.BytesIO(encoded.getvalue())).size==(32,32)
tone=np.sin(2*np.pi*440*np.arange(16000)/16000).astype(np.float32)*0.1
loudness=pyloudnorm.Meter(16000).integrated_loudness(tone)
assert np.isfinite(loudness)
audio=io.BytesIO(); soundfile.write(audio,np.zeros(16000,dtype=np.float32),16000,format='WAV',subtype='FLOAT')
samples,rate=soundfile.read(io.BytesIO(audio.getvalue()),dtype='float32'); assert rate==16000 and samples.shape==(16000,)
extractor=WhisperFeatureExtractor(feature_size=128,sampling_rate=16000,chunk_length=30)
mel=extractor(samples,sampling_rate=16000,return_tensors='np').input_features
assert mel.shape==(1,128,3000)
tokenizer=T5TokenizerFast.from_pretrained(str(root/'tokenizer'),local_files_only=True)
counts={text:len(tokenizer(text).input_ids) for text in ['Hello and welcome.','नमस्ते, आपका स्वागत है।','Aaj we discuss science.']}
assert all(count>1 for count in counts.values())
ff=pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve(); assert ff.is_relative_to(root)
subprocess.run([str(ff),'-version'],check=True,stdout=subprocess.DEVNULL)
# Inspect the actual dylibs loaded by all imports and the Metal probe.
dyld=ctypes.CDLL(None)
dyld._dyld_image_count.restype=ctypes.c_uint32
dyld._dyld_get_image_name.argtypes=[ctypes.c_uint32];dyld._dyld_get_image_name.restype=ctypes.c_char_p
loaded=[dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
external=[name for name in loaded if not name.startswith(('/System/Library/','/usr/lib/','/Library/Apple/System/Library/','/private/preboot/Cryptexes/OS/')) and not pathlib.Path(name).resolve().is_relative_to(root)]
assert not external, ('external loaded native libraries',external)
print(json.dumps(dict(runtime_probe_passed=True,generated_video=False,quality_approved=False,
    python=sys.version.split()[0],prefix=sys.prefix,base_prefix=sys.base_prefix,
    loaded_native_images=len(loaded),external_native_images=external,metal_tensor_test=True,
    mel_shape=list(mel.shape),token_counts=counts,ffmpeg=str(ff),loudness_analysis_test=True,python_network_connections_disabled=True)))
'''

FLASHHEAD_COMMON = r'''
import ctypes, io, json, os, pathlib, socket, subprocess, sys
root=pathlib.Path(sys.argv[1]).resolve()
def blocked(*args, **kwargs): raise RuntimeError('Network disabled for runtime validation')
socket.socket.connect=blocked
assert sys.flags.no_user_site == 1
for path in sys.path:
    if path: assert pathlib.Path(path).resolve().is_relative_to(root), ('external sys.path',path)
def audit_loaded():
    dyld=ctypes.CDLL(None);dyld._dyld_image_count.restype=ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes=[ctypes.c_uint32];dyld._dyld_get_image_name.restype=ctypes.c_char_p
    loaded=[dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    external=[p for p in loaded if not p.startswith(('/System/Library/','/usr/lib/','/Library/Apple/System/Library/','/private/preboot/Cryptexes/OS/')) and not pathlib.Path(p).resolve().is_relative_to(root)]
    assert not external, ('external loaded native images',external)
    return len(loaded)
'''

FLASHHEAD_TORCH = FLASHHEAD_COMMON + r'''
import ssl,sqlite3,bz2,lzma,numpy as np,scipy,soundfile,pyloudnorm
import torch,torchvision,imageio_ffmpeg
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor
assert torch.backends.mps.is_available()
x=torch.ones((8,8),device='mps');y=x@x;torch.mps.synchronize();assert float(y[0,0])==8
sys.path[:0]=[str(root/'backend'),str(root/'research'),str(root/'vendor/SoulX-FlashHead')]
from flashhead_mps import load_model_module,sinusoidal_embedding
module=load_model_module()
position=torch.tensor([0,250,750,1000],dtype=torch.bfloat16,device='mps')
actual=sinusoidal_embedding(256,position).cpu()
expected=module.reference_sinusoidal(256,position.cpu())
assert torch.equal(actual,expected)
samples=np.zeros(16000,dtype=np.float32)
values=Wav2Vec2FeatureExtractor(sampling_rate=16000)(samples,sampling_rate=16000,return_tensors='pt').input_values
assert tuple(values.shape)==(1,16000)
memory=io.BytesIO();soundfile.write(memory,samples,16000,format='WAV',subtype='FLOAT')
decoded,rate=soundfile.read(io.BytesIO(memory.getvalue()),dtype='float32');assert rate==16000 and np.array_equal(decoded,samples)
image=Image.new('RGB',(32,32));memory=io.BytesIO();image.save(memory,format='PNG');assert Image.open(io.BytesIO(memory.getvalue())).size==(32,32)
ff=pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve();assert ff.is_relative_to(root)
subprocess.run([str(ff),'-version'],check=True,stdout=subprocess.DEVNULL)
print(json.dumps(dict(component='torch',passed=True,metal_tensor_test=True,timestep_test=True,
    speech_preprocessor_test=True,loaded_native_images=audit_loaded(),python=sys.version.split()[0],
    generated_video=False,quality_approved=False,python_network_connections_disabled=True)))
'''

FLASHHEAD_TRACKING = FLASHHEAD_COMMON + r'''
import numpy as np,cv2,mediapipe as mp
# Isolated from the process that owns Metal inference. This synthetic image is
# only a native graph/resource smoke test; it is never an avatar result.
with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1) as mesh:
    result=mesh.process(np.zeros((512,512,3),dtype=np.uint8))
    assert result.multi_face_landmarks is None
print(json.dumps(dict(component='tracking',passed=True,loaded_native_images=audit_loaded(),
    generated_video=False,quality_approved=False,python_network_connections_disabled=True)))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runtime',type=Path);args=parser.parse_args()
    root=args.runtime.resolve()
    packaging=json.loads((root/'packaging-report.json').read_text())
    profile=packaging.get('engine_profile','longcat')
    env={key:value for key,value in os.environ.items() if not key.startswith(('PYTHON','DYLD_'))}
    env.update(PATH='/usr/bin:/bin',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
               DO_NOT_TRACK='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    env.pop('IMAGEIO_FFMPEG_EXE',None)
    with tempfile.TemporaryDirectory(prefix='avatar-runtime-cwd-') as directory:
        if profile=='flashhead':
            parts=[]
            for source in (FLASHHEAD_TORCH,FLASHHEAD_TRACKING):
                result=subprocess.run([str(root/'.venv/bin/python'),'-c',source,str(root)],cwd=directory,
                                      env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                parts.append(json.loads(result.stdout))
            report=dict(runtime_probe_passed=True,engine_profile=profile,components=parts,
                        generated_video=False,quality_approved=False)
        else:
            result=subprocess.run([str(root/'.venv-longcat/bin/python'),'-c',PROBE,str(root)],cwd=directory,
                                  env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            report=json.loads(result.stdout)
    (root/'runtime-probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
