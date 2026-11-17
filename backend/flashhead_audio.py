"""Offline, bounded speech conditioning for the tested FlashHead streaming recipe."""
import json
from pathlib import Path

import numpy as np

from flashhead_cache import sha,atomic_json
from flashhead_timeline import NEW_SAMPLES,HISTORY_SAMPLES


def history_for(speech,index):
    end=(index+1)*NEW_SAMPLES;start=end-HISTORY_SAMPLES
    left=max(0,start);right=min(len(speech),end)
    history=np.zeros(HISTORY_SAMPLES,dtype=np.float32)
    if right>left:history[left-start:right-start]=speech[left:right]
    return history


def contexts(speech,count,directory,identity,audio_folder,check=lambda:None,progress=lambda *_:None):
    """Cache one six-MB context at a time; return paths, never all feature tensors."""
    import gc
    import sys
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'research'),str(root/'vendor/SoulX-FlashHead')]
    from flashhead_mps import load_audio_encoder
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    paths=[folder/f'{index:04d}.npy' for index in range(count)]
    ready=set()
    for index,path in enumerate(paths):
        check();record=path.with_suffix('.json')
        if not(path.exists() and record.exists()):continue
        meta=json.loads(record.read_text())
        if meta.get('request')!=identity or meta.get('sha256')!=sha(path):
            raise ValueError('Saved speech features failed verification; start a new render')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.shape!=(1,33,5,12,768) or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid saved speech features')
        ready.add(index)
    if len(ready)==count:return paths
    torch.set_num_threads(2)
    encoder=load_audio_encoder(audio_folder)
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(audio_folder),local_files_only=True)
    indices=(torch.arange(167,200)[:,None]+torch.arange(-2,3)[None,:]).clamp(0,199)
    try:
        with torch.inference_mode():
            for index,path in enumerate(paths):
                check()
                if index in ready:continue
                values=extractor(history_for(speech,index),sampling_rate=16000,return_tensors='pt').input_values
                result=encoder(values,seq_len=200,output_hidden_states=True)
                features=torch.stack(result.hidden_states[1:],dim=1).squeeze(0).permute(1,0,2)
                context=features[indices].unsqueeze(0).contiguous()
                if not torch.isfinite(context).all():raise RuntimeError('Speech analysis produced invalid features')
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.save(stream,context.numpy(),allow_pickle=False)
                temporary.replace(path)
                atomic_json(path.with_suffix('.json'),dict(request=identity,sha256=sha(path)))
                progress(index+1,count)
                del values,result,features,context
    finally:del encoder;gc.collect()
    return paths
