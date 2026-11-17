"""Real pretrained window/recovery equivalence; no fabricated avatar output."""
import gc
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research'),str(ROOT/'vendor/SoulX-FlashHead')]
from inference_lease import GPUInferenceLease


def main():
    import imageio_ffmpeg
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Wav2Vec2FeatureExtractor
    from flashhead_mps import load_audio_encoder
    from flashhead_engine import Engine
    from flashhead_cache import ChunkCache,sha
    from flashhead_benchmark import verify_models
    folder=ROOT/'.local/flashhead-engine-recovery';folder.mkdir(exist_ok=False)
    start=time.monotonic();peak=0
    def progress(stage,**details):
        nonlocal peak
        peak=max(peak,torch.mps.driver_allocated_memory())
        state=dict(stage=stage,elapsed_seconds=time.monotonic()-start,max_observed_driver_bytes=peak,**details)
        (folder/'progress.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
    verify_models(ROOT/'.local/flashhead-models');verify_models(ROOT/'.local/flashhead-audio')
    torch.set_num_threads(2)
    audio='/path/to/your/local/media/My_voice.wav'
    decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',audio,'-t','2.24',
                            '-f','f32le','-ac','1','-ar','16000','-'],capture_output=True,check=True)
    speech=np.frombuffer(decoded.stdout,dtype=np.float32);history=np.zeros(128000,dtype=np.float32)
    encoder=load_audio_encoder(ROOT/'.local/flashhead-audio')
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(ROOT/'.local/flashhead-audio'),local_files_only=True)
    contexts=[]
    with torch.inference_mode():
        for index in range(2):
            history=np.concatenate((history[17920:],speech[index*17920:(index+1)*17920]))
            values=extractor(history,sampling_rate=16000,return_tensors='pt').input_values
            output=encoder(values,seq_len=200,output_hidden_states=True)
            features=torch.stack(output.hidden_states[1:],dim=1).squeeze(0).permute(1,0,2)
            indices=(torch.arange(167,200)[:,None]+torch.arange(-2,3)[None,:]).clamp(0,199)
            contexts.append(features[indices].unsqueeze(0).contiguous())
    del encoder,output,features,values;gc.collect()
    source=Image.open('/path/to/your/local/media/Portrait_BlackT.png').convert('RGB').crop((445,92,969,616))
    engine=None
    with ChunkCache(folder/'cache',dict(purpose='real-neural-recovery-equivalence',seed=42)) as cache:
        try:
            engine=Engine(source,ROOT/'.local/flashhead-models',progress=progress)
            first=engine.generate(contexts[0]);state=engine.state();cache.commit(0,first,state,28)
            expected=np.load(ROOT/'.local/flashhead-detail-five/chunk-0000.npy',allow_pickle=False)
            if not np.array_equal(first,expected):raise RuntimeError('Extracted engine differs from the tested benchmark first window')
            uninterrupted=engine.generate(contexts[1]);engine.close();engine=None;gc.collect()
            # Reload all actual models and only the committed first-window state.
            cache.verify(0);state=cache.paths(0)[1].read_bytes()
            engine=Engine(source,ROOT/'.local/flashhead-models',recovery=state,progress=progress)
            resumed=engine.generate(contexts[1])
            if not np.array_equal(resumed,uninterrupted):raise RuntimeError('Resumed neural window differs from uninterrupted generation')
            expected_second=np.load(ROOT/'.local/flashhead-detail-five/chunk-0001.npy',allow_pickle=False)
            if not np.array_equal(resumed,expected_second):raise RuntimeError('Extracted engine differs from the tested benchmark continuation')
            cache.commit(1,resumed,engine.state(),28)
            report=dict(actual_pretrained_windows_executed=3,first_window_matches_benchmark_exactly=True,
                continuation_matches_benchmark_exactly=True,resumed_window_matches_uninterrupted_exactly=True,
                elapsed_seconds=time.monotonic()-start,max_observed_driver_bytes=peak,quality_passed=False,
                complete_video_generated=False,cache_frame_sha256=[record['frames_sha256'] for record in cache.records],
                limitation='Window/state equivalence only. Native integration, full-length quality and packaged inference remain unverified.')
            (folder/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        finally:
            if engine is not None:engine.close()


if __name__=='__main__':
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active render',flush=True)):
        main()
