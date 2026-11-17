"""Real pretrained FlashHead Pro evaluation, local MPS only; never a quality pass.

Uses official architecture, four-step shifted distilled sampling, streaming audio
context, five conditioning frames and decoded motion carry. The external adapter
handles unsupported rotary arithmetic and attention memory. No model training.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(ROOT/'backend'))
sys.path.insert(0,str(ROOT/'vendor/SoulX-FlashHead'))
from inference_lease import GPUInferenceLease


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(8*1024*1024):value.update(data)
    return value.hexdigest()


def verify_models(folder):
    manifest=json.loads((folder/'download-manifest.json').read_text())
    for item in manifest['files']:
        path=folder/item['path']
        if path.stat().st_size!=item['bytes'] or (item.get('sha256') and digest(path)!=item['sha256']):
            raise ValueError('Model verification failed: '+item['path'])
    return manifest


def main(args):
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image,ImageOps
    import torch
    from safetensors import safe_open
    from transformers import Wav2Vec2FeatureExtractor
    from flash_head.utils.utils import match_and_blend_colors_torch
    from flashhead_mps import load_model_module,load_audio_encoder

    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if args.output.exists():raise ValueError('Use a fresh output folder; unfinished evidence is retained')
    args.output.mkdir(parents=True)
    args.output_created=True
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.65)
    start=time.monotonic();peak_driver=0
    def progress(stage,**details):
        nonlocal peak_driver
        peak_driver=max(peak_driver,torch.mps.driver_allocated_memory())
        state=dict(stage=stage,elapsed_seconds=time.monotonic()-start,
                   max_observed_driver_bytes=peak_driver,quality_passed=False,**details)
        temporary=args.output/'progress.tmp';temporary.write_text(json.dumps(state,indent=2))
        temporary.replace(args.output/'progress.json');print(json.dumps(state),flush=True)

    progress('verifying local weights')
    model_manifest=verify_models(ROOT/'.local/flashhead-models')
    audio_manifest=verify_models(ROOT/'.local/flashhead-audio')
    sources=dict(photo=digest(args.photo),audio=digest(args.audio))
    photo=ImageOps.exif_transpose(Image.open(args.photo)).convert('RGB')
    box=tuple(args.crop)
    if not(0<=box[0]<box[2]<=photo.width and 0<=box[1]<box[3]<=photo.height):
        raise ValueError('Crop must stay inside the uploaded photograph')
    if box[2]-box[0]!=box[3]-box[1]:raise ValueError('Use a square crop without stretching')
    crop=photo.crop(box).resize((512,512),Image.Resampling.LANCZOS);crop.save(args.output/'input-crop.png')
    request=dict(source_photo=str(args.photo),source_audio=str(args.audio),source_sha256=sources,
                 crop=box,model_revision=model_manifest['revision'],audio_revision=audio_manifest['revision'],
                 seconds=args.seconds,seed=args.seed,steps=4,shift=5,frames_per_chunk=33,
                 analysis_lead_frames=args.analysis_lead_frames,
                 motion_frames=5,output_frames_per_chunk=28,model='FlashHead Pro',dtype='bfloat16',
                 device='mps',audio_device='cpu',training_performed=False,quality_passed=False)
    import importlib.metadata
    request['runtime_versions']={name:importlib.metadata.version(name) for name in
                                 ('torch','diffusers','transformers','safetensors','numpy','Pillow')}
    request['code_sha256']={str(path.relative_to(ROOT)):digest(path) for path in
                           (Path(__file__).resolve(),ROOT/'research/flashhead_mps.py',
                            ROOT/'vendor/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py',
                            ROOT/'vendor/SoulX-FlashHead/flash_head/wan/modules/vae.py',
                            ROOT/'vendor/SoulX-FlashHead/flash_head/audio_analysis/wav2vec2.py',
                            ROOT/'vendor/SoulX-FlashHead/flash_head/audio_analysis/torch_utils.py',
                            ROOT/'vendor/SoulX-FlashHead/flash_head/utils/utils.py')}
    (args.output/'request.json').write_text(json.dumps(request,indent=2))
    ff=imageio_ffmpeg.get_ffmpeg_exe();required_frames=round(args.seconds*25)
    lead=args.analysis_lead_frames
    chunk_count=math.ceil((lead+required_frames)/28)
    # Match the official streaming setup: eight-second rolling audio context,
    # five frames before the current 28-frame speech segment, five-wide windows.
    decoded=subprocess.run([ff,'-v','error','-i',str(args.audio),'-t',str(args.seconds),
                            '-f','f32le','-ac','1','-ar','16000','-'],check=True,capture_output=True)
    speech=np.frombuffer(decoded.stdout,dtype=np.float32)
    if len(speech)<round(args.seconds*16000):raise ValueError('Voice-over is shorter than the requested test')
    lead_samples=lead*640
    speech=np.pad(speech,(lead_samples,chunk_count*17920-lead_samples-len(speech)))
    progress('extracting actual speech features')
    feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(ROOT/'.local/flashhead-audio'),local_files_only=True)
    encoder=load_audio_encoder(ROOT/'.local/flashhead-audio')
    (args.output/'audio-encoder-loading.json').write_text(json.dumps(encoder.avatar_loading_report,indent=2))
    contexts=[];history=np.zeros(128000,dtype=np.float32)
    with torch.inference_mode():
        for chunk in range(chunk_count):
            history=np.concatenate((history[17920:],speech[chunk*17920:(chunk+1)*17920]))
            values=feature_extractor(history,sampling_rate=16000,return_tensors='pt').input_values
            result=encoder(values,seq_len=200,output_hidden_states=True)
            features=torch.stack(result.hidden_states[1:],dim=1).squeeze(0).permute(1,0,2)
            indices=(torch.arange(167,200)[:,None]+torch.arange(-2,3)[None,:]).clamp(0,199)
            contexts.append(features[indices].unsqueeze(0).contiguous())
    del encoder,result,features,values;gc.collect()

    progress('encoding uploaded appearance')
    spec=importlib.util.spec_from_file_location('avatar_flashhead_vae',ROOT/'vendor/SoulX-FlashHead/flash_head/wan/modules/vae.py')
    vae_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(vae_module)
    dtype=torch.bfloat16;device='mps'
    vae=vae_module.WanVAE(vae_path=str(ROOT/'.local/flashhead-models/VAE_Wan/Wan2.1_VAE.pth'),
                          dtype=dtype,device=device,parallel=False,use_tiling=False)
    original=torch.from_numpy(np.array(crop)).permute(2,0,1)[None,:,None].to(device=device,dtype=dtype)/127.5-1
    with torch.inference_mode():reference=vae.encode(original.repeat(1,1,33,1,1)).cpu()
    if not torch.isfinite(reference).all():raise RuntimeError('Appearance encoder produced non-finite latents')
    vae.model.to('cpu');gc.collect();torch.mps.empty_cache()

    progress('loading pretrained portrait transformer')
    module=load_model_module();config=json.loads((ROOT/'.local/flashhead-models/Model_Pro/config.json').read_text())
    with torch.device('meta'):model=module.WanModelAudioProject.from_config(config)
    state={}
    with safe_open(str(ROOT/'.local/flashhead-models/Model_Pro/diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as weights:
        for name in weights.keys():state[name]=weights.get_tensor(name).to(dtype=dtype)
    model.load_state_dict(state,strict=True,assign=True);del state
    if any(p.is_meta for p in model.parameters()):raise RuntimeError('Unresolved model parameters')
    model.eval().requires_grad_(False);model.to(device)
    reference=reference.to(device);motion=reference[:,:1].clone()
    generator=torch.Generator(device='cpu').manual_seed(args.seed)
    def noise(shape):return torch.randn(shape,generator=generator,dtype=dtype,device='cpu').to(device)
    def timestep(t):
        t=t/1000
        return torch.tensor([1000*5*t/(1+4*t)],dtype=torch.float32,device=device)
    times=[timestep(t) for t in (1000,750,500,250,0)]
    with torch.inference_mode():
        for chunk,context in enumerate(contexts):
            # Carry needs the VAE; swap stages instead of retaining both on GPU.
            model.to(device);vae.model.to('cpu');torch.mps.empty_cache()
            latents=noise((16,9,64,64));context=context.to(device)
            for step in range(4):
                latents[:,:motion.shape[1]]=motion
                flow=model(x=latents[None],timestep=times[step],context=context,y=reference[None])[0]
                current=(times[step]/1000).to(dtype)[:,None,None,None]
                following=(times[step+1]/1000).to(dtype)[:,None,None,None]
                clean=latents-flow*current
                latents=(1-following)*clean+following*noise(clean.shape)
                if not torch.isfinite(latents).all():raise RuntimeError('Non-finite denoising output')
                torch.mps.synchronize();progress('denoising',chunk=chunk+1,chunks=chunk_count,step=step+1)
            latents[:,:motion.shape[1]]=motion
            del flow,clean,context;model.to('cpu');gc.collect();torch.mps.empty_cache()
            vae.model.to(device)
            progress('decoding actual generated frames',chunk=chunk+1,chunks=chunk_count)
            decoded_chunks=[];decoded_frames=0
            for decoded_chunk in vae.decode_stream(latents):
                if not torch.isfinite(decoded_chunk).all():raise RuntimeError('Non-finite decoded frames')
                decoded_chunks.append(decoded_chunk.cpu());decoded_frames+=decoded_chunk.shape[2]
                progress('decoding actual generated frames',chunk=chunk+1,chunks=chunk_count,decoded_frames=decoded_frames)
            if decoded_frames!=33:raise RuntimeError('Incomplete temporal decode')
            del decoded_chunk;torch.mps.empty_cache()
            videos=torch.cat(decoded_chunks,dim=2).to(device);del decoded_chunks
            videos=match_and_blend_colors_torch(videos,original,1.0)
            if not torch.isfinite(videos).all():raise RuntimeError('Non-finite color matching')
            if chunk+1<chunk_count:motion=vae.encode(videos[:,:,-5:].to(dtype))
            frames=((videos[0,:,5:].float().permute(1,2,3,0)+1)*127.5).clamp(0,255).byte().cpu().numpy()
            np.save(args.output/f'chunk-{chunk:04d}.npy',frames)
            del videos,latents,frames;gc.collect();torch.mps.empty_cache()
    del model,vae;gc.collect();torch.mps.empty_cache()
    # The silence only shifts the neural window grid. The viewer gets the
    # generated speech frames at their original times and the original audio.
    output=args.output/'review.mov';partial=args.output/'review.partial.mov'
    progress('exporting original audio',frames=required_frames)
    process=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','512x512','-r','25','-i','-',
                              '-i',str(args.audio),'-map','0:v','-map','1:a','-t',str(args.seconds),'-c:v','libx264',
                              '-crf','16','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],stdin=subprocess.PIPE)
    count=0
    try:
        for index in range(chunk_count):
            frames=np.load(args.output/f'chunk-{index:04d}.npy',mmap_mode='r',allow_pickle=False)
            if frames.shape!=(28,512,512,3) or frames.dtype!=np.uint8:raise ValueError('Unexpected generated RGB cache')
            start=index*28
            for frame in frames[max(0,lead-start):min(28,lead+required_frames-start)]:
                process.stdin.write(frame.tobytes());count+=1
    finally:process.stdin.close();return_code=process.wait()
    if return_code or count!=required_frames:raise RuntimeError('Video encoding did not cover the requested timeline')
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),str(args.audio),str(partial),
                    '--seconds',str(args.seconds),'--expected-frames',str(required_frames),'--fps','25'],check=True)
    if sources!={'photo':digest(args.photo),'audio':digest(args.audio)}:raise ValueError('Source changed during generation')
    partial.replace(output)
    report=dict(**request,generated_video=True,output=str(output),frames=count,
                elapsed_seconds=time.monotonic()-start,max_observed_driver_bytes=peak_driver,
                visual_review_required=True,source_pixels_unchanged=False,
                output_chunk_join_frames=[n-lead for n in range(28,chunk_count*28,28) if 0<n-lead<required_frames],
                note='Actual generated head crop; original full-body photo projection has not been performed.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));progress('complete',frames=count)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--photo',type=Path,required=True);p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--crop',type=int,nargs=4,required=True);p.add_argument('--seconds',type=float,default=1.12)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analysis-lead-frames',type=int,choices=range(21),default=0,
                   help='Research only: prepend analysis silence and discard matching generated frames')
    args=p.parse_args()
    if not .64<=args.seconds<=15 or abs(args.seconds*25-round(args.seconds*25))>1e-6:p.error('Use 0.64–15 seconds on a 25 fps boundary')
    args.photo=args.photo.resolve();args.audio=args.audio.resolve();args.output=args.output.resolve()
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active local render',flush=True)):
        try:main(args)
        except BaseException as error:
            if getattr(args,'output_created',False):
                state=dict(stage='cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',
                           error=str(error),quality_passed=False)
                (args.output/'failure.json').write_text(json.dumps(state,indent=2))
                temporary=args.output/'progress.tmp';temporary.write_text(json.dumps(state,indent=2))
                temporary.replace(args.output/'progress.json')
            raise
