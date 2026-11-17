"""Experimental local quality benchmark. Not exposed as a production render backend."""
import argparse,hashlib,json,os,sys,time,types,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.local/matplotlib')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(ROOT/'vendor/LatentSync'))

def detector_module():
    import cv2,mediapipe as mp,numpy as np
    class Detector:
        def __init__(self,device=None):
            self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=.5)
        def __call__(self,frame,threshold=.5):
            result=self.mesh.process(frame) # pipeline frames are RGB
            if not result.multi_face_landmarks:return None,None
            h,w=frame.shape[:2]
            pts=np.array([[p.x*w,p.y*h] for p in result.multi_face_landmarks[0].landmark])
            # Match the three alignment anchors used by upstream (brows and nose).
            landmarks=np.zeros((106,2))
            landmarks[[43,48,49,51,50]]=pts[[70,63,105,66,107]].mean(0)
            landmarks[101:106]=pts[[336,296,334,293,300]].mean(0)
            landmarks[[74,77,83,86]]=pts[[1,2,98,327]].mean(0)
            return [*pts.min(0),*pts.max(0)],landmarks
    module=types.ModuleType('latentsync.utils.face_detector');module.FaceDetector=Detector
    sys.modules[module.__name__]=module

def main():
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--audio',required=True);p.add_argument('--seconds',type=float,default=2);p.add_argument('--steps',type=int,default=20);p.add_argument('--version',choices=['1.5','1.6'],default='1.5');p.add_argument('--output');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if not .64 <= args.seconds <= 10: p.error('Benchmark is bounded to 0.64–10 seconds')
    if not 1 <= args.steps <= 60: p.error('Use 1–60 inference steps')
    args.video=str(Path(args.video).resolve()); args.audio=str(Path(args.audio).resolve())
    import torch,numpy as np,cv2,soundfile as sf,imageio_ffmpeg
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL,DDIMScheduler
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    out=Path(args.output).resolve() if args.output else ROOT/('.local/latentsync-benchmark' if args.version=='1.5' else '.local/latentsync-16-benchmark');out.mkdir(parents=True,exist_ok=True)
    ff=imageio_ffmpeg.get_ffmpeg_exe();bindir=ROOT/'.local/bin';bindir.mkdir(exist_ok=True)
    if not (bindir/'ffmpeg').exists():(bindir/'ffmpeg').symlink_to(ff)
    os.environ['PATH']=str(bindir)+os.pathsep+os.environ.get('PATH','')
    expected={'source_audio':args.audio,'source_video':args.video,'seconds':args.seconds,'steps':args.steps,'version':args.version,'seed':1247}
    fingerprints={key:hashlib.sha256(Path(path).read_bytes()).hexdigest() for key,path in [('audio',args.audio),('video',args.video)]}
    chunk_folder=out/'chunks';chunk_folder.mkdir(exist_ok=True)
    cached_chunks=sorted(chunk_folder.glob('*.npz'))
    request_path=out/'request.json'
    if args.resume:
        previous=json.loads(request_path.read_text())
        if any(previous.get(key)!=value for key,value in expected.items()):
            raise ValueError('Resume settings or media paths differ from the cached run')
        if previous.get('source_sha256') and previous['source_sha256']!=fingerprints:
            raise ValueError('Resume source media contents changed')
        if not previous.get('source_sha256') and any(Path(path).stat().st_mtime>request_path.stat().st_mtime for path in (args.audio,args.video)):
            raise ValueError('Legacy cache source was modified after the render began')
        source_path=out/'source.mkv'
        if not source_path.exists():source_path=out/'source.mp4'
        if not source_path.exists() or not (out/'analysis.wav').exists():raise ValueError('Resume needs the unchanged prepared media')
        for index,path in enumerate(cached_chunks):
            if path.stem!=f'{index:04d}':raise ValueError('Resume chunk sequence has a gap')
            values=np.load(path)['faces']
            expected_frames=min(16,round(args.seconds*25)-index*16)
            size=512 if args.version=='1.6' else 256
            if values.shape!=(expected_frames,3,size,size) or not np.isfinite(values).all():
                raise ValueError('Invalid decoded chunk cache')
    else:
        if cached_chunks:raise ValueError('Use --resume with matching inputs, or a fresh output directory')
        source_path=out/'source.mkv'
        subprocess.run([ff,'-y','-v','error','-i',args.video,'-t',str(args.seconds),'-an','-vf','fps=25,scale=1280:1280:force_original_aspect_ratio=decrease','-c:v','ffv1',str(source_path)],check=True)
        subprocess.run([ff,'-y','-v','error','-i',args.audio,'-t',str(args.seconds),'-ac','1','-ar','16000',str(out/'analysis.wav')],check=True)
    capture=cv2.VideoCapture(str(source_path))
    available_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
    if available_frames < round(args.seconds*25):
        raise ValueError(f'Source has only {available_frames} frames; provide enough motion for the requested duration')
    if sf.info(str(out/'analysis.wav')).duration < args.seconds-.02:
        raise ValueError('Voice-over is shorter than the requested duration')
    detector_module()
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.whisper.audio2feature import Audio2Feature
    import latentsync.pipelines.lipsync_pipeline as pipeline_module
    from latentsync.utils.image_processor import ImageProcessor
    from latentsync.utils.util import read_video_cv2
    pipeline_module.read_video=lambda path,**kwargs:read_video_cv2(path)
    pipeline_module.read_audio=lambda path:torch.from_numpy(sf.read(path,dtype='float32')[0])
    pipeline_module.ImageProcessor=lambda resolution,device,mask_image:ImageProcessor(resolution,device='mps',mask_image=mask_image)
    checkout=ROOT/'vendor/LatentSync'
    cfg=OmegaConf.load(checkout/('configs/unet/stage2.yaml' if args.version=='1.5' else 'configs/unet/stage2_512.yaml'))
    resolution=int(cfg.data.resolution)
    checkpoint_folder=checkout/('checkpoints-v'+args.version.replace('.',''))
    scheduler=DDIMScheduler.from_pretrained(str(checkout/'configs'))
    audio=Audio2Feature(model_path=str(checkout/'checkpoints-v15/whisper/tiny.pt'),device='cpu',num_frames=16)
    # Upstream emits two extra endpoint feature chunks and then loops video.
    # Retain their audio context, but render only the requested timeline frames.
    feature_chunks=audio.feature2chunks
    def exact_chunks(*values,**options):
        chunks=feature_chunks(*values,**options)
        required=round(args.seconds*25)
        if len(chunks)<required:
            raise ValueError('Audio features do not cover the requested timeline')
        return chunks[:required]
    audio.feature2chunks=exact_chunks
    vae=AutoencoderKL.from_pretrained(str(ROOT/'vendor/musetalk-mac/upstream/models/sd-vae'),torch_dtype=torch.float16,local_files_only=True)
    vae.config.scaling_factor=.18215;vae.config.shift_factor=0
    # Avoid allocating a second 5 GB float32 model during checkpoint loading.
    # Strict loading also refuses silently missing or mismatched model weights.
    with torch.device('meta'):
        unet=UNet3DConditionModel.from_config(OmegaConf.to_container(cfg.model))
    checkpoint=torch.load(str(checkpoint_folder/'latentsync_unet.pt'),map_location='cpu',weights_only=True,mmap=True)
    torch.nn.Module.load_state_dict(unet,checkpoint['state_dict'],strict=True,assign=True)
    del checkpoint
    if any(value.is_meta for value in list(unet.parameters())+list(unet.buffers())):
        raise RuntimeError('Checkpoint left unresolved model tensors')
    unet=unet.half();vae.enable_slicing()
    pipeline=pipeline_module.LipsyncPipeline(vae=vae,audio_encoder=audio,unet=unet,scheduler=scheduler).to('mps')
    (out/'request.json').write_text(json.dumps({**expected,'resolution':resolution,'source_sha256':fingerprints},indent=2))
    pipeline.avatar_chunk_cache=str(chunk_folder)
    print(f'Reusing {len(cached_chunks)} completed chunks',flush=True)
    original_loop=pipeline.loop_video
    def cache_alignment(chunks,frames):
        result=original_loop(chunks,frames)
        _,_,boxes,matrices=result
        np.savez(out/'alignment.npz',affine=np.stack([m.detach().cpu().numpy().reshape(2,3) for m in matrices]),boxes=np.asarray(boxes))
        return result
    pipeline.loop_video=cache_alignment
    original_decode=pipeline.decode_latents
    chunk_index=0
    def cache_decode(latents):
        nonlocal chunk_index
        chunk_index=pipeline._avatar_chunk_index
        decoded=original_decode(latents)
        if not torch.isfinite(decoded).all():
            raise RuntimeError('Decoder produced non-finite pixels; refusing to save a corrupted chunk')
        temporary=chunk_folder/f'{chunk_index:04d}.tmp'
        with temporary.open('wb') as stream:
            np.savez(stream,faces=decoded.detach().cpu().numpy())
        temporary.replace(chunk_folder/f'{chunk_index:04d}.npz')
        chunk_index+=1
        return decoded
    pipeline.decode_latents=cache_decode
    # Cache generated aligned faces once, enabling compositor comparisons without
    # repeating expensive diffusion. Preserve the upstream export as the baseline.
    original_restore=pipeline.restore_video
    def cache_restore(faces,video_frames,boxes,affine_matrices):
        np.savez(out/'restoration.npz',
                 faces=faces.detach().cpu().numpy(),
                 affine=np.stack([matrix.detach().cpu().numpy().reshape(2,3) for matrix in affine_matrices]),
                 boxes=np.asarray(boxes))
        return original_restore(faces,video_frames,boxes,affine_matrices)
    pipeline.restore_video=cache_restore
    os.chdir(out)
    torch.manual_seed(1247);start=time.monotonic()
    sys.path.insert(0,str(ROOT/'backend'))
    from mps_attention import bounded_attention, bounded_feed_forward
    torch.mps.set_per_process_memory_fraction(.7)
    observations={'denoising_steps_completed':len(cached_chunks)*args.steps,'resumed_chunks':len(cached_chunks),'max_observed_driver_bytes':0,'max_observed_tensor_bytes':0}
    def progress(step,timestep,latents):
        observations['denoising_steps_completed']+=1
        observations['max_observed_driver_bytes']=max(observations['max_observed_driver_bytes'],torch.mps.driver_allocated_memory())
        observations['max_observed_tensor_bytes']=max(observations['max_observed_tensor_bytes'],torch.mps.current_allocated_memory())
        state={**observations,'elapsed':time.monotonic()-start,'status':'rendering'}
        temporary=out/'progress.tmp'
        temporary.write_text(json.dumps(state,indent=2));temporary.replace(out/'progress.json')
    with bounded_attention(), bounded_feed_forward(unet):
        pipeline(video_path=str(source_path),audio_path=str(out/'analysis.wav'),video_out_path=str(out/'candidate.mp4'),num_frames=16,num_inference_steps=args.steps,guidance_scale=1.5,weight_dtype=torch.float16,width=resolution,height=resolution,mask_image_path=str(checkout/'latentsync/utils/mask.png'),temp_dir=str(out/'temp'),callback=progress,callback_steps=1)
    # Remux the original audio, never the inference-only resampled audio.
    subprocess.run([ff,'-y','-v','error','-i',str(out/'candidate.mp4'),'-i',args.audio,'-map','0:v','-map','1:a','-t',str(args.seconds),'-c:v','libx264','-crf','16','-frames:v',str(round(args.seconds*25)),'-c:a','pcm_f32le',str(out/'review.mov')],check=True)
    subprocess.run([ff,'-v','error','-i',str(out/'review.mov'),'-f','null','-'],check=True)
    report={'backend':'LatentSync '+args.version,'device':'mps','seconds':args.seconds,'steps':args.steps,'elapsed':time.monotonic()-start,'source_video':args.video,'source_audio':args.audio,'resolution':resolution,'seed':1247,'memory_fraction':.7,'attention_query_chunk':256,'feed_forward_token_chunk':256,'quality_passed':False,'review_required':True}
    report.update(observations)
    (out/'progress.json').write_text(json.dumps({**report,'status':'complete'},indent=2))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'backend'))
    from inference_lease import GPUInferenceLease
    with GPUInferenceLease(on_wait=lambda:print('Waiting for the active local render',flush=True)):
        main()
