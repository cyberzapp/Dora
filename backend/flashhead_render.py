"""Experimental native-app worker using the genuinely tested FlashHead recipe.

Full-length/recovery integration requires actual execution before release. Visual
quality is never inferred from file integrity or a successful process exit.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from worker import Job,Cancelled,validate_request,atomic_json,ffmpeg
from inference_lease import InferenceLease,GPUInferenceLease
from flashhead_cache import ChunkCache,sha
from flashhead_timeline import plan,NEW_FRAMES
from head_composite import source_face_bounds,crop_box


def verify_weights(job):
    for name,folder in [('flashhead-pro','flashhead-models'),('flashhead-audio','flashhead-audio')]:
        manifest=json.loads((ROOT/'backend/model_manifests'/f'{name}.json').read_text())
        for entry in manifest['files']:
            job.check();path=ROOT/'.local'/folder/entry['path']
            if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise ValueError('Local model file is missing or damaged: '+entry['path'])


def raw_frames(cache,total):
    import numpy as np
    emitted=0
    for index,record in enumerate(cache.records):
        cache.verify(index)
        values=np.load(cache.paths(index)[0],mmap_mode='r',allow_pickle=False)
        for frame in values[:record['append_frames']]:
            if emitted>=total:raise ValueError('Generated cache exceeds the requested timeline')
            yield frame;emitted+=1
    if emitted!=total:raise ValueError('Generated cache does not cover the voice-over')


def motion_corrections(cache,total,job):
    import numpy as np
    output=job.directory/'head-stabilization.npz'
    job.status(.88,'Checking generated head motion')
    job.run([sys.executable,ROOT/'backend/flashhead_motion.py','--cache',cache.directory,
             '--frames',total,'--output',output])
    corrections=np.load(output,allow_pickle=False)['transforms']
    if corrections.shape!=(total,2,3) or not np.isfinite(corrections).all():raise ValueError('Invalid stabilization result')
    return corrections


def run(request,job):
    validate_request(request)
    if request.get('source_video') or request.get('motion'):
        raise ValueError('FlashHead currently uses the outfit photo and voice-over. Personal reference-motion conditioning is not implemented.')
    bounds=source_face_bounds(request)
    width,height=request.get('width',1920),request.get('height',1080)
    if any(type(value) is not int or value<128 or value>3840 or value%2 for value in (width,height)):
        raise ValueError('Use even output dimensions from 128 to 3840 pixels')
    stabilize=request.get('stabilize_head',False)
    if type(stabilize) is not bool:raise ValueError('stabilize_head must be Boolean')
    seed=request.get('seed',42)
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Invalid render seed')
    job.status(.01,'Checking local models and source media');verify_weights(job)
    photo_hash,audio_hash=sha(Path(request['photo'])),sha(Path(request['audio']))
    import numpy as np
    import soundfile as sf
    from PIL import Image,ImageOps
    from flashhead_layout import Layout
    photo=ImageOps.exif_transpose(Image.open(request['photo'])).convert('RGB')
    box=crop_box(photo,bounds,2.0);layout=Layout(photo,(width,height),bounds,box)
    if shutil.disk_usage(job.directory).free<2*1024**3:raise ValueError('Free at least 2 GB of storage before rendering')
    job.status(.03,'Preparing the original voice-over locally')
    native_audio=job.directory/'voice-native.wav';analysis=job.directory/'voice-analysis.wav'
    # Decode at native rate/channels to determine exact original sample duration.
    # Neither this file nor the mono analysis file replaces the export's audio.
    job.run([ffmpeg(),'-v','error','-y','-i',request['audio'],'-t','301','-vn',
             '-map_metadata','-1','-c:a','pcm_f32le',native_audio])
    info=sf.info(native_audio)
    if not .5<=info.frames/info.samplerate<=300:raise ValueError('Use a voice-over between 0.5 and 300 seconds')
    with sf.SoundFile(native_audio) as source:
        for values in source.blocks(blocksize=65536,dtype='float32',always_2d=True):
            job.check()
            if not np.isfinite(values).all():raise ValueError('The voice-over contains invalid audio samples')
    job.run([ffmpeg(),'-v','error','-y','-i',native_audio,'-ac','1','-ar','16000',
             '-map_metadata','-1','-c:a','pcm_f32le',analysis])
    speech,sample_rate=sf.read(analysis,dtype='float32')
    if sample_rate!=16000 or speech.ndim!=1 or not np.isfinite(speech).all():raise ValueError('Voice analysis failed')
    total_frames=(info.frames*25+info.samplerate-1)//info.samplerate
    # Plan against the exact original duration, avoiding a resampling-rounding
    # loss at an exact frame boundary. Analysis padding is inference-only.
    _,windows=plan(len(speech),output_frames=total_frames)
    atomic_json(job.directory/'flashhead-timeline.json',dict(frames=total_frames,fps=25,
        source_audio_frames=info.frames,source_audio_rate=info.samplerate,windows=[asdict(w) for w in windows]))
    import importlib.metadata
    identity=dict(photo_sha256=photo_hash,audio_sha256=audio_hash,analysis_sha256=sha(analysis),
        frames=total_frames,crop=list(box),seed=seed,recipe='pro4-shift5-history8-carry5-v1',
        runtime={name:importlib.metadata.version(name) for name in ('torch','transformers','diffusers','numpy','safetensors')},
        code={name:sha(ROOT/name) for name in ['backend/flashhead_engine.py','backend/flashhead_audio.py',
            'backend/flashhead_timeline.py','research/flashhead_mps.py',
            'vendor/SoulX-FlashHead/flash_head/src/modules/flash_head_model.py',
            'vendor/SoulX-FlashHead/flash_head/wan/modules/vae.py',
            'vendor/SoulX-FlashHead/flash_head/audio_analysis/wav2vec2.py',
            'vendor/SoulX-FlashHead/flash_head/audio_analysis/torch_utils.py',
            'vendor/SoulX-FlashHead/flash_head/utils/utils.py']},
        manifests={name:sha(ROOT/'backend/model_manifests'/name) for name in ['flashhead-pro.json','flashhead-audio.json']})
    estimate=len(windows)*(28*512*512*3+33*5*12*768*4)+width*height*total_frames//4+2*1024**3
    if shutil.disk_usage(job.directory).free<estimate:raise ValueError(f'Free at least {math.ceil(estimate/1024**3)} GB to finish this render')
    from flashhead_audio import contexts
    with ChunkCache(job.directory/'flashhead-cache',identity) as cache:
        performance_path=cache.directory/'performance.json'
        performance=json.loads(performance_path.read_text()) if performance_path.exists() else {}
        if performance.get('request')!=cache.fingerprint:performance={}
        if len(cache.records)>len(windows):raise ValueError('Cached timeline exceeds the current voice-over')
        for index,record in enumerate(cache.records):
            if record['append_frames']!=windows[index].append_frames:raise ValueError('Cached timeline coverage changed')
        job.status(.06,'Preparing speech features')
        feature_paths=contexts(speech,len(windows),cache.directory/'speech',cache.fingerprint,
            ROOT/'.local/flashhead-audio',job.check,lambda done,count:job.status(.06+.04*done/count,'Preparing speech features'))
        del speech
        engine=None
        try:
            if len(cache.records)<len(windows):
                from flashhead_engine import Engine
                import torch
                current=len(cache.records)
                def progress(stage,**details):
                    portion=.65*details['step']/4 if stage=='Generating speech' else (
                        .65+.3*details['frames']/33 if stage=='Decoding video' else 0)
                    job.status(.1+.76*(current+portion)/len(windows),
                        f'{stage} • section {current+1} of {len(windows)}')
                recovery=cache.paths(len(cache.records)-1)[1].read_bytes() if cache.records else None
                engine=Engine(photo.crop(box),ROOT/'.local/flashhead-models',seed,recovery,job.check,progress)
                for current in range(len(cache.records),len(windows)):
                    job.check();context=torch.from_numpy(np.load(feature_paths[current],allow_pickle=False))
                    frames=engine.generate(context)
                    cache.commit(current,frames,engine.state(),windows[current].append_frames)
                    observed=engine.observations()
                    for key in ('max_observed_driver_bytes','max_observed_tensor_bytes'):
                        observed[key]=max(observed[key],performance.get(key,0))
                    performance=dict(request=cache.fingerprint,committed_windows=current+1,**observed)
                    atomic_json(performance_path,performance)
                    del frames,context
        finally:
            if engine is not None:engine.close()
        corrections=motion_corrections(cache,total_frames,job) if stabilize else None
        job.status(.92,'Exporting video with the original audio')
        partial=job.directory/'flashhead-output.partial.mov';output=job.directory/'output.mov'
        process=subprocess.Popen([ffmpeg(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width}x{height}','-r','25','-i','-','-i',request['audio'],'-map','0:v','-map','1:a',
            '-t',str(total_frames/25),'-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],
            stdin=subprocess.PIPE,start_new_session=True)
        try:
            for index,frame in enumerate(raw_frames(cache,total_frames)):
                job.check();composed=layout.frame(frame,None if corrections is None else corrections[index])
                process.stdin.write(composed.tobytes())
                if index%25==0:job.status(.92+.05*index/total_frames,'Exporting video with the original audio')
            process.stdin.close()
            if process.wait():raise RuntimeError('Video encoding failed')
        finally:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            if not process.stdin.closed:process.stdin.close()
        job.status(.98,'Verifying every frame and original audio')
        job.run([sys.executable,ROOT/'scripts/verify_export.py',request['audio'],partial,
                 '--expected-frames',total_frames,'--fps','25'])
        if photo_hash!=sha(Path(request['photo'])) or audio_hash!=sha(Path(request['audio'])):
            raise ValueError('Source media changed during rendering')
        job.check();partial.replace(output)
        provenance=dict(engine='FlashHead Pro',generated_video=True,quality_passed=False,training_performed=False,
            source_photo_sha256=photo_hash,source_audio_sha256=audio_hash,frames=total_frames,fps=25,
            output_size=[width,height],model_crop_size=[512,512],source_crop=list(box),framing=layout.mode,
            stabilize_head=stabilize,original_audio_verified=True,all_frames_verified=True,
            source_body_preserved_outside_head_before_encoding=True,request=identity)
        provenance['memory_observations']=performance or None
        provenance['native_launcher_network_policy']=request.get('network_policy','not-declared-by-launcher')
        provenance['export_implementation']={name:sha(ROOT/name) for name in
            ['backend/flashhead_render.py','backend/flashhead_layout.py','backend/head_composite.py',
             'backend/head_stabilization.py','backend/flashhead_motion.py']}
        atomic_json(job.directory/'provenance.json',provenance)
        job.status(1,'Video rendered locally • visual review required',status='Completed',output=output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path);args=parser.parse_args()
    request=json.loads(args.request.read_text())
    directory=request.get('directory') or request.get('output_dir')
    if not directory:raise ValueError('Render directory is missing')
    job=Job(directory,attempt=request.get('attempt'))
    def cancel(*_):job.cancelled=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        with InferenceLease(job.directory/'.flashhead-job.lock',job.check,
                            lambda:job.status(0,'Waiting for this render to finish')):
            with GPUInferenceLease(job.check,
                                lambda:job.status(0,'Waiting for another local render')):run(request,job)
        return 0
    except Cancelled:
        job.status(0,'Render cancelled • completed sections retained',status='Cancelled');return 2
    except Exception as error:
        traceback.print_exc();job.status(0,str(error),status='Failed');return 1


if __name__=='__main__':sys.exit(main())
