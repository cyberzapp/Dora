"""Local CPU SyncNet timing diagnostic, with MediaPipe face crops.

Uses the SyncNet architecture vendored by LatentSync from Joon Son Chung's MIT
licensed syncnet_python. Crop replacement means scores must be calibrated on
real footage and deliberate offsets; they are not paper-comparable benchmarks.
"""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.local/matplotlib')
sys.path.insert(0,str(ROOT/'.local/eval-deps'))
import cv2
import imageio_ffmpeg
import numpy as np
import python_speech_features
import torch
from scipy.ndimage import median_filter
from scipy.io import wavfile


def box_iou(left,right):
    x0=max(left[0],right[0]);y0=max(left[1],right[1])
    x1=min(left[0]+left[2],right[0]+right[2]);y1=min(left[1]+left[3],right[1]+right[3])
    intersection=max(0,x1-x0)*max(0,y1-y0)
    return intersection/(left[2]*left[3]+right[2]*right[3]-intersection)


def summarize_distances(distances, search_frames):
    minimum,index=distances.min(0)
    offset=search_frames-int(index)
    return dict(estimated_offset_frames=offset, estimated_offset_seconds=offset/25,
                distance=float(minimum), confidence=float(distances.median()-minimum),
                search_boundary_hit=abs(offset)==search_frames)


def supported_timing(visual, sound, search_frames, clip_start, samples):
    """Compare every offset over the same visual windows using only real audio."""
    positions=torch.arange(search_frames,len(visual)-search_frames)
    if len(positions)<20:
        raise ValueError('Common support requires at least 20 complete visual windows')
    offsets=torch.arange(-search_frames,search_frames+1)
    sound_windows=sound[positions[:,None]+offsets[None,:]]
    distances=torch.nn.functional.pairwise_distance(visual[positions,None,:],sound_windows)
    chunks=[]
    # Use adjacent visual intervals, combining a short final interval with its
    # neighbor so every estimate has at least 20 five-frame visual windows.
    for indices in torch.tensor_split(torch.arange(len(positions)),max(1,len(positions)//20)):
        first=int(positions[indices[0]])
        last=int(positions[indices[-1]])+1
        interval=samples[first*640:last*640].astype(np.float64)/32768
        chunks.append(dict(first_window_seconds=clip_start+first/25,
                           end_window_seconds=clip_start+last/25,
                           visual_windows=len(indices),
                           zero_offset_audio_rms=float(np.sqrt(np.mean(interval**2))),
                           **summarize_distances(distances[indices].mean(0),search_frames)))
    return dict(visual_windows=len(positions),
                **summarize_distances(distances.mean(0),search_frames), intervals=chunks,
                interpretation='Equal real audio support across offsets. Interval times denote visual-window starts. '
                               'Low confidence, low speech energy and search-edge minima require playback review.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--audio',help='Optional separate audio for a mismatch control')
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--start',type=float,default=0,help='Start the same interval in both streams')
    parser.add_argument('--audio-shift',type=float,default=0,help='Positive values delay audio; calibration only')
    parser.add_argument('--output',required=True)
    parser.add_argument('--track-primary-face',action='store_true',help='Follow the single initial presenter by box overlap; reject missing or ambiguous matches')
    parser.add_argument('--common-support',action='store_true',
                        help='Also score whole-clip and local offsets with real audio at every candidate shift')
    parser.add_argument('--search-frames',type=int,default=15,
                        help='Maximum offset in either direction at 25 fps')
    parser.add_argument('--short-diagnostic', action='store_true',
                        help='Allow short, low-confidence local comparisons without common support')
    args=parser.parse_args()
    minimum_seconds = .8 if args.short_diagnostic else 2
    if not minimum_seconds <= args.seconds <= 15:
        parser.error(f'Use a {minimum_seconds}–15 second diagnostic')
    if args.short_diagnostic and (args.common_support or args.search_frames > 2):
        parser.error('Short diagnostics require an offset search of at most two frames without common support')
    if not 1 <= args.search_frames <= 15:
        parser.error('--search-frames must be between 1 and 15')
    if args.common_support and round(args.seconds*25)<25+2*args.search_frames:
        parser.error('Clip too short for common support at the requested search range')
    if not np.isfinite(args.start) or args.start<0:parser.error('Start must be a finite nonnegative time')
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'vendor/LatentSync/checkpoints-evaluation/syncnet_v2.model'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442':
        raise RuntimeError('Unverified evaluation checkpoint')
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('local_syncnet_model',ROOT/'vendor/LatentSync/eval/syncnet/syncnet.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.S().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix='syncnet-',dir=output.parent) as temporary:
        temporary=Path(temporary)
        video=temporary/'video.mp4';audio=temporary/'audio.wav'
        interval=['-ss',str(args.start),'-t',str(args.seconds)]
        subprocess.run([ff,'-y','-v','error','-i',args.video,*interval,'-an','-vf','fps=25',str(video)],check=True)
        subprocess.run([ff,'-y','-v','error','-i',args.audio or args.video,*interval,'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(audio)],check=True)
        import mediapipe as mp
        capture=cv2.VideoCapture(str(video));boxes=[];previous=None;ignored=[]
        with mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=.5) as detector:
            while True:
                ok,frame=capture.read()
                if not ok:break
                result=detector.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                detections=list(result.detections or [])
                def region(detection):
                    b=detection.location_data.relative_bounding_box
                    return b.xmin,b.ymin,b.width,b.height
                matches=([d for d in detections if box_iou(region(d),previous)>=.3]
                         if args.track_primary_face and previous is not None else detections)
                if len(matches)!=1:
                    capture.release()
                    raise RuntimeError(f'Timing diagnostic requires one continuously visible face (frame {len(boxes)}, found {len(detections)}, matched {len(matches)})')
                if len(detections)>1:ignored.append(dict(frame=len(boxes),ignored_detections=len(detections)-1))
                previous=region(matches[0])
                h,w=frame.shape[:2];box=matches[0].location_data.relative_bounding_box
                boxes.append([(box.xmin+box.width/2)*w,(box.ymin+box.height/2)*h,max(box.width*w,box.height*h)/2])
        capture.release()
        boxes=median_filter(np.asarray(boxes),size=(13,1),mode='nearest')
        capture=cv2.VideoCapture(str(video));crops=[]
        for x,y,size in boxes:
            ok,frame=capture.read()
            if not ok:raise RuntimeError('Frame decode failed during face cropping')
            pad=int(size*1.8);frame=np.pad(frame,((pad,pad),(pad,pad),(0,0)),constant_values=110)
            x+=pad;y+=pad
            crop=frame[int(y-size):int(y+size*1.8),int(x-size*1.4):int(x+size*1.4)]
            crops.append(cv2.resize(crop,(224,224)))
        capture.release()
        rate,samples=wavfile.read(audio)
        if abs(len(crops)-round(args.seconds*25))>1 or abs(len(samples)-round(args.seconds*rate))>640:
            raise RuntimeError(f'Timing diagnostic interval was not completely decoded (video {len(crops)} frames, audio {len(samples)} samples at {rate} Hz; requested {args.seconds}s)')
        shift=round(args.audio_shift*rate)
        if shift>0:samples=np.pad(samples,(shift,0))[:len(samples)]
        elif shift<0:samples=np.pad(samples, (0,-shift))[-shift:len(samples)-shift]
        mfcc=python_speech_features.mfcc(samples,rate).T.astype(np.float32)
        images=torch.from_numpy(np.stack(crops).transpose(3,0,1,2)).float().unsqueeze(0)
        audio_features=torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)
        last=min(len(crops),len(samples)//640)-5
        visual=[];sound=[]
        with torch.inference_mode():
            for start in range(0,last,2):
                indices=range(start,min(last,start+2))
                visual.append(model.forward_lip(torch.cat([images[:,:,i:i+5] for i in indices])).cpu())
                sound.append(model.forward_aud(torch.cat([audio_features[:,:,:,i*4:i*4+20] for i in indices])).cpu())
        visual=torch.cat(visual);sound=torch.cat(sound)
        window=args.search_frames;padded=torch.nn.functional.pad(sound,(0,0,window,window))
        distances=torch.stack([torch.nn.functional.pairwise_distance(visual[i:i+1].expand(2*window+1,-1),padded[i:i+2*window+1]) for i in range(len(visual))]).mean(0)
        timing=summarize_distances(distances,window)
        report={'video':str(Path(args.video).resolve()),'audio':str(Path(args.audio or args.video).resolve()),
                'start_seconds':args.start,'seconds':args.seconds,'introduced_audio_delay_seconds':args.audio_shift,'device':'cpu',
                **timing,
                'offset_search_frames':window,'checkpoint_sha256':digest,'cropper':'MediaPipe, median-smoothed SyncNet-style crop',
                'primary_face_tracking':args.track_primary_face,'tracking_iou_threshold':.3 if args.track_primary_face else None,
                'additional_detections_excluded':ignored,'missing_detections_interpolated':False,
                'quality_passed':False,'short_diagnostic':args.short_diagnostic,
                'limitation':'Timing diagnostic only; short windows have limited support and use padded offsets.'}
        if args.common_support:
            report['common_support']=supported_timing(visual,sound,window,args.start,samples)
        output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
