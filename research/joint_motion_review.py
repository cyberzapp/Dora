"""Find review moments where head, eye, and mouth estimates change during speech."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from scipy.spatial.transform import Rotation


def motion_events(track, audio_energy):
    samples = {row['frame']: row for row in track['samples']}
    gaze_names = [k for k in samples[min(samples)]['blendshapes'] if k.startswith('eyeLook')]
    rotations = {n: Rotation.from_euler('xyz',row['relative_xyz_degrees'],degrees=True) for n,row in samples.items()}
    deltas = {n: np.rad2deg((rotations[n]*rotations[n-1].inv()).as_rotvec()) for n in rotations if n-1 in rotations}
    events = {'head_reversals':[], 'gaze_changes_with_eyes_open':[], 'eyelid_asymmetry_changes':[]}
    for n,row in samples.items():
        if n-1 not in samples:continue
        a,b = samples[n-1]['blendshapes'],row['blendshapes']
        blink_keys = ('eyeBlinkLeft','eyeBlinkRight')
        entry = dict(frame=n, seconds=n/25,
                     estimated_pose_step_degrees=float(np.linalg.norm(deltas[n])),
                     bilateral_closure=[b[k] for k in blink_keys], jaw_open=b['jawOpen'],
                     mouth_close=b['mouthClose'], mouth_pucker=b['mouthPucker'], audio_rms=float(audio_energy[n]))
        if n-1 in deltas:
            previous,current = deltas[n-1],deltas[n]
            speed_a,speed_b = np.linalg.norm(previous),np.linalg.norm(current)
            if min(speed_a,speed_b)>0.4:
                cosine=float(previous@current/(speed_a*speed_b))
                if cosine<-.5:
                    events['head_reversals'].append(dict(**entry,strength=float(-cosine*min(speed_a,speed_b)),direction_cosine=cosine))
        if max(c[k] for c in (a,b) for k in blink_keys)<.35 and gaze_names:
            change=max(abs(b[k]-a[k]) for k in gaze_names)
            events['gaze_changes_with_eyes_open'].append(dict(**entry,strength=change))
        change=abs((b['eyeBlinkLeft']-b['eyeBlinkRight'])-(a['eyeBlinkLeft']-a['eyeBlinkRight']))
        events['eyelid_asymmetry_changes'].append(dict(**entry,strength=change))
    return ({name: sorted(rows,key=lambda r:r['strength'],reverse=True)[:8] for name,rows in events.items()},
            {name: len(rows) for name,rows in events.items()})


def audio_rms(source,count):
    pcm=subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(source),'-vn',
        '-t',str(count/25),'-ac','1','-ar','16000','-f','f32le','-'])
    samples=np.frombuffer(pcm,dtype=np.float32)
    if len(samples)!=count*640 or not np.isfinite(samples).all():raise ValueError('Audio does not cover the review timeline')
    return np.sqrt(np.mean(samples.astype(np.float64).reshape(count,640)**2,axis=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expressions',type=Path,required=True)
    parser.add_argument('--audio',type=Path,help='Shared original voice-over for a matched comparison; otherwise each clip uses its own audio')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a fresh review output')
    tracks=json.loads(args.expressions.read_text())['videos']
    if not tracks or len({v['frames_at_25fps'] for v in tracks})!=1:raise ValueError('Matched expression timelines are required')
    count=tracks[0]['frames_at_25fps']
    shared=audio_rms(args.audio,count) if args.audio else None
    reports=[]
    for track in tracks:
        with Path(track['path']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=track['sha256']:raise ValueError('Compared video changed')
        points,counts=motion_events(track,shared if shared is not None else audio_rms(track['path'],count))
        reports.append(dict(video=track['path'],missing_frames=track['missing_frames'],
                            review_points=points,detected_event_counts=counts))
    result=dict(videos=reports,frames=count,shared_audio=args.audio is not None,quality_passed=False,
                interpretation='Ranks moments to watch; thresholds are diagnostic and uncalibrated. '
                'Legitimate fast motion, winks, or tracking noise can be flagged. RMS is not phoneme alignment. '
                'When audio differs, same-time frames are not equivalent speech events.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
