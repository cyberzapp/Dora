"""Prepare a real-camera conditioning clip, explicitly not a generated result.

Uses one fixed head crop across the performance. The old recorded mouth motion
is intentionally paired with the new voice-over as input to a dubbing test; it
is never presented as an already lip-synchronized avatar.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from download_model_manifest import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['video','audio','output']:parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    parser.add_argument('--start',type=float,default=1)
    parser.add_argument('--frames',type=int,choices=(49,121),default=49)
    parser.add_argument('--consent',action='store_true')
    parser.add_argument('--kj-provenance', type=Path,
        help='Identify an existing KJ photo-animation video as derived conditioning')
    parser.add_argument('--fixed-crop', nargs=4, type=int, metavar=('LEFT','TOP','RIGHT','BOTTOM'),
        help='Reuse the same crop across a controlled comparison')
    args=parser.parse_args()
    if not args.consent:parser.error('Likeness consent is required')
    if not np.isfinite(args.start) or args.start<0:parser.error('Use a nonnegative source start time')
    derived = None
    if args.kj_provenance:
        path = args.kj_provenance.resolve()
        prior = json.loads(path.read_text())
        if prior.get('backend') != 'ComfyUI-LivePortraitKJ actual node classes' or args.video != path.with_name('motion.mp4'):
            raise ValueError('Expected the KJ motion video alongside its original provenance')
        photo = Path(prior['photo']).resolve()
        derived = dict(backend=prior['backend'], provenance_sha256=digest(path),
                       photo=str(photo), photo_sha256=digest(photo))
        if prior.get('held_source_mouth_frames') == prior.get('frames') and prior.get('frames', 0) > 0:
            derived['driving_mouth_keypoints_held'] = True
    args.output.mkdir(parents=True,exist_ok=False)
    source_video_sha=digest(args.video);source_audio_sha=digest(args.audio);ff=imageio_ffmpeg.get_ffmpeg_exe()
    normalized=args.output/'camera-normalized.mkv';seconds=args.frames/25
    # FFmpeg applies the camera's display rotation once, before fixed cropping.
    subprocess.run([ff,'-v','error','-i',str(args.video),'-ss',str(args.start),'-an',
        '-vf','fps=25,scale=1280:720:force_original_aspect_ratio=decrease',
        '-frames:v',str(args.frames),'-c:v','ffv1',str(normalized)],check=True)
    capture=cv2.VideoCapture(str(normalized));boxes=[]
    width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        with mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=.5) as detector:
            while True:
                ok,frame=capture.read()
                if not ok:break
                result=detector.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if len(result.detections or [])!=1:raise ValueError(f'Reference needs one visible face at frame {len(boxes)}')
                b=result.detections[0].location_data.relative_bounding_box
                boxes.append([b.xmin*width,b.ymin*height,b.width*width,b.height*height])
    finally:capture.release()
    if len(boxes)!=args.frames:raise ValueError('Camera does not cover the requested interval')
    box=np.median(np.asarray(boxes),axis=0);side=min(width,height,round(max(box[2:])*2))
    cx=box[0]+box[2]/2;cy=box[1]+box[3]/2
    left=max(0,min(width-side,round(cx-side/2)));top=max(0,min(height-side,round(cy-side/2)))
    if args.fixed_crop:
        left, top, right, bottom = args.fixed_crop
        side = right-left
        if side != bottom-top or not (0<=left<right<=width and 0<=top<bottom<=height):
            raise ValueError('Fixed crop must be square and inside the normalized video')
    if any(x<left or y<top or x+w>left+side or y+h>top+side for x,y,w,h in boxes):
        raise ValueError('The fixed camera crop loses the face; choose a steadier interval')
    movie=args.output/'camera-motion-reference.mkv'
    subprocess.run([ff,'-v','error','-i',str(normalized),'-i',str(args.audio),
        '-map','0:v:0','-map','1:a:0','-t',str(seconds),'-vf',f'crop={side}:{side}:{left}:{top}:exact=1,scale=512:512:flags=lanczos',
        '-c:v','ffv1','-pix_fmt','bgr0','-c:a','pcm_f32le',str(movie)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_export.py'),str(args.audio),str(movie),
        '--seconds',str(seconds),'--expected-frames',str(args.frames),'--fps','25'],check=True)
    if digest(args.video)!=source_video_sha or digest(args.audio)!=source_audio_sha:raise ValueError('Original media changed')
    record=dict(generated_video=False,training_performed=False,reference_only=True,
        description='Real camera motion with new audio; old mouth motion is not a lip-sync result',
        video=str(args.video),source_video_sha256=source_video_sha,source_start_seconds=args.start,
        audio=str(args.audio),source_audio_sha256=source_audio_sha,originals_unchanged=True,
        normalized_video_sha256=digest(normalized),fixed_crop=[left,top,left+side,top+side],crop_exact=True,
        normalized_size=[width,height],output_size=[512,512],frames=args.frames,fps=25,seconds=seconds,
        source_motion_retimed=False,source_resampled_to_25fps=True,per_frame_crop_tracking=False,
        output=str(movie),output_sha256=digest(movie),original_decoded_audio_verified=True,quality_passed=False)
    if derived:
        record.update(conditioning_kind='kj-photo-motion', derived_conditioning=derived,
                      description='Derived KJ photo animation used as visual dubbing input')
    (args.output/'reference.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
