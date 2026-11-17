"""Diagnostic: actual Dub-It mouth over the unchanged conditioning frames.

The generated mouth comes from the verified Dub-It output. Before encoding,
every pixel outside a fixed mouth mask is copied from the corresponding real
camera or LivePortraitKJ photo-motion frame.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from scipy.ndimage import median_filter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'backend')]
from download_model_manifest import digest
from ltx_video_export import export_original_audio

OUTER_LIPS=[61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
ANCHORS=[33,133,362,263,168,6,197]
EYES=[33,133,362,263]


def read(path, expected_frames):
    capture=cv2.VideoCapture(str(path));frames=[];points=[]
    try:
        if abs(capture.get(cv2.CAP_PROP_FPS)-25)>.01:raise ValueError('Expected 25 fps source')
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
                refine_landmarks=True,min_detection_confidence=.5) as mesh:
            while True:
                ok,frame=capture.read()
                if not ok:break
                result=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if len(result.multi_face_landmarks or [])!=1:
                    raise ValueError(f'Face tracking failed at frame {len(frames)}; no invented landmarks allowed')
                h,w=frame.shape[:2]
                points.append(np.array([[p.x*w,p.y*h] for p in result.multi_face_landmarks[0].landmark],np.float32))
                frames.append(frame)
    finally:capture.release()
    if len(frames)!=expected_frames or any(frame.shape!=(512,512,3) for frame in frames):
        raise ValueError(f'This diagnostic requires {expected_frames} actual 512-square frames')
    return frames,points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,type=lambda p:Path(p).resolve())
    parser.add_argument('--output',required=True,type=lambda p:Path(p).resolve())
    parser.add_argument('--kj-photo-motion',action='store_true',
                        help='Keep the tracked eyes, brows and head from the KJ photo reference')
    args=parser.parse_args();started=time.monotonic()
    report=json.loads((args.candidate/'provenance.json').read_text())
    movie=args.candidate/'review.mov';reference=report['reference'];camera=Path(reference['output'])
    audio=Path(reference['audio'])
    if (report.get('generated_video') is not True or report.get('adapter_inference_performed') is not True
            or report.get('output_sha256')!=digest(movie) or reference['output_sha256']!=digest(camera)
            or reference['source_audio_sha256']!=digest(audio)):
        raise ValueError('Requires the unchanged actual Dub-It result and its verified inputs')
    frames=49
    if args.kj_photo_motion:
        from project_ltx_candidate import validate_kj_dub
        photo=Path(reference['derived_conditioning']['photo'])
        report,movie=validate_kj_dub(args.candidate,photo,audio)
        frames=report['frames']
        if reference['frames']!=frames:
            raise ValueError('The KJ reference and Dub-It clip must cover the same timeline')
    elif reference.get('conditioning_kind')=='kj-photo-motion':
        raise ValueError('Use --kj-photo-motion for KJ photo conditioning')
    camera_frames,camera_points=read(camera,frames)
    generated_frames,generated_points=read(movie,frames)
    transforms=[]
    for source,target in zip(generated_points,camera_points):
        matrix,inliers=cv2.estimateAffinePartial2D(source[ANCHORS],target[ANCHORS],method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():raise ValueError('Cannot align generated mouth to the real face')
        scale=np.hypot(matrix[0,0],matrix[1,0]);angle=np.arctan2(matrix[1,0],matrix[0,0])
        if not .8<scale<1.25 or abs(angle)>np.deg2rad(15) or np.max(np.abs(matrix[:,2]))>80:
            raise ValueError('Generated face drift exceeds this restrained alignment control')
        transforms.append([scale,angle,matrix[0,2],matrix[1,2]])
    transforms=median_filter(np.asarray(transforms),size=(5,1),mode='nearest')
    matrices=[];hulls=[]
    for parameters,source,target in zip(transforms,generated_points,camera_points):
        scale,angle,tx,ty=parameters;c=scale*np.cos(angle);s=scale*np.sin(angle)
        matrix=np.array([[c,-s,tx],[s,c,ty]],np.float32);matrices.append(matrix)
        moved=cv2.transform(source[OUTER_LIPS][None],matrix)[0]
        combined=np.concatenate([target[OUTER_LIPS],moved]);center=combined.mean(axis=0)
        hulls.append(cv2.convexHull(((combined-center)*1.45+center).astype(np.float32)))
    binary=np.zeros((512,512),np.uint8)
    for hull in hulls:cv2.fillConvexPoly(binary,np.round(hull).astype(np.int32),1)
    alpha=np.clip(cv2.distanceTransform(binary,cv2.DIST_L2,5)/8.,0,1).astype(np.float32)
    for points in camera_points:
        for x,y in points[EYES]:
            if alpha[int(np.clip(y,0,511)),int(np.clip(x,0,511))]!=0:
                raise ValueError('Mouth control overlaps the original eyes')
    if np.count_nonzero(alpha)>512*512*.18:raise ValueError('Mouth region is too broad for this diagnostic')
    args.output.mkdir(parents=True,exist_ok=False)
    cv2.imwrite(str(args.output/'mouth-mask.png'),np.round(alpha*255).astype(np.uint8))
    silent=args.output/'mouth-control-silent.mp4';ff=imageio_ffmpeg.get_ffmpeg_exe()
    encoder=subprocess.Popen([ff,'-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','512x512','-r','25',
        '-i','-','-an','-c:v','libx264','-crf','12','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
    try:
        for original,generated,matrix in zip(camera_frames,generated_frames,matrices):
            warped=cv2.warpAffine(generated,matrix,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            composite=original.copy();active=alpha>0;a=alpha[active,None]
            composite[active]=np.clip(np.rint(original[active]*(1-a)+warped[active]*a),0,255).astype(np.uint8)
            if not np.array_equal(composite[~active],original[~active]):raise ValueError('Camera pixels outside mouth changed')
            encoder.stdin.write(composite.tobytes())
        encoder.stdin.close()
        if encoder.wait():raise RuntimeError('Mouth-control encoding failed')
    finally:
        if encoder.poll() is None:encoder.kill();encoder.wait()
    output=export_original_audio(silent,audio,args.output/'review.mov',frames=frames,fps=25)
    evidence=dict(generated_mouth_from_actual_model=True,new_model_inference=False,
        source_generated_video=str(movie),source_generated_sha256=digest(movie),camera_input_sha256=digest(camera),
        source_generation_provenance_sha256=digest(args.candidate/'provenance.json'),
        conditioning_kind=reference.get('conditioning_kind','camera-motion'),
        source_photo_sha256=reference.get('derived_conditioning',{}).get('photo_sha256'),
        source_audio_sha256=digest(audio),original_audio_verified=True,frames=frames,fps=25,
        camera_head_and_eyes_preserved_before_encoding=True,global_frame_stabilization=False,
        fixed_mask=True,mask_active_pixels=int(np.count_nonzero(alpha)),mask_sha256=digest(args.output/'mouth-mask.png'),
        transforms=transforms.tolist(),output=str(output),output_sha256=digest(output),
        quality_passed=False,photo_avatar_feature_complete=False,elapsed_seconds=time.monotonic()-started,
        limitation='Source gestures may differ from the new speech. Blending and mouth timing require playback review. Pixel preservation is verified before codec compression.')
    (args.output/'provenance.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
