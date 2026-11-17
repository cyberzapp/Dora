"""Place verified LTX head frames on the unchanged real source body.

Research composition only: no new motion inference, body/outfit synthesis, eye
replacement, audio retiming or visual-quality approval.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research')]
from flashhead_layout import Layout
from inference_lease import GPUInferenceLease
from review_ltx_candidate import validate_candidate
from compare_flashhead_lead import digest,call


def pixels(movie):
    reader=imageio_ffmpeg.read_frames(str(movie),pix_fmt='rgb24')
    try:
        metadata=next(reader)
        if tuple(metadata['size'])!=(512,512):raise ValueError('Expected actual 512-square model frames')
        for data in reader:yield np.frombuffer(data,dtype=np.uint8).reshape(512,512,3)
    finally:reader.close()


def validate_kj_dub(candidate, photo, audio):
    report = json.loads((candidate/'provenance.json').read_text())
    reference = report['reference']
    derived = reference.get('derived_conditioning', {})
    official = json.loads((ROOT/'research/dubit-source-lock.json').read_text())
    movie = candidate/'review.mov'
    if (report.get('conditioning_kind') != 'kj-photo-motion' or report.get('generated_video') is not True
            or not report.get('adapter_inference_performed') or report.get('adapter') != official
            or report.get('fps') != 25 or report.get('frames') not in (49, 121)
            or set(report.get('adapter_binding_evidence', {})) != {'1', '2'}
            or report.get('output_sha256') != digest(movie)
            or derived.get('photo_sha256') != digest(photo)
            or reference.get('source_audio_sha256') != digest(audio)
            or digest(Path(reference['output'])) != reference['output_sha256']):
        raise ValueError('Expected completed KJ-conditioned Dub-It generation from the same real photo and audio')
    # The KJ worker fits the original photo before export, and reference
    # preparation scales the full frame again. Map the recorded crop back
    # onto the original photo before applying the native framing transform.
    with Image.open(photo) as source:
        sw, sh = ImageOps.exif_transpose(source).size
    width, height = reference['normalized_size']
    left, top, right, bottom = reference['fixed_crop']
    if not reference.get('crop_exact', False):
        # The existing reference was prepared from FFV1 yuv420p with FFmpeg's
        # default chroma-aligned crop. Round origins and dimensions separately.
        right = (left & ~1) + ((right-left) & ~1)
        bottom = (top & ~1) + ((bottom-top) & ~1)
        left, top = left & ~1, top & ~1
    report['crop'] = [round(left*sw/width), round(top*sh/height),
                      round(right*sw/width), round(bottom*sh/height)]
    report['seconds'] = report['frames']/report['fps']
    call('scripts/verify_export.py', audio, movie, '--seconds', report['seconds'],
         '--expected-frames', report['frames'], '--fps', report['fps'])
    return report, movie


def dubit_head_corrections(movie, frame_count):
    from dubit_mouth_control import ANCHORS, EYES, read
    from slow_head_path import transforms

    decoded, landmarks = read(movie, frame_count)
    del decoded
    samples = []
    reference = landmarks[0][ANCHORS]
    for index, points in enumerate(landmarks):
        matrix,_ = cv2.estimateAffinePartial2D(reference,points[ANCHORS],method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError(f'Unable to measure upper-face pose at frame {index}')
        roll = np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))
        center = points[EYES].mean(axis=0)/512
        samples.append([index,roll,*center])
    return transforms(samples,(512,512))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','native-reference','photo','audio','output']:
        parser.add_argument('--'+name,type=lambda x:Path(x).resolve(),required=True)
    parser.add_argument('--kj-dub', action='store_true', help='Compose a verified KJ-conditioned Dub-It result')
    parser.add_argument('--mouth-control',type=lambda x:Path(x).resolve(),
                        help='Compose the verified mouth-only comparison for the same KJ-conditioned Dub-It result')
    parser.add_argument('--smooth-head-path',action='store_true',
                        help='Test limited zero-phase rigid stabilization on the verified full-face Dub-It result')
    args=parser.parse_args()
    if args.mouth_control and not args.kj_dub:
        parser.error('--mouth-control requires --kj-dub')
    if args.smooth_head_path and (not args.kj_dub or args.mouth_control):
        parser.error('--smooth-head-path requires a full-face --kj-dub render')
    if args.output.exists() or args.output.with_suffix('.partial.mov').exists():raise ValueError('Retain previous output; choose a fresh path')
    validator = validate_kj_dub if args.kj_dub else validate_candidate
    report,movie=validator(args.candidate,args.photo,args.audio)
    generated_movie = movie
    mouth_control_provenance = None
    if args.mouth_control:
        mouth_control_provenance = args.mouth_control/'provenance.json'
        control = json.loads(mouth_control_provenance.read_text())
        movie = args.mouth_control/'review.mov'
        if (control.get('generated_mouth_from_actual_model') is not True
                or control.get('camera_head_and_eyes_preserved_before_encoding') is not True
                or control.get('source_generated_sha256') != report['output_sha256']
                or control.get('source_generation_provenance_sha256') != digest(args.candidate/'provenance.json')
                or control.get('camera_input_sha256') != report['reference']['output_sha256']
                or control.get('source_photo_sha256') != report['reference']['derived_conditioning']['photo_sha256']
                or control.get('source_audio_sha256') != report['reference']['source_audio_sha256']
                or control.get('frames') != report['frames'] or control.get('fps') != report['fps']
                or control.get('output_sha256') != digest(movie)):
            raise ValueError('The mouth-only comparison does not match the verified Dub-It generation')
        call('scripts/verify_export.py',args.audio,movie,'--seconds',report['seconds'],
             '--expected-frames',report['frames'],'--fps',report['fps'])
    request_path=args.native_reference/'request.json'
    native_request=json.loads(request_path.read_text())
    native_provenance_path=args.native_reference/'provenance.json'
    native=json.loads(native_provenance_path.read_text())
    photo=ImageOps.exif_transpose(Image.open(args.photo)).convert('RGB')
    native_source=Path(native_request['photo'])
    if (digest(native_source)!=native['source_photo_sha256'] or
            not np.array_equal(np.asarray(photo),np.asarray(ImageOps.exif_transpose(Image.open(native_source)).convert('RGB')))):
        raise ValueError('Native framing reference used a different source photograph')
    bounds=native_request['face_bounds'];size=(native_request['width'],native_request['height'])
    if len(bounds)!=4 or not np.isfinite(bounds).all() or any(v<0 or v>1 for v in bounds) or min(bounds[2:])<=0:
        raise ValueError('Invalid source face bounds')
    layout=Layout(photo,size,bounds,report['crop'])
    if list(size)!=native['output_size'] or layout.mode!=native['framing']:
        raise ValueError('Native framing reference differs from recorded source layout')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    corrections, head_path = None, None
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active local inference before segmentation',flush=True)):
        import mediapipe as mp
        if args.smooth_head_path:
            corrections,head_path=dubit_head_corrections(movie,report['frames'])
        support=np.zeros((512,512),dtype=np.float32)
        source=np.asarray(photo.crop(report['crop']).resize((512,512),Image.Resampling.LANCZOS)
            if args.kj_dub else Image.open(args.candidate/'source-crop.png').convert('RGB'))
        count=0
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
            support=np.maximum(support,segmenter.process(np.array(source,copy=True)).segmentation_mask)
            for frame in pixels(movie):
                mask=segmenter.process(np.array(frame,copy=True)).segmentation_mask
                if mask.shape!=(512,512) or not np.isfinite(mask).all():raise ValueError('Invalid person segmentation')
                support=np.maximum(support,mask);count+=1
        if count!=report['frames']:raise ValueError('Incomplete segmentation timeline')
        components,labels,stats,_=cv2.connectedComponentsWithStats((support>.2).astype(np.uint8),8)
        if components<2:raise ValueError('Empty subject mask')
        support=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.float32)
        support=cv2.GaussianBlur(cv2.dilate(support,np.ones((7,7),np.uint8)),(9,9),1.6)
        left,top,right,bottom=report['crop'];sx=(right-left)/512;sy=(bottom-top)/512
        projection=np.array([[sx,0,left+(sx-1)/2],[0,sy,top+(sy-1)/2],[0,0,1.]])
        matrix=layout.source_transform@projection
        fixed=cv2.warpAffine(support,matrix[:2],layout.size,flags=cv2.INTER_LINEAR)
        old=layout.mask.copy();layout.mask=np.minimum(layout.mask,fixed)
        if np.count_nonzero(layout.mask>.5)<100:raise ValueError('The fixed matte hides the generated face')
        restored=(old>.01)&(layout.mask<.001)
        Image.fromarray((layout.mask*255).astype(np.uint8)).save(args.output.with_suffix('.alpha.png'))
        partial=args.output.with_suffix('.partial.mov');width,height=size
        process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width}x{height}','-r','25','-i','-','-i',str(args.audio),'-map','0:v:0','-map','1:a:0',
            '-t',str(report['seconds']),'-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','pcm_f32le',str(partial)],stdin=subprocess.PIPE)
        emitted=0
        try:
            for frame in pixels(movie):
                # Layout performs one spatial resampling and asserts original
                # body/background pixels outside the fixed head matte exactly.
                correction = corrections[emitted] if corrections is not None else None
                process.stdin.write(layout.frame(frame,correction=correction).tobytes());emitted+=1
        finally:process.stdin.close();code=process.wait()
        if code or emitted!=report['frames']:raise ValueError('Incomplete presenter composition')
    call('scripts/verify_export.py',args.audio,partial,'--seconds',report['seconds'],'--expected-frames',emitted,'--fps',25)
    partial.replace(args.output)
    result=dict(output=str(args.output),output_sha256=digest(args.output),quality_passed=False,
        new_motion_inference=False,source_generated_video_sha256=digest(generated_movie),
        source_generation_provenance_sha256=digest(args.candidate/'provenance.json'),
        source_conditioning_kind=report.get('conditioning_kind', 'ltx-photo'),
        source_photo_sha256=digest(args.photo),source_audio_sha256=digest(args.audio),
        native_framing_request_sha256=digest(request_path),native_framing_provenance_sha256=digest(native_provenance_path),
        frames=emitted,fps=25,seconds=report['seconds'],output_size=list(size),framing=layout.mode,
        body_and_background_preserved_outside_head_before_encoding=True,restored_static_pixels=int(restored.sum()),
        original_decoded_audio_verified=True,all_frames_verified=True,regular_presentation_times_verified=True,
        head_motion_stabilized=head_path is not None,face_enhancement_applied=False,
        matte='One fixed union of actual source and generated subject masks; largest component, threshold .2, dilate7, blur9 sigma1.6',
        limitation='Fixed matte can reveal original head edges or retain background inside the motion envelope; visual review remains required')
    if mouth_control_provenance:
        result.update(source_mouth_control_sha256=digest(movie),
                      source_mouth_control_provenance_sha256=digest(mouth_control_provenance))
    if head_path:
        result['head_path']=head_path
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
