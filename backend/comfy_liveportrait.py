"""Execute actual ComfyUI-LivePortraitKJ node classes locally (no substitute implementation)."""
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.local/matplotlib')

def main():
    p=argparse.ArgumentParser();p.add_argument('--photo',required=True);p.add_argument('--video',required=True);p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--motion-reference');p.add_argument('--pose-strength',type=float,default=.65);p.add_argument('--expression-strength',type=float,default=.65)
    p.add_argument('--source-anchored-mouth',action='store_true',help='Experimental: keep source pose/upper-face keypoints, transfer generated mouth keypoints only')
    p.add_argument('--hold-source-mouth', action='store_true',
                   help='Keep photo mouth keypoints fixed in relative-motion conditioning for a later dubbing pass')
    a=p.parse_args()
    if not .5<=a.seconds<=15: p.error('Use 0.5–15 seconds for the KJ benchmark')
    if a.source_anchored_mouth and a.motion_reference:p.error('Source-anchored mouth cannot be combined with personal motion')
    if a.hold_source_mouth and (a.source_anchored_mouth or a.motion_reference):
        p.error('Holding the source mouth requires ordinary relative motion')
    photo=Path(a.photo).resolve();video=Path(a.video).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    # Comfy parses process arguments at import; retain our parsed values above.
    sys.argv=[sys.argv[0]]
    sys.path.insert(0,str(ROOT/'vendor/ComfyUI'))
    import torch,cv2,numpy as np,imageio_ffmpeg
    from PIL import Image,ImageOps
    import folder_paths
    import comfy.model_management as mm
    folder_paths.models_dir=str(ROOT/'vendor/ComfyUI/models')
    package=ROOT/'vendor/ComfyUI-LivePortraitKJ'
    spec=importlib.util.spec_from_file_location('avatar_liveportrait_kj',package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    classes=module.NODE_CLASS_MAPPINGS
    print('Actual KJ nodes loaded; device:',mm.get_torch_device(),flush=True)
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff,'-y','-v','error','-i',str(video),'-t',str(a.seconds),'-an','-vf','fps=25,scale=640:640:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2',str(out/'driving.mp4')],check=True)
    im=ImageOps.exif_transpose(Image.open(photo)).convert('RGB');im.thumbnail((1280,1280))
    source=torch.from_numpy(np.asarray(im).copy()).float().unsqueeze(0)/255
    v=cv2.VideoCapture(str(out/'driving.mp4'));frames=[]
    while True:
        ok,f=v.read()
        if not ok:break
        frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    v.release()
    if not frames:raise RuntimeError('No driving frames decoded')
    driving=torch.from_numpy(np.stack(frames)).float()/255
    start=time.monotonic()
    with torch.inference_mode():
        pipeline,=classes['DownloadAndLoadLivePortraitModels']().loadmodel(precision='fp32',mode='human')
        cropper,=classes['LivePortraitLoadMediaPipeCropper']().crop('CPU',True)
        crop=classes['LivePortraitCropper']()
        crop_args=dict(pipeline=pipeline,cropper=cropper,dsize=512,scale=2.3,vx_ratio=0,vy_ratio=-.125,face_index=0,face_index_order='large-small',rotate=True)
        _,source_info=crop.process(source_image=source,**crop_args)
        drive_crop,drive_info=crop.process(source_image=driving,**{**crop_args,'rotate':not bool(a.motion_reference)})
        if any(x is None for x in drive_info['crop_info_list']):raise RuntimeError('KJ failed to detect a driving face; no frames silently skipped')
        original_get_info = pipeline.live_portrait_wrapper.get_kp_info
        mixer = None
        if a.motion_reference:
            from personal_motion import PersonalMotionMixer
            reference_path=Path(a.motion_reference).resolve()
            subprocess.run([ff,'-y','-v','error','-i',str(reference_path),'-t',str(a.seconds),'-an','-vf','fps=25,scale=640:640:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2',str(out/'personal-reference.mp4')],check=True)
            reference_capture=cv2.VideoCapture(str(out/'personal-reference.mp4'));reference_info=[]
            try:
                for index in range(len(frames)):
                    ok,reference_frame=reference_capture.read()
                    if not ok:raise ValueError('Personal reference must cover the entire generated segment; no motion loop is used')
                    reference_rgb=cv2.cvtColor(reference_frame,cv2.COLOR_BGR2RGB)
                    info,reference_crop=cropper.crop_single_image(reference_rgb,512,2.3,-.125,0,0,'large-small',False)
                    if info is None:raise ValueError(f'Personal reference face is not visible at frame {index}')
                    tensor=pipeline.live_portrait_wrapper.prepare_source(reference_crop)
                    measured=original_get_info(tensor)
                    reference_info.append({key:value.detach().cpu() if isinstance(value,torch.Tensor) else value for key,value in measured.items()})
                    del tensor,measured
            finally:reference_capture.release()
            mixer=PersonalMotionMixer(reference_info,pose_strength=a.pose_strength,expression_strength=a.expression_strength)
            def measured_personal_motion(image,*args,**kwargs):
                return mixer(original_get_info(image,*args,**kwargs))
            pipeline.live_portrait_wrapper.get_kp_info=measured_personal_motion
        if a.source_anchored_mouth:
            from personal_motion import SourceAnchoredMouth
            mixer=SourceAnchoredMouth(source_info['source_info'][0])
            def anchored_mouth(image,*args,**kwargs):
                return mixer(original_get_info(image,*args,**kwargs))
            pipeline.live_portrait_wrapper.get_kp_info=anchored_mouth
        held_mouth_frames = 0
        if a.hold_source_mouth:
            from personal_motion import LIP_POINTS

            def hold_source_mouth(image, *args, **kwargs):
                nonlocal held_mouth_frames
                info = original_get_info(image, *args, **kwargs)
                expression = info['exp'].clone()
                # The relative branch subtracts frame zero. A constant value
                # makes that delta zero for the mouth while retaining other motion.
                expression[:, LIP_POINTS, :] = 0
                info['exp'] = expression
                held_mouth_frames += 1
                return info

            pipeline.live_portrait_wrapper.get_kp_info = hold_source_mouth
        # KJ's relative + explicit retargeting branch replaces the motion keypoints
        # with x_s + eye/lip deltas, discarding the computed head transform.
        # Relative expressions already carry eye/lip motion; leave the separate
        # retargeting node disconnected to preserve the actual head performance.
        try:
            cropped,result=classes['LivePortraitProcess']().process(source_image=source,driving_images=drive_crop,crop_info=source_info,pipeline=pipeline,lip_zero=not (a.source_anchored_mouth or a.hold_source_mouth),lip_zero_threshold=.03,stitching=True,relative_motion_mode='single_frame' if a.source_anchored_mouth else 'relative',driving_smooth_observation_variance=3e-6,delta_multiplier=1.0,mismatch_method='constant',expression_friendly=False,expression_friendly_multiplier=1.0)
        finally:pipeline.live_portrait_wrapper.get_kp_info=original_get_info
        if mixer is not None and mixer.index!=len(frames):raise RuntimeError('Personal motion did not cover all generated frames')
        if a.hold_source_mouth and held_mouth_frames != len(frames):
            raise RuntimeError('Mouth constraint did not cover the driving timeline')
        if len(result['out_list'])!=len(frames) or any(not item for item in result['out_list']):
            raise RuntimeError('KJ did not animate every requested frame; refusing a static-frame fallback')
        full,_=classes['LivePortraitComposite']().process(source,cropped,result)
        if not torch.isfinite(full).all():
            raise RuntimeError('KJ compositor returned non-finite pixels')
    pixels=(full.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
    h,w=pixels.shape[1:3]
    encoder=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25','-i','-','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(out/'motion.mp4')],stdin=subprocess.PIPE)
    for frame in pixels:encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():raise RuntimeError('FFmpeg encoding failed')
    # This transfers a recorded performance; audio is from that same performance.
    subprocess.run([ff,'-y','-v','error','-i',str(out/'motion.mp4'),'-i',str(video),'-map','0:v:0','-map','1:a:0','-t',str(len(pixels)/25),'-c:v','copy','-c:a','pcm_f32le',str(out/'review.mov')],check=True)
    subprocess.run([ff,'-v','error','-i',str(out/'review.mov'),'-f','null','-'],check=True)
    report=dict(backend='ComfyUI-LivePortraitKJ actual node classes',frames=len(pixels),fps=25,device=str(mm.get_torch_device()),elapsed=time.monotonic()-start,photo=str(photo),motion=str(video),audio='original driving performance audio; not arbitrary new-voice lip synchronization',quality_verified=False)
    report.update(relative_motion_mode='relative',explicit_eye_lip_retargeting=False,lip_zero=True,stitching=True)
    if a.hold_source_mouth:
        report.update(lip_zero=False, held_source_mouth_frames=held_mouth_frames,
                      audio='Original driver audio for reference only. Mouth motion is constrained for dubbing.')
    if a.source_anchored_mouth:report.update(relative_motion_mode='single_frame',lip_zero=False,source_anchored_mouth=True,
        preserved_source_keypoints='head pose, translation, scale and non-mouth expression coordinates',
        limitation='Experimental neural mouth transfer. No generated head movement or blinks are requested; neural rendering may still alter pixels. Not personalized training or accepted realism.')
    if a.motion_reference:report.update(personal_motion_reference=str(Path(a.motion_reference).resolve()),pose_strength=a.pose_strength,expression_strength=a.expression_strength,audio_mouth_keypoints_preserved=True,personalized_training=False)
    (out/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
