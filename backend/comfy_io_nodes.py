"""Local media I/O for the Avatar Studio ComfyUI-LivePortraitKJ graph."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
import torch
import folder_paths
import imageio_ffmpeg

class AvatarStudioLoadPerformance:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'video_path':('STRING',{'default':''}),'seconds':('FLOAT',{'default':2,'min':.5,'max':15})}}
    RETURN_TYPES=('IMAGE','STRING');RETURN_NAMES=('frames','original_video');FUNCTION='load';CATEGORY='Avatar Studio'
    def load(self,video_path,seconds):
        import uuid
        target=Path(folder_paths.get_temp_directory())/f'avatar-{uuid.uuid4()}.mp4';target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',video_path,'-t',str(seconds),'-an','-vf','fps=25,scale=640:640:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2',str(target)],check=True)
        video=cv2.VideoCapture(str(target));frames=[]
        try:
            while True:
                ok,frame=video.read()
                if not ok:break
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:video.release();target.unlink(missing_ok=True)
        if not frames:raise ValueError('Video has no readable frames')
        return torch.from_numpy(np.stack(frames)).float()/255,video_path

class AvatarStudioSavePerformance:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'images':('IMAGE',),'original_video':('STRING',),'filename_prefix':('STRING',{'default':'AvatarStudio_KJ'})}}
    RETURN_TYPES=();OUTPUT_NODE=True;FUNCTION='save';CATEGORY='Avatar Studio'
    def save(self,images,original_video,filename_prefix):
        import uuid
        prefix=Path(filename_prefix).name
        folder=Path(folder_paths.get_output_directory());folder.mkdir(parents=True,exist_ok=True)
        output=folder/f'{prefix}_{uuid.uuid4().hex[:8]}.mov';silent=output.with_suffix('.silent.mp4')
        pixels=(images.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8);h,w=pixels.shape[1:3];ff=imageio_ffmpeg.get_ffmpeg_exe()
        proc=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25','-i','-','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
        try:
            for frame in pixels:proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            if proc.wait():raise RuntimeError('Encoding failed')
            subprocess.run([ff,'-y','-v','error','-i',str(silent),'-i',original_video,'-map','0:v:0','-map','1:a:0','-t',str(len(pixels)/25),'-c:v','copy','-c:a','pcm_f32le',str(output)],check=True)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
            silent.unlink(missing_ok=True)
        return {'ui':{'text':[str(output)]}}

NODE_CLASS_MAPPINGS={'AvatarStudioLoadPerformance':AvatarStudioLoadPerformance,'AvatarStudioSavePerformance':AvatarStudioSavePerformance}
