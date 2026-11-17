"""Decode the real photo latent at the planned 49-frame video geometry.

This is a repeated-photo codec/memory check, not a talking avatar or a motion
generation result. Use the isolated LTX runtime with OS networking denied.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research'),str(ROOT/'scripts')]
from download_model_manifest import digest,matches
from inference_lease import GPUInferenceLease
from verify_export import verify_video


def main():
    source=ROOT/'.local/ltx-real-photo-probe'
    output=ROOT/'.local/ltx-real-video-codec-probe'
    model=ROOT/'.local/ltx-2.3-q8-weights'
    report=json.loads((source/'report.json').read_text())
    if (report.get('actual_photo_encoder') is not True or report.get('actual_video_decoder') is not True
            or report.get('generated_video') is not False):
        raise ValueError('Requires the completed real-photo codec probe')
    lock=json.loads((ROOT/'research/ltx-source-lock.json').read_text())
    record=next(x for x in lock['models'][0]['files'] if x['path']=='vae_decoder.safetensors')
    if not matches(model/record['path'],record['bytes'],record['sha256']):
        raise ValueError('The video decoder failed pinned integrity verification')
    if output.exists():raise ValueError('Retain the previous temporal codec probe')
    with GPUInferenceLease(on_wait=lambda:print('Waiting for the active local renderer',flush=True)):
        import os
        import imageio_ffmpeg
        import mlx.core as mx
        import numpy as np
        from ltx_pipelines_mlx.utils import blocks
        output.mkdir()
        started=time.monotonic()
        try:
            mx.set_default_device(mx.gpu);mx.set_memory_limit(11*1024**3);mx.set_cache_limit(0)
            latent_path=source/'reference-latent.safetensors'
            latent=mx.load(str(latent_path))['photo']
            if latent.shape!=(1,128,1,16,16) or not bool(mx.all(mx.isfinite(latent)).item()):
                raise ValueError('Invalid reference photo latent')
            # Seven latent time planes decode to 49 pixel frames. Repetition
            # exercises the real temporal decoder without claiming new motion.
            latent=mx.repeat(latent,7,axis=2);mx.eval(latent)
            bin_dir=output/'bin';bin_dir.mkdir()
            (bin_dir/'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
            os.environ['PATH']=f'{bin_dir}:/usr/bin:/bin'
            decoder=blocks.VideoDecoder(model)
            partial=output/'partial.mp4'
            print('Decoding repeated source-photo latents at 49 x 512 x 512',flush=True)
            decoder.decode_and_stream(latent,str(partial),frame_rate=25)
            peak=mx.get_peak_memory()
            decoder.free();del latent;blocks.aggressive_cleanup()
            count=verify_video(partial,49,25)
            reader=imageio_ffmpeg.read_frames(str(partial),pix_fmt='rgb24')
            means=[]
            try:
                metadata=next(reader)
                if tuple(metadata['size'])!=(512,512):raise ValueError('Unexpected decoded size')
                for frame in reader:
                    pixels=np.frombuffer(frame,dtype=np.uint8)
                    means.append(float(pixels.mean()))
            finally:reader.close()
            if len(means)!=49 or min(means)<1:raise ValueError('Missing or black photo reconstruction frames')
            movie=output/'static-photo-codec.mp4';partial.replace(movie)
            result=dict(actual_video_decoder=True,generated_video=False,quality_passed=False,
                motion_generated=False,description='Repeated real-photo latents; codec test only',
                reference_probe_sha256=digest(source/'report.json'),latent_sha256=digest(latent_path),
                decoder_sha256=record['sha256'],latent_shape=[1,128,7,16,16],decoded_frames=count,
                output_size=[512,512],fps=25,frame_mean_range=[min(means),max(means)],
                regular_presentation_times_verified=True,elapsed_seconds=time.monotonic()-started,
                max_observed_mlx_peak_bytes=peak,output_sha256=digest(movie))
            (output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        except Exception as error:
            (output/'failure.json').write_text(json.dumps(dict(error=str(error),generated_video=False),indent=2))
            raise


if __name__=='__main__':main()
