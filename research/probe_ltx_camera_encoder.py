"""Encode actual camera motion at both reference-conditioning resolutions.

Compatibility/memory probe only. No talking avatar, dubbing or training occurs.
Run in .venv-ltx with OS networking denied.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'research')]
from download_model_manifest import digest,matches
from inference_lease import GPUInferenceLease
from ltx_reference_video_io import bundled_video_probe,probe_reference_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'.local/dubit-camera-reference-two')
    parser.add_argument('--output', type=Path, default=ROOT/'.local/dubit-camera-encoder-probe')
    args = parser.parse_args()
    source = args.reference.resolve()
    output = args.output.resolve()
    model=ROOT/'.local/ltx-2.3-q8-weights'
    record=json.loads((source/'reference.json').read_text())
    video=source/'camera-motion-reference.mkv'
    if record.get('reference_only') is not True or record.get('generated_video') is not False or digest(video)!=record['output_sha256']:
        raise ValueError('Requires the unchanged prepared conditioning input')
    lock=json.loads((ROOT/'research/ltx-source-lock.json').read_text())
    weight=next(x for x in lock['models'][0]['files'] if x['path']=='vae_encoder.safetensors')
    if not matches(model/weight['path'],weight['bytes'],weight['sha256']):raise ValueError('Unverified video encoder')
    if output.exists():raise ValueError('Retain the previous camera encoder evidence')
    with GPUInferenceLease(on_wait=lambda:print('Waiting for active local inference',flush=True)):
        import imageio_ffmpeg
        import mlx.core as mx
        from ltx_pipelines_mlx.utils import blocks
        from ltx_pipelines_mlx.utils.media_io import decode_video_by_frame,video_preprocess
        from ltx_core_mlx.model.video_vae.tiling import TilingConfig, TemporalTilingConfig
        output.mkdir();started=time.monotonic()
        bin_dir=output/'bin';bin_dir.mkdir();(bin_dir/'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ['PATH']=f'{bin_dir}:/usr/bin:/bin'
        mx.set_default_device(mx.gpu);mx.set_memory_limit(11*1024**3);mx.set_cache_limit(0)
        try:
            info=probe_reference_video(str(video))
            frames = record['frames']
            if frames not in (49,121) or (info.width,info.height,info.num_frames,info.fps)!=(512,512,frames,25):raise ValueError('Reference video geometry changed')
            tiling = TilingConfig(temporal_config=TemporalTilingConfig(40,24)) if frames > 49 else None
            phases=[]
            with bundled_video_probe():
                for size in (256,512):
                    print(f'Encoding {frames} reference frames at {size} square',flush=True)
                    pixels=video_preprocess(decode_video_by_frame(video,frame_cap=frames),size,size)
                    if pixels.shape!=(1,3,frames,size,size):raise ValueError('Unexpected decoded reference tensor')
                    conditioner=blocks.ImageConditioner(model);encoder=conditioner.load()
                    latent=encoder.tiled_encode(pixels,tiling) if tiling else encoder.encode(pixels)
                    mx.eval(latent)
                    if latent.shape!=(1,128,(frames-1)//8+1,size//32,size//32) or not bool(mx.all(mx.isfinite(latent)).item()):raise ValueError('Invalid encoded reference')
                    path=output/f'camera-{size}-latent.safetensors';mx.save_safetensors(str(path),{'video':latent})
                    phases.append(dict(size=size,latent_shape=list(latent.shape),latent_sha256=digest(path),observed_mlx_peak_bytes=mx.get_peak_memory()))
                    del encoder,pixels,latent;conditioner.free();blocks.aggressive_cleanup()
            kind = record.get('conditioning_kind', 'real-camera')
            result=dict(actual_camera_video_encoded=kind=='real-camera', conditioning_kind=kind,
                frames=frames, temporal_tiling=dict(frames=40, overlap=24) if tiling else None,
                generated_video=False,training_performed=False,
                dubbing_performed=False,quality_passed=False,reference_sha256=digest(video),
                reference_provenance_sha256=digest(source/'reference.json'),encoder_sha256=weight['sha256'],
                phases=phases,elapsed_seconds=time.monotonic()-started,max_observed_mlx_peak_bytes=mx.get_peak_memory())
            (output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        except Exception as error:
            (output/'failure.json').write_text(json.dumps(dict(error=str(error),generated_video=False,quality_passed=False),indent=2))
            raise


if __name__=='__main__':main()
