"""Decode a completed Dub-It latent checkpoint without rerunning denoising.

Requires a real completed inference checkpoint; camera encoding probes and
partial denoising are rejected. Run with OS networking denied.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT/'research')]
from download_model_manifest import digest
from inference_lease import GPUInferenceLease
from ltx_audio_preview import validate_model, MODEL, SOURCE_REVISION
from ltx_video_export import export_original_audio

CODE_HASH = digest(Path(__file__))


def validate_checkpoint(directory, model):
    record = json.loads((directory/'checkpoint.json').read_text())
    if (record.get('adapter_inference_performed') is not True or
            record.get('audio_conditioning_verified') is not True or
            record.get('conditioning_stages') != 2 or
            record.get('source_revision') != SOURCE_REVISION or
            record.get('frames') not in (49,121) or
            (record.get('fps'), record.get('output_geometry')) != (25,[512,512])):
        raise ValueError('Expected a completed real Dub-It inference checkpoint')
    mode = record.get('audio_conditioning_mode')
    if mode not in ('reference','frozen-target') or record.get('original_target_audio_frozen') != (mode=='frozen-target'):
        raise ValueError('Checkpoint audio conditioning mode is missing or inconsistent')
    official = json.loads((ROOT/'research/dubit-source-lock.json').read_text())
    if record.get('adapter') != official:
        raise ValueError('Checkpoint does not identify the pinned official adapter')
    latent = directory/'video-latent.safetensors'
    if digest(latent) != record['latent_sha256']:
        raise ValueError('Generated latent checkpoint failed integrity verification')
    bindings = record.get('adapter_binding_evidence', {})
    for stage in ('1','2'):
        evidence = bindings.get(stage, {})
        if (not evidence.get('matched_blocks') or evidence.get('attached_layer_count',0) <= 0 or
                evidence.get('mode') != 'unfused' or evidence.get('strength') != 1.0):
            raise ValueError('Checkpoint lacks observed Dub-It adapter bindings for both stages')
    audio = Path(record['reference']['audio'])
    if digest(audio) != record['reference']['source_audio_sha256']:
        raise ValueError('Original voice-over changed since generation')
    if validate_model(model, MODEL) != record['base_manifest_sha256']:
        raise ValueError('Decoder model differs from the generation checkpoint')
    return record, latent, audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=lambda p:Path(p).resolve())
    parser.add_argument('--model',type=lambda p:Path(p).resolve(),default=ROOT/'.local/ltx-2.3-q8-weights')
    parser.add_argument('--output',required=True,type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    if args.output.exists():parser.error('Use a fresh recovery directory; retain the original attempt')
    record, latent_path, audio = validate_checkpoint(args.checkpoint,args.model)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    with GPUInferenceLease():
        import imageio_ffmpeg
        import mlx.core as mx
        from ltx_pipelines_mlx.utils.blocks import VideoDecoder
        args.output.mkdir(parents=True,exist_ok=False)
        start=time.monotonic()
        try:
            if shutil.disk_usage(args.output).free < 10*1024**3:
                raise ValueError('Leave 10 GiB free for decoding and system swap')
            bin_dir=args.output/'bin';bin_dir.mkdir()
            (bin_dir/'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
            os.environ['PATH']=f'{bin_dir}:/usr/bin:/bin'
            mx.set_default_device(mx.gpu);mx.set_memory_limit(11*1024**3);mx.set_cache_limit(0)
            video=mx.load(str(latent_path))['video']
            frames = record['frames']
            if tuple(video.shape)!=(1,128,(frames-1)//8+1,16,16) or not bool(mx.all(mx.isfinite(video)).item()):
                raise ValueError('Checkpoint video geometry or values are invalid')
            decoder=VideoDecoder(args.model)
            silent=args.output/'generated-silent.mp4'
            decoder.decode_and_stream(video,str(silent),frame_rate=25)
            output=export_original_audio(silent,audio,args.output/'review.mov',frames=frames,fps=25)
            result=dict(record,generated_video=True,quality_passed=False,
                recovered_from=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint/'checkpoint.json'),
                recovery_code_sha256=CODE_HASH,
                export_helper_sha256=digest(ROOT/'research/ltx_video_export.py'),
                output=str(output),output_sha256=digest(output),
                original_decoded_audio_verified=True,denoising_rerun=False,
                recovery_seconds=time.monotonic()-start,recovery_mlx_peak_bytes=mx.get_peak_memory())
            (args.output/'provenance.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(dict(output=str(output),denoising_rerun=False,quality_passed=False)),flush=True)
        except Exception as error:
            (args.output/'failure.json').write_text(json.dumps(dict(error=str(error),generated_video=False,
                checkpoint_retained=True,quality_passed=False),indent=2))
            raise


if __name__=='__main__':main()
