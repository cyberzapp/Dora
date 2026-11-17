"""Run one real TBDub Student window offline on the verified Black T reference."""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'vendor/TBDub-19e18e4ba8e2a19e6716cb38110ee2de1dae9bb3'
REVIEW = ROOT/'.local/tbdub-source-review'
PATCH_RUNTIME = REVIEW/'ComfyUI-TBDub-8a7028e433356d96f6fb66a048c4b646c930c951/runtime'
FPS, FRAMES = 25, 72
PATCH_SHA256 = '3b900b88835b68ddfdb6b35857a97dbdae77a6e5a4302c7eb00568394935bc4b'
SOURCE_MANIFEST_SHA256 = '154d930df4d8c3ae499fa6b639fb873bb27071bdb543ac727eddc1b4d7846d62'
MODEL_MANIFESTS = (
    ('TaoLiveAIGC--TBDub', 'tbdub-student-weights',
     'd78fc3db5c583451742cb584bac4567208207930c4ff560c26856c724d7efd13'),
    ('KlingTeam--X-Dub', 'tbdub-vae-weights',
     '6f51819cfa196ff96043565cde1bee86d1b6ecd40585a5e229533e4ac029ebb2'),
    ('facebook--hubert-large-ll60k', 'tbdub-hubert-weights',
     'e97c4083ef46cbd812f1b2666d76e96d260afd3d0a55c92d30541ac200caac1a'),
)
sys.path[:0] = [str(UPSTREAM), str(PATCH_RUNTIME), str(ROOT/'backend')]
from download_model_manifest import digest, matches
from inference_lease import GPUInferenceLease
from ltx_video_export import export_original_audio
from mps_attention import bounded_attention
from tbdub_mps_math import mps_frequency_tables, mps_rotary_embedding, mps_timestep_embedding


def validate_models():
    inventories = []
    for name, directory, expected_hash in MODEL_MANIFESTS:
        manifest_path = REVIEW/f'{name}.json'
        if digest(manifest_path) != expected_hash:
            raise ValueError(f'Unpinned checkpoint manifest for {name}')
        manifest = json.loads(manifest_path.read_text())
        folder = ROOT/'.local'/directory
        marker = folder/'download-complete.json'
        if not marker.is_file() or json.loads(marker.read_text()) != dict(
                repo=manifest['repo'], revision=manifest['revision'], verified=True):
            raise ValueError(f'Checkpoint transfer has not completed for {name}')
        for item in manifest['files']:
            path = folder/item['path']
            if not matches(path, item['bytes'], item.get('sha256'), item.get('git_blob_sha1')):
                raise ValueError(f'Unverified checkpoint file: {path.name}')
        inventories.append(manifest)
    return inventories


def validate_source():
    manifest_path = PATCH_RUNTIME/'upstream_manifest.json'
    if digest(manifest_path) != SOURCE_MANIFEST_SHA256 or digest(PATCH_RUNTIME/'tbdub_patches.py') != PATCH_SHA256:
        raise ValueError('Unexpected revision of the TBDub streaming patch or source manifest')
    pinned = json.loads(manifest_path.read_text())
    if not UPSTREAM.name.endswith(pinned['commit']):
        raise ValueError('Unexpected TBDub revision')
    for name, expected in pinned['files'].items():
        data = (UPSTREAM/name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Pinned TBDub source changed: {name}')


def prepare_sample(upstream, reference_dir, output):
    import imageio_ffmpeg
    reference = json.loads((reference_dir/'reference.json').read_text())
    video = Path(reference['output'])
    audio = Path(reference['audio'])
    photo = reference.get('derived_conditioning', {})
    if (reference.get('conditioning_kind') != 'kj-photo-motion' or reference['fps'] != FPS
            or reference['output_size'] != [512, 512] or reference['frames'] < FRAMES
            or digest(video) != reference['output_sha256'] or digest(audio) != reference['source_audio_sha256']
            or digest(Path(photo['photo'])) != photo['photo_sha256']):
        raise ValueError('Expected the unchanged KJ photo reference and original voice-over')
    excerpt = output/'target-excerpt.wav'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-i',str(audio),'-vn',
                    '-t',str(FRAMES/FPS),'-c:a','pcm_f32le',str(excerpt)],check=True)
    decode = [imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-i']
    pcm = subprocess.check_output([*decode, str(excerpt), '-map','0:a:0','-f','f32le','-'])
    original_pcm = subprocess.check_output([*decode, str(audio), '-map','0:a:0',
                                            '-t',str(FRAMES/FPS),'-f','f32le','-'])
    if not pcm or pcm != original_pcm:
        raise ValueError('Prepared audio does not match the original decoded voice-over')
    sample = upstream.load_cropped_inputs(str(video), str(excerpt))
    upstream.validate_sample(sample)
    if len(sample.reference_video) != reference['frames'] or upstream.get_audio_length(str(excerpt)) != FRAMES:
        raise ValueError('Prepared inputs do not match the one-window timeline')
    return sample, reference, audio


def install_bounded_vae_decode(vae, report):
    import torch

    original_decode = vae.decode
    tile_size, tile_stride = (20, 20), (12, 12)

    def decode(latents, device, **_):
        torch.mps.synchronize()
        torch.mps.empty_cache()
        decoder = dict(mode='full-width')
        report['decoder'] = decoder
        try:
            decoded = original_decode(latents, device=device, tiled=False)
        except RuntimeError as error:
            if not str(error).startswith('MPS backend out of memory'):
                raise
            decoder['full_width_memory_error'] = str(error)
        if 'full_width_memory_error' in decoder:
            # Retry after the failed call's traceback has released its tensors.
            # The upstream tiled decoder blends overlaps on CPU.
            vae.model.clear_cache()
            torch.mps.empty_cache()
            print('Full-width decode reached the Metal memory cap. Retrying with overlapping tiles.', flush=True)
            decoder.update(mode='overlapping-tiles', tile_size_latents=tile_size,
                           tile_stride_latents=tile_stride)
            decoded = original_decode(latents, device=device, tiled=True,
                                      tile_size=tile_size, tile_stride=tile_stride)
        torch.mps.synchronize()
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError('The VAE returned non-finite video pixels')
        decoder.update(decoded_shape=list(decoded.shape),
                       sampled_driver_bytes=torch.mps.driver_allocated_memory())
        return decoded

    vae.decode = decode


def run(args):
    import imageio_ffmpeg
    import numpy as np
    import torch
    import inference as upstream
    import tbdub_patches as streaming
    from diffsynth.models import tbdub_dit
    from diffsynth.pipelines import tbdub as pipeline_module

    start = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    bin_dir = out/'bin'
    bin_dir.mkdir()
    (bin_dir/'ffmpeg').symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ['PATH'] = f'{bin_dir}:/usr/bin:/bin'
    report = dict(generated_video=False, quality_passed=False, frames=FRAMES, fps=FPS,
                  upstream_revision=UPSTREAM.name.removeprefix('TBDub-'),
                  streaming_patch_sha256=PATCH_SHA256)
    try:
        sample, reference, audio = prepare_sample(upstream, args.reference, out)
        report.update(reference=reference, reference_frames=len(sample.reference_video),
                      target_excerpt_sha256=digest(Path(sample.audio_path)),
                      target_excerpt_pcm_verified=True)
        runtime_args = upstream.parse_args([
            '--video', reference['output'], '--audio', sample.audio_path, '--device', 'mps',
            '--cropped-input', '--inference-mode', 'student', '--seed', '42', '--output-dir', str(out),
            '--checkpoint-dir', str(ROOT/'.local/tbdub-student-weights'),
            '--config', str(ROOT/'.local/tbdub-student-weights/config.json'),
            '--vae-checkpoint', str(ROOT/'.local/tbdub-vae-weights/Wan2.2_VAE.safetensors'),
            '--hubert-checkpoint', str(ROOT/'.local/tbdub-hubert-weights'),
        ])
        if (not runtime_args.student_first_clip_padding or not runtime_args.motion_from_latents
                or runtime_args.num_student_steps != 2 or runtime_args.sigma_shift != 1.0
                or upstream.CLIP_NUM_FRAMES - upstream.MOTION_NUM_FRAMES != FRAMES):
            raise ValueError('The published Student recipe does not match the expected one-window benchmark')
        indices = upstream.make_pingpong_indices(len(sample.reference_video), FRAMES)
        if list(indices) != list(range(FRAMES)):
            raise ValueError('The single-window reference would repeat or reverse source motion')
        report.update(student_steps=runtime_args.num_student_steps,
                      sigma_shift=runtime_args.sigma_shift,
                      motion_preroll_frames=upstream.MOTION_NUM_FRAMES,
                      conditioning_video_repeated=False)
        if args.check_inputs:
            (out/'preflight.json').write_text(json.dumps(report,indent=2))
            return

        report['weights'] = validate_models()
        if not torch.backends.mps.is_available():
            raise RuntimeError('This benchmark requires an available Apple Metal device')
        torch.mps.set_per_process_memory_fraction(0.85)
        torch.set_num_threads(4)
        os.environ['TBDUB_CACHE_DIR'] = str(out/'audio-cache')
        # DiffSynth recognizes torch.device.type for Metal. Its string parser
        # treats unfamiliar device names as CPU.
        runtime_args.device = torch.device('mps')
        tbdub_dit.precompute_freqs_cis_3d = partial(mps_frequency_tables, tbdub_dit.precompute_freqs_cis_3d)
        timestep = partial(mps_timestep_embedding, tbdub_dit.sinusoidal_embedding_1d)
        tbdub_dit.sinusoidal_embedding_1d = pipeline_module.sinusoidal_embedding_1d = timestep
        tbdub_dit.rope_apply = partial(mps_rotary_embedding, tbdub_dit.rope_apply)
        report['streamed_reader_modules'] = streaming.install_reader()

        silent = out/'student-silent.mp4'
        def save_silent(frames, *_args, **_kwargs):
            if len(frames) != FRAMES or any(frame.size != (512,512) for frame in frames):
                raise ValueError('Upstream generation returned an unexpected frame count or geometry')
            command = [imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24',
                       '-s','512x512','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','14',
                       '-pix_fmt','yuv420p',str(silent)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
                for frame in frames:
                    writer.stdin.write(np.asarray(frame.convert('RGB')).tobytes())
                writer.stdin.close()
                if writer.wait():
                    raise RuntimeError('Silent output encoding failed')

        upstream.save_video_with_audio = save_silent
        with GPUInferenceLease(on_wait=lambda:print('Waiting for active local inference',flush=True)), torch.inference_mode():
            pipeline, prompt = streaming.build_pipeline(upstream, runtime_args, 'stream')
            # The upstream feature extractor can release its Metal weights as
            # soon as it has produced the conditioning features on CPU.
            pipeline.hubert_processor.model.to('cpu')
            pipeline.hubert_processor.cpu_offload = True
            report['hubert_offload_after_features'] = True
            coverage = streaming.dit_weight_coverage(pipeline.dit)
            report['weight_coverage'] = coverage
            if coverage['not_disk_backed_count'] or coverage['non_meta_count'] or coverage['buffers_not_from_checkpoint']:
                raise ValueError('The streamed transformer did not account for every pretrained tensor')
            report['chunked_decoder_layers'] = streaming.install_decoder_conv_chunks(pipeline.vae)
            install_bounded_vae_decode(pipeline.vae, report)
            peaks = []
            def finish_block(_module, _inputs, _output):
                torch.mps.synchronize()
                peaks.append(torch.mps.driver_allocated_memory())
            hooks = [block.register_forward_hook(finish_block) for block in pipeline.dit.blocks]
            with bounded_attention(query_chunk=128):
                try:
                    upstream.run_inference(pipeline,prompt,sample,'student',runtime_args)
                finally:
                    for hook in hooks:
                        hook.remove()
                    report['transformer_blocks_executed'] = len(peaks)
                    report['max_sampled_dit_driver_bytes'] = max(peaks, default=0)
            if len(peaks) != 2*len(pipeline.dit.blocks):
                raise ValueError('The expected two pretrained Student denoising steps did not execute')

        movie = export_original_audio(silent,audio,out/'review.mov',frames=FRAMES,fps=FPS)
        report.update(generated_video=True, output=str(movie), output_sha256=digest(movie),
                      original_decoded_audio_verified=True,elapsed_seconds=time.monotonic()-start)
        (out/'provenance.json').write_text(json.dumps(report,indent=2))
    except Exception as error:
        report['error'] = str(error)
        (out/'failure.json').write_text(json.dumps(report,indent=2))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=lambda p:Path(p).resolve())
    parser.add_argument('--output', required=True, type=lambda p:Path(p).resolve())
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--consent', action='store_true')
    args=parser.parse_args()
    if not args.consent:
        parser.error('Explicit likeness consent is required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    validate_source()
    run(args)


if __name__ == '__main__':
    main()
