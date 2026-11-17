"""Run the genuine LongCat Whisper checkpoint on a local voice-over.

Produces features and a diagnostic report, never a generated-video substitute.
The original audio file is only read and is not modified.
"""
import argparse
import hashlib
import json
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/longcat-avatar-mlx'))
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import imageio_ffmpeg
from transformers import WhisperFeatureExtractor
from longcat_video_avatar.models.whisper import WhisperEncoder
from longcat_video_avatar.audio_process import group_pool_whisper_hidden_states, linear_interpolate_features
from longcat_streaming import TensorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / '.local/longcat-models/q4')
    parser.add_argument('--audio', type=Path, required=True)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--seconds', type=float, default=5)
    duration.add_argument('--full-audio', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        raise ValueError('Use 1–300 seconds of voice-over')
    args.output.mkdir(parents=True, exist_ok=True)
    with args.audio.open('rb') as source:
        source_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    started = time.monotonic()
    checkpoint = args.weights / 'audio_encoder/model.safetensors'
    manifest = json.loads((args.weights / 'download-manifest.json').read_text())
    entry = next(x for x in manifest['files'] if x['path'] == 'audio_encoder/model.safetensors')
    digest = hashlib.sha256()
    with checkpoint.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != entry['sha256']:
        raise ValueError('Audio checkpoint hash mismatch')
    print('Verified pretrained audio weights', flush=True)

    # FFmpeg also accepts native recorder formats such as M4A. Only analysis
    # is converted; the export always uses the original input audio.
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error',
                              '-i', str(args.audio), '-t', str(301 if args.full_audio else args.seconds),
                              '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                             check=True, stdout=subprocess.PIPE).stdout
    audio = np.frombuffer(decoded, dtype='<f4').copy()
    if len(audio) > 300 * 16000:
        raise ValueError('Split voice-overs longer than five minutes into projects')
    if len(audio) < 16000 or not np.isfinite(audio).all():
        raise ValueError('Use at least one second of valid recorded voice-over')
    import pyloudnorm as pyln
    loudness = pyln.Meter(16000).integrated_loudness(audio)
    if np.isfinite(loudness) and abs(loudness) <= 100:
        audio = pyln.normalize.loudness(audio, loudness, -23.0)
    # Explicit Whisper-large-v3 configuration; no implicit network download.
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                        hop_length=160, chunk_length=30, n_fft=400)
    config = json.loads((args.weights / 'audio_encoder/config.json').read_text())
    model = WhisperEncoder.from_config(config)
    model.load_weights(TensorStore(args.weights / 'audio_encoder').weights(''), strict=True)
    mx.eval(model.parameters())
    print('Strictly loaded complete Whisper encoder', flush=True)
    def encode_groups(mel, chunk_index):
        # HF hidden_states ends with the post-final-LayerNorm state.
        x = mx.array(mel).astype(model.conv1.weight.dtype).transpose(0, 2, 1)
        x = nn.gelu(model.conv1(x)); x = nn.gelu(model.conv2(x))
        x = x + model.embed_positions.weight[:x.shape[1]][None]
        hidden = [x]
        for index, layer in enumerate(model.layers):
            x = layer(x)
            mx.eval(x)
            if index < len(model.layers) - 1:
                hidden.append(x)
        hidden.append(model.layer_norm(x))
        pooled = group_pool_whisper_hidden_states(hidden)
        mx.eval(pooled)
        print(f'Encoded audio chunk {chunk_index + 1}', flush=True)
        return pooled
    chunks = []
    for index, start in enumerate(range(0, len(audio), 30 * 16000)):
        mel = extractor(audio[start:start + 30 * 16000], sampling_rate=16000, return_tensors='np').input_features
        chunks.append(encode_groups(mel, index))
    # Integer arithmetic avoids dropping the 29th frame for 1.16-second clips
    # (binary floating point represents 1.16 * 25 as 28.999999999999996).
    frames = (len(audio) * 25) // 16000
    # Trim padded 30-second encoder states BEFORE interpolation, as upstream does.
    groups = mx.concatenate(chunks, axis=1)[:, :frames * 2]
    # Explicit length also avoids the same floating-point truncation in the port.
    flat = groups.reshape(1, groups.shape[1], 5 * 1280)
    per_frame = linear_interpolate_features(flat, 50, 25, output_len=frames).reshape(1, frames, 5, 1280)
    indices = mx.clip(mx.arange(frames)[:, None] + mx.arange(5)[None, :] - 2, 0, frames - 1)
    embeddings = per_frame[:, indices]
    if embeddings.shape != (1, frames, 5, 5, 1280):
        raise RuntimeError('Audio feature frame count mismatch')
    mx.eval(embeddings)
    if not bool(mx.all(mx.isfinite(embeddings))):
        raise RuntimeError('Nonfinite pretrained audio features')
    mx.save_safetensors(str(args.output / 'audio.safetensors'), {'audio': embeddings})
    report = dict(status='audio_encoder_passed', generated_video=False,
                  shape=list(embeddings.shape), seconds=len(audio)/16000,
                  analysis_samples=len(audio), sample_rate=16000, fps=25,
                  elapsed_seconds=time.monotonic()-started,
                  peak_mlx_bytes=mx.get_peak_memory(), checkpoint_sha256=digest.hexdigest(),
                  audio_path=str(args.audio.resolve()), source_sha256=source_hash,
                  full_audio=args.full_audio, encoder_chunks=len(chunks), device=mx.device_info(),
                  note='Pretrained audio feature extraction only; no visual quality claim')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
