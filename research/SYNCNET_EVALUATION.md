# Local lip-sync timing diagnostics

`syncnet_timing.py` runs the SyncNet architecture already vendored by LatentSync on CPU. It uses a pinned `python_speech_features==0.6` installation in `.local/eval-deps`, and a local checkpoint downloaded from ByteDance's LatentSync 1.6 auxiliary files. No media is uploaded.

Checkpoint SHA-256: `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442` (54,573,114 bytes). Loader verifies the hash and loads state tensors strictly with `weights_only=True`.

The original SyncNet implementation is by Joon Son Chung and carries an MIT license: https://github.com/joonson/syncnet_python/blob/master/LICENSE.md . Its original model download is documented at https://github.com/joonson/syncnet_python/blob/master/download_model.sh . The selected model mirror is https://huggingface.co/ByteDance/LatentSync-1.6/tree/main/auxiliary (repository model-card license: OpenRAIL++). Keep license/provenance notices when packaging dependencies; this evaluator is currently research-only.

Our adapter replaces the upstream S3FD detector with MediaPipe and retains the median-smoothed, downward-offset face crop. It avoids shell-string FFmpeg commands and runs inference with gradients disabled, batch two and two CPU threads. These changes require calibration: its scores must not be presented as directly comparable to published benchmark scores.

Required calibration and interpretation:

1. Evaluate an original camera recording with its original audio.
2. Delay a temporary copy of the audio by a known interval; check the measured offset changes accordingly. Never modify the source file.
3. Evaluate generated candidates and the HeyGen reference through the same cropper and settings.
4. Use estimated offset and confidence as evidence alongside playback. Do not infer photorealism, identity fidelity, or user acceptance from these numbers. Do not automatically shift output audio to improve the score.

Example:

```sh
.venv/bin/python research/syncnet_timing.py --video /absolute/path/to/recording.mp4 --seconds 5 --output .local/syncnet-controls/original.json
.venv/bin/python research/syncnet_timing.py --video /absolute/path/to/recording.mp4 --seconds 5 --audio-shift 0.2 --output .local/syncnet-controls/delayed.json
```

Calibration completed on the first five seconds of the supplied camera recording: estimated offset +6 frames (240 ms), confidence 6.4133. A temporary 200 ms audio delay changed the estimate to +1 frame (40 ms), confidence 6.2698: exactly the expected five-frame change. This validates relative offset sensitivity for this control; the absolute camera offset still deserves playback confirmation. Source audio was not modified. Results are private local artifacts, not a production-quality certification.
