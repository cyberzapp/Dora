# Local LongCat setup and recovery

This renderer remains experimental until actual video generation and perceptual
review pass. The native app's existing renderer has not been promoted to LongCat.

Run from the project folder on an Apple Silicon Mac, with Python 3.12, git and
curl installed:

```sh
python3.12 scripts/install_longcat.py --check-only
python3.12 scripts/install_longcat.py
```

The installer checks source-file hashes against the pinned MLX port, installs an
isolated runtime from exact dependency versions, executes a real Metal tensor
test and downloads the pinned checkpoint revision. It does not upload personal
media. Installation needs network access; inference uses local files. The model
requires about 25.01 GB of disk storage, with additional space for dependencies,
derived caches, temporary download blocks and exports. Runtime wheels may
require a recent macOS release; the actual tested machine is the M5 in this log.

Re-running installation resumes completed byte ranges and verifies complete
large files by SHA-256. The downloader refuses concurrent access to the same
destination. A completed download does not prove successful generation.

For a new short compatibility run, supply real consented photo/audio paths:

```sh
.venv-longcat/bin/python research/longcat_benchmark.py \
  --image /absolute/path/to/photo.png \
  --audio /absolute/path/to/voice.wav \
  --output .local/my-longcat-test
```

The default 29-frame 384×256 output is a compatibility diagnostic. Higher-quality
acceptance requires longer, larger actual outputs and visual review. Input photos
are fitted with their aspect ratio preserved. Exported audio is the original
excerpt, verified as exact decoded floating-point PCM.

To cancel, create a file named `cancel` in that output directory. The worker
checks it between transformer blocks, guidance passes and decoder chunks. To
resume, remove the cancellation marker and run the same command with `--resume`.
Input content, dimensions, prompt, model revision and checkpoint metadata must
match. A process killed before saving a step will redo that step. Failures and
progress are recorded in `progress.json`; success requires `report.json` and the
fully decoded, audio-verified output. Review is always marked pending.

The current personal benchmark lives in `.local/longcat-first-render/`. Its
logs and progress appear in the native app's Settings page. Keep the Mac on power
and awake for lengthy downloads/renders. See `LONGCAT_EVALUATION.md` for measured
compatibility, memory peaks, limitations and remaining gates.
