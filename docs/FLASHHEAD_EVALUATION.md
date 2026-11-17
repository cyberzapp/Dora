# FlashHead Pro local evaluation

Status 2026-11-07: genuine pretrained generation runs on this M5 / 16 GB Mac and
is available in the native app for evaluation. The user preferred the sharper
Black T photo version to the HQvideo frame, but rejected motion quality in the
15-second native result. The full-length job was cancelled at section 23/48.
No current FlashHead result meets the release quality gate.

Official source: <https://github.com/Soul-AILab/SoulX-FlashHead>, commit
`9bc03de06bb0de82cd6bc477804512ae06144bf2`. Model:
<https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B>, revision
`5fbd2e44ba08ab2deb28bd55cff2c15523e51ba5`. The selected Pro transformer, Wan VAE
and Wav2Vec2 encoder are declared Apache-2.0 in the official model card. Lite's
different VAE is excluded. Exact file sizes, hashes and audio revision are in
`research/flashhead-source-lock.json`. All selected weights passed full SHA checks.

## Actual implementation

`research/flashhead_mps.py` is an external adapter; vendor files remain unchanged.
It performs rotary arithmetic using real float32 operations equivalent to the
official complex/double CPU calculation, computes small double-precision timestep
constants on CPU and bounds independent attention queries to 256 rows. Distributed
CUDA paths explicitly fail. Five mathematical/device checks pass, including the
actual MPS/bfloat16 timestep path. A combined CPU+double conversion initially failed
on MPS; transferring to CPU before changing dtype fixed it. No failed attempt is
presented as generated video.

The speech encoder loads strictly. Old positional-convolution weight-g/v names
are mapped to current Torch parametrizations. The automatic Transformers loader
otherwise left those trained weights random; that path was rejected. Training-only
SpecAugment is disabled, and its absent mask token is not invented. The unused CTC
text head is explicitly excluded. All speech analysis is CPU-local.

`research/flashhead_benchmark.py` uses the official Pro architecture, bfloat16
weights, four-step shifted distilled sampling, 33-frame windows, five conditioning
frames, 28 new output frames and an eight-second rolling speech context. It carries
the actual last five decoded frames through the trained VAE between chunks. Model
and VAE stages swap GPU residency. The VAE streams decoded chunks to CPU. CPU noise
uses an explicit seed; this is reproducible here, not bit-identical to CUDA RNG.
Source files remain unchanged and the export uses original audio, never the
16 kHz analysis track. No personalized training was performed.

## Measured results

- Pretrained speech encoder: finite `[1,33,5,12,768]` features from the user's voice,
  0.610 seconds including loading. Not a video result.
- Video VAE: real 33-frame photo reconstruction, 23.98 seconds, maximum observed
  driver allocation 6.71 GB. Labeled comparison retains visible teeth/face detail.
- First actual talking crop: 28 frames / 1.12 seconds in 60.82 seconds, maximum
  observed driver allocation 6.71 GB. Original audio and all frames verified.
- Meaningful sample: 121 frames / 4.84 seconds in 279.18 seconds, maximum observed
  driver allocation 6.67 GB. All five temporal windows completed. Original decoded
  PCM and every frame verified. Not a full-duration or peak-system-memory test.
- Local calibrated SyncNet diagnostic: -1 frame / -40 ms, distance 8.5132,
  confidence 5.3453. It is not a lip-shape or realism score; no audio shift applied.
- All 121 frames tracked; eye-line roll range 7.22 degrees. This is a 2D diagnostic,
  not a claim that movement is natural.

Private evidence: `.local/flashhead-quality-five/`. `review.mov` is the actual
512×512 generated crop. `source-body.mov` projects its generated RGB caches onto
the original seated photograph in one video encode, preserving source pixels
outside the head mask before compression. Its 1536×1024 canvas includes one padding
column; effective generated facial detail is still the model crop's roughly
166-pixel face, not new high-resolution detail. `source-body.json` records hashes
and provenance. The supplied HeyGen reference has about a 306-pixel face.

`research/review_chunk_boundaries.py` creates adjacent-frame strips at every join
from actual RGB caches. `research/project_head_candidate.py` applies source-body
preservation and independently verifies the final audio/frame count. These are
review tools, not a quality-approved production backend.

## Refinement and app integration

The tighter crop produced 121 frames in 291.03 seconds, with observed MPS driver
allocation up to 7.01 GB. The user reports both resemblance and motion improved,
while still needing refinement. Optional rigid stabilization uses a nine-frame
smooth motion path, with corrections capped at three model pixels and 0.75 degrees.
It does not retarget nonrigid expressions or change the mouth's timing.

`backend/flashhead_engine.py` reproduces the tested neural recipe. An actual
three-window execution proved exact first-window and continuation equality with
the benchmark and exact resumed-versus-uninterrupted output after reloading models.
`backend/flashhead_cache.py` commits RGB frames and recovery tensors transactionally,
verifies their hashes and rejects changed inputs/settings. Audio contexts are saved
separately and verified, without keeping all tensors resident. The timeline covers
the exact original audio duration, including a fractional final video frame.

The real worker was cancelled after three committed sections and resumed to five.
A native tracking-stage crash was fixed by isolating MediaPipe and using writable
RGB copies. Export then reused all sections and passed original-audio/full-frame
verification. The actual native Swift store/queue subsequently completed a cold
4.84-second generation, verified 1920×1080 export and persisted completion in
SQLite. It uses full-resolution orientation-normalized source pixels, not the older
downscaled outfit cache. The visible app now exposes FlashHead, source-photo
selection, optional jitter reduction, 1080p framing and resume controls.

A standalone runtime with public models is installed in the app's own Application
Support folder. Separate Metal and tracking probes passed with Python network
connections disabled, temporary CWD, minimal PATH and no external non-system loaded
libraries. Actual app-launched diagnostics passed. A real 375-frame, 15-second
render from the relocated runtime finished in 1,352.74 seconds, with exact
decoded original audio and a complete frame decode. The longer continuation was
tested with networking denied by macOS but cancelled after the quality rejection.
Full-length completion, visual quality, multilingual acceptance, license review
and notarization remain open. A shared per-user lock prevents concurrent large
models across the research checkout and installed runtime.

The user's newer `HQvideo.mp4` is confirmed camera footage with a replaced
background, not an unedited capture. A closed-mouth frame at 1.125 seconds was
tested as the appearance source, preserving its provenance and visible background.
The user preferred the earlier photo. It was not used as a motion-conditioning
video.

The 4.84-second source-anchored LivePortraitKJ control used the same photo and
mouth motion extracted from a real FlashHead result. KJ ran on MPS and finished
all 121 frames in 382.65 seconds; original decoded audio matched. The result is
visibly steadier but has a softer mouth and nearly static eyes. It is retained as
a diagnostic comparison only; do not promote it as a production mode.
