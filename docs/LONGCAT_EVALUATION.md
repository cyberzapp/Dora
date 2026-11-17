# LongCat-Video-Avatar 1.5 evaluation

Current status on 2026-11-06: genuine local pretrained generation has completed,
including 121 frames at 768×512, tighter head-crop generation and actual KJ transfers.
The user rejected the personal-motion transfer for unrealistic movement,
expressions, teeth and mouth. No quality-approved renderer exists. LongCat is
selectable as an evaluation engine; it is not the accepted final generation path.
The dated implementation notes below retain the earlier compatibility history.

## Sources and selected candidate

- Official model: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5
- Official code: https://github.com/meituan-longcat/LongCat-Video
  (inspected checkout: `6b3f4b8582a8bc3f20f795735f5383716c4ba794`).
- Apple MLX port: https://github.com/xocialize/longcat-avatar-mlx
  (inspected checkout: `e2e1e8701424cef0e601281b62e228e5289ed032`).
- Converted checkpoint: https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged
  revision `5d5b5d61ce6c206930a94c760f6941aff03f9389`.

Official weights and the selected port declare MIT licenses. The official path
targets CUDA/FlashAttention. The community port reports a successful 29-frame
256×432 render on a 128 GB M5 Max; this is not evidence of 16 GB compatibility.
Use the published 8-step DMD weights and schedule, not an arbitrary reduced-step
base-model run. This is image-and-audio-conditioned video generation; use the
user's real portrait and do not generate replacement portrait/outfit assets.

## Measured local evidence

- Device: Apple M5, 17,179,869,184 bytes physical memory. MLX reports recommended
  working set 12,713,115,648 bytes. This is a runtime limit, not available free RAM.
- Isolated `.venv-longcat`: MLX/Metal 0.32.4; frozen dependencies in
  `research/longcat-runtime-lock.txt`. Main application environment unchanged.
- Actual Metal matrix multiplication passed.
- The port's small synthetic avatar-transformer forward passed on Metal.
- Four tests in `research/test_longcat_streaming.py` passed: BF16/F32/U32 value
  preservation; quantized resident-versus-streamed equivalence; incomplete-shard
  rejection; actual small avatar-transformer resident-versus-streamed equivalence.
  These use synthetic test weights and establish no pretrained quality result.

The pinned checkpoint totals approximately 25.0 GB on disk. Its transformer is
12.1 GB, text encoder 11.36 GB, audio encoder 1.27 GB, VAE 0.254 GB. Loading every
component together is unsuitable for this machine. `research/longcat_streaming.py`
implements selective tensor reads and one-block-at-a-time execution with strict
parameter matching, preserving the published 4-bit weights and FP32 AdaLN values.
Full-size pretrained block execution, encoder staging, scheduler integration and
end-to-end memory behavior remain unvalidated.

## Download blocker

Update 2026-11-06: native curl successfully transferred model ranges. Replaced the
stalled Python request path with native curl, bounded wall-clock deadlines,
range validation and visible retry errors. Resumed existing byte ranges. The
complete audio encoder passed SHA-256 verification and the transformer download
is now progressing. The earlier connection failure below is historical.

The resumable download retained 105 of 152 audio-encoder ranges from the earlier
attempt. On resumption, the small Hugging Face config endpoint returned HTTP 200,
but following the large-weight redirect to `us.aws.cdn.hf.co` timed out connecting.
No partial weight file is considered a valid checkpoint or used for inference.
Manifest, byte ranges, partial data and logs are under `.local/longcat-models/`
and `.local/longcat-download.log`. LFS files must pass full SHA-256 verification.

## Next gates

1. Complete and hash-verify the pinned downloads; resolve CDN connectivity first.
2. Run the genuine audio encoder on `My_voice.wav`, then a full-size pretrained
   transformer block with bounded memory. Record finite outputs and peak memory.
3. Implement and validate sequential text/audio/VAE stages, sequential guidance
   passes, and the official DMD schedule. Evaluate tensors between stages/steps.
4. Generate a short compatibility clip using `Portrait_BlackT.png` and new audio,
   preserving aspect ratio. Label low-resolution diagnostics explicitly.
5. Produce a meaningful higher-resolution comparison against the HeyGen reference;
   inspect identity, lips, teeth, skin, clothing, flicker and audio timing. Preserve
   original export audio. Overnight runtime is acceptable to the user.
6. Integrate into SwiftUI only after genuine generation and perceptual acceptance.

Existing user media is sufficient. No additional recording, cloud service, or
hardware purchase is required merely to finish this compatibility investigation.

## 2026-11-06 implementation and measured results

`research/longcat_audio_probe.py` strictly loaded all pretrained Whisper encoder
parameters and processed the first five seconds of `My_voice.wav`. Completed in
2.63 seconds with 1,458,694,158 bytes peak MLX allocations; this is not total
system memory. Produced finite features of shape `[1,125,5,5,1280]`. Report:
`.local/longcat-audio-probe/report.json`. Original voice-over is unchanged.

Audio preprocessing now follows the inspected upstream pipeline: LUFS
normalization for analysis only; 128-bin Whisper features; post-final-LayerNorm
last hidden state (matching Hugging Face Whisper, unlike the selected port's
pre-LayerNorm final state); trim padded encoder features before interpolation.
No audio/video quality claim follows from this encoder test.

The experimental benchmark `research/longcat_benchmark.py` stages audio, streamed
text encoding, portrait encoding, streamed video diffusion, and video decoding.
It verifies checkpoint hashes, keeps the original aspect ratio, preserves the
reference latent and its zero timestep, and samples only the remaining latents.
It decodes the reference-inclusive latent sequence as Meituan's AI2V path does.
The default port demo instead adds a latent, updates the reference, and strips it
before decode; those behaviors are deliberately not used here.

The eight-step shifted Euler schedule matches the installed Diffusers CPU
reference within 1e-7 absolute tolerance. The official v1.5 distilled recipe uses
unit text/audio guidance and one conditional pass. Optional experimental CFG uses
three sequential passes. The code requests an 8 GiB MLX allocation budget and
256 MiB cache limit, records failures and per-step latents, then verifies full
frame decoding and exact decoded PCM against the source excerpt on completion.

Seven local checks passed: five streaming tests (including actual small text and
video architecture equivalence) and two sampler tests. Full-size resident and
quantized block parameter schemas exactly match the pinned checkpoint index.
Full-size pretrained diffusion and the staged VAE path still need actual runs.

The first queued benchmark waits for all verified weights, then attempts 29
frames at 384×256 using `Portrait_BlackT.png` and `My_voice.wav`. It is explicitly
a compatibility test, not the meaningful higher-resolution quality comparison.
Logs: `.local/longcat-download.log`, `.local/longcat-first-render.log`; live state:
`.local/longcat-first-render/progress.json`. A completed report is required before
claiming generation succeeded. Native-app renderer remains unchanged.

The full-size **trained** block subsequently passed both a small token input and
3072 tokens matching the planned 29-frame 384×256 diagnostic. The latter used
1,437,348,492 bytes peak MLX allocation and 0.72 seconds for block loading/forward
(excluding shard hash verification). This uses genuine verified block-0 weights,
but synthetic activations, so it establishes block compatibility and no generated
video quality. Reports: `.local/longcat-block-report.json` and
`.local/longcat-block-render-shape-report.json`.

Recovery update: the staged benchmark now supports `--resume` using an atomic
safetensors checkpoint with embedded completed-step and request SHA-256 metadata.
Changed source content, settings or model revisions are rejected. The sampler's
interrupted-versus-uninterrupted test passes with exact tensor equality. Eight
checks now pass across streaming and sampling tests. Corrected an integer frame
count bug: floating-point `int(1.16*25)` drops a frame; sample-count arithmetic
retains all 29 audio feature frames. The actual pretrained encoder was rerun on
that exact 1.16-second excerpt to check this boundary before rendering.

The genuine VAE also strictly loaded and reconstructed the user's real portrait
at 384×256 and 768×512. Both produced finite outputs and were visually inspected
side by side: identity, seated framing, T-shirt and background appear closely
preserved, with minor reconstruction softness. This is a still reconstruction,
not audio-driven animation. The 768×512 test took 2.69 seconds and peaked at
4,077,804,868 MLX allocation bytes; mean absolute pixel difference was 1.61/255.
These numbers do not measure perceptual video quality. Reports/images are under
`.local/longcat-portrait-probe/` (256) and `.local/longcat-portrait-probe-512/`.
Verified VAE files were copied into the main model folder to reuse that download.

Native camera capture now selects 1080p when supported, with 720p/high fallback,
and reports failure if the movie output cannot be added. Release compilation
passed; hardware capture with this change remains to be exercised. Worker tests
now include nested-encoder cancellation: all nine passed.

Temporal VAE decoding now runs eagerly through the same helper used by the
benchmark. Verified 29-frame output from repeated portrait latents at 384×256
(5.57 seconds, peak 3,046,661,346 MLX bytes) and 768×512 (25.23 seconds, peak
9,900,039,394 bytes). These are decoder tests, with no synthetic test media
exported or presented as talking-avatar results. The higher-resolution peak
exceeded the requested 8 GiB allocator budget: this setting must not be described
as a guaranteed hard memory cap. Full pipeline peak and system pressure still
need measurement. Cancellation now checks between streamed DiT blocks and decode
chunks and records a distinct cancelled state, retaining completed checkpoints.

Audio analysis decoding now uses local FFmpeg so native recording formats such
as M4A are accepted as well as WAV. The trained encoder passed on the real WAV
excerpt and a locally derived M4A format fixture, both producing exactly 29 video
feature frames. Only the analysis signal is converted to mono/16 kHz; original
export audio still comes directly from the input and must pass decoded PCM
verification. Invalid or sub-one-second probe input produces an actionable error.
The reusable benchmark prompt now refers to reference appearance/clothing rather
than hard-coding a black T-shirt. The already queued personal test was parsed
with the earlier black-T-shirt prompt, which matches its actual supplied photo.

The final text-encoder shard was hash-verified and published. A trained full-size
UMT5 block with 512 prompt tokens passed finite-output and strict-loading checks
at peak 630,219,306 MLX bytes. This establishes that block's compatibility with
synthetic hidden states, not full prompt encoding or video quality. The second
video-transformer shard also completed SHA-256 verification. Remaining components
are still downloading; the render's full-checkpoint gate remains in force.

The trained block also passed at 47,616 video tokens (31 latent frames at the
768×512 image geometry, corresponding to a 121-frame/~4.84-second clip). Measured
block load/forward time was 8.74 seconds and peak MLX allocation 8,919,799,904 bytes.
This exceeds the requested 8 GiB guideline but stayed within the runtime's
available resources in this isolated probe. It does not establish whole-pipeline
peak, throughput, generated lip sync or visual quality. Report:
`.local/longcat-block-presenter-shape-report.json`.

Reused pinned cached tokenizer/configuration/index files from the prior metadata
inspection after checking expected sizes and, for tokenizer.json, its published
SHA-256. Offline T5TokenizerFast successfully prepared 512-token tensors for
English, Hindi, Hinglish and an empty negative prompt, with in-vocabulary IDs
and nonempty masks. Report: `.local/longcat-tokenizer-report.json`. This verifies
text preparation only, not spoken-language lip-sync or video quality. The complete
trained text encoder and full video generation remain pending downloads.

Reference-anchored continuation sampling is now implemented at the sampler
level, following the inspected upstream non-KV-cache AVC path. It fixes all
carried video latents plus an optional portrait anchor at timestep zero and
updates only the remaining latents. Anchor/time-position arguments are passed
to the actual MLX model. Invariant tests and resident-versus-streamed numerical
checks using the actual small model architecture passed (six streaming checks,
four sampler checks). Synthetic-weight wiring checks do not establish trained
long-video continuity. Long-form audio windowing, actual multi-window generation,
transition review and final original-audio export still require implementation
and validation before this can satisfy full-length voice-over generation.

Full voice-over preparation now passes on all 53.1093125 seconds of My_voice.wav.
The trained encoder processed two 30-second analysis windows with normalization
computed over the whole recording, retained valid encoder samples, and produced
1327 timed 25-fps feature frames. Peak MLX allocations were 1,581,541,398 bytes.
Report/cache: `.local/longcat-full-voice/`. This is analysis only; no full-length
avatar video has been generated. Audio reports now include sample count, timebase
and source SHA-256; exports reject media changed during rendering.

Added exact continuation-window planning in `backend/longcat_timeline.py`. Three
checks passed for boundary durations, invalid geometry and the real voice length.
It maps the 53.109-second track to 1328 output frames in 13 overlapping windows,
without gaps or duplicate output indices. The final fractional frame is retained;
only derived feature context may be padded, while original audio is remuxed once.
This planner still needs integration and actual multi-window render validation.

All three trained DiT shards completed SHA-256 verification. The complete
48-block trained transformer passed one finite-output forward call at the planned
29-frame diagnostic geometry with the real portrait and voice embeddings, using
zero diagnostic text embeddings while the full text model downloads. It took
33.63 seconds including image preparation/loading and peaked at 1,763,769,668 MLX
bytes. No video was generated by this test. Report: `.local/longcat-dit-report.json`.

A critical recipe difference was found in the pinned official demo:
`run_demo_avatar_single_audio_to_video.py:71` forces eight steps and text/audio
guidance 1.0 for distilled avatar-v1.5. The community demo's 4.0 defaults are not
used. The sampler now defaults to the official single conditional prediction,
and the still-waiting first render was refreshed before generation to use this
recipe. Requests record both guidance scales and a sampler revision so old
checkpoints cannot silently resume with different dynamics. Ten streaming/sampler
checks pass, including eight conditional predictions per DMD clip versus 24 calls
when explicit experimental guidance 4.0 is requested. Full text encoding,
complete denoising, export and visual quality remain pending.

Implemented the experimental full-length worker `backend/longcat_render.py`.
It connects complete-recording features, exact window planning, reference-anchored
continuation, streamed models, per-step recovery, SHA-verified completed-window
caches, single-encode assembly and original-audio verification. Source/model/code
fingerprints prevent incompatible cache recovery. Completed windows are read from
disk rather than keeping the lesson in unified memory. This code has not yet run
end to end with the complete checkpoint and is not the native default renderer.
Its actual-media preflight correctly recorded Failed with no output when model
installation was incomplete. No simulated result was produced.

Export verification now uses bundled FFmpeg with fatal decode errors and
passthrough frame timing, eliminating an OpenCV/legacy-environment dependency.
It passed exact PCM, all 48 frames and 25 fps on the existing partial candidate
from the isolated LongCat environment. Full-worker neural execution, continuity,
visual quality and native queue integration remain acceptance gates.

## First actual end-to-end LongCat generation

All 25,013,365,609 model bytes completed installation and large-weight hash
verification. The real eight-step unit-guidance pipeline generated 29 frames at
384×256 from Portrait_BlackT.png and the first 1.16 seconds of My_voice.wav.
Render preparation/inference/export took 257.14 seconds excluding download wait;
peak MLX allocation was 3,051,056,384 bytes. Every frame decoded successfully and
original excerpt PCM matched exactly. Output:
`.local/longcat-first-render/longcat-compatibility.mov`; provenance and checks are
in its report.json. Independent export verification also passed.

Inspected actual full-frame and face-crop sheets. Facial animation, blinking and
head motion are visible and identity is recognizable at this diagnostic size.
However, the output is low resolution, facial detail is soft, and an invented hand
enters the frame near the end. This is NOT an accepted quality result or the final
software example. Preserving the source body/background and restraining motion
remain necessary; do not claim that prompt text alone guarantees preservation.

Started a genuine 121-frame (~4.84-second) 768×512 test with the same source/audio
and a restrained-motion prompt. It uses the released 8-step guidance-1 recipe and
is logged at `.local/longcat-quality-five.log`. Quality remains unapproved until
this longer actual output and any preservation compositor are reviewed. Native
full-length generation and production acceptance remain outstanding.

Implemented a bounded head/neck compositor in `backend/head_composite.py` using
validated Vision face bounds and the exact aspect-fit transform. It retains source
RGB pixels outside that region before encoding. Three CPU checks pass for pixel
preservation, padded geometry and invalid input rejection. Applied it to the
actual first LongCat clip using source-matched native face metadata; the invented
hand is removed while facial animation remains. Inspected the raw/preserved
comparison sheet. Output: `.local/longcat-first-render/preserved-photo.mov`.
All 29 frames and original source PCM passed verification.

This is still a short diagnostic. The postprocessed comparison entails an extra
encode; the final worker should composite raw decoded frames before its single
encode. Head motion, blend boundaries, identity and sharpness require the longer
higher-resolution review. A fixed head mask also cannot guarantee absence of
occluding gestures inside the animated region; review remains necessary.

Source-body preservation is connected to the experimental native worker before
its single final encode. Native jobs persist the validated normalized face bounds;
older prepared-photo requests can read their existing native landmark sidecar.
Missing metadata produces reimport guidance. Masks/appearance flags and compositor
code are included in recovery fingerprints; continued windows use preserved
frames as context. Four compositor checks and native saved-job migration checks
pass, and the app compiles. Longer rendered appearance/boundary review is pending.

The 121-frame 768×512 run completed its first denoising step at render elapsed
711.46 seconds, with peak 9,085,385,082 MLX bytes. This measured stage is slower
than the isolated-block estimate. A live process sample showed Metal evaluation;
the render was not restarted based on the initially quiet progress log.

Queued another actual implementation comparison: ComfyUI-LivePortraitKJ transfers
the locally generated LongCat performance back onto the uploaded source photo.
The helper uses the shared inference lease and currently waits behind the larger
render, preventing concurrent GPU model loading. This honors the requested KJ
candidate and may provide another appearance-preservation route, but no quality
improvement is claimed until its actual output is reviewed. Log:
`.local/longcat-kj-comparison.log`.

### Current quality sequence (2026-11-06)

The real 121-frame 768×512 evaluation remains active, with two of eight denoising
steps completed at the latest inspection. The actual KJ appearance-transfer
comparison waits on the shared inference lease. A 512×512 head-crop comparison
is queued after successful KJ completion, using the same photo, voice prefix and
trained model. It reserves more input pixels for facial detail while keeping the
original full-photo body/background during reprojection. These queued evaluations
are not generated results or quality approvals. See VALIDATION.md for measured
crop geometry, native draft/playback checks and the recovered queue error.

### Continuation-reference correction

The local upstream attention graph separates portrait and carried-motion queries
even when `mask_frame_range=0`; the pinned community port did not. An external
attention subclass now corrects that branch without modifying the vendor checkout
or model weights. Numerical checks, exact context-isolation and unchanged
image-to-video checks pass. Full-length pretrained continuation still requires a
real render and visual inspection; see VALIDATION.md. Current in-flight short
image-to-video inference has not been restarted or altered.
