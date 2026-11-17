# Validation — 2026-11-04

Passed: native SwiftUI debug/release compilation and signed app launch; native UI imports of both Suman photos and 53.1-second audio; Vision facial landmark validation and person segmentation; byte-identical original copies; SQLite round-trip; actual MPS tensor operation; actual MediaPipe face tracking on Apple M5; seven worker lifecycle/error/cancellation tests. Missing-model error and Retry control verified in app.

Real MuseTalk inference is being tested. No perceptual quality, reference-motion success, camera capture success, or production-readiness claim follows from build/import success.

Pending release QA: physical camera/microphone, reference-motion fidelity, light mode/accessibility, long render memory pressure, crash recovery, all export audio codecs, notarization and self-contained installer.

## Actual MuseTalk success

Four-second user voice excerpt + checkered-shirt portrait: MuseTalk v1.5, PyTorch 2.15.0, MPS, float32, batch 1. Produced 100 frames at 25 fps in 49.14 seconds. Export fully decoded successfully and had expected frame count. Original audio stream copied into MOV; independent decoded float PCM byte comparison passed. Four sampled frames inspected: visibly different mouth shapes, preserved torso/clothing/background; generated face region shows smoothing. Listening/perceived synchronization review remains necessary.

Evidence: `.local/smoke/provenance.json`, `.local/smoke/export.mov`, `.local/smoke/review.jpg`. These private artifacts are excluded from version control. Full 53.109-second render started separately.

Optional motion: model weights installed; MediaPipe crop and LivePortrait CPU ONNX landmark refinement executed successfully on Suman's portrait, producing 203 landmarks. Full animation/motion transfer is still unverified without a user reference video.

Full voice-over render PASSED: 53.109 seconds, 1,328 generated frames, 25 fps, landscape 1280×720, MuseTalk v1.5 on MPS; elapsed 504.34 seconds. Full export decoding and frame-count validation passed. Original stereo float audio stream copied. Independent original/export decoded-audio SHA-256 comparison passed. Output `.local/full/export.mov`; provenance `.local/full/provenance.json`.

## Quality correction: real video source

The user rejected still-photo quality. The supplied `my_video.mp4` is 56.3 seconds with 90-degree display rotation; FFmpeg normalization correctly produces upright landscape frames. Added direct video source mode, forward sequential playback with no reversed gesture loops, source-duration validation, deterministic VAE latent encoding, and smoothed face boxes. First 4-second actual new-audio render passed decoding and audio equality but visual comparison exposed severe lower-face smoothing/moustache loss. That candidate does not meet the quality goal. A tighter mouth blend is under evaluation to retain source moustache/chin. No HeyGen-equivalence claim is justified.

Tight-mouth-mask experiment also REJECTED: source facial hair is more visible but ghost edges and seams appear around the changed mouth. The experiment is opt-in (`experimental_mouth_mask`) and is not the default. Real-video mode works technically, but the current MuseTalk reconstruction still does not meet the requested likeness/quality target. Model replacement/benchmarking remains necessary; UI polish cannot resolve this limitation. Eight worker tests pass; updated native app release build passes.

User-directed ComfyUI-LivePortraitKJ path now installed and actually executed on M5 MPS. Two-second/50-frame KJ retargeting test and 16-frame real ComfyUI graph execution passed. See `docs/COMFYUI_KJ.md`. This uses driving-performance audio and is not a claim of arbitrary new-audio lip synchronization or native-app backend integration.

Trained LatentSync 1.5 checkpoint completed SHA-256 verification and actual MPS inference on KJ output: 20 denoising steps, 16-frame temporal batch plus short remainder (18 inferred frames), 0.64-second trimmed sample. Inference completed; initial final mux failed from a relative-path bug, corrected in runner and independently remuxed. Sampled faces retain visible moustache detail but quality is not approved. Longer 2-second/40-step test started. The earlier random-weight probe is superseded only for pretrained execution compatibility, not perceptual quality. Source KJ output remains a performance transfer; LatentSync supplies the changed audio-driven lip motion.

## Higher-resolution feasibility — 2026-11-05

KJ + trained LatentSync 1.5, 2-second/40-step MPS test completed in 384.45 seconds. Export corrected to exactly 50 video frames at 25 fps (stream-copy trimming retained two unwanted packets initially). Independent decoded-audio equality passed. Close-ups show more retained facial-hair detail than the rejected MuseTalk sample, but face softness remains below the user's HeyGen target; not approved as final renderer.

A 512-pixel/16-frame architecture memory probe first failed on a 4 GiB attention-score allocation. Query-chunked attention then completed the same architecture probe with finite output: 7.622 GB Metal driver memory, 9.33 seconds including construction. Random weights were used for the memory probe, so this is NOT a high-resolution generation-quality result. MPS fp16 attention equivalence test max error: 0.0000038147; masked-attention comparison also passed. Memory caps remain enabled.

The experimental benchmark now supports the actual 1.6 (512-resolution) configuration, query-chunked attention and sequential classifier-free guidance passes to reduce peak memory. Its separate checkpoint download has started; pretrained 1.6 video quality remains unverified. KJ remains the motion stage. The native app has not been promoted to these experimental quality backends.

## Trained 1.6 execution — 2026-11-05

Checkpoint SHA-256 verified: `0a478e89eb660f82da4c35dbdde8a5adfb27f99d1b4e50edd03729e1e98316d3`.
The first trained run exceeded the 8.29 GiB MPS process limit in feed-forward GELU. Added inference-only token chunking for Diffusers FeedForward, leaving temporal attention intact. Actual fp16 MPS feed-forward comparison had maximum absolute difference 0.0. Memory caps remain enabled.

Retry completed actual trained 512-resolution inference: 20 steps, 0.64-second diagnostic, approximately 347.96 seconds. Independent complete frame decoding found exactly 16 frames at 25 fps; decoded source/export audio SHA-256 matched. Artifact: `.local/latentsync-16-benchmark/review.mov`; labeled close-ups: `.local/latentsync-16-benchmark/review/faces.jpg`.

Visual inspection still shows softness, including resampling of the upper face during upstream paste-back. This diagnostic does not pass the realism goal and is too short to establish temporal quality. A separate CPU float32 compositor restricted to the model synthesis mask is implemented for controlled comparison; three tests establish unchanged pixels outside the mask, no-op behavior for an empty mask and correct inverse-affine positioning. Perceptual quality remains unverified.

The benchmark now validates source duration, trims audio-feature chunks to the exact requested timeline rather than allowing upstream's two extra endpoint frames to loop the motion, records observed MPS memory at each denoising step, and caches aligned generated faces for compositor comparison without rerunning diffusion. A five-second KJ source is being prepared for an actual new-voice test with `My_voice.wav`.

First five-second KJ execution completed (125 frames, 292.19 seconds). The subsequent long lip-sync run was deliberately cancelled after discovering that KJ's explicit retargeting branch overwrites relative head motion with the source pose. Its `.local/quality-five/progress.json` is marked cancelled; it is not a completed five-second result. A corrected source is being generated in `.local/kj-five-motion`.

The next benchmark saves alignment metadata and each decoded chunk atomically, allowing partial compositor inspection while later chunks run. These caches do not yet implement automatic diffusion resume. Strict meta-device/memory-mapped checkpoint loading avoids constructing a second full float32 model; all 1.6 parameter keys matched. Two actual MPS numerical regression tests and four CPU compositor tests passed. Eight existing worker lifecycle tests also passed.

Corrected KJ relative-motion run completed: 125 frames / five seconds in 324.62 seconds. All frames decoded and original performance audio matched. Eye-line roll standard deviation measured 0.95 degrees versus 0.37 in the prior retargeted output and 1.30 in the driving recording; these are supporting 2D diagnostics, not 3D-pose or perceptual acceptance.

The new-audio run (`.local/quality-five-motion`) saved two completed 16-frame chunks, then exceeded MPS memory at the next chunk. Completed decoded frames had accumulated on the GPU. The pipeline now transfers completed frames to CPU, restores one face at a time on MPS, clears unused GPU allocations between chunks, and supports explicit `--resume` from validated decoded chunks. Resume reused the first two chunks and passed the previous failure point; full-duration stability remains under test. Source paths/settings and source hashes are checked for resume. The older first cache lacks original hashes, so its first resume additionally checked source modification times before recording hashes.

Partial new-audio output (16 frames / 0.64 seconds) passed exact decoded-audio comparison. Masked CPU compositing retains surrounding detail but mouth softness remains; no perceptual pass. Subsequent fresh runs use FFV1 input preparation instead of an extra lossy H.264 encode. A five-frame check of the selected FFV1 format decoded pixel-for-pixel identically to the KJ source. RGB-format conversion was rejected for this check because it changed some values by up to 3; preserving the input pixel format passed.

Local CPU SyncNet timing calibration detected exactly the five-frame change produced by a deliberate 200 ms audio delay. Camera-control estimated offset: +240 ms, confidence 6.41; delayed control: +40 ms, confidence 6.27. HeyGen-reference estimate: -80 ms, confidence 8.32. These are diagnostic observations with a substituted MediaPipe cropper, not published benchmark scores or realism approval. See `research/SYNCNET_EVALUATION.md`.

### Native setup progress (2026-11-06)

Settings now displays LongCat model transfer totals and the first-render stage,
with an evaluation-log link. It counts verified-size completed files and unique
saved ranges only when their partial file exists with the expected size.
Downloaded weights remain distinct from completed generation and visual review.
The native status smoke check passed (orphan ranges, duplicate/out-of-bounds
ranges, completed files, and pending evaluation). Release compilation passed.
Native rendering integration and final application acceptance remain pending.

### LongCat installation reproducibility

Pinned the public model manifest in `backend/model_manifests/longcat-q4.json`
and source hashes in `research/longcat-source-lock.json`. The new installer
check-only path verified the current source and reported the exact 25,013,365,609
byte checkpoint total. A real subprocess lock-contention check confirmed the
updated downloader refuses a second writer before creating a completion marker.
Fresh-machine installation remains untested; this does not establish release
readiness. The already-running transfer predates the lock change and was left
undisturbed; subsequent invocations use the new lock.

### Library recovery and UI inspection

Fixed two data-safety paths: SQLite errors no longer return a default empty
library, and StudioStore assigns the writable database only after successful
loading. Saving without a loaded database now reports an error rather than
silently doing nothing. Native fixture checks passed for empty-library loading,
normal round-trip, corrupt-payload failure and preservation of the original
unreadable payload. Release compilation passed.

The packaged app was launched through computer use and its process remained
running, but both accessibility inspections failed (AX cannotComplete followed by
server timeoutReached). This is not a passed UI inspection. No screenshots or
native interaction results were available, so layout/flow acceptance remains open.

### Independent export verification strengthened

`scripts/verify_export.py` now decodes every frame and compares the decoded
count with expected/container counts, optionally checks fps, and supports an
explicit source-audio excerpt. It no longer represents frame-count metadata as
proof of complete decoding. The existing 48-frame/1.92-second partial KJ +
LatentSync candidate passed exact source PCM, complete frame decoding and 25 fps
checks. This verifies export integrity only; that candidate remains unapproved
for visual quality. The old 16-frame `.local/latentsync-16-benchmark/review.mov`
did not match either original voice/camera PCM in this check; its old report lacks
source-audio provenance. Treat its original-audio-preservation claim as unverified
and do not present it as an accepted deliverable. New LongCat runs verify their
explicit source before reporting success.

Independent text-encoder downloads now use a separate staging folder and reverse
shard order. Completed files are hash-verified and atomically linked into the
main installation. A fixture check passed for idempotent publication and
preserving conflicting destination files. Subset downloads do not publish the
full installation's completion marker.

### Shared inference coordination

Added an OS file lease shared by the legacy worker, full-length LongCat worker
and queued LongCat benchmark. It is acquired before memory-heavy model stages,
not while model downloads are pending. Waiting callers remain cancellable and
show a native wait state. Two real-process checks passed: cancellation while a
separate process owns the lease, automatic release on owner termination, and
release after a model exception. All nine existing worker checks also pass.
The still-waiting benchmark was refreshed to use the lease. Its report now
separates actual render elapsed time from total time including downloads.

### Native LongCat evaluation routing

Added a persisted engine setting and per-job engine snapshot. Existing saved jobs
without an engine field decode to the legacy route; the migration smoke check
passed, including LongCat encode/decode. The native queue now launches the
isolated LongCat worker when explicitly selected. It blocks incomplete model
installation and unsupported reference-video/motion inputs rather than silently
ignoring them. LongCat jobs can resume the same verified cache or start a new job
without deleting the previous cache. The default remains legacy pending visual
acceptance, and the UI labels LongCat as evaluation.

LongCat diagnostics ran on the actual M5: real Metal tensor test and bundled
FFmpeg passed, both remaining text shards were reported missing, and full video
inference/quality remained false. Native build passed. End-to-end queue generation
and UI interaction still require actual testing once models finish downloading.

### Actual native queue failure integration

StudioStore now accepts isolated storage/runtime/preferences for integration
checks. The command-line Swift harness launched the actual LongCat Python worker
through the same Process route used by the app. A missing-media job reached Failed
with the worker's actionable error, empty output and persisted SQLite state, while
retaining the job ID for recovery. User library/preferences were not used.
The check passed after adding a terminal-state cancellation guard: cancelling or
shutting down cannot relabel an already failed/completed job as cancelled. Native
shutdown also stops polling. Release compilation passed. This is failure-path
integration evidence, not a successful neural generation or visual UI pass.

### Actual native UI observations

Computer-use access recovered. Inspected the running app's Dashboard, Settings,
Render Queue and Create Video. Confirmed live model-transfer totals, no active
native render and no draft audio/notes before closing the older idle build. The
updated build initially timed out through the UI connection; a thread sample
showed the normal AppKit event loop, and attaching by app name then succeeded.
Verified the new LongCat evaluation selector, selected it as requested, and ran
its actual native diagnostics (Metal and FFmpeg true, missing text shards listed,
video verification and quality approval false).

Imported the supplied My_voice.wav through the real file picker into the Suman /
Black T studio. The app displayed 53.1 seconds, an enabled audio player and the
waveform, with LongCat's actual 768×432 export setting and explicit evaluation
limitations. Pressing Add to render queue before installation completed produced
an actionable downloading alert instead of a fake render. Dismissed the alert and
left the prepared studio open. This validates these UI paths only; successful
native neural generation, playback of that output and final visual quality remain
pending. The imported project name is LongCat quality review.

### Native audio timeline transport

Added a dedicated AVPlayer transport with live position observation, scrubber,
pause/resume, natural completion/restart, replacement cleanup and stale-observer
protection. The studio now binds its displayed clock and seek slider to actual
playback. A muted local silence fixture exercised real AVFoundation advancement,
pause, seeking/resume, end notification, restart and recording replacement; the
check passed. This fixture is transport test data, not avatar-generation output.
The integrated app compiled; visual playback-control interaction in the updated
bundle still needs inspection. The active high-resolution neural render and
queued KJ comparison were left running.

### Draft persistence and real playback UI (2026-11-06)

Added backward-compatible SQLite draft storage for the selected avatar/outfit,
voice-over path/duration/waveform, production notes, project name, framing,
background settings, reference choices and engine. Autosaves changed drafts while
the app is open and on normal shutdown. Missing files are cleared with recovery
guidance while the remaining draft is retained. Isolated native checks passed
periodic autosave, reopen, Unicode notes, settings restoration, absent-audio
recovery and decoding a library from before drafts existed.

Built and launched the native bundle. Imported the supplied recording through
NSOpenPanel, observed actual AVPlayer clock advancement, paused it and sought
forward using the native slider's accessibility increment. Direct accessibility
setValue is unavailable for this SwiftUI slider; its supported increment worked.
Quit and reopened the actual app: Create Video restored LongCat quality review,
Suman / Black T, the 53.1-second recording, waveform, original background and
LongCat evaluation mode. Screenshot inspected. Playback resets to the beginning
on reopening; the playback cursor is not part of the saved draft.

### Camera-save and invalid-request recovery

Camera capture now removes its temporary recording only after the library copy
and database save report success. On failure, it retains the temporary file and
shows its recovery path. A delegate-level test passed failed-save retention and
successful-save cleanup without accessing camera/microphone hardware. Actual
camera-device recording remains unverified.

Repeating native queue recovery during the active high-resolution render exposed
an error: the missing-media test waited for the shared inference lease and timed
out. Stopped only that isolated test worker. LongCat now validates consent/input
existence before waiting for the lease, and revalidates inside the render. The
native queue → real worker → actionable missing-media failure → SQLite recovery
check then passed. The quality render was never interrupted.

### Head-detail comparison prepared and queued

Added research/longcat_head_benchmark.py. It crops the real reference photograph
around the prepared Vision face bounds, runs the actual trained LongCat benchmark
at 512×512, reprojects that generated crop onto the original photo, retains source
pixels outside the feathered head region before encoding, and independently
verifies the original audio excerpt and full frame count. It creates no replacement
portrait or body. The original 1535×1024 source needs one black padding column for
even H.264 dimensions; source pixels are not stretched.

The measured crop is [353,1,1060,708]. It allocates approximately 189 model pixels
across the face, versus 131 in the active full-frame 768×512 test. This geometric
increase does not establish improved quality. Syntax and actual-source geometry
checks passed. A 29-frame actual comparison is queued in
.local/run_head_comparison_after_kj.py; it will start only if the pending KJ
comparison completes. No neural head-crop result exists yet. Review the real
output before integrating this approach into the native renderer.

### Full-render disk preflight

The full LongCat worker now checks available space before description/reference
inference, budgeting cached RGB windows, final raw assembly and 2 GiB of working
headroom. Low-space failure tells the user to free space and resume; caches are
retained. A fault-injected low-space → restored-space retry check passed. This
is a conservative check; free space may still change during a long render and
filesystem write errors continue through the normal recoverable job failure.
The current 121-frame benchmark is separate and was not restarted for this fix.

At the latest live inspection the high-resolution benchmark completed step 3/8
in 2385.55 seconds of rendering, with tracked MLX peak 9,121,560,954 bytes. This is
not total process/system memory. Neither that incomplete run nor its queued
comparisons has a visual approval.

### Full-resolution photo preparation (2026-11-06)

Found that native preparation resized both its face-analysis image and the
render-cache photograph to a maximum side of 1280. Face analysis remains bounded,
but new imports now cache an orientation-normalized PNG at full source resolution.
Original imported files remain byte-for-byte unchanged. Metadata version 2 records
analysis and prepared sizes, and failed preparation cleans all of its partial
artifacts. Actual Vision validation of both supplied portraits, full prepared-size
checks, original-byte checks, person segmentation and SQLite round trip passed.
Existing outfits are not silently rewritten because old render/cache fingerprints
may depend on their prepared image. Replace/reimport an existing outfit to obtain
the new full-resolution preparation; the active research renders use the original
full-resolution source directly and were not changed.

### Corrected long-video reference attention

Comparing the pinned MLX port with Meituan's local reference source found that
`num_cond_latents > 1, mask_frame_range=0` incorrectly merged the portrait-anchor
and carried-video conditioning queries. The reference always isolates portrait
self-attention from carried-context self-attention; generated queries attend to
all tokens. This can affect the continuation path, which has not yet had an
accepted full neural run.

Added research/longcat_attention.py as an external subclass, preserving parameter
names, published weights and the pinned vendor checkout. Streamed blocks install
it before quantization/strict loading. A full-render cache fingerprint includes
the new helper. Image-to-video (`num_cond_latents=1`) delegates unchanged to the
pinned implementation, so the active short benchmark is unaffected.

Checks passed: independent NumPy dense-attention reference for mixed and entirely
conditioned sequences; exact carried-context invariance when portrait tokens are
perturbed; identical image-to-video outputs with the same parameters; and the
resident/streamed architecture, quantized tensor loading, cancellation and sampler
suite (12 checks total). The initial CPU-vs-Metal test used 1e-5 tolerance and
failed with max difference 0.00193489. The pinned port documents SDPA numerical
accumulation differences; the independent comparison now uses 3e-3 while the
context-isolation and unchanged-path invariants remain exact. This is numerical
infrastructure evidence, not pretrained full-video or visual-quality evidence.
Head-composite (4) and exact timeline (3) checks also passed. The pinned vendor
source still passes its complete source-hash lock.

### Per-attempt queue status and custom-background errors

Native render requests now carry a unique attempt ID. Both real Python worker
entry points include it in every status update; active native polling accepts
only its current attempt. This prevents stale failure/completion files or a
shutting-down earlier process from changing a resumed job's live state. Job IDs,
verified neural caches and model/source fingerprints remain stable across resume.
The actual LongCat missing-media worker/native SQLite recovery check passed after
this protocol change, and all 9 worker lifecycle tests passed.

A separate native subprocess fixture deliberately published an older attempt's
completion before its current-attempt failure. The app ignored the stale output
and remained Rendering, then accepted the current worker's state: passed. This
fixture performs no inference and creates no video. Its first launch used an
Apple Python shim relocated through a symlink and failed before publishing any
status; switched the fixture to the verified local Python binary and added an
explicit startup check before assessing the status race.

Custom background composition previously fell back silently to a solid color
when a selected image could not be decoded. It now rejects missing/corrupt
backgrounds before segmentation, with import-again guidance, and checks color
conversion instead of force-unwrapping it. Native checks passed both missing and
damaged backgrounds, retaining source bytes and a pre-existing output unchanged.

The latest native bundle compiled and passed strict ad-hoc signature verification.
Closed the older idle app, launched the updated bundle, and inspected Create Video:
the LongCat quality review draft still restores Suman / Black T, 53.1-second audio,
waveform, original background and the selected LongCat evaluation format. No full
native neural job was enqueued during the ongoing research render.

### Fine-grained render progress

Streamed DiT blocks now support an optional completion callback, invoked only
after their actual result has been evaluated. The full LongCat worker uses it
to report segment, denoising step and completed block, improving feedback during
very long steps. Progress is based on completed computation, not a timer or
simulated frames. The streamed/resident architecture check verifies block events;
all 12 streaming, attention and sampler checks passed after the change. The
already-running benchmark retains its original per-step reporting and was not
restarted. Latest live observation: step 4/8, render elapsed 3288.13 seconds,
tracked MLX peak 9,121,560,954 bytes. KJ and head-crop evaluations remain queued.

Visually inspected the native full-resolution gray-background composite from the
actual Black T photo: face, hair and clothing are retained with the source framing.
This is static segmentation/compositing evidence, not an AI-generated video or
proof of temporal compositing quality.

### Native capture lifecycle serialization (2026-11-06)

Moved camera session configuration/start/stop and movie-output changes into a
private serial controller. The observable capture model is main-actor isolated;
delegate callbacks transfer only UI updates to that actor. The controller's
unchecked Sendable boundary is limited to queue-confined capture operations and
the immutable session shared with Apple's preview layer. Release compilation no
longer reports the previous camera weak-capture/Sendable warnings (the existing
SDK linker-search warnings remain).

New lifecycle checks inject a controller/permission provider and access no actual
camera or microphone. They pass denied permission, a stopped permission request,
countdown cancellation across rapid preview restart and duplicate-record guards.
The delegate-level failed-save retention/successful-save cleanup check also
passes. Native recordings are now staged under the persistent local library's
recordings folder, rather than a temporary directory, until confirmed saved.

Asked the user for a five-second hardware recording test. They chose to test
recording themselves later. Camera and microphone were not activated. Actual
hardware recording remains an open user-assisted verification item; do not claim
it has passed from these injected lifecycle tests.

### Portable runtime component and relocated native worker

Added a builder that stages Python 3.12's framework/stdlib, the MLX environment,
non-system dylib dependencies, loader-relative paths and explicit engine source
files. Strict vendor-source hashes are retained. No user photographs, recordings,
project data, model weights or generated results are copied. The initial audit
mistook wheel dylib install IDs under /DLC for unresolved dependencies; all
absolute install IDs are now normalized before the final audit.

A moved artifact with spaces in its path passed actual Metal computation, PNG and
float-WAV codecs, pyloudnorm analysis, Whisper feature preparation, offline
multilingual tokenization and FFmpeg startup from an unrelated working directory.
Python network connections were blocked. All 562 loaded native images resolved
to the artifact or Apple system locations. The package audit covered 338 native
binaries, with no external symlinks or absolute non-system dylib links remaining.
The actual native queue → relocated LongCat worker → missing-media error → local
SQLite recovery check also passed. No neural generation was claimed by these
checks. The app's default runtime was not changed during the active render.

Collected dependency notices, the Python relocation-change summary and the media
binary's own licensing output. The FFmpeg binary is GPL-enabled; imageio-ffmpeg's
wrapper license alone does not cover it. Distribution review remains explicitly
required. Found the pinned mlx-metal wheel requires macOS 26; added an early
installer guard and diagnostic metadata. Actual runtime tests were on macOS 27.2.
See PORTABLE_RUNTIME.md for reproducible commands and remaining release gates.

The latest camera build was launched through the native UI. Motion Studio listed
MacBook Air Camera, showed Start preview, and kept Record with microphone disabled.
Neither button was pressed; camera and microphone stayed off as requested. Returned
to Create Video and verified the 53.1-second draft, outfit and waveform remained
restored. The research render is independent of this app restart and stayed live.
Latest checkpoint at inspection: step 6/8, render elapsed 4981.55 seconds.

### Source-resolution-aware appearance review

Added research/appearance_review.py to extract independently located fixed face
crops and full framing from actual generated clips. Columns identify the static
original photo, generated candidates and external HeyGen reference separately.
Rows sample each clip's own duration and explicitly are not speech-synchronized.
Source dimensions and approximate detected face-pixel widths are displayed;
ordinary display interpolation is not presented as recovered detail or AI output.
The reference may use an explicitly recorded pre-outro sampling window (0–13 s
here); generated candidates always include their final frame.

The first run lacked a native OpenGL context in the restricted tool environment;
the same MediaPipe analysis completed with normal native graphics access. Actual
inspection confirms the existing 384×256 diagnostic loses substantial detail:
detected face width about 60 pixels, versus about 237 in the original photo and
306 in the external reference. Body-preserving post-composition does not restore
that lost facial detail. These are review observations, not identity or realism
scores. The 768×512 test and queued head-crop comparison remain necessary.

Added a configurable 2–4× face crop to the head benchmark and a cosine edge taper
for reprojection, keeping pixels outside the crop unchanged. Actual-source
geometry checks give crop [445,92,969,616] at factor 2.0 (about 256 Vision-face-box
model pixels), versus [353,1,1060,708] at factor 2.7 (about 189). No tighter-crop
neural result is claimed; it is available for a controlled follow-up if needed.
The queued default stays 2.7. A review runner now waits for the real full-photo
benchmark's completed, decoded/audio-verified report, verifies its source hashes,
then creates a body-preserved export and appearance sheets. It does not restart
or generate a substitute for the active neural render.

### Atomic voice-over replacement

Moved native audio decoding and copying off the UI actor into AudioPreparation.
A new recording replaces the existing draft only after complete validation.
Failed copies are removed without clearing the selected recording, duration or
waveform. Invalid non-finite samples fail with re-export guidance. Waveforms now
consider every channel and bound only their displayed amplitude; original copied
audio bytes remain unchanged. Keyboard/menu imports also respect the busy guard.

A native test passed right-channel-only audio, exact original/copy bytes, damaged
file rejection, non-finite float sample rejection, failed-copy cleanup and saved
draft retention. Fixtures were local audio-test data, not avatar output or
microphone recordings. Hardware capture remains deferred at the user's request.

### Actual 121-frame LongCat result and user review

The 768×512 / 121-frame render completed. Actual rendering took 6766.21 seconds;
tracked MLX peak reached 10,027,117,824 bytes including temporal VAE decoding.
Every exported frame decoded and the 4.84-second original audio excerpt matched
exactly as decoded float PCM. The automatic review runner produced and separately
verified a body/background-preserved version plus labeled appearance sheets.

Visual inspection shows substantially better facial detail than the earlier
384×256 diagnostic, with a detected face width around 120 rather than 60 pixels.
The supplied HeyGen reference is still around 306 pixels across its detected face.
Presented the real 4.84-second preserved-body video to the user. Their feedback:
"Lip sync or motion needs work." This is not an accepted-quality result and must
not be promoted to the final renderer or production release.

The calibrated local CPU SyncNet diagnostic estimated -2 frames / -80 ms,
distance 8.3625, confidence 5.7524. The external-reference control also estimated
-80 ms but had distance 6.9309/confidence 8.3223. These values do not establish
natural mouth shapes, motion, language accuracy or perceptual acceptance. No
audio shift was applied; the original recording remains untouched.

### Actual KJ bridge retry and queued comparisons

The first LongCat→KJ bridge attempt failed before animation: resizing a 3:2 driver
to a maximum 640 pixels produced height 427, which libx264 rejects. Added one-pixel
padding to even dimensions in the actual KJ adapter, the Comfy graph I/O node and
legacy video normalization. A new KJ attempt then completed all 29 frames in
64.88 seconds of node execution, with exact original audio verified. Its labeled
appearance sheet shows preserved source detail versus its low-resolution driver;
full-duration perceptual quality still requires review.

The queued head-crop attempt initially failed at MLX import because its observer
had been started in the restricted tool environment. It produced no generated
frames. Restarted only that failed comparison with normal Metal access and a new
output directory; the 512×512 pretrained head-crop run is live. The completed
121-frame benchmark was never restarted. A KJ transfer driven by that complete
4.84-second output is queued behind the active head test to investigate the user's
lip/motion feedback. Both remain unapproved.

### Personal-motion test: explicit quality rejection

The personal reference variant completed 121 frames in 347.76 seconds of KJ node
execution. Original audio and full decoding passed. It used measured pose and
non-mouth expression changes from the supplied my_video.mp4, retained audio-driver
mouth coordinates, and used the real ComfyUI-LivePortraitKJ nodes. This is motion
transfer, not personalized model training. Numerical mouth-isolation and rotation
checks passed before the actual run.

Observable eye-line roll range fell from 13.81 degrees in the raw LongCat clip,
to 6.63 in its generic KJ transfer, to 4.17 in the personal-motion version. All 121
frames were tracked. These 2D measures are not perceptual quality scores. The
personal result's CPU SyncNet estimate remained -80 ms, distance 8.8048/confidence
4.4665, also not a quality pass.

Presented the actual personal-motion clip. The user rejected it: "completely
unnatural, very unrealistic movement, expressions look unnatural, teeth and mouth
movement unrealistic". Do not promote it, extend it overnight as the selected
renderer, or describe reduced motion as achieved realism. Earlier clarification
identified head/expressions as the main initial issue; the final rejection also
explicitly includes teeth and mouth.

Starting a controlled LatentSync 1.6 run on the actual camera footage with the new
voice-over. Recorded head/eye motion can be retained while the synthesis-mask
compositor restricts changed pixels. This isolates reconstruction damage and does
not replace the original photo/outfit requirement. The benchmark now participates
in the shared inference lease. No new camera/microphone capture was activated.

### Camera-mouth control: stopped after visible reconstruction failure

The first 16 pretrained LatentSync 1.6 frames completed (0.64 seconds, 40 steps).
The same-frame comparison shows smeared lips and mouth cavity and loss of lip/
facial-hair detail versus the actual camera frames. Stopped the diagnostic rather
than spending roughly two more hours on the same failed reconstruction. The
requested 4.84-second render did not finish; do not describe it as completed.
Evidence and the explicit rejection decision are in `.local/camera-mouth-control/`.

The new `--audit` compositor option checked every pre-encode frame: original RGB
outside the full projected synthesis support remains exact. All 16 encoded frames
decode and the original 0.64-second audio excerpt is exactly preserved as decoded
PCM. Four compositor checks pass. These checks establish preservation, not mouth
realism, articulation or perceptual acceptance. The research comparison initially
failed because the saved affine was float16; the OpenCV-only diagnostic now casts
it to float32. The model was not rerun to correct that review-tool failure.

### FlashHead Pro: compatibility work, no generated result yet

Pinned official source `9bc03de06bb0de82cd6bc477804512ae06144bf2`, model revision
`5fbd2e44ba08ab2deb28bd55cff2c15523e51ba5`. Selected Pro transformer + Wan VAE:
6,538,483,761 bytes including configuration/card. Download and LFS SHA checks are
in progress. The official model card identifies those components and its Wav2Vec2
encoder as Apache-2.0; the excluded Lite VAE has a different license. Source lock
and exact model inventory are in `research/flashhead-source-lock.json`.

The external adapter leaves pinned vendor files untouched, replaces unsupported
float64/complex rotary math with real float32 operations, disables distributed
CUDA paths and splits independent attention queries into groups of 256. Four
CPU mathematical checks pass against the official operations. The first test
loader restored the entire Python module dictionary, accidentally unloading
Torch internals; corrected it to restore only its temporary NVIDIA import names.

Actual Metal probe: the official small transformer block matches the CPU result
with maximum absolute difference 1.55e-6. A synthetic full-sequence 9,216×1,536
attention call produced finite output in 0.210 seconds; observed driver allocation
1,108,033,536 bytes. This is an operation probe, not full-model memory or quality.
No avatar video was produced by this probe.

The separately SHA-verified Wav2Vec2 checkpoint strictly loaded on CPU and processed
the user's actual first 1.12 seconds in the official eight-second streaming
context. Output `[1,33,5,12,768]` was finite; loading + feature extraction took
0.610 seconds. Automatic Transformers loading initially left trained positional
convolution weights random because of old weight-g/v names versus current Torch
parametrizations; this was caught before inference. The adapter now maps those
trained tensors explicitly and rejects missing/mismatched weights. Inference-only
SpecAugment is disabled; its absent training mask token is not invented. The
unused CTC text head is explicitly excluded. No training was performed.

The new benchmark implements the official four-step Pro sampler, shifted
timesteps, five conditioning frames, rolling audio context, actual VAE motion
carry and original-audio verification. Full pretrained generation, visual quality
and app integration remain unverified. It must pass those gates before selection.

The actual pretrained Wan video VAE then passed a 33-frame 512×512 reconstruction
of the uploaded photo crop on MPS/bfloat16. Elapsed 23.98 seconds; maximum observed
driver allocation 6,710,837,248 bytes; finite latent shape `[16,9,64,64]` and all 33
decoded frames. The labeled comparison retains visible face, hair, moustache and
teeth detail; mid-frame mean absolute RGB difference was 1.20/255. This is a still
reconstruction through a temporal autoencoder, not generated talking video and
not a perceptual score. The full portrait transformer's weight names and shapes
also match all 843 checkpoint tensors / 1,507,694,912 parameters from its fully
transferred header; no incomplete weight data was loaded. Full-file SHA validation
is still required before pretrained transformer inference.

### FlashHead generation, user review and native integration

All selected FlashHead weights subsequently passed full-file SHA verification.
Real 121-frame / 4.84-second generation completed in 279.18 seconds, observed MPS
driver maximum 6.67 GB. The user reported “Clear improvement, but still needs work.”
A tighter real-photo crop increased detected model face width from about 166 to
223 pixels. Its 121-frame generation took 291.03 seconds, observed driver maximum
7.01 GB. The user then confirmed “Both improved, still needs refinement” after
bounded rigid stabilization. Neither response is final quality acceptance.

The stabilization applies only small rigid coordinate corrections, capped at
3 model pixels and 0.75 degrees, without transferring expressions or changing
audio. It shares a single resampling with final source-photo projection. The
source body/background outside the head mask remain exact before encoding.
All 121 output frames and original decoded PCM passed in both variants.

Actual pretrained recovery: three windows executed, including model reload from
committed state. Resumed output exactly matched uninterrupted output, and the
extracted engine's first/second windows exactly matched the research benchmark.
The worker then passed a real cancellation after three committed windows and
resumed without regenerating those windows. A native MediaPipe tracking crash
interrupted the first export attempt; isolating tracking in its own process and
passing writable RGB copies fixed the stage. All five generated sections were
reused and the verified 1080p export completed in 15.41 seconds. The first four
worker windows match the research benchmark exactly. The last differs because the
worker test uses a physically cut voice file: resampling differs in its final 16
analysis samples by at most 4.00e-5 versus analysis of the full original recording.
This is a source-boundary difference, not a claimed exact whole-clip equivalence.

The actual Swift store/queue cold test imported a consenting real photo and voice
excerpt, generated all sections, exported verified 1920×1080 video and persisted
Completed in SQLite. Original photo bytes and full 1535×1024 render-source size
were checked. Render elapsed 297.78 seconds. Evidence:
`.local/native-flashhead-queue/last-result.json`. This is a native integration test,
not a GUI click test. Existing project migration and setup-status tests pass.

The visible app relaunch exposed a real UI stall: LocalModelStatus file IO was
blocking the main thread in open(). Moved setup reads into a background task with
runtime/engine generation guards. A real blocked-FIFO test proves the UI caller
returns promptly and stale results do not replace newer settings. The rebuilt app
then opened immediately; actual Settings showed FlashHead selected, complete local
models and passing Metal diagnostics. Create Video exposes optional head-jitter
reduction and 1080p export. No camera or microphone was activated.

A standalone FlashHead runtime was packaged under the app's Application Support
folder. First relocation failed on differing Intel/ARM LC_ID_DYLIB names in
universal protobuf/MediaPipe wheels; selecting and re-signing the arm64 slices
fixed relocation. The second package passed separate Torch/Metal and MediaPipe
probes with Python networking blocked, temporary CWD and minimal PATH. 704 and
682 loaded native images respectively remained in the runtime/system libraries.
No personal media was bundled. The public model files were locally cloned and
verified by app-launched diagnostics. Full inference from this installation and
full-length generation remain to be exercised. Dependency/model redistribution
license review and notarization remain incomplete.

Shared per-user GPU locking now coordinates source-checkout and installed-runtime
renderers. A second-runtime subprocess test confirms contention/cancellable wait,
preventing simultaneous large-model residency across different runtime roots.

### New HQvideo.mp4 source

The user supplied HQvideo.mp4, SHA-256
`b0f235793b9b359b34b1aacab881e89d76bcdf5ce4e475e492b77d3fd989d942`,
and explicitly confirmed camera footage with a replaced background after a visible
watermark prompted a provenance check. Do not describe it as an unedited original
camera file. It is 1920×1080, 24 fps, 239 frames, 9.96 seconds with 48 kHz stereo
AAC. The full original file is unchanged. All frames were tracked.

Selected actual frame 27 (1.125 seconds): open eyes, closed mouth, eye-line tilt
about 1.21 degrees. Selection used observed landmarks only, without image synthesis
or beautification. Native photo validation and unchanged-copy/full-resolution
checks passed. A 4.84-second local FlashHead worker comparison is running from
that real frame with the existing voice excerpt and stabilization. The supplied
video is the appearance source here, not an implemented motion driver. Results
and provenance are under `.local/flashhead-hq-five/`; no quality verdict yet.

Current Python suite: 41 tests passed, 2 skipped (environment-dependent Metal
checks). Separate actual Metal compatibility and neural recovery tests described
above were executed with native device access.

### HQ-source outcome and preferred-candidate selection

The HQ-frame worker completed all 121 frames at 1920×1080 in 387.21 seconds.
Original audio decoded exactly and every frame verified. The user reviewed the
actual clip and said “Previous photo version looks better.” The sharper original
Black T photo remains the preferred candidate; the new-source clip is retained as
a comparison, not promoted. No claim that a higher input resolution improved the
perceived output follows. Local timing diagnostic: -1 frame / -40 ms, distance
8.8905, confidence 4.8031; not a realism score.

The actual visible app queued the full 53.1-second voice-over with the preferred
photo and stabilization. It correctly waits behind a native-queue 15-second test
using the installed runtime. A brief app update cancelled only that waiting job;
the saved queue entry was restored and resumed through the UI. Its job ID remains
43F32244-6B58-43B4-A2FD-1ACB67A1FA4C. The ongoing shorter test and queued full voice
are not completed results. No generated progress was discarded by the UI update.

The new native “Choose a frame from video” flow passed real-media checks and
visible UI execution: open HQvideo.mp4, scrub preview, save frame, view new outfit.
It preserves an unchanged complete local copy of the video, a separate derived
frame/cache and the actual frame time. GUI import at a requested 0.99 s stored
the decoded 0.958333 s frame. The local video-copy SHA exactly matches the source.
Native invalid-time testing leaves no orphan files and preparation retains full
1920×1080 image dimensions. No recording hardware was activated. The new library
entry is labeled “HQ video picker • edited background”; Black T was reselected.

A file-picker automation attempt from a hidden research folder left Open disabled;
a normal visible file location worked. A later stack sample showed the main thread
idle rather than blocked, distinguishing this from the earlier genuine main-thread
setup-read stall. The new direct-video picker successfully selected HQvideo.mp4
from the project root. Remaining hidden-folder import behavior is not claimed
fixed. Light-mode workspace readability and responsive navigation were inspected
during active inference; dark appearance was restored afterward.

### Preview and portrait export checks

Replaced synchronous full-image opening in PhotoView with a bounded background
thumbnail cache (maximum 1600 pixels); full render inputs remain unchanged. The
real 1080p-source test checks decoding, bounded dimensions, missing-file behavior
and unchanged source bytes. Current Swift debug build passes. The visible release
already includes the real-media-validated video-frame picker; the preview cache
change is not yet loaded into that running app.

Portrait export was independently exercised from the verified real FlashHead RGB
cache at 1080×1920. All 121 frames and the original complete voice excerpt pass
verification. This is a framing/export test reusing actual generation, not new
inference or a visual-quality approval. Source body pixels outside the head mask
are preserved before encoding. Evidence: `.local/flashhead-portrait-review/`.

A labeled comparison of the actual native-queue cold render against the preferred
research sample shows consistent identity/detail in the inspected frames. The
export canvases differ (1920×1080 vs 1536×1024); neither display upscaling nor the
comparison constitutes a higher-resolution neural face result.

### Installed-runtime 15-second generation completed

The real native queue launched the relocated runtime and completed 375 frames /
15.0 seconds at 1920×1080. Original decoded audio, all frames and SQLite completion
passed. Elapsed 1,352.74 seconds (22.55 minutes), longer than extrapolating the
short diagnostic. This is an observed sustained render time, not an FPS claim.
Evidence: `.local/flashhead-installed-fifteen/last-result.json`. The full voice job
then acquired the shared GPU lease and started through the visible app's queue.

The full 15-second local SyncNet timing diagnostic estimated -1 frame / -40 ms,
distance 8.5863, confidence 4.5768. This does not demonstrate natural teeth, mouth
shapes or pose. No audio was shifted. Adjacent actual generated RGB frames at all
13 window joins have paginated review sheets. Inspected early and final joins do
not show a gross identity replacement, but normal-speed perceptual review and
the user's feedback remain necessary. Do not claim final acceptance.

The queued full-length worker now saves its own stage-callback memory observations
per committed window. The first observed driver allocation maximum is
6,706,528,256 bytes and tensor allocation maximum 3,051,764,736 bytes. These are
observations, not continuous peaks, total system memory or guaranteed caps. This
reporting change does not alter neural equations, weights, sampling or RNG.

### OS-enforced offline continuation

The native launcher now invokes render workers and their descendants with macOS
network access denied, without changing system network settings. A test connection
was refused with PermissionError, while actual Metal computation still passed.
The separate MediaPipe tracking graph also passed under the same OS profile. The
native stale-attempt protocol test passes with that launcher. This enforcement
is separate from Hugging Face's existing offline/local-files-only settings.

After eight committed full-voice sections, the job was safely cancelled, the app
reloaded, and the same job resumed at section nine under the network restriction.
The recorded launch.json identifies the OS profile and current attempt. Saved
model state and speech contexts were reused without changing neural cache identity.
The first 13 windows / 364 generated frames of the full run exactly match the
installed-runtime 15-second test, including newly computed offline continuation
after resume. The short test's final window has different padded future-audio
context and was deliberately excluded from that equality comparison. This is
reproducibility evidence, not visual approval.

Worker retry now archives prior request, status, launch record and worker log in
an attempts subfolder before launching a fresh attempt. The updated native protocol
test verifies archive retention and rejects stale completion status. This source
change is prepared for the next build; the running full render is left undisturbed.
# Motion quality rejection and source-anchored experiment

The user rejected the actual installed-runtime 15-second FlashHead output: only
roughly the first second was somewhat acceptable; face/expression motion and some
head movements in the remaining fourteen seconds were unrealistic. This is a
failed quality gate despite correct decoding, audio preservation and successful
local inference. The visible full-voice job was cancelled through its native
Cancel button at section 23/48. Worker status confirms cancellation and retained
completed sections; no full-length output is claimed.

`research/expression_review.py` tracked all 375 normalized frames of the generated
clip, all 249 frames of HQvideo and all 1,407 frames of the older camera clip at
25 fps without missing face detections. FFmpeg autorotation precedes tracking.
The pinned local task SHA-256 is
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
Model-estimated relative XYZ rotation ranges were about 11.64/13.90/8.42 degrees
for the generated clip, versus 19.37/22.66/14.64 for HQvideo and
25.14/30.19/17.59 for the older recording. These are different performances and
uncalibrated estimates, not ground truth or a realism score. Smaller estimated
motion plainly did not meet the user's perceptual requirement. Full coefficient
traces and candidate frame times are retained in
`.local/expression-review-v1/report.json`. No capture devices were activated.

An experimental `--source-anchored-mouth` mode now runs the actual KJ nodes with
source pose, position, scale and non-mouth expression keypoints, replacing only
the six mouth expression coordinates from the genuinely generated driver.
It uses KJ's absolute `single_frame` branch and disables lip-zero adjustment so
the source anchor is not accidentally reinterpreted as first-frame-relative motion.
It requests no head movement or blink animation. This is a deliberately restrained
comparison, not personalized training or the default application pipeline.
Three personal-motion tests pass, including anchor invariance under changing
driver pose, input/output mutation isolation and non-finite mouth rejection.

The actual KJ run completed in 382.65 seconds on MPS at 1280×854 and 25 fps.
All 121 frames decoded and the 4.84-second audio excerpt matched the original
decoded samples exactly. Estimated pose range dropped from 10.93/13.52/5.63 to
2.45/0.42/1.35 degrees, but the eyes are nearly static and the mouth looks softer.
SyncNet estimated zero-frame offset for both, with a higher distance and lower
confidence in the KJ result; this is diagnostic only. Keep the new result in
`.local/flashhead-kj-anchored-five/` as a control and do not promote it.

The earlier FlashHead output had a bilateral closure estimate above 0.6 for
eleven consecutive frames starting at 0.96 seconds. The longest such spans in
the two camera clips were six and one frames. This flags a concrete moment to
inspect, but the threshold is uncalibrated and it does not explain every reported
motion problem. The cancelled full job retained 22 committed sections.

The independent [LTX comparison](LTX_EVALUATION.md) is queued behind its public
weight download. Actual audio encoder and Metal tests passed with OS networking
denied; there is no LTX video result yet. Python suite: 43 tests run, 2 skipped.

## Native five-second test clip

The signed build in `dist/test-clip/Avatar Studio.app` opened on macOS, restored
the Black T outfit and 53.1-second recording, and exposed **Render test clip**
alongside the full render action. The actual Swift store/queue integration test
imported the same full recording and invoked the new short-render action in a
separate local library. It kept the draft audio path and duration, created a
separate five-second float PCM excerpt, and ran the installed FlashHead worker
with networking denied by macOS. The excerpt matched the source's first
1,920,000 decoded bytes exactly (48 kHz stereo float PCM).

The worker completed five sections and verified the 1920×1080 MOV in 279.50
seconds. An independent check against the original recording decoded all 125
frames at 25 fps and matched the original five-second audio excerpt. Completion
was persisted in SQLite; the original photo was not modified. The real output
and test report are under `.local/native-flashhead-test-clip/`. This validates
the short-render workflow, not higher visual quality or a new engine.

Restart recovery now checks the completion against the current request's attempt
ID and requires the engine's expected nonempty local output. It does not revive
explicitly cancelled or never-started jobs. The app saves the in-flight job before
launching its worker. The protocol smoke tests rejected stale attempts, empty
files and outputs from another directory, preserved cancellation and prior logs,
and recovered a copy of the verified five-second neural export byte-for-byte.
The updated `dist/Avatar Studio.app` built and passed strict local signature
verification. These recovery tests did not generate new video.

## FlashHead seed comparison

An alternate seed (`431`) produced a real 84-frame, 3.36-second render on M5
with OS networking denied, in 166.86 seconds. The original photo, crop, pinned
model revisions and the opening 215,040 speech analysis bytes matched the earlier
seed `42` run. The new generated RGB was projected onto the same photograph
with bounded stabilization, and the export matched the original decoded audio.

The equal-length diagnostic tracked every frame. The earlier sample's longest
estimated bilateral eye closure was 0.44 seconds; the alternate had two 0.20-second
spans near the same section boundary. Both had their largest estimated adjacent
pose change at 1.12 seconds (about 2.24° and 2.44°, respectively). The matched
close-ups still show closed eyes at the join and similar generated mouth detail.
Changing the seed altered the motion, but did not resolve the observed issue.
No default was changed. Generated output, motion traces, and the matched sheet
are under `.local/flashhead-seed-431-three/`; quality remains unapproved.

Localized rigid join damping was then tested on both actual clips without new
inference. It leaves standard corrections identical outside the join windows,
keeps the transform rigid, and caps eye-center displacement at six model pixels.
Both full exports passed exact original decoded audio and frame checks. Across
the matched 84-frame span, the estimated rotation change at 1.12 seconds fell
from 2.52° to 2.33° for seed 42 and from 2.84° to 2.80° for seed 431. Matched
close-ups still show the eye closure and visible pose change. The first seed's
estimated long closure remained 0.44 seconds; rigid warping does not address the
expression or lip synthesis. Keep the treatment research-only and the native
default unchanged. Comparison: `.local/flashhead-seed-431-three/join-comparison.jpg`.

A second experiment shifted the FlashHead generation grid without changing the
exported recording: prepend 12 frames of analysis-only silence, generate and
discard the matching 12 output frames, then export the 56 frames at their
original speech timestamps. The projected result and the native-resolution crop
both passed complete decoding and exact audio comparison. Three windows completed
in 157.93 seconds with networking denied. The visible joins moved from the earlier
1.12-second boundary to 0.64 and 1.76 seconds. The strongest estimated change in
the new result was at 0.12 seconds (2.46°); the two joins measured 0.43° and 0.39°,
and the old join timestamp measured 0.29° versus 2.24° in the earlier result.
The longest estimated bilateral closure fell from 0.44 to 0.20 seconds. These
are diagnostics, not verified lip synchronization or perceptual acceptance.
The lead-in changes model context as well as boundary position, so it does not
isolate a causal mechanism. The sample and actual boundary frames are retained in
`.local/flashhead-shifted-join/`. A 4.84-second matched follow-up is running; the
native engine still uses its original timeline.

The matched 4.84-second follow-up completed in 310.11 seconds. Both composed
clips passed original-audio and all 121 frame checks. On the shifted grid the
four estimated join changes were 0.47°, 0.59°, 0.19°, and 0.02°, versus the
earlier maximum join change of 2.52°. The largest change still occurred at
0.12 seconds (2.54°), and the longest estimated eye closure was 0.20 seconds.
The timing diagnostic estimated 0 frames for the baseline and −1 frame for the
shifted result, with distance 8.89 versus 8.29 and confidence 4.02 versus 4.74.
Matched close-ups show open eyes and a more centered face at 1.20 seconds.
Neither estimated timing nor still frames prove natural articulation. Retain the
actual review clip and matched evidence in `.local/flashhead-shifted-join-five/`.
A true 15-second follow-up was launched to compare with the previously rejected
duration. The research export now reads one generated RGB chunk at a time and
promotes a MOV only after independent frame and original-audio checks.

The 1080p matching path reused the native job's exact decoded source photo,
framing, raw RGB eye-corner tracking method and stabilization settings. Both
121-frame, 25 fps results matched the same original decoded audio excerpt. The
new joint-motion diagnostic flagged a head reversal at 1.40 seconds in the
earlier clip and none above the same threshold in the shifted clip; this is a
review cue, not a naturalness result. The middle 5–10 seconds of the earlier
15-second result also passed localized timing analysis. The completed follow-up
will be compared in matched framing, with one shared original audio track and
separate timing checks for all three five-second intervals.

The first eight seconds of HQ camera footage, the supplied external HeyGen clip
and the earlier local 15-second render were tracked as independent performances
at 25 fps. Each retained its own audio; they are not word-aligned comparisons.
The same reversal detector flagged two points in the camera clip, one in the
HeyGen reference and one in the rejected local render. Peak adjacent estimated
rotation was 4.94°, 3.45°, and 2.35°, respectively. Thus neither smaller maximum
rotation nor zero threshold flags can establish realism. The five-second matched
review now also includes consecutive frames at the highest-ranked head, gaze,
and eyelid events. It shows a steadier head around 1.40 seconds, while an uneven
blink remains visible around 4.56 seconds. This remains visual review evidence,
not approval of the model or a substitute for listening to the full clip.

The 15-second shifted render and matched 1080p review both completed. All 375
frames at 25 fps and exact original decoded audio passed in the shifted export
and the labeled side-by-side video. Identical source pixels, native framing and
stabilization algorithm were used for the comparison. The final elapsed clock
jumped between the last decode callback and completion, so do not use it as a
clean throughput measurement. Observed MPS allocation reached 6.67 GB.

The estimated worst section-join rotation step dropped from 2.32° to 0.60°;
two earlier reversal flags (1.40s and 12.16s) became zero at the same threshold.
Longest estimated bilateral closure fell from 0.44s to 0.20s. This does not
guarantee natural motion: the shifted clip still has a 2.32° estimated movement
near 0.12s, and later close-ups show uneven eyelids around 7.96s and 12.24s.
Global timing estimated −1 frame for both. In the 5–10s interval both estimated
−1 frame but the shifted clip had weaker audio-visual matching evidence (distance
9.05 vs 7.69; confidence 3.27 vs 5.06). Do not change the default or call the eyes
or lip sync solved. The full playback comparison, actual flags and localized
reports are under `.local/flashhead-shifted-join-fifteen/native-matched-review/`.

## Reported screen vibration

The user still sees vibration. Regional registration of all 375 decoded frames
showed the distant left/right background and shirt essentially stationary
(maximum first-frame translation estimates below 0.013 output pixels). Motion
and luma changes concentrate around the animated head and its surrounding blend;
the test does not cover player/display judder and does not invalidate the user's
visible complaint. `research/screen_stability_review.py` retains per-frame
registration and luma measurements in `screen-stability.json` beside the render.

`research/static_subject_blend.py` tests one fixed person-support mask computed
from the source crop and all generated RGB frames. It removes a large amount of
generated surrounding background without freezing face pixels or varying the
mask frame by frame. The first mask contained a disconnected background island;
the v2 comparison retains only the largest connected person region. The actual
15-second v2 export and labeled comparison passed all 375 frame and original
audio checks. Sampled head-edge close-ups were inspected at 0, 1.20, 7.96 and
12.24 seconds. Uneven eyelids and intrinsic head motion remain visible; the blend
is research-only and not a claim that vibration or realism is solved.

The original fixed-mask experiment reduced the 95th-percentile adjacent luma
change in 157,378 restored surrounding pixels from 0.514 to 0.0043 (8-bit scale).
This isolates a compositing contribution, not a quality score for the animated
face. The v2 report and playback comparison are under
`.local/flashhead-shifted-join-fifteen/static-subject-blend-v2*` and
`static-background-comparison.mov`. Native defaults remain unchanged.

The next control applied a 1.5 Hz forward/backward rigid head-path correction
with smoothly bounded displacement (8 model pixels / 2 degrees), retaining the
same generated mouth frames and source audio. A synthetic check retained 99.5%
of a slow 0.4 Hz nod and 2% of a 5 Hz oscillation; transforms remained rigid.
The real 375-frame render's tracked eye-center step at the 95th percentile fell
from 2.75 to 1.14 model pixels, using at most 4.54 pixels / 0.75 degrees of
correction. All frames and exact original decoded audio passed in the export
and its side-by-side comparison.

This does not solve the head shake: independent estimated 3D pose's largest
adjacent change was nearly unchanged (2.47° vs 2.44°). Sampled frames at 0.12,
1.20, 7.96 and 12.24 seconds still show uneven eyelids and intrinsic face motion.
Thus reducing 2D center jitter cannot be presented as a correction of the
model's generated pitch/yaw or expressions. Retain `slow-head-subject-blend.mov`
and `head-path-comparison.mov` as diagnostics; no native default change.

All three 15-second exports (native framing, fixed subject mask and slow rigid
path) were independently decoded with per-frame presentation timestamps. Each
had 375 frames, first PTS 0.00, last PTS 14.96, and every adjacent interval 0.04
seconds within one microsecond. Thus irregular encoded frame cadence was not
found in these files. This does not test display refresh behavior or establish
natural motion; intrinsic generated head/eye defects remain. Evidence:
`.local/flashhead-shifted-join-fifteen/presentation-timing-review.json`.

`scripts/verify_export.py` now checks every decoded frame's integer presentation
time against the requested frame-rate grid, using the stream time base rather
than rounded printed seconds. It streams decoder diagnostics in bounded memory
and still rejects decode errors, missing frames and changed decoded source audio.
Three real-encoding regression tests passed, including a 20-frame clip whose
tenth frame is 10 ms late: frame count alone accepts that fixture, while the new
cadence check rejects it. The actual 15-second FlashHead export (375 frames) and
earlier full-length MuseTalk export (1,328 frames) passed original-audio and cadence
checks. This is file-integrity validation, not visual acceptance.

After confirming the native render worker was idle, the installed FlashHead
runtime helper and its checksum manifest were updated; the previous files are
retained in `.local/export-cadence-runtime-backup/`. All 23 installed engine file
hashes matched afterward. The installed isolated Python runtime also checked the
actual 15-second export and exact source audio successfully with OS networking
denied. No camera or microphone was activated and inference defaults are unchanged.

## Separate visual review state

Completed native renders now show visual review pending until the user records
Looks natural or Needs refinement. The decision and date persist in SQLite
separately from the worker's completion status. The review can be cleared, and
a fresh render starts unreviewed even if copied from a reviewed job. Legacy jobs
decode without any implied visual approval. A native storage/queue smoke check
passed those cases, including preventing review of a queued result. The signed
build `dist/review/Avatar Studio.app` compiled and passed strict signature
verification. The actual signed app was subsequently opened, the native queue
confirmed idle, and the older user-rejected MuseTalk video marked Needs refinement
through Projects. After quitting and reopening the same build, Projects retained
Needs refinement while the technical render status remained Completed. No result
was marked visually accepted and no capture device was activated.
This adds no automatic quality acceptance and changes no inference defaults.

## LTX real-photo and temporal decoder compatibility

With OS networking denied, the pinned actual image encoder and video decoder
encoded/reconstructed the consenting Black T photograph's square face crop. Both
latent `[1,128,1,16,16]` and 512-square pixels were finite. Observed MLX peak was
4,604,814,220 bytes; elapsed time 1.60 seconds. Original and reconstructed images
were inspected and closely retain the visible facial detail. Reconstruction PSNR
39.02 dB is not a realism or lip-sync metric.

The actual photo latent was then repeated into seven temporal planes and decoded
at the first preview's planned size: 49 frames of 512-square pixels at 25 fps.
All frames decoded, had regular integer presentation times and nonblack pixels.
The upstream decoder selected 40-frame tiles with 8-frame overlap; observed MLX
peak was 8,675,670,644 bytes, elapsed time 6.13 seconds. This is a repeated-photo
codec fixture, with no new motion or speech generation. It is not an avatar result.

Evidence: `.local/ltx-real-photo-probe/report.json` and
`.local/ltx-real-video-codec-probe/report.json`. Actual text conditioning and
end-to-end video transformer execution are still pending verified weight downloads.
No full-model compatibility, sustained natural motion or production-quality claim
follows from these component checks.

## Functional native background preview

Create Video now updates the source-photo canvas for Original, Solid color and
Image backgrounds before a render is queued. Preview and export preparation
share the native Vision/Core Image compositor. The preview uses a serial actor,
coalesces picker changes, bounds display dimensions to 1,600 pixels and drops
cancelled requests; it creates no photo files or generated-avatar results.

The real-photo smoke check passed original/color/image backgrounds, preserved
an interior subject pixel and the unchanged original file hash, handled a missing
background and cancellation, and bounded a larger copy of the real photograph.
The first check exposed a 159-level maximum pixel mismatch when segmentation
received an ImageIO thumbnail instead of export's Core Image input. Matching the
decode path resolved it; preview and export-preparation pixels for the actual
1,535-pixel photograph now differ by at most three 8-bit levels. Larger preview
images are reduced for display, so this is not an exact final-framing preview.
Evidence: `tests/studio_background_preview_smoke.swift` and
`.local/studio-background-preview-smoke.log`.

Signed build `dist/background-preview-v2/Avatar Studio.app` compiled and was opened
through its full path. Native UI inspection showed the actual Black T photograph
against the selected gray background. Selecting Image without a file displayed
Import a background image to preview it. An initial container label obscured that
message from accessibility; the revised build exposes the real message and a
separately labeled photo. Switching back to Original restored the original image
and removed the error. The user's Original selection was restored, no queue job
was added, and neither camera nor microphone was activated. This is source-preview
functionality; generated motion and realism remain unresolved.

## Overnight render activity

The native queue now holds a Foundation user-initiated activity only while an
actual render worker is running. This requests prevention of automatic idle
system sleep and allows idle display sleep. Worker exit ends the activity before
the next job starts; app shutdown also explicitly releases it. It changes no
system preference and does not claim to override lid closure or forced sleep.

`tests/native_render_activity_smoke.swift` runs the real native queue with bounded,
protocol-only Python subprocesses and reads the test process's actual IOKit power
assertions. It passed acquisition during a worker, absence of display-sleep
prevention, and release on normal worker exit, cancellation, preparation failure
and shutdown. No neural result or video is fabricated by these fixtures. The test
uses the SDK's assertion-on value (255), not the global summary's boolean value.
The existing attempt/retry test also passed, including stale-status rejection and
retained previous inputs/logs. Evidence: `.local/native-render-activity-smoke.log`
and `.local/native-attempt-power-regression.log`.

The standard `dist/Avatar Studio.app` was rebuilt, ad-hoc signed and passed strict
signature verification. After confirming the old native queue was idle, the
standard bundle was opened through its full path. Create Video retained the Black
T outfit, full recording and Original background and displayed the overnight-render
information. The running executable was independently confirmed inside that bundle.
No old rejected render was resumed and no capture hardware was activated.

The separate research preview process is also protected by a temporary
`caffeinate -i -w` guard bound to that verified process's lifetime. A real macOS
assertion inspection confirmed its idle-system-sleep prevention and no corresponding
display-sleep prevention. This guard covers the current download/wait/generation
test only; it is not substituted for native queue behavior or a quality pass.

## Matched LTX review preparation

`research/review_ltx_candidate.py` validates genuine completed generation records,
unchanged movie hashes, identical photo/crop pixels, original source-audio hashes
and full frame/audio checks before producing a labeled comparison. The earlier
FlashHead raw crop was checked against its retained compositor provenance and all
121 frames/original decoded audio passed. Its crop pixels match the actual Black
T source exactly. No new inference was performed for this control check.

The tool deliberately rejected both actual codec-test directories and the still
unfinished LTX directory. A codec fixture cannot become an avatar candidate just
because it contains a decodable video. Future review includes shared original
audio, equal opening frame counts, expression/motion review points and appearance
sheets. The 49-frame run is below the timing diagnostic's two-second minimum;
the tool records a skipped timing check instead of padding footage or inventing
a score. Both inputs are raw model crops, so final native body composition and
full-duration quality remain separate gates. No actual LTX generation comparison
has run yet. Evidence: `.local/ltx-review-control-validation.json` and
`.local/ltx-review-fixture-rejection.json`.

Before inference, the queued preview was explicitly restarted to fix code
provenance: it now hashes the loaded script at startup, before waiting, rather
than hashing a potentially edited file when inference eventually begins. The
old process had no initialized output directory or generated frames; downloads
were left running. The new offline process uses unchanged inputs/settings, and
the old process's temporary sleep guard exited normally. A new guard is attached
to the new verified process. Current preview log:
`.local/ltx-photo-two-seconds-staged-v3.log`.

## Native appearance QA

The standard app's main Settings page and Create Video canvas were inspected in
Light appearance. The separate native Settings window initially remained dark
despite its Light selection. `SettingsView` now propagates its selected color
scheme to that window as well. The signed standard bundle was rebuilt and passed
strict signature verification. Actual UI inspection then confirmed a light native
Settings window, preserved Light after quit/relaunch, and System following this
Mac's dark appearance. The original explicit Dark preference was restored through
the app and verified in its picker. Source outfit, audio and model settings were
unchanged; no job or capture device was started. This is scoped appearance QA,
not a complete accessibility audit or any improvement claim for generated video.

After a native Settings close, UI observation temporarily timed out. A process
sample showed the app's main thread idle in the normal event loop; it subsequently
quit normally and relaunched. Retain `.local/native-appearance-window-sample.txt`
as observation evidence rather than reporting an unproved application hang.

## Resuming missing research processes

The previous tool-session handles became unavailable. Independent OS process
inspection found neither downloaders nor the text/preview test processes running;
no initialized generation output or text-test output existed. The retained ranges
contained 29.15/29.75 GB of video-model files and 7.20/8.07 GB of text-model files.
The two downloaders were therefore resumed from saved ranges, and both missing
offline waiters relaunched with unchanged inputs/settings. This restart was based
on confirmed missing processes, not an observation timeout. A new idle-sleep guard
is attached to the verified new preview PID; live process/session checks remain
authoritative. Current logs: `.local/ltx-q8-download-recovered.log`,
`.local/ltx-gemma-download-recovered.log`, `.local/ltx-real-text-probe-recovered.log`,
and `.local/ltx-photo-two-seconds-recovered.log`.



## 2026-11-08: rejected LTX performance and camera-conditioned Dub-It preparation

The user rejected the actual 4.84-second LTX output for doll-like human movement. The completed steady-prompt control and native-size composites are not accepted either. The default backend was not changed. See `docs/LTX_EVALUATION.md` for the updated decision.

Actual 49-frame camera reference preparation preserved originals and used a fixed crop with no per-frame recentering. Actual camera VAE encoding passed at 256 and 512 square (observed MLX peaks 6.53 GB and 9.32 GB). Reports: `.local/dubit-camera-reference-two/reference.json`, `.local/dubit-camera-encoder-probe/report.json`. These files are conditioning inputs, not generated videos.

After correcting the bundled FFmpeg PATH, `.local/dubit-conditioning-probe/report.json` passed actual source-audio encoding, both frozen audio states, both cached video conditioning stages, and restoration of scoped functions. MLX peak was 423,119,366 bytes. No Dub-It adapter inference occurred. The original failed attempt remains retained separately.

The new `research/dubit_preview.py` compiles and its CLI loads without heavy inference. `.local/dubit-runner-preflight.json` confirms missing and invalid adapter files are rejected before the GPU lease. It verifies the official adapter, cache and original input hashes; stages text models; freezes actual target speech; checkpoints generated latents; and verifies full export before publishing a review file. These generation paths are implemented but not executed yet. The official gated adapter is pending the local path promised by the user. No model access terms were accepted on their behalf.


## 2026-11-08: recoverable Dub-It decoding and verified export retry

`research/recover_dubit_export.py` implements decoder-only continuation from a hash-verified, completed inference checkpoint. It verifies the pinned adapter identity, original audio, base inventory, geometry and finite latent values. Missing adapter inference evidence, unfinished states and corrupted latent files are rejected before GPU use; `.local/dubit-checkpoint-rejection-check.json` records the executed checks. Actual generated Dub-It latent recovery remains untested because the adapter file is pending.

The shared `research/ltx_video_export.py` was exercised with the genuine existing `.local/ltx-photo-two-seconds/generated-silent.mp4`. An intentionally incomplete partial export was replaced; all 49 actual frames, cadence and original decoded audio passed. Published-file overwrite and a deliberate 50-vs-49 frame mismatch were rejected. Original video and audio hashes remained unchanged. Evidence: `.local/dubit-export-recovery-check/report.json`. No new inference was performed and visual quality remains rejected.

The Dub-It runner now reports actual per-step denoising progress without loading a preview decoder alongside the transformer. All three affected Python modules compile. Runner and recovery invocation instructions are retained in `research/README.md`; actual inference must run with OS networking denied.


## 2026-11-08: official Dub-It recipe and actual reference-conditioned base execution

Source review identified a material experimental change in the prepared runner: upstream jointly denoises audio in stage 1 and freezes the generated conditioning audio in stage 2, whereas the initial local adaptation froze original target speech in both stages. Neither is yet tested with the downloaded Dub-It adapter. The runner now defaults to upstream `reference` behavior and retains `frozen-target` as an explicit comparison. Original voice-over is exported in both modes; internal conditioning audio is saved only as latent diagnostic data. Exact official source identity is retained in `research/dubit-upstream-recipe.json` (revision `4f521619ae0029e8fbb64d555030c2b121a33b4f`). Recovery rejects missing or inconsistent mode metadata and corrupted checkpoints for either mode; six executed checks are in `.local/dubit-checkpoint-modes-check.json`.

`research/probe_dubit_streaming.py` actually ran the pinned pretrained base transformer with real camera and audio reference tokens, offline under macOS network denial. One 256-square forward with reference-audio denoising passed in 6.67 s; the frozen-audio control passed in 6.62 s. One full-resolution 512-square forward over 3,584 video and 98 audio tokens passed in 23.06 s. All results were finite; frozen target audio stayed bit-exact. Maximum observed MLX allocation was 4,226,133,310 bytes. Evidence: `.local/dubit-streaming-base-probe/report.json` and its log.

This was a base-model geometry and memory probe: no Dub-It adapter was loaded, no full denoising schedule or video decoding was run, and no generated avatar was produced. It excludes adapter memory and does not establish full Dub-It compatibility or perceptual quality. The official adapter local file remains pending.


## 2026-11-08: durable avatar/outfit edits and asynchronous import isolation

Source inspection found that ordinary photo import could change an existing default outfit because it compared the previous selection to the default even when not replacing. It also found that create/rename/default/delete and ordinary photo import did not roll back visible library changes when SQLite saving failed. Photo and video-frame completion could change the outfit selection after the user had switched to another avatar.

`StudioStore` now commits avatar/outfit edits with a visible-state rollback on persistence failure. Import retains an existing default; replacement updates the default only when replacing it, preserves the outfit name and list position, and uses a new source/cache identity. Old photographs and derived files are retained unchanged. Both photo and video-frame imports share the commit path and update selection only if the original target avatar is still selected. Invalid replacement requests are rejected before preparation.

`tests/outfit_library_recovery_smoke.swift` was compiled with the actual native store and run with the genuine Black T portrait in isolated storage. The final run passed first import, additional import without changing default, named replacement with new cache identity, replacement of the default, unchanged old source/prepared files, damaged-photo rejection without derived-file changes, switching avatars during actual Vision preparation, explicit SQLite write-failure rollback for rename/default/delete/photo replacement/avatar creation, and restart persistence. The original uploaded photo hash stayed unchanged. No camera/microphone or model generation was used. Evidence: `.local/outfit-library-recovery-check-v2.log`.

The normal `dist/Avatar Studio.app` was rebuilt against the installed runtime and signed; strict deep signature verification passed. Build log: `.local/native-outfit-recovery-build.log`. Binary SHA-256: `91823f6d78f3dcb65c69c193f2616800d7b469c560cd8e1a3ac564cea6ffb1ee`. This release build check does not establish generation quality or complete production readiness. Dub-It adapter inference remains pending the user-provided file.


## 2026-11-08: queue persistence failures and durable active cancellation

Fixed retry/resume and copy actions that continued into worker setup after SQLite saving failed. They now use the current stored job state, restore failed in-memory edits and do not prepare or launch a worker unless the queue mutation is saved. Worker-exit and preparation-error handlers stop advancing the queue when persistence fails, avoiding a chain of unrelated failed jobs during a storage outage.

Active cancellation now writes an atomic per-attempt `cancellation.json` before requesting process termination. On restart, a cancellation matching both the saved request's attempt UUID and job ID overrides an incomplete active job, even if SQLite did not save the cancellation. A mismatched old cancellation is ignored. Persisted resume supersedes the previous cancellation record; new attempts also clear it before preparing their request. This is additional recovery evidence, not an output-generation mechanism.

`tests/queue_persistence_recovery_smoke.swift` passed with actual native queue code, SQLite write-rejection triggers and a protocol-only subprocess. It verified no worker setup or job duplication after rejected saves, cancellation journaling while SQLite failed, no launch of subsequent queued work after the failed termination save, cancellation recovery after restart, and rejection of a stale attempt's cancellation. Evidence: `.local/queue-persistence-recovery-check.log`. The fixture loaded no models and produced no video.

Both existing regressions were recompiled against the changed store and passed: `.local/native-attempt-recovery-v3.log` confirms stale-status rejection and preservation of previous requests/logs; `.local/native-render-activity-v3.log` confirms actual idle-system-sleep assertions, display sleep allowance, and release after worker exit, cancellation, preparation failure and shutdown. The normal app release build and strict deep signature verification passed. Build: `.local/native-queue-persistence-build.log`; source and binary hashes: `.local/queue-persistence-recovery-result.json`. No claim of production readiness or accepted realism follows from these queue tests.


## 2026-11-08: external model-file gate

The official Dub-It adapter remains absent from the checked Downloads, repository and named Hugging Face cache locations; `.local/dubit-adapter-availability-audit.json` records the latest check. The user has offered to download it but has not supplied the path. This same generation blocker has persisted across multiple consecutive goal turns while preparation, real base-component tests and native recovery defects were completed. No full Dub-It inference or accepted realism result exists.

The next meaningful generation action requires that file: verify its pinned official hash, run the prepared offline short camera-conditioned test, and inspect the actual result before deciding whether to extend it or integrate it. Repeating rejected model settings, generating an unconditioned substitute without the adapter, or promoting the existing backend would not satisfy the quality gate. The project remains an engineering preview, not production ready; the full objective remains unmet. The blocked audit and remaining dependent work are recorded in `.local/quality-gate-blocked-audit.json`.


## 2026-11-08: resumed real native UI and resource recheck

After the goal resumed, the latest normal app bundle was inspected through the native UI, explicitly quit with no active renders, and relaunched by its full path. The real Suman avatar and four outfit entries remained visible. Create Video restored the Black T selection, full-voice evaluation name, 53.1-second voice-over, waveform, original background and 1920x1080 format. The image remained clearly labeled as a source photograph. Projects retained both the user's cancelled full render and the earlier output's Needs refinement review. The app was left on Create Video. No camera, microphone, settings changes, new inference or render were activated. Evidence and current binary hash: `.local/native-resumed-ui-recovery.json`.

A fresh local search included Downloads, Desktop, the repository, named official Hugging Face caches, case variants, partial downloads and files matching the exact expected adapter byte size; no candidate was found and no traversal error was reported. Evidence: `.local/dubit-resumed-availability.json`. The user was asked whether the download completed or is failing, and for its path if complete. This is the first fresh blocked audit after resume; the goal is not complete and no full Dub-It test is claimed. The stale final roadmap instruction to finish an already rejected LTX run was replaced with the actual next camera-conditioned test and its remaining appearance/quality gates.


## 2026-11-08: reject silently absent streamed Dub-It adapters

The pinned port source documents that incorrect LoRA block prefixes can silently discard deltas. Added `research/dubit_adapter_evidence.py` to inspect the actual streamed model after the first executed step of each stage: require the requested single adapter file, unfused strength 1, nonzero matching blocks/ranks, a synchronized runtime layout, nonempty attached layers and no skipped layer targets. The runner records both stages' binding evidence before checkpointing or publishing an output. Decoder recovery now requires that recorded binding evidence as well. These checks inspect execution bindings; they do not prove numerical parity or perceptual quality.

Six rejection cases passed with clearly labeled test doubles: no streamed model, no source, no matched blocks, unsynchronized bindings, empty layer handle and unsupported/skipped layers. Evidence: `.local/dubit-adapter-binding-rejections.json`. A real positive adapter-binding case remains unexecuted until the user supplies the official model file. All changed Python modules compile. This is preparation work during the second resumed goal turn with the same external-file gate; no Dub-It generation or accepted realism is claimed.


## 2026-11-09: user-authorized Hugging Face token download attempt

The user explicitly supplied a token for the official adapter download. `research/download_dubit_authorized.py` reads it through an echo-disabled terminal prompt, keeps it out of argv/project files/logs, and adds the bearer header as an unredirected header so it is not forwarded to a CDN redirect. No Hugging Face login configuration was saved and no model access conditions were accepted.

The actual authenticated pinned-file request returned HTTP 403 `GatedRepo`. Hugging Face reports that the account is not in the authorized list and directs it to request model access. The read-only identity check returned success; the token is valid and has the fineGrained role. No account identity fields were persisted. Sanitized evidence: `.local/ltx-dubit-weights/access.json`. No model bytes were downloaded and no inference was started. The user must review/request access to this model on the account associated with the token before the same downloader can proceed. This is a model-access gate rather than an invalid-token diagnosis.


## 2026-11-09: authorized official adapter download completed; actual Dub-It test started

After the user enabled access and requested a retry, the authenticated official transfer succeeded. The complete 2,466,665,072-byte adapter matched SHA-256 `fc415b12cb639e78511bc264f85080c2f7b188e334c1d9fade76b310e2bc419c`, pinned revision `cc18226c909e87aa8a1a079896ad46a4ec01e4d1`. Download took 850.23 seconds. Evidence: `.local/ltx-dubit-weights/download-complete.json`. The token was entered through an echo-disabled prompt and was not written into argv, project files, Hugging Face login configuration or download logs. The initial 403 evidence remains in `access-before-approval.json`; current `access.json` records the successful verified download. No access agreements were accepted by the agent.

Actual inference started in `.local/dubit-camera-two-reference/`, with `--audio-conditioning reference`, the real prepared camera clip, original My_voice.wav and the verified official adapter. macOS `sandbox-exec` denies all worker networking; `caffeinate -i` prevents idle system sleep for the active process. At this entry the model is still running; no compatibility success or quality acceptance is claimed.

While downloading, the prepared five-second real-camera conditioning input passed all 121 frames, cadence and original decoded audio checks. It uses a single fixed crop, retains original media hashes and is explicitly reference-only, not a generated/lip-synced result: `.local/dubit-camera-reference-five/reference.json`. CPU FaceLandmarker tracked all 49 short-reference frames without misses; `.local/dubit-camera-reference-two/expression-review.json` is a diagnostic baseline, not a naturalness score. A separate review helper was prepared to label camera input (old mouth) distinctly from actual generated speech.
