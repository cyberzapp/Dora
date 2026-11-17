# Avatar Studio: roadmap to the realism target

Current direction, 10 November 2026. The user stopped the previous experimental direction and requested research first. The research decision in [AVATAR_QUALITY_RESEARCH.md](AVATAR_QUALITY_RESEARCH.md) takes precedence over the execution steps below. Next work is a source-level audit of personalized training, data requirements, Mac feasibility and licensing. Earlier queued benchmark commands are historical and should not be relaunched from this roadmap.

11 November update. The compact-training audit is complete and the [audio-driven motion comparison](AUDIO_DRIVEN_MOTION_RESEARCH.md) now prioritizes Ditto and JoyVASA for the next research gates. Resolve checkpoint/configuration compatibility, auxiliary terms, full Mac execution paths and timing semantics before an implementation choice. Their shortlist position is based on architectural fit, with actual visual quality still unverified.

The published Ditto configuration matches the 1,103-feature training schema. Separate eye/emotion reference controls remain a naturalness risk to audit; no checkpoint execution or Mac compatibility is established. The effective configuration must be explicit before any later comparison.

A limited FP32 volumetric-sampling probe now passes on the installed MPS runtime with CPU fallback disabled. This is primitive evidence only. Source tracing also identifies Ditto's inner sampler retaining a CUDA device default because its loader does not forward the selected constructor device. Resolve this before candidate inference; do not replace interpolation math unnecessarily.

The untrained Ditto motion architecture now passes a CPU/MPS guided-forward comparison and a 50-step MPS sampling check with fallback disabled. This narrows the remaining feasibility work to pretrained compatibility and the other pipeline stages; it is not a quality result.

Updated 2026-11-10. Target: user-approved realism comparable to the supplied HeyGen reference, produced locally on the M5/16 GB Mac through the native app, using the user's real likeness and a new voice-over.

User preference confirmed: prioritize realism; overnight renders are acceptable. Retain quality settings when slower local inference is necessary. Still measure performance and memory accurately.

## Verified starting point

- Native SwiftUI app, local library, media import, queue, cancellation/recovery, logs, original-audio export and short test-clip rendering exist. A signed local review build and relocated FlashHead runtime run on this Mac. Production QA remains incomplete.
- MuseTalk, KJ/LatentSync and LongCat have real local inference evidence, but their tested outputs failed the realism target. Preserve their evidence; do not repeat a rejected configuration as a release candidate.
- FlashHead initially improved short-clip resemblance. Its installed-runtime 15-second native render passed technical checks but was rejected for unnatural head, eyes and expressions. That longer review supersedes earlier encouraging short-clip feedback.
- Shifted model windows reduced some transition spikes; fixed subject blending reduced surrounding background flicker. Stronger 2D stabilization still left intrinsic head rotation and uneven eyelids. These are research controls, not accepted defaults.
- Three actual 15-second exports have regular encoded frame timing. Export validation now checks each frame's presentation time, full decoding and original decoded audio. This cannot establish natural motion or exclude display-specific playback problems.
- The pinned LTX 2.3 MLX candidate completed actual 49- and 121-frame audio-conditioned generation. The user rejected the full-face free-motion output for doll-like performance. The official Dub-It adapter is downloaded and verified. Actual camera-only Dub-It was also rejected. The strongest current local baseline uses the Black T photograph, LivePortraitKJ motion and Dub-It with frozen target-speech conditioning. It passed original-audio and timestamp verification, but local lip timing remains inconsistent. See [LTX evaluation](LTX_EVALUATION.md) and [validation evidence](VALIDATION.md).

## 1. Establish a repeatable acceptance benchmark

Use Portrait_BlackT.png for appearance, my_video.mp4 for motion and Timeline 1.mov solely as the comparison reference. Keep originals unchanged. Compare identical audio where practical and match framing without stretching the face. Separately test My_voice.wav as a genuinely new voice-over. Record versions, settings, durations, memory, source hashes and audio provenance for every candidate.

Review at normal playback speed and face-crop scale: identity; skin and facial-hair detail; lip closure and consonants; teeth/tongue consistency; blinking and gaze; head/neck stability; temporal flicker; compositing boundaries; clothing and background preservation. Measure audio offset and inspect chunk boundaries. Automated metrics are supporting evidence, never substitutes for perceptual review.

The user's latest priority is joint motion quality: stable head and neck with
no brief shake, natural eyelids and gaze, and clear lip timing together during
speech. Watch the full comparison with sound, then inspect consecutive frames
around sudden reversals, gaze shifts, blink asymmetries, and section joins.
Check opening, middle, and closing speech intervals separately; an average
offset must not hide a few visibly bad frames. Do not accept a candidate
simply because it reduces motion, freezes the eyes, or passes a timing metric.

Gate: a repeatable comparison set and an agreed visual review rubric. A successful export alone never passes.

## 2. Prove the quality backend before expanding the UI

Hold the exact 121-frame Black T reference, seed and original audio constant while comparing the upstream Dub-It audio recipe to the frozen-target baseline. Check whether either recipe keeps the visual speech aligned locally while retaining recognizable eyes and natural head motion. The mouth-only composite preserved the KJ upper face but produced weaker measured timing. Keep the full-face render as the primary comparison. TBDub Student is a separate local compatibility experiment pending the pinned model downloads and trained render.

Gate: a 5–10 second new-audio sample with no obvious identity drift, mouth seams, unstable teeth, flicker or head/neck distortion at intended viewing size. Present the strongest candidate beside the HeyGen reference for user review. Do not promote an inadequate model merely because it runs.

## 3. Stabilize a complete speaking segment

Render 15–30 seconds, then the full 53-second voice-over. Implement bounded temporal chunks with sufficient model context and inspected overlap/transition handling. Cache source preparation; release stage-specific model memory; instrument peak memory and elapsed time. Define honest motion behavior when audio outlasts the reference: use tested restrained motion, or request a longer reference; do not silently repeat conspicuous gestures.

Preserve original audio in the archival export and verify decoded equality. Validate frame counts, duration, A/V alignment, full decoding and absence of black frames. Test offline with networking unavailable, cancellation, restart recovery, low disk space and invalid media.

Gate: full-duration quality holds, every chunk boundary passes inspection, memory stays within a safe operating budget, audio is preserved and failures are recoverable. Overnight rendering is already authorized; report measured runtime and system responsiveness without requesting that preference again.

## 4. Integrate the accepted pipeline into SwiftUI

Replace the default generation path only with a perceptually accepted local pipeline. If it uses ComfyUI, keep the server on loopback and manage worker/server startup, readiness, progress, cancellation and shutdown from the application. Expose appearance photo, motion reference, new voice-over, restrained motion controls and export framing. Show stage-based progress and actionable failures. Hide experimental tuning from the normal creation flow.

Cache preparation by source and model/settings hashes. Record output provenance. Validate camera capture, outfit replacement, custom backgrounds, native preview, queue recovery and generated-video playback through the actual app. Preserve source framing; do not synthesize missing body parts, chairs or gestures.

Gate: a user can import their assets and produce the approved-quality result through the app without Terminal commands.

## 5. Release qualification

Package pinned runtime dependencies and a verified model installer with resumable downloads, license notices and storage estimates. Audit model and transitive dependency licenses before distribution. Test a fresh installation, model corruption, network-free generation, app restart, permissions, accessibility and both appearance modes. Add signing/notarization as distribution credentials permit.

Gate: repeatable installation and end-to-end native-app generation; all remaining limitations documented, with user approval of full-length visual quality.

## Decision if the quality gate fails

First identify whether source capture, KJ motion, audio-driven reconstruction or compositing causes the defect. Test a sharper recording only when evidence implicates capture; keep the original-input benchmark for a fair comparison. Evaluate another local audio-driven backend only with an explicit compatibility/license/memory test and a controlled quality comparison. Do not spend indefinitely adjusting a backend that consistently destroys identity.

If realistic output cannot be demonstrated under M5/16 GB and local-only constraints, report the exact limiting stage and measured evidence. Discuss any hardware or scope change with the user; never silently move inference to the cloud or claim parity.

## Resources from the user

Existing personal media and the official Dub-It adapter are available. The official 2,466,665,072-byte file passed its pinned SHA-256. Later acceptance needs short English, Hindi and Hinglish voice-over examples and visual feedback. Signing credentials apply when distribution is ready.

## Immediate next action

The research phase has identified a concrete next experiment: render one 4.84-second new-audio Ditto clip using Portrait_BlackT.png and My_voice.wav. The isolated runner is `scripts/run_ditto_quality_trial.py`. It reuses the five byte-identical local renderer weights, verifies the pinned motion/audio/auxiliary assets, corrects sampler device propagation and uses the numerically checked FP32 path. It supplies source-derived eye conditioning and neutral emotion instead of inheriting the packaged reference controls. Original float audio samples are preserved in the MOV export.

The first real trial is now rendered and technically validated: `.local/ditto-experiment/output/ditto-blackt-trial.mov`, 121 frames, 25 fps, unchanged decoded audio, 266.02 seconds for the run. It remains unaccepted for quality. Review the actual playback before another run or integration.

The required deliverable is the actual clip with full decode, timing and audio checks, followed by a visual comparison against the previous baseline. Success is a visibly stronger speaking performance, not a download, completed test or valid file. Do not expand the app or pursue additional synthetic probes before this result is available. If the result fails, record the visible defects and make a backend decision; do not silently promote it into the application.

The LTX/TBDub commands earlier in this roadmap are historical. They are not the active execution plan and must not be relaunched.

The preferred appearance source remains Portrait_BlackT.png. HQvideo.mp4 is
camera footage with a replaced background, but its tested appearance was rejected
in favor of the photograph. Timeline 1.mov is the externally supplied HeyGen
reference only. Originals and all rejected diagnostic caches stay separate from
application-generated exports.

Camera/microphone recording QA is deferred at the user's request; do not activate
either device. Existing assets suffice for the next model test. Native long-run
quality, multilingual acceptance, capture QA, installation/distribution and final
user approval remain open. See [LTX evaluation](LTX_EVALUATION.md),
[FlashHead evaluation](FLASHHEAD_EVALUATION.md), [LongCat evaluation](LONGCAT_EVALUATION.md)
and [validation evidence](VALIDATION.md).
