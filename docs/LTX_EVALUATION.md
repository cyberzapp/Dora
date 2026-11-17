# LTX 2.3 local comparison

The first actual end-to-end local test passed technical validation on the M5/16 GB Mac: 49 frames at 512 × 512 and 25 fps, using the Black T photograph and the first 1.96 seconds of My_voice.wav. It completed in 122.18 seconds with observed MLX peak allocation of 8,676,752,006 bytes. Every frame decoded with regular presentation times, and decoded original stereo audio matched exactly. Networking was denied by macOS throughout inference. This is a real pretrained output, not a codec fixture or a personalized training result.

**Visual quality was rejected by the user.** The completed 121-frame (4.84-second) test passed technical export checks but looked like doll-like head movement. A second, steadier prompt reduced head turns without establishing natural human performance. Neither candidate is accepted or promoted to the native app. Technical execution does not establish HeyGen-level quality or production readiness.

## Pinned runtime and public weights

The source and complete selected inventory are in `research/ltx-source-lock.json`.

- MLX port: dgrauet/ltx-2-mlx, revision `747b30ef10e92a011e85b2aa0f6c4ee86ced5218`.
- Video model: dgrauet/ltx-2.3-mlx-q8, revision `69f8f11996b7b2ee54e82d0db7a5acad14b8685d`, distilled v1.1 transformer.
- Text model: mlx-community/gemma-3-12b-it-4bit, revision `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.
- Selected files: 37,822,540,222 bytes. Both full inventories finished downloading and passed their pinned hashes before inference.
- Isolated Python 3.12 environment: MLX 0.32.2, mlx-lm 0.31.1, transformers 5.3.0, ltx-core-mlx/ltx-pipelines-mlx 0.18.0; dependency consistency checked.

The [port](https://github.com/dgrauet/ltx-2-mlx) streams transformer blocks from disk. Its [maturity notes](https://github.com/dgrauet/ltx-2-mlx/blob/main/docs/PIPELINE_MATURITY.md) classify distilled audio conditioning as experimental. Runtime support here is based on the executed short test, not reported support alone. Source code is MIT; model weights have separate LTX-2 Community and Gemma terms. Redistribution clearance is not established.

## Executed pipeline

`research/ltx_audio_preview.py` uses the real photo crop `[445,92,969,616]`, resized conventionally to 512 square. `research/ltx_staged_text.py` encodes the actual prompt with Gemma, materializes its hidden states, frees Gemma, then loads and strictly executes the actual connector. Its overridden `load()` defers the pipeline's otherwise simultaneous preload of these large components.

The actual audio VAE encodes the supplied recording. Its tokens are frozen as conditioning in both denoising stages. The streamed distilled transformer ran all eight half-resolution steps and three full-resolution refinement steps. Video latents were decoded separately; the original recording was attached as float PCM and independently compared after decoding. No generated voice or reconstructed audio was substituted.

The script hashes its loaded source before a download wait and records the actual staged-text helper hash. Full source, model/input hashes, crop, seed, runtime versions and memory observations are retained in provenance. The GPU lease serializes heavy inference. A temporary idle-sleep guard covers the current research run without holding the display awake or changing system power settings. No camera or microphone was activated.

Evidence for the 49-frame run:

- `.local/ltx-photo-two-seconds/provenance.json`
- `.local/ltx-photo-two-seconds/review.mov`
- `.local/ltx-photo-two-seconds/progress.json`
- `.local/ltx-photo-two-seconds/actual-frames.jpg`
- `.local/ltx-photo-two-seconds-recovered.log`

The observed allocator limit is a guideline, not a hard system-RAM guarantee. This run does not prove long-duration memory behavior or native application integration.

## Real component checks

The separate actual Gemma/connector test also passed: 49 real hidden-state layers, finite video embeddings `[1,1024,4096]` and audio embeddings `[1,1024,2048]`, 9.92 seconds, observed MLX peak 8,091,573,276 bytes. Evidence: `.local/ltx-real-text-probe/report.json`.

Earlier isolated checks passed actual audio encoding, connector execution, spatial upscaling and real-photo encoding/decoding. The photo reconstruction retained close visible detail (39.02 dB reconstruction PSNR, observed MLX peak 4,604,814,220 bytes). Repeating its latent over seven time planes exercised 49-frame temporal decoding at observed peak 8,675,670,644 bytes. These isolated tests are explicitly codec/component fixtures, not generated talking avatars or motion-quality evidence. Full records remain in VALIDATION.md and `.local/ltx-*-probe*`.

## Controlled visual review

`research/review_ltx_candidate.py` checked the actual LTX output against the genuine earlier FlashHead raw crop using identical real reference pixels, source crop and original audio. It checked both movies' provenance/hashes, every frame and decoded source audio. Its guards rejected both real codec fixtures and the unfinished LTX directory before accepting an actual completed generation.

The comparison is under `.local/ltx-photo-two-seconds/matched-review/`: a labeled side-by-side movie with shared original audio, face/appearance sheets and consecutive frames at ranked motion/eye events. The supplied HeyGen clip is explicitly an external reference with a different performance, not word-aligned footage or an app result. Both local candidates are raw model head crops; neither is the final seated body composition.

Eight sampled LTX frames show changing mouth shapes and a turn toward the camera; facial softness and resemblance still require playback review. The estimated largest adjacent pose step was 1.37 degrees versus 2.25 for the earlier FlashHead control; longest estimated bilateral closure was 0.08 versus 0.44 seconds. These uncalibrated measurements are review cues, not naturalness scores. SyncNet timing was explicitly skipped because 49 frames is below its two-second minimum; no footage was padded and no score invented.

## Completed longer tests and current direction

The actual 121-frame test in `.local/ltx-photo-five-seconds/` completed in 254.98 seconds, with observed MLX peak allocation 9,515,861,142 bytes. All frames, cadence and unchanged source audio passed. The user's explicit doll-like movement rejection is retained in `quality-decision.json`. The steady-prompt control in `.local/ltx-steady-five/` completed in 227.21 seconds with the same peak; it is not user-approved. Native-size seated-photo compositions also passed export checks without solving unnatural performance. Do not render longer versions of these rejected settings as release candidates.

SyncNet initially found an extra face in a background lamp. Inspection identified a genuine face plus the false detection. Explicit overlap-based primary-face tracking resolved that diagnostic without lowering thresholds or inventing missing detections. On original camera footage starting at one second, an intentional 200 ms audio delay changed the estimated offset by exactly five 25 fps frames. This validates that timing control, not realism. Both generated LTX controls measured about -80 ms relative offset; their exported original audio was not shifted.

The next candidate preserves actual camera performance through video-conditioned Dub-It rather than asking the model to invent the entire head performance. This is a compatibility experiment, not an accepted implementation. The official adapter and its checksum are pinned in `research/dubit-source-lock.json`. The official adapter was downloaded after the user granted access, and its pinned SHA-256 was verified on November 9.

Actual camera preparation and VAE encoding passed for 49 frames at 256 and 512 square, with observed MLX peaks 6,531,704,316 and 9,320,743,862 bytes. These are real camera encodings, not generated outputs. The conditioning probe encoded My_voice.wav through the actual audio VAE and verified fixed target speech and cached real-camera references through both state constructions; it did not execute adapter inference. The prepared MKV still has the camera recording's old mouth motion and must never be presented as a lip-synced result.

`research/dubit_preview.py` now provides the initial offline experiment: verify the official adapter hash and source/cache integrity, stage text models, reuse the measured camera encodings, retain upstream reference-audio denoising by default, optionally compare frozen original target speech in both stages, save a generated-latent checkpoint, decode video only, and independently verify every frame and original audio. Its syntax/CLI and invalid-adapter rejection were checked. Actual Dub-It inference subsequently completed on the M5. The user rejected the camera-only result as unnatural. This first camera-only control has no outfit-photo conditioning and does not fulfill the final reusable photo-avatar requirement.

If the actual model result warrants it, test 5–10 seconds, 15–30 seconds and then full duration. Inspect normal-speed head/eyes/brows, teeth, lip timing, identity and background stability, including continuation boundaries. Do not interpret lower motion metrics or valid exports as perceptual approval.

## Upstream recipe check — 2026-11-08

The official Dub-It source was retrieved at revision `4f521619ae0029e8fbb64d555030c2b121a33b4f` and SHA-256 `47b963c712c21532f6d6721a08e1104f411f80fb3d135cb90f7b203570fe4251`. It uses the distilled schedule and jointly generates stage-1 audio from appended reference speech, then freezes that generated audio in stage 2. Freezing the supplied target tokens in both stages is a local experimental change, not an upstream requirement or an established quality improvement. The runner now defaults to upstream `reference` behavior, with `frozen-target` as an explicit comparison. Original audio is exported in both cases; the generated conditioning-audio latent is checkpointed for diagnosis but not substituted for the voice-over. Actual lip timing must be inspected in either mode.

A subsequent actual base-transformer probe exercised the larger appended-reference geometry: two 256-square forwards and one 512-square forward. All finite-output checks passed, including unchanged frozen audio, with maximum observed MLX allocation 4,226,133,310 bytes. The 512-square step took 23.06 seconds. This excludes adapter memory, full schedules and video decode and must not be represented as a successful Dub-It render. Evidence: `.local/dubit-streaming-base-probe/report.json`.

## Black T appearance and articulation — 2026-11-10

The user specified `Portrait_BlackT.png` as the avatar face source. A direct frame-zero photo anchor against the raw camera reference switched to the camera appearance at frame 1. The comparison in `.local/dubit-photo-camera-two/appearance-review.jpg` records the failure.

The next control uses the existing genuine LivePortraitKJ animation from that photo as the visual conditioning sequence. The camera recording supplies driving motion and `My_voice.wav` supplies target speech. Both completed 49-frame conditions retain the photo appearance at the inspected points. `reference` follows the upstream joint audio procedure. `frozen-target` uses the supplied speech tokens throughout and is an experimental comparison. The reference run stopped during VAE decode and recovered from its verified generated latent without rerunning generation. The frozen run completed normally in 179.36 seconds with peak observed MLX allocation of 8.09 GB. The independently verified exports preserve the original decoded voice-over and all 49 frame timestamps.

The verified 1920 × 1080 seated composition is `.local/dubit-blackt-kj-two-recovered/presenter-1080.mov`. It projects the generated face onto the original photo framing, with exact preservation outside the fixed head matte before video encoding. Sampled mouth shapes are still restrained. The short duration falls below the calibrated SyncNet interval, so the export has no timing or perceptual acceptance claim. The native app continues to use its existing evaluation engine.

The two-second landmark comparison measured median inner-lip gap divided by lip width at 0.003 for the input, 0.012 for upstream reference mode and 0.074 for fixed speech. Correlation to the old driving mouth opening fell from 0.683 to 0.384 in the fixed-speech condition. This supports measuring timing over a longer interval. The 1080p sample’s decoded background and shirt patches each measured under 0.01 pixel maximum translation. The head patch moves and requires independent perceptual review.

The 121-frame input preparation now records exact crop coordinates, and the encoder uses overlapping temporal tiles for this length. The real 512-square reference encode completed with peak observed MLX allocation of 8.71 GB. The deliberately mismatched KJ input audio control scored SyncNet distance 12.13 and confidence 1.18, with offset at the 15-frame search boundary. Compare the new render to that control over identical audio and duration.

The 4.84-second fixed-speech render completed in 491.55 seconds at 9.54 GB observed peak MLX allocation. The raw result and the 1080p seated composition independently decoded all 121 frames at 25 fps and preserved the original audio excerpt. The actual render retains the Black T appearance through the sampled final frame. SyncNet measured -2 frames (-80 ms) for both the raw and composed results, with confidence 2.24 and 1.59 respectively. The score improved relative to the mismatched input but confidence remains modest and the mouth motion still needs playback review. Neither score establishes a reliable correction or perceptual acceptance.

- Actual seated export: `.local/dubit-blackt-kj-five-frozen/presenter-1080.mov`
- Generated-versus-input comparison: `.local/dubit-blackt-kj-five-frozen/review/comparison.mov`
- Timing reports: `syncnet-timing.json` and `presenter-syncnet.json` in the result folder.

## Matched reference-mouth experiment — 2026-11-10

The earlier seated export measured -2 frames in the first 2.4 seconds and -13 frames in the second, with confidence 2.07 and 1.34. A deliberately delayed whole-clip control moved the whole-clip estimate from -2 to -7 frames as expected for a 200 ms delay. The two interval estimates do not support one global timing correction.

A new LivePortraitKJ input constrained the six mouth expression keypoints to zero relative delta while retaining the remaining motion. This also disabled KJ's lip-zero retargeting so the still photo defined the neutral mouth shape. Both inputs used the same fixed crop, original Black T photograph and audio excerpt. The face tracker covered all 121 frames and the maximum estimated jaw-open coefficient fell from 0.06908 to 0.00142. Rotation ranges remained broadly similar.

The actual Dub-It render with this input completed in 601.23 seconds at 9.53 GB observed MLX peak. Both the raw and seated output passed frame cadence and original audio verification. The new mouth movements are visible against the constrained input. Timing distance worsened from 11.22 to 12.15 over the whole seated clip, and confidence declined from 1.59 to 1.13. The second-half estimate hit the search boundary at -15 frames. This candidate is rejected as an improvement. The stronger comparison remains `.local/dubit-blackt-kj-five-frozen/presenter-1080.mov`.

## TBDub Student compatibility screening

[TBDub V1.1](https://github.com/TaoLiveAIGC/TBDub/tree/19e18e4ba8e2a19e6716cb38110ee2de1dae9bb3) offers a video-to-video student model that could use the same Black T and KJ workflow. Its NVIDIA memory results do not establish M5 feasibility. The pinned source was inspected alongside the [community streaming implementation](https://github.com/hiroki-abe-58/ComfyUI-TBDub/tree/8a7028e433356d96f6fb66a048c4b646c930c951).

Actual M5 operator probes used the upstream attention function and existing query chunking at the full 10,240-token, 3,072-channel Student geometry with BF16 synthetic inputs. The output was finite, with 1.14 GB observed MPS driver allocation. Three upstream float64 position operations fail on MPS. The Metal math in `research/tbdub_mps_math.py` passed the same geometry with 1.20e-5 rotary RMS difference after BF16 rounding and bit-exact timestep output relative to upstream CPU. These operator checks do not load trained weights. Full model streaming, VAE and generation are still unverified. The public, pinned 15.27 GB Student and auxiliary download was started. Reports and manifests are in `.local/tbdub-source-review/`.

The offline preview preflight now reads the actual 121-frame LivePortraitKJ reference derived from `Portrait_BlackT.png` and checks the photo, video and voice-over hashes. It uses the first 72 frames of new speech, compares the decoded float PCM to the corresponding 2.88 seconds of `My_voice.wav` byte for byte, and verifies the published two-step Student recipe and five-frame pre-roll. The longer visual reference covers the complete window without looping the recorded performance. Preflight evidence is in `.local/tbdub-blackt-preflight-v3/preflight.json`. A separate execution with incomplete weights rejected the render and produced no video.

A full upstream transformer block ran with 39 disk-backed synthetic tensors and the full 10,240-token geometry. It returned finite output in 1.24 seconds, with 3.33 GB observed MPS driver allocation and zero block parameters remaining on Metal. In the separate codec test, the upstream Wan 2.2 encoder accepted 77 frames at 512 square. The full-width decoder exceeded an 8.29 GiB cap in a nine-frame test. Four overlapping spatial tiles with the community causal-convolution patch decoded the full 77-frame test in 116.73 seconds, with finite pixels and 5.47 GB observed driver allocation after decoding. After a Metal cache flush and with the benchmark's 85% per-process allocator fraction, the full-width decoder completed all 77 frames in 78.10 seconds with finite pixels and 9.74 GB observed driver allocation. These were synthetic-weight compatibility tests. Driver samples are not measured system peak usage.

The trained benchmark tries full-width decode first to retain the face's complete spatial context. If Metal reports an out-of-memory failure, it clears the upstream causal cache and retries the same generated latent with overlapping tiles. It propagates unrelated decoder errors and rejects non-finite pixels. Four injected cases exercised the full-width path, memory fallback, unrelated error and non-finite output. The first trained benchmark still requires playback and inspection of skin and mouth detail, including overlap regions if tiles are used.

The trained benchmark is queued locally behind all three pinned download inventories. It rechecks every model hash, accounts for all streamed transformer weights and persistent buffers, checks the execution of both denoising steps and verifies the original decoded audio and all presentation timestamps before publishing a clip. Output is intended for `.local/tbdub-blackt-student-72/`. The earlier Dub-It raw crop over the same 2.88-second interval measured SyncNet offset -2 frames, distance 11.04 and confidence 2.36. This provides a matched diagnostic baseline and does not determine realism or release readiness.

## LTX Black T mouth-only comparison

The earlier LTX/Dub-It render is the primary quality baseline. The new comparison keeps the same 121-frame LivePortraitKJ Black T conditioning sequence for the eyes, brows and head. It aligns the actual Dub-It mouth with upper-face landmarks and composites it within a fixed feathered region. The compositor verified every pixel outside that region against the KJ reference before encoding. This is a recomposition of trained inference already performed, with no new denoising or audio shift. The selected source motion still comes from a different performance than the voice-over.

The raw and 1080p variants independently passed decoded audio, frame count and timestamp verification. A comparison from a different Dub-It run was rejected before the projector published a movie. Six matched frame samples showed visible mouth articulation and the retained upper face. In the matched 1080p diagnostic, both variants measured -2 frames. Distance worsened from 11.22 to 11.63, and confidence declined from 1.59 to 1.34 with the mouth-only blend. The earlier LTX render remains the baseline. Inspect normal-speed eye and head motion independently of the timing score.

- Earlier full-face export: `.local/dubit-blackt-kj-five-frozen/presenter-1080.mov`
- Matched side-by-side export: `.local/dubit-blackt-kj-mouth-only-five/presenter-side-by-side.mov`
