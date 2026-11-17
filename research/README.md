# Black T quality baseline — 2026-11-10

The earlier LTX/Dub-It Black T render remains the primary baseline. The experimental `dubit_mouth_control.py --kj-photo-motion` route retains the matched LivePortraitKJ upper face and composites the verified generated mouth. The corresponding `project_ltx_candidate.py --kj-dub --mouth-control` route checks provenance and exports the same framing and original voice-over. The mouth-only comparison had weaker measured lip timing than the earlier LTX render. See [the evaluation record](../docs/LTX_EVALUATION.md).

## Parallel TBDub compatibility experiment

`tbdub_mps_preview.py` runs one 2.88-second TBDub Student window on the Black T photo and the original voice-over. It consumes the locally prepared LivePortraitKJ visual reference, checks its photo, video and audio hashes, and runs the upstream two-step inference loop. The worker requires verified model inventories, uses disk-backed transformer layers and tries full-width Wan decoding with an overlapping-tile fallback for Metal memory errors. It verifies the original decoded audio and all frame timestamps before publishing a research preview.

Run from the repository root with the isolated `.venv-tbdub` environment and macOS network denial. Add `--check-inputs` to test the actual input and published recipe without loading the trained model. Use a new output directory each time.

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv-tbdub/bin/python research/tbdub_mps_preview.py --consent --reference .local/dubit-kj-reference-five --output .local/tbdub-blackt-student-retry
```

The trained benchmark is queued behind the three model downloads. Benchmark and transfer logs are under `.local/tbdub-source-review/`. The actual input preflight and synthetic-weight Metal checks have passed. The benchmark does not establish production readiness or naturalness before the trained output is reviewed. See [the evaluation record](../docs/LTX_EVALUATION.md).

## Earlier Dub-It experiment — 2026-11-08

The user rejected the actual LTX free-motion sample for doll-like performance. The current candidate is real-camera-conditioned Dub-It with new voice-over. Actual camera encodings and audio-conditioning states passed; adapter inference is pending the official gated file that the user is downloading. No accepted realism or production-readiness claim follows from these probes. See `../docs/LTX_EVALUATION.md` for measured evidence and decisions.

`dubit_preview.py` requires `--lora` pointing to the official local file, `--consent` and a fresh `--output` directory. It verifies the pinned adapter/source/cache hashes and requires observed nonempty runtime adapter bindings after an actual step in each denoising stage, uses offline model directories and emits actual denoising progress. The default `--audio-conditioning reference` retains upstream joint audio denoising; `--audio-conditioning frozen-target` is a separate experimental comparison. Both exports preserve the original voice-over. The exact upstream source review is pinned in `dubit-upstream-recipe.json`. Run with macOS networking denied, using the existing `.venv-ltx` environment. Example (from the repository root; substitute the actual downloaded path):

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv-ltx/bin/python research/dubit_preview.py --lora '/actual/local/path/ltx-2.3-22b-ic-lora-dubit-0.9.safetensors' --consent --output .local/dubit-camera-two
```

The initial 49-frame test uses actual camera performance and no photo appearance anchor. It is not the completed reusable photo-avatar feature. Prepared camera references have the old camera mouth motion and are never generated-video deliverables.

After both denoising stages and original-audio conditioning checks pass, the runner writes `video-latent.safetensors` and `checkpoint.json`. If decoding or export fails, retain that directory. `recover_dubit_export.py --checkpoint <attempt> --output <fresh recovery directory>` verifies the checkpoint and original audio, loads only the video decoder, and exports without rerunning denoising. Use the same offline environment and network-denied invocation. Actual generated-latent recovery remains to be exercised after the first Dub-It render; invalid and corrupted checkpoint rejection passed.

The shared export helper was exercised on an existing genuine LTX-generated silent movie: recovery from an interrupted partial passed; all 49 frames and original decoded audio passed; incorrect expected length and overwriting a published output were rejected. This is export-recovery evidence, not new model generation or perceptual acceptance.

# Realism benchmark work

Goal: realistic identity-preserving lipsync, not merely render completion. Neither current MuseTalk output nor a successful tensor probe meets that goal.

Candidate: ByteDance LatentSync 1.5, source a229c3948406bc2cf6eaf4873e662e70c6a04746. Code Apache-2.0; checkpoint model card identifies OpenRAIL++. Inspect these terms and dependencies before distribution. Upstream 1.6 advertises 18 GB inference VRAM, 1.5 8 GB. These CUDA figures do not establish Mac memory compatibility.

`probe_latentsync.py`: actual full-architecture random-weight 16-frame denoising step on M5 MPS; finite output, 24.40 seconds including model construction, Metal driver memory 6.32 GB. This is a compatibility test ONLY.

`latentsync_benchmark.py`: bounded experimental trained-model render adapter. MediaPipe alignment replaces upstream InsightFace; audio encoder runs on CPU, temporal UNet/VAE on MPS. Original voice-over remuxed separately. Not yet an approved app backend. Test results must be visually reviewed.

`download_latentsync.py`: resumable parallel download; validates byte ranges and complete SHA-256 against Hugging Face metadata. Downloads weights locally, never sends user media.

`latentsync-port.patch`: defers Decord imports so model operations and OpenCV video path can run on Mac without a Decord wheel. Apply to pinned checkout before running.

Quality review must cover: moustache/lip identity, temporal shimmer, seams, phoneme synchronization, eyes/head naturalness, original audio preservation, and long-duration drift. Face similarity alone or changed mouth shapes are insufficient. No model is promoted based solely on runtime success.

Sources: https://github.com/bytedance/LatentSync and https://huggingface.co/ByteDance/LatentSync-1.5

Actual additional tests: pretrained tiny Whisper audio encoder produced `[200,5,384]` features from the four-second user voice excerpt; MediaPipe/LatentSync affine alignment produced `[3,256,256]` aligned face on MPS. Checkpoint download is still pending; these do not validate trained video generation.

`run_pending_benchmark.py` waits up to one hour for the verified checkpoint, then runs a 0.64-second/20-step local benchmark on the supplied camera footage with the supplied voice excerpt. Logs: `/tmp/latentsync-benchmark.log`. It does not promote or publish the result. Outputs remain under `.local/latentsync-benchmark/`. Parallel download logs: `/tmp/latentsync-download.log`.

Transfer update: plain/range transfers were too slow; installed hf_xet 1.1.10 and started accelerated transfer into `vendor/LatentSync/xet-download`. The checkpoint is moved into the benchmark path only after its SHA-256 equals `6440b49a7ccceff56cdc001f5f17605216337f5bbd66fa360139768926e23f51`. The queued benchmark has not yet generated a quality-reviewed result. No user media is transmitted by model downloads.

2026-11-05: the goal remains unachieved. KJ + trained LatentSync 1.5 40-step sample executed but remains too soft against the reference. Work advanced to the 1.6/512-resolution candidate. `backend/mps_attention.py` bounds attention-score buffers without changing query/key/value math; measured fp16 max attention error 3.8e-6, masked test passed, full-network sequential-guidance comparison max error 0.0. Random-weight 512 memory probe passed at 7.62 GB Metal driver memory after the unsliced operation failed allocating a 4 GiB buffer.

`research/run_pending_16.py` waits up to two hours for the SHA-verified 1.6 checkpoint and runs a bounded 20-step actual-generation test on KJ output with the reference audio. Download log `/tmp/latentsync-16-download.log`; render log `/tmp/avatar-kj-latentsync-16.log`. Quality and full native-app integration are still pending; no completion or HeyGen-equivalence claim is warranted.
