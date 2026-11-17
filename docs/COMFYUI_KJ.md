# ComfyUI-LivePortraitKJ backend

User explicitly requested Kijai's implementation. Source pinned for this test:
- ComfyUI-LivePortraitKJ: `4d9dc6205b793ffd0fb319816136d9b8c0dbfdff`
- ComfyUI: `4ac1c173afd42f1141dcebace42d8091998d3b8c`

This is the actual KJ node implementation. `backend/comfy_liveportrait.py` executes its registered classes, using ComfyUI's real model management, cropper, retargeting, process and composite stages. It does not rename the earlier adapter or insert MuseTalk after KJ.

Environment `.venv-comfy` contains its own dependency upgrades, sharing existing PyTorch/media libraries through a `.pth` file. It leaves the app's `.venv` untouched. `research/comfy-runtime-requirements.txt` excludes optional sample/template media and pins ONNX/protobuf for MediaPipe compatibility. Exact installed versions: `research/comfy-requirements.lock.txt`.

The existing upstream human checkpoints are converted losslessly to KJ's safetensors layout by `scripts/prepare_kj_models.py`. Strict state-dict loading in KJ verifies architecture/key compatibility. MediaPipe and CPU ONNX landmarks are selected. KJ uses fp32 on MPS and CPU compositing, matching its explicit MPS workaround.

Run a bounded actual-node test:

```sh
.venv-comfy/bin/python backend/comfy_liveportrait.py \
  --photo /absolute/path/photo.png --video /absolute/path/performance.mp4 \
  --seconds 2 --output .local/kj-test
```

Start local ComfyUI: `bash scripts/start_comfy_kj.sh`. It listens only on 127.0.0.1:8188, in offline mode. The actual API-format graph is `workflows/liveportrait-kj-api.json`; submit it as the `prompt` value to `/prompt`. This is an API graph, not a claim of a UI-layout workflow JSON. Local media I/O nodes are in `backend/comfy_io_nodes.py`, symlinked under ComfyUI custom_nodes alongside KJ. The source photo is staged into ComfyUI input.

The graph transfers a recorded performance—including lip motion—from the camera video onto the real photo, and uses that same performance's audio. This is not arbitrary new-voice audio-driven synchronization. Its output must not be mislabeled as a generated response to unrelated audio. No photorealism or HeyGen parity claim follows from successful rendering.

## Executed verification — 2026-11-05

- Actual KJ node-class run on Apple M5 MPS completed: 50 frames / 2 seconds, fp32, 110.68 seconds including node stages, output `.local/kj-test/review.mov`.
- Sampled close-ups retain visible moustache detail and show transferred mouth changes. This is a visual observation, not a claim of HeyGen parity or perfect synchronization.
- Full ComfyUI API graph accepted with no node errors. A 16-frame validation executed in 37.36 seconds and history returned `status_str: success`, `completed: true`.
- API output: `vendor/ComfyUI/output/Suman_LivePortraitKJ_9f0e274c.mov`. History retained privately in `.local/kj-api-history.json`.
- Source repositories are unmodified; model layout conversion is separate. This path contains no MuseTalk pass.
- Native SwiftUI app has not yet been switched to this backend; this is an installed and tested local ComfyUI backend plus executable adapter/workflow.

## Relative head-motion correction

Inspection of the pinned KJ implementation showed that its relative-motion branch with explicit eye/lip retargeting replaces the computed driving transform with source keypoints plus eye/lip deltas. Earlier tests therefore demonstrate expression retargeting, not faithful head-pose transfer. The adapter and API graph now leave the separate retargeting node disconnected, using relative expressions, lip-zero and stitching so the driving head transform is retained. A new five-second execution is being validated; the earlier API success does not by itself validate the modified graph.

`research/motion_review.py` measures eye-line angle and eye-center displacement to help detect frozen motion. These are 2D diagnostics, not 3D-pose accuracy or perceptual-quality scores.
