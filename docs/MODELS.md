# Model selection and license inventory

Inspected source on 2026-11-04. Licenses below describe the checked-out sources; they are not a legal determination about your intended distribution.

| Component | Source revision | Selection |
|---|---|---|
| MuseTalk Mac port | `7fd019315127d7f31e2e7f9853547ddceabbeb6e` | MediaPipe detector and Metal-compatible parsing |
| MuseTalk upstream | `0a89dec45a0192b824e3cf4daf96c239440c5ed8` | Missing VAE/UNet wrappers restored from upstream; UNet loads CPU weights safely before moving to MPS |
| LivePortrait | `9b294b3d0536135442ea73cb01e6cb3ca7029dd3` | Optional motion transfer through our local MediaPipe adapter |
| MediaPipe | `0.10.14` | Actual face-mesh detection tested on both supplied portraits, Apple M5 |

Sources: [Mac port](https://github.com/barnent1/musetalk-mac), [MuseTalk](https://github.com/TMElyralab/MuseTalk), [LivePortrait](https://github.com/KlingAIResearch/LivePortrait), [MediaPipe](https://github.com/google-ai-edge/mediapipe).

MuseTalk and LivePortrait source are MIT licensed. MediaPipe source is Apache-2.0. MuseTalk's LICENSE documents separate dependency licenses. Preserve each upstream LICENSE when distributing. FFmpeg build/license information is available via the packaged executable's `-version` / `-L`; distribution of an app with H.264 and FFmpeg warrants a separate release review.

Upstream LivePortrait includes InsightFace models restricted to non-commercial research. Avatar Studio's `backend/motion.py` replaces that detector with MediaPipe and uses LivePortrait's own 203-point landmark refinement. The installer downloads only `liveportrait/*`, not InsightFace or animal models. Do not invoke the upstream CLI directly: it would use the upstream detector instead. The replacement adapter needs real reference-motion validation; its existence is not evidence that reference motion is production-ready.

The Mac port's cloud scripts, TTS integration, web server, and hosted idle generation are not used. There is no cloud endpoint in Avatar Studio's worker. The VAE is used within MuseTalk's lip-sync model, never to generate outfit photos or bodies.

## Memory and execution

- Metal availability is tested by an actual tensor operation. CUDA is not selected.
- MuseTalk uses float32 and batch size 1 until other configurations are measured.
- Audio features use bounded 10-second chunks with contextual overlap.
- Video frames stream to FFmpeg; full generated videos are not retained in RAM.
- Still-photo face parsing and VAE reference latents are reused within a render.
- LivePortrait runs as a separate process and exits before MuseTalk models load.
- Reference motion is limited to 15 seconds; source image fits within 1280 pixels.
- Audio is limited to 300 seconds per render in this release.
- No automatic CUDA fallback, custom training, voice cloning, TTS, or synthetic idle gestures.

## Dependency caveats

Python 3.12 was used. The source port recommends 3.11, but 3.12 wheels installed and MediaPipe detection executed successfully here. `pip check` under the restricted tool sandbox reports MediaPipe unsupported, while its real arm64 detection test succeeds outside that sandbox. Record actual execution evidence rather than interpreting installation/import success as compatibility.

`backend/requirements.lock.txt` captures the installed versions. Source repositories are pinned. Model files should be accompanied by the generated SHA-256 manifest after installation; model downloads are not currently a signed supply-chain system.
