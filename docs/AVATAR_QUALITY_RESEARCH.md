# Research decision for Suman's avatar

Reviewed 10 November 2026. The current instruction is research first. The previous rendering and model-porting direction has stopped. A process inventory found no active Python render, model-download or queued benchmark processes at this review. Earlier completed experiments and partial downloads remain evidence for comparison.

The [11 November source audit](PERSONALIZED_TRAINING_AUDIT.md) refines this recommendation. Compact personal training does not automatically learn head or gaze behavior for new speech. SyncTalk_2D retains a recorded pose path, MuseTalk's full FP32 training has an impractical persistent memory budget on this Mac, and MimicTalk's personal training adapts its renderer while motion uses additional components. Select appearance and speech-conditioned motion together before choosing the implementation.

The subsequent [audio-driven motion comparison](AUDIO_DRIVEN_MOTION_RESEARCH.md) prioritizes Ditto for further feasibility and quality research, with JoyVASA as the LivePortrait-based control. This is a research shortlist. Both still need complete Mac, timing, asset-term and actual-quality checks before an implementation choice.

The personalized audit now also checks Gaussian avatars and a differentiable Metal backend. Metal training components exist, but no audited combination yet supplies a cleared, complete personal renderer plus natural audio-driven head/eye behavior and arbitrary real-photo outfits. Repository release availability is checked separately from paper demonstrations.

## Recommendation

The strongest next research direction is a personalized avatar learned from clear video of Suman, with joint attention to appearance, articulation and speaking style. For the current 16 GB Mac, first investigate a compact person-specific video renderer and its training feasibility. This is an engineering recommendation based on the evidence below. No reviewed implementation has demonstrated all the requested quality, photo-outfit flexibility and M5 constraints together.

A useful near-term control is dubbing original camera footage while preserving its photographed eyes, head and torso. That can establish whether new speech can be rendered convincingly in Suman's face. It still inherits the source performance's gestures and requires a separate solution for newly timed expressions, long videos and arbitrary outfit photographs. Camera dubbing alone does not satisfy the final application objective.

Keep `Portrait_BlackT.png` as the required appearance and outfit reference. Test that requirement explicitly after the renderer passes on genuine footage. A method needing a separate recording or trained appearance for every outfit has a product tradeoff that must be reported.

## What the current evidence establishes

The user rejected the earlier long FlashHead output for head, eye, expression and stability problems. LTX free-motion and camera-only Dub-It were also rejected. The later Black T KJ/Dub-It result is a comparison baseline with unresolved quality. No person-specific training on Suman has been completed.

The latest existing Black T timing report estimated -2 frames globally, but local estimates varied from -2 to +14 and -8 frames with modest confidence. The same diagnostic found a five-frame shift after a deliberate 200 ms delay on real camera audio. This supports investigating local articulation, rather than applying one global audio shift. These estimates remain diagnostic and need playback confirmation. See [timing evidence](../.local/dubit-blackt-baseline-diagnostics/dubit-timing.json) and [camera delay control](../.local/dubit-blackt-baseline-diagnostics/camera-delay-timing.json).

Registration on all 121 frames found maximum shifts below 0.007 output pixels in the chosen background patch and 0.003 in the shirt patch. The head region changes substantially. This narrows the encoded-video investigation to head motion, facial deformation and blending. It does not assess display or player judder. See [regional evidence](../.local/dubit-blackt-baseline-diagnostics/canvas-stability.json).

The previous pipeline uses a neural photo animation followed by another generative pass over the face. My inference is that each stage can introduce identity, eyelid and mouth changes, while unrelated source gestures can disagree with new speech. Reduced motion and sharper output alone cannot establish natural performance.

## What HeyGen's research actually suggests

HeyGen's Avatar V report describes conditioning on video references, a motion representation for personal talking style and training losses covering identity, synchronization and perception. It reports distributed inference across eight GPUs. It also describes reference conditioning without identity-specific fine-tuning at inference time. Consequently, “we must train a custom model because HeyGen does” would be an unsupported conclusion. The transferable principle is to model dynamic identity and speech together. Its proprietary foundation model and infrastructure are a substantial gap from an app combining public pretrained components. The user's supplied comparison clip has an unknown engine version. [HeyGen technical report](https://dynamic.heygen.ai/www/Paper%20Links/avatarv_tech_report.pdf)

## Candidate assessment

| Approach | Why it matters | Evidence and decision |
| --- | --- | --- |
| Personalized 2D renderer, including SyncTalk_2D | Learns a particular person's face from recorded video and retains much of the photographed performance | Priority source audit for a compact renderer. Mac training, likeness, language generalization and licensing remain unresolved. |
| Personalized 3D, including MimicTalk and SyncTalk | Explicit identity, motion and expression modeling is relevant to the reported defects | Strong research comparison. CUDA, tracking and renderer dependencies make a Mac port substantial work. |
| MuseTalk 1.5 adaptation | Public training code plus an existing Apple inference port provide a practical engineering starting point | A portability/control candidate. Existing face-detail and jitter limitations prevent selecting it as a quality winner. |
| LTX/Dub-It | Existing local inference provides a comparison against further work | Retain as a baseline. Tested output has unresolved motion and timing. |
| LongCat Avatar 1.5 | Joint image/audio generation and long-video handling | Relevant quality reference. Published Apple configuration exceeds the current Mac's memory. |
| TBDub | Specialized video dubbing with explicit temporal and identity objectives | Relevant dubbing research. Published resource requirements exceed a straightforward 16 GB deployment. |
| EchoMimicV3-Flash | Smaller generative model, published 12 GB GPU claim | Secondary source review. NVIDIA VRAM figures do not establish 16 GB unified-memory feasibility. |
| LiveAvatar, InfiniteTalk, SoulX-FlashTalk | Joint audio-driven video generation with temporal context | Useful architecture comparisons. CUDA-oriented implementations and larger models have an unresolved local cost. |
| HeyGem / Duix Avatar | Offline toolkit with a cloning workflow | Published deployment requires NVIDIA and substantial host resources. Its name does not establish equivalence to HeyGen. |

### Compact person-specific training

SyncTalk_2D publishes an actual training pipeline, recommends a five-minute stable frontal recording and uses a 328-pixel working resolution. Source inspection shows explicit CUDA calls in training and inference. The inference script traverses the recorded frames forwards and backwards, so its default motion behavior conflicts with realistic long-form performance. I did not find a repository LICENSE at the checked path. That needs resolution before adopting or distributing its code. The method is a candidate to study, with clear changes and quality gates required. [README](https://github.com/ZiqiaoPeng/SyncTalk_2D), [training source](https://github.com/ZiqiaoPeng/SyncTalk_2D/blob/main/train_328.py), [inference source](https://github.com/ZiqiaoPeng/SyncTalk_2D/blob/main/inference_328.py)

MuseTalk provides training scripts with perceptual, synchronization and adversarial objectives. Its documentation acknowledges a 256-pixel face region, changes to moustache and lip detail, and jitter. Fine-tuning on Suman could be evaluated for identity improvement, but that improvement is unproven. Its Apple port demonstrates an inference route on an M3 Ultra, with a separate hosted-services layer that would be excluded from this application. Training on M5 requires its own audit and measured backward pass. [MuseTalk](https://github.com/TMElyralab/MuseTalk), [training configuration](https://github.com/TMElyralab/MuseTalk/blob/main/configs/training/stage2.yaml), [Apple port](https://github.com/barnent1/musetalk-mac)

### Personalized 3D and motion style

MimicTalk publishes training and inference for a specific person, optional style and pose videos, and a reported batch-one training requirement of approximately 8 GB VRAM. Its installation is verified on A100/V100 with CUDA 12.1 and includes PyTorch3D/MMCV. That figure cannot be translated into a tested 16 GB Mac budget. Its default inference pose is static unless pose information is supplied. [MimicTalk](https://github.com/yerfor/MimicTalk), [installation](https://github.com/yerfor/MimicTalk/blob/main/docs/prepare_env/install_guide.md)

SyncTalk++ describes synchronization across identity, speech, expression, head and torso. It is closely aligned with the user's complaints. However, its project page's Code link resolves to the original SyncTalk repository. The original implementation targets CUDA and recommends approximately four to five minutes of training video. Availability of the complete improved implementation must be confirmed before treating it as deployable. [SyncTalk++ project](https://ziqiaopeng.github.io/synctalk%2B%2B/), [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk)

### Larger video and dubbing models

LongCat Avatar 1.5 has image/audio generation and continuation support. Its Apple MLX port reports testing on a 128 GB M5 Max and recommends 32–48 GB Macs for its 4-bit variant. That is materially different from this M5/16 GB machine. Our custom streaming experiments demonstrate engineering possibilities, but have not yielded accepted realism. [Official LongCat](https://github.com/meituan-longcat/LongCat-Video), [Apple port](https://github.com/xocialize/longcat-avatar-mlx)

TBDub reports 15.01 GiB sampled NVIDIA process memory for Student with CPU offload, plus 21.3 GiB host RSS on its stated workload. Its weights and auxiliaries total about 15.27 GB on disk. Those are separate resource measurements, not a Mac fit claim. Synthetic Metal block and codec tests in this project do not establish trained-model quality. [TBDub](https://github.com/TaoLiveAIGC/TBDub)

LTX's official Dub-It guide focuses on replacement dialogue and generated speech. It validates five listed languages, which exclude Hindi. Freezing supplied target-audio latents and later remuxing the original recording is a local experiment with a separate synchronization question. The MLX project's maturity notes also classify several audio-conditioned paths as experimental. [Official Dub-It guide](https://docs.ltx.io/open-source-model/feature-guides/audio/dub-it-beta), [MLX maturity notes](https://github.com/dgrauet/ltx-2-mlx/blob/main/docs/PIPELINE_MATURITY.md)

EchoMimicV3-Flash advertises a 1.3B model and 12 GB VRAM operation. Its tested environments are CUDA/Linux with NVIDIA GPUs. LiveAvatar advertises 48 GB GPU inference with FP8. InfiniteTalk documents both video and image inputs, but also color/identity limitations and CUDA dependencies. Duix's documented setup requires NVIDIA and recommends at least 32 GB host RAM. These remain comparisons rather than immediate Mac deployment selections. [EchoMimic](https://github.com/antgroup/echomimic_v3), [LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar), [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk), [Duix](https://github.com/duixcom/Duix-Avatar)

## Proposed research sequence

1. Audit compact personalized training end to end. Inspect the data pipeline, trainable parameters, backwards operations, memory, checkpoint handling and licenses for MuseTalk adaptation and SyncTalk_2D. Compare MimicTalk's stronger style modeling against its porting cost. Produce a feasibility decision before another model download or implementation effort.
2. Establish two independent evaluation tasks. Same-speech reconstruction uses held-out camera footage with its own audio and exposes identity/reconstruction defects. New-speech generation uses a separate English, Hindi or Hinglish recording and exposes lip timing and motion generalization. Keep held-out frames and speech out of training.
3. For the first quality comparison, use original captured video as visual input. Record whether the eyes and head are preserved footage, generated motion or transferred motion. Reusing a genuine performance may give natural movement while still having gesture/content mismatch.
4. Require visible improvement over the existing Black T baseline at normal speed, including pauses, closed lips, teeth, blink transitions and the final seconds. Check source audio equality and timestamps separately. Score-based timing correction is insufficient.
5. Test the original Black T photograph and at least one other real outfit. A video-trained avatar that cannot preserve separately uploaded outfits leaves an explicit requirement incomplete. Resolve that gap before native integration or long renders.

## Capture resource with the highest likely value

My proposed dataset is one continuous five-minute recording in the black T-shirt and one separate 30–60 second validation take. Use the original camera file, a fixed camera, stable focus/exposure, even lighting and clear synchronized audio. Frame the head, shoulders and torso while retaining several hundred real pixels across the face. Include English, Hindi and Hinglish speech, natural blinking, small turns, pauses, relaxed mouth closure and visible consonant articulation. Preserve source frame rate and audio until preprocessing. A closer still in the same lighting can supplement `Portrait_BlackT.png`. Background replacement and beauty filtering reduce the value of the capture for evaluating fine detail.

This is a proposed protocol for the selected research direction. The five-minute duration is consistent with SyncTalk_2D's guidance. HeyGen's own capture guide recommends at least two uninterrupted minutes and favors good camera footage. The final dataset should follow the chosen model's actual requirements. [HeyGen capture guide](https://help.heygen.com/en/articles/8389138-digital-twin-video-avatar-filming-tips)

## Decision boundary

Keep the current M5 and fully local requirement as the default. If the source audit shows that a promising personalized method needs an infeasible port or memory budget, document that result before selecting another method. A separately owned local GPU workstation could expand the candidate set, but represents a hardware change that has not been authorized. More render time alone cannot restore missing facial information or remove a model's quality ceiling.

The evidence supports prioritizing personalization and better reference coverage. It does not support naming a universally best model or promising HeyGen parity on the current hardware. The next deliverable is a source-level training feasibility decision, followed by a controlled test only when the research direction is settled.
