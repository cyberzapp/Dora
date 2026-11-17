"""Isolated real-media Ditto trial; not an application backend or quality claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pause_weights(samples, rate, frame_count, *, rest_at_start=False, fade_frames=3):
    """Find sustained quiet spans and ease the mouth toward rest inside them."""
    import numpy as np

    edges = np.minimum(np.arange(frame_count + 1, dtype=np.int64) * rate // 25, len(samples))
    energy = np.array([np.sqrt(np.mean(samples[a:b] ** 2)) if a < b else 0
                       for a, b in zip(edges[:-1], edges[1:])])
    weights = np.zeros(frame_count, dtype=np.float32)
    reference = np.percentile(energy, 85)
    if not np.isfinite(reference) or reference < .002:
        return weights
    quiet_edges = np.flatnonzero(np.diff(np.r_[False, energy < .12 * reference, False]))
    for start, end in zip(quiet_edges[::2], quiet_edges[1::2]):
        if end - start < 5:
            continue
        distance = np.minimum(np.arange(1, end - start + 1), np.arange(end - start, 0, -1))
        weights[start:end] = np.sin(np.pi / 2 * np.minimum(distance / fade_frames, 1)) ** 2
    if rest_at_start:
        # Use a stricter threshold at the leading edge to leave quiet speech
        # onsets under the original articulation model's control.
        lead = int(np.argmax(np.r_[energy >= .02 * reference, True]))
        if lead >= 2:
            remaining = np.arange(lead, 0, -1)
            opening = np.sin(np.pi / 2 * np.minimum(remaining / min(lead, 3), 1)) ** 2
            weights[:lead] = np.maximum(weights[:lead], opening)
    return weights


def smooth_pose(motions, sigma_frames):
    """Zero-phase smoothing of rotation logits before motion stitching."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    for key in ("pitch", "yaw", "roll"):
        values = np.stack([frame[key] for frame in motions])
        filtered = gaussian_filter1d(values, sigma_frames, axis=0, mode="reflect")
        # Preserve the first pose used as the reference for relative driving.
        filtered += values[0] - filtered[0]
        filtered[0] = values[0]
        for frame, logits in zip(motions, filtered):
            frame[key] = logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=4.84)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress-json", type=Path,
                        help="Write render progress for a supervising local worker")
    parser.add_argument("--lip-zero-delta", type=Path,
                        help="Optional upstream source-lip normalization delta, not a motion track")
    parser.add_argument("--lip-zero-strength", type=float,
                        help="Scale the source-mouth correction from zero to one")
    parser.add_argument("--rest-mouth-in-pauses", action="store_true",
                        help="Apply the source-mouth correction only in sustained quiet spans")
    parser.add_argument("--rest-mouth-at-start", action="store_true",
                        help="Also relax the mouth during a short quiet lead-in")
    parser.add_argument("--pause-fade-frames", type=int, default=3,
                        help="Frames to ease toward mouth rest at each edge of a quiet span")
    parser.add_argument("--neutralize-speech-mouth", action="store_true",
                        help="Reduce the source smile along the lip closure direction during speech")
    parser.add_argument("--personal-lip-track", type=Path,
                        help="Verified reference-conditioned TalkLikeYou lip trajectory")
    parser.add_argument("--pose-smoothing", type=float, metavar="SIGMA_FRAMES",
                        help="Apply offline Gaussian smoothing to the three head rotation logits")
    parser.add_argument("--keep-render-features-on-metal", action="store_true",
                        help="Keep intermediate warp features on Metal until image decoding")
    parser.add_argument("--blink-at", type=float, action="append", default=[],
                        metavar="SECONDS", help="Center a Ditto eyelid gesture at this time")
    args = parser.parse_args()
    if not 0 < args.seconds <= 15:
        parser.error("Trial duration must be between zero and 15 seconds")
    if args.lip_zero_strength is not None and (
        not args.lip_zero_delta or not 0 <= args.lip_zero_strength <= 1
    ):
        parser.error("--lip-zero-strength requires a delta and must be between zero and one")
    if args.rest_mouth_in_pauses and not args.lip_zero_delta:
        parser.error("--rest-mouth-in-pauses requires a lip-zero delta")
    if args.rest_mouth_at_start and not args.rest_mouth_in_pauses:
        parser.error("--rest-mouth-at-start requires --rest-mouth-in-pauses")
    if not 3 <= args.pause_fade_frames <= 12:
        parser.error("--pause-fade-frames must be between 3 and 12")
    if args.neutralize_speech_mouth and not args.rest_mouth_in_pauses:
        parser.error("--neutralize-speech-mouth requires --rest-mouth-in-pauses")
    if args.pose_smoothing is not None and not .5 <= args.pose_smoothing <= 5:
        parser.error("--pose-smoothing must be between 0.5 and 5 frames")
    root, audit = args.trial_root.resolve(), args.audit_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "output"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ditto-blackt-trial.mov").exists():
        raise ValueError("Use a fresh output directory; existing trial must be retained")

    def report_progress(stage, written=0, total=None):
        if args.progress_json is None:
            return
        progress_path = args.progress_json.resolve()
        temporary = progress_path.with_name(f".{progress_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"stage": stage, "written_frames": written,
                                         "total_frames": total}) + "\n")
        temporary.replace(progress_path)

    report_progress("validating")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["MPLCONFIGDIR"] = str(root / "mpl-cache")
    sys.path.insert(0, str(root / "python-deps"))
    sys.path.insert(0, str(root / "upstream"))
    import numpy as np
    import soundfile as sf
    import torch
    import imageio_ffmpeg

    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal is not available in this process")
    metadata = json.loads((audit / "digital-avatar--ditto-talkinghead-metadata.json").read_text())
    weights_root = root / "checkpoints/ditto_pytorch"
    checked = []
    for item in metadata["siblings"]:
        if not item["rfilename"].startswith("ditto_pytorch/"):
            continue
        path = root / "checkpoints" / item["rfilename"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Incomplete model asset: {path.name}")
        if sha256(path) != item["lfs"]["sha256"]:
            raise ValueError(f"Model asset hash mismatch: {path.name}")
        checked.append(item["rfilename"])
    config_source = audit / "ditto-config.pkl"
    if sha256(config_source) != "8e4c9ecb8dbb5aa4d22cb3f9b2c6ceabc62499b04af5fe074826d5b1044c4146":
        raise ValueError("Published configuration hash mismatch")
    # Only this previously opcode-audited, pinned upstream pickle is accepted.
    with config_source.open("rb") as stream:
        config = pickle.load(stream)
    blink_template = config["default_kwargs"].get("delta_eye_arr")
    for stage in config["base_cfg"].values():
        stage["device"] = "mps"
    config["audio2motion_cfg"]["device"] = "mps"
    controls = {"emo": 4, "ch_info": None, "delta_eye_arr": None,
                "eye_f0_mode": True, "drive_eye": True, "relative_d": True,
                "sampling_timesteps": 50, "online_mode": False,
                "overall_ctrl_info": {}, "ctrl_info": {}}
    config["default_kwargs"].update(controls)
    config_path = output / "mps-explicit-controls.pkl"
    with config_path.open("wb") as stream:
        pickle.dump(config, stream)
    loader = root / "upstream/core/utils/load_model.py"
    text = loader.read_text()
    old = "    model = module(**kwargs)\n    model.load_model(model_path).to(device)"
    new = "    if module_name == 'LMDM':\n        kwargs['device'] = device\n    model = module(**kwargs)\n    model.load_model(model_path).to(device)"
    if old in text:
        loader.write_text(text.replace(old, new, 1))
    elif new not in text:
        raise ValueError("Unexpected upstream model loader")
    # Use the FP32 path already checked numerically, rather than assume MPS FP16.
    for name in ["motion_extractor", "appearance_extractor", "warp_network", "decoder"]:
        path = root / f"upstream/core/models/{name}.py"
        text = path.read_text()
        path.write_text(text.replace("dtype=torch.float16, enabled=True", "dtype=torch.float16, enabled=False"))
    blend = root / "upstream/core/utils/blend/__init__.py"
    blend.write_text("import pyximport\nimport numpy as np\npyximport.install(build_dir=" + repr(str(root / "cython-build")) + ", setup_args={'include_dirs': np.get_include()})\nfrom .blend import blend_images_cy\n")
    source_rate = sf.info(args.audio).samplerate
    frames = round(args.seconds * source_rate)
    num_frames = math.ceil(frames / source_rate * 25)
    frame_controls = {}
    if args.blink_at:
        eye_delta = np.asarray(blink_template, dtype=np.float32)
        if eye_delta.ndim != 2 or eye_delta.shape[1] != 63 or not np.isfinite(eye_delta).all():
            raise ValueError("Invalid Ditto eyelid template")
        eye_points = [11, 13, 15, 16, 18]
        eyelids = np.zeros((len(eye_delta), 21, 3), dtype=np.float32)
        eyelids[:, eye_points] = eye_delta.reshape(-1, 21, 3)[:, eye_points]
        # Return to zero at the ends so the gesture joins the generated motion smoothly.
        blend = np.linspace(0, 1, len(eyelids), dtype=np.float32)[:, None, None]
        eyelids -= (1 - blend) * eyelids[:1] + blend * eyelids[-1:]
        peak = int(np.linalg.norm(eyelids.reshape(-1, 63), axis=1).argmax())
        for seconds in args.blink_at:
            if not math.isfinite(seconds):
                parser.error("--blink-at must be a finite time")
            first = round(seconds * 25) - peak
            if first < 0 or first + len(eyelids) > num_frames:
                parser.error("Each blink must fit inside the render")
            for index, delta in enumerate(eyelids, first):
                if index in frame_controls:
                    parser.error("Blink gestures must not overlap")
                frame_controls[index] = {"delta_exp": delta.reshape(1, 63)}
    original, sample_rate = sf.read(args.audio, frames=frames, dtype="float32", always_2d=True)
    if len(original) != frames:
        raise ValueError("Voice-over is shorter than requested trial")
    rest_weights = (pause_weights(original, sample_rate, num_frames, rest_at_start=args.rest_mouth_at_start,
                                  fade_frames=args.pause_fade_frames)
                    if args.rest_mouth_in_pauses else None)
    excerpt = output / "original-voice-excerpt.wav"
    sf.write(excerpt, original, sample_rate, subtype="FLOAT")
    reread, _ = sf.read(excerpt, dtype="float32", always_2d=True)
    if not np.array_equal(original, reread):
        raise ValueError("Voice excerpt did not preserve decoded samples")
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    lip_delta = None
    if args.lip_zero_delta:
        baseline = json.loads((root / "output/provenance.json").read_text())
        if baseline["photo_sha256"] != sha256(args.photo):
            raise ValueError("Lip-normalization source differs from the baseline photograph")
        lip_delta = np.load(args.lip_zero_delta, allow_pickle=False)
        if lip_delta.shape != (1, 21, 3) or lip_delta.dtype != np.float32 or not np.isfinite(lip_delta).all():
            raise ValueError("Invalid lip-normalization delta")
        strength = 1.0 if args.lip_zero_strength is None else args.lip_zero_strength
        lip_delta *= strength
    personal_lips = None
    articulation_provenance = None
    if args.personal_lip_track:
        articulation_provenance = json.loads((args.personal_lip_track.parent / "provenance.json").read_text())
        if (articulation_provenance["track_sha256"] != sha256(args.personal_lip_track)
            or articulation_provenance["photo_sha256"] != sha256(args.photo)
            or articulation_provenance["voice_sha256"] != sha256(args.audio)):
            raise ValueError("Articulation source or trajectory hash mismatch")
        personal_lips = np.load(args.personal_lip_track, allow_pickle=False)
        expected = (math.ceil(frames / sample_rate * 25), 6, 3)
        if personal_lips.shape != expected or personal_lips.dtype != np.float32 or not np.isfinite(personal_lips).all():
            raise ValueError("Articulation trajectory does not cover this exact clip")
    provenance = {"status": "running", "engine": "Ditto", "seed": seed,
        "source_commit": "c3e47eee2e626500017a0556b470d6d4182f85e8",
        "model_revision": metadata["sha"], "verified_assets": checked,
        "photo": str(args.photo.resolve()), "photo_sha256": sha256(args.photo),
        "voice": str(args.audio.resolve()), "voice_sha256": sha256(args.audio),
        "excerpt_samples": frames, "excerpt_sample_rate": sample_rate,
        "controls": controls, "fallback": "0", "quality_accepted": False,
        "precision": "float32",
        "patches": ["Forward sampler constructor device", "Local Cython build/cache paths", "Disable wrapper FP16 autocast for tested FP32 path"]}
    if lip_delta is not None:
        provenance["lip_normalization"] = {"delta_file": str(args.lip_zero_delta.resolve()),
            "sha256": sha256(args.lip_zero_delta), "application": "Add upstream source-lip-zero delta after stitching, before feature warp",
            "new_source_image_generated": False, "new_motion_generated_by_other_model": personal_lips is not None}
        provenance["lip_normalization"]["strength"] = strength
        if rest_weights is not None:
            provenance["lip_normalization"]["pause_weights"] = rest_weights.tolist()
            provenance["lip_normalization"]["rest_at_start"] = args.rest_mouth_at_start
            provenance["lip_normalization"]["pause_fade_frames"] = args.pause_fade_frames
        if args.neutralize_speech_mouth:
            provenance["lip_normalization"]["speech"] = {
                "application": "Shift toward source-mouth closure with soft saturation",
                "max_shift": 0.85,
                "saturation": 1.0,
            }
    if personal_lips is not None:
        provenance["articulation"] = articulation_provenance
        provenance["articulation"]["application"] = "Replace six lip-expression triplets in absolute motion space; retain Ditto pose and other expression channels"
    if args.pose_smoothing is not None:
        provenance["pose_smoothing"] = {
            "sigma_frames": args.pose_smoothing, "channels": ["pitch", "yaw", "roll"],
            "first_pose_preserved": True,
        }
    if frame_controls:
        provenance["blink_centers_seconds"] = args.blink_at
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    start = time.monotonic()
    try:
        from stream_pipeline_offline import StreamSDK
        report_progress("loading_models", total=num_frames)
        print("Loading verified local models", flush=True)
        sdk = StreamSDK(str(config_path), str(weights_root))
        if args.keep_render_features_on_metal:
            warp = sdk.warp_f3d.warp_net
            decoder = sdk.decode_f3d.decoder
            if any(stage.model_type != "pytorch" or stage.device != "mps"
                   for stage in (warp, decoder)):
                raise RuntimeError("Metal feature retention requires both PyTorch stages on MPS")

            def render_on_metal(features, source, driving):
                with torch.no_grad():
                    warped = warp.model(*(torch.from_numpy(value).to("mps")
                                          for value in (features, source, driving))).float()
                    image = decoder.model(warped).float().cpu().numpy()
                return np.transpose(image[0], [1, 2, 0]).clip(0, 1) * 255

            # The existing stage queues still carry frame indices and preserve order.
            sdk.warp_f3d = render_on_metal
            sdk.decode_f3d = lambda image: image
            provenance["patches"].append("Retain intermediate warp tensor on Metal")
        if lip_delta is not None:
            original_warp = sdk.warp_f3d
            if rest_weights is None:
                sdk.warp_f3d = lambda features, source, driving: original_warp(features, source, driving + lip_delta)
            else:
                mouth_points = [6, 12, 14, 17, 19, 20]
                closure = lip_delta[:, mouth_points]
                closure_energy = float(np.sum(closure ** 2))
                frame_index = 0

                def warp_with_rest_mouth(features, source, driving):
                    nonlocal frame_index
                    weight = rest_weights[frame_index]
                    frame_index += 1
                    if weight or (args.neutralize_speech_mouth and closure_energy > 0):
                        driving = driving.copy()
                        source_mouth = source[:, mouth_points]
                        mouth = driving[:, mouth_points]
                        if args.neutralize_speech_mouth and closure_energy > 0:
                            projected = float(np.sum((mouth - source_mouth) * closure) / closure_energy)
                            available = max(0.0, 1.0 - projected)
                            # Saturate smoothly near closure without flattening the
                            # components perpendicular to the closure direction.
                            mouth = mouth + (0.85 * available / (available + 1.0)) * closure
                        rest = source_mouth + closure
                        driving[:, mouth_points] = (1 - weight) * mouth + weight * rest
                    return original_warp(features, source, driving)

                sdk.warp_f3d = warp_with_rest_mouth
        lip_indices = [6, 12, 14, 17, 19, 20]
        if personal_lips is not None or args.pose_smoothing is not None:
            original_convert = sdk.audio2motion.cvt_fmt
            def convert_motion(sequence):
                motions = original_convert(sequence)
                if personal_lips is not None and len(motions) != len(personal_lips):
                    raise ValueError("Generated motion and personal articulation lengths differ")
                np.savez_compressed(output / "ditto-base-motion.npz",
                    **{key: np.stack([row[key] for row in motions]) for key in motions[0]})
                if args.pose_smoothing is not None:
                    smooth_pose(motions, args.pose_smoothing)
                    np.savez_compressed(output / "ditto-filtered-pose.npz",
                        **{key: np.stack([row[key] for row in motions]) for key in ("pitch", "yaw", "roll")})
                if personal_lips is not None:
                    for index, row in enumerate(motions):
                        expression = row["exp"].copy().reshape(1, 21, 3)
                        expression[0, lip_indices] = personal_lips[index]
                        row["exp"] = expression.reshape(1, 63)
                return motions
            sdk.audio2motion.cvt_fmt = convert_motion
        if personal_lips is not None:
            from core.atomic_components import motion_stitch
            original_mix = motion_stitch._mix_s_d_info
            def mix_absolute_lips(source, driving, use_d_keys, initial):
                lips = driving["exp"].reshape(1, 21, 3)[:, lip_indices].copy()
                mixed = original_mix(source, driving, use_d_keys, initial)
                expression = mixed["exp"].copy().reshape(1, 21, 3)
                expression[:, lip_indices] = lips
                mixed["exp"] = expression.reshape(1, 63)
                return mixed
            motion_stitch._mix_s_d_info = mix_absolute_lips
        final = output / "ditto-blackt-trial.mov"
        print("Preparing real source photograph", flush=True)
        sdk.setup(str(args.photo.resolve()), str(final), **controls)
        sdk.setup_Nd(N_d=num_frames, fade_in=-1, fade_out=-1, ctrl_info=frame_controls)
        original_update = sdk.writer_pbar.update
        def update_written_frames(count=1):
            original_update(count)
            report_progress("writing_frames", sdk.writer_pbar.n, num_frames)
        if args.progress_json is not None:
            sdk.writer_pbar.update = update_written_frames
        import librosa
        analysis_audio, _ = librosa.load(excerpt, sr=16000)
        report_progress("analyzing_audio", total=num_frames)
        print(f"Generating {num_frames} actual model frames", flush=True)
        features = sdk.wav2feat.wav2feat(analysis_audio)
        sdk.audio2motion_queue.put(features)
        sdk.close()
        report_progress("muxing", num_frames, num_frames)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", sdk.tmp_output_path,
            "-i", str(excerpt), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "pcm_f32le", str(final)], check=True)
        provenance.update(status="rendered_unreviewed", output=str(final), video_frames=num_frames,
                          elapsed_seconds=time.monotonic() - start)
        print(str(final), flush=True)
        report_progress("exported", num_frames, num_frames)
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}",
                          elapsed_seconds=time.monotonic() - start)
        report_progress("failed", total=num_frames)
        raise
    finally:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
