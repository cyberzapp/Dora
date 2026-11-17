"""Measure the rendered effect of each implicit lip coordinate on a cached trial."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

LIPS = [6, 12, 14, 17, 19, 20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--response", type=Path, help="Test separate depth and jaw directions inferred from paired probes")
    parser.add_argument("--stitch-ablation", action="store_true",
                        help="Remove only per-point lip offsets, preserving shared stitch translation")
    parser.add_argument("--lip-swap", type=int, nargs="+",
                        help="Render one donor lip pose for each requested diagnostic frame")
    parser.add_argument("--trace-track", type=Path,
                        help="Compare cached articulation, upstream articulation, and pause normalization")
    parser.add_argument("--candidate-track", type=Path,
                        help="Replay a matched derived articulation track and record its geometry")
    parser.add_argument("frames", nargs="+", type=int)
    args = parser.parse_args()
    if sum(bool(mode) for mode in (args.response, args.stitch_ablation, args.lip_swap,
                                   args.trace_track, args.candidate_track)) > 1:
        parser.error("Choose one diagnostic mode")
    if args.lip_swap and len(args.lip_swap) != len(args.frames):
        parser.error("Provide one donor lip frame for each requested frame")
    trial = args.trial.resolve()
    root = trial.parent
    report = json.loads((trial / "provenance.json").read_text())
    from hashlib import sha256
    for filename, expected in ((report["photo"], report["photo_sha256"]),
                               (trial / "track/personal-lip-track.npy", report["articulation"]["track_sha256"])):
        if sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input differs from trial: {filename}")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["MPLCONFIGDIR"] = str(root / "mpl-cache")
    sys.path[:0] = [str(root / "python-deps"), str(root / "upstream")]
    import cv2
    import mediapipe as mp
    import torch
    import stream_pipeline_offline as upstream
    from core.atomic_components import motion_stitch

    original_mix = motion_stitch._mix_s_d_info

    def mix(source, driving, keys, initial):
        lips = driving["exp"].reshape(21, 3)[LIPS].copy()
        result = original_mix(source, driving, keys, initial)
        expression = result["exp"].copy().reshape(21, 3)
        expression[LIPS] = lips
        result["exp"] = expression.reshape(1, 63)
        return result

    motion_stitch._mix_s_d_info = mix
    sdk = upstream.StreamSDK(str(trial / "mps-explicit-controls.pkl"), str(root / "checkpoints/ditto_pytorch"))
    # Keep SDK setup and stitching, then run selected frames synchronously.
    for name in ("audio2motion", "motion_stitch", "warp_f3d", "decode_f3d", "putback", "writer"):
        setattr(sdk, f"{name}_worker", lambda: None)
    sdk.setup(report["photo"], str(args.output / "unused.mov"))
    base = np.load(trial / "ditto-base-motion.npz", allow_pickle=False)
    pose = np.load(trial / "ditto-filtered-pose.npz", allow_pickle=False)
    lips = np.load(trial / "track/personal-lip-track.npy", allow_pickle=False)
    upstream_lips = None
    alternate_track = args.trace_track or args.candidate_track
    if alternate_track:
        trace_provenance = json.loads(alternate_track.with_name("provenance.json").read_text())
        if (trace_provenance["voice_sha256"] != report["voice_sha256"]
            or trace_provenance["photo_sha256"] != report["photo_sha256"]
            or trace_provenance["track_sha256"] != sha256(alternate_track.read_bytes()).hexdigest()):
            raise ValueError("Trace track does not match the cached inputs")
        upstream_lips = np.load(alternate_track, allow_pickle=False)
        if upstream_lips.shape != lips.shape or not np.isfinite(upstream_lips).all():
            raise ValueError("Invalid upstream articulation")
    requested = args.frames + (args.lip_swap or [])
    if min(requested) < 0 or max(requested) >= len(lips):
        raise ValueError("Probe frames must be inside the cached timeline")
    info = sdk.source_info["x_s_info_lst"][0]
    features = torch.from_numpy(sdk.source_info["f_s_lst"][0]).to("mps")
    warp = sdk.warp_f3d.warp_net.model
    decoder = sdk.decode_f3d.decoder.model
    with (root.parent / "audio-motion-source-audit-2026-11-11/ditto-config.pkl").open("rb") as stream:
        import pickle
        data = stream.read()
    if sha256(data).hexdigest() != "8e4c9ecb8dbb5aa4d22cb3f9b2c6ceabc62499b04af5fe074826d5b1044c4146":
        raise ValueError("Original config hash mismatch")
    eye = np.asarray(pickle.loads(data)["default_kwargs"]["delta_eye_arr"], dtype=np.float32).reshape(-1,21,3)
    eyelid = np.zeros_like(eye)
    eyelid[:, [11,13,15,16,18]] = eye[:, [11,13,15,16,18]]
    blend = np.linspace(0,1,len(eye),dtype=np.float32)[:,None,None]
    eyelid -= (1-blend)*eyelid[:1] + blend*eyelid[-1:]
    peak = np.linalg.norm(eyelid.reshape(-1,63),axis=1).argmax()
    blink = {round(seconds*25)-int(peak)+j: value.reshape(1,63)
             for seconds in report["blink_centers_seconds"] for j,value in enumerate(eyelid)}
    normalization = report["lip_normalization"]
    weights = normalization["pause_weights"]
    closure = np.load(normalization["delta_file"],allow_pickle=False)[:, LIPS] * normalization["strength"]

    def motion(index, lip):
        row = {key: (pose[key] if key in pose.files else base[key])[index].copy() for key in base.files}
        row["exp"].reshape(21,3)[LIPS] = lip
        return row

    def render(index, lip, *, bypass_lip_stitch=False, normalize=True):
        controls = {"delta_exp": blink[index]} if index in blink else {}
        stitch = sdk.motion_stitch.stitch_net
        unstitched_lips = None
        shared_translation = None

        def capture_prediction(_module, _inputs, prediction):
            nonlocal shared_translation
            shared_translation = prediction[:, -2:].detach().cpu().numpy()

        def capture_stitch(source, driving):
            nonlocal unstitched_lips
            unstitched_lips = driving[:, LIPS].copy()
            return stitch(source, driving)

        hook = None
        if args.stitch_ablation or alternate_track:
            hook = stitch.model.mlp.register_forward_hook(capture_prediction)
            sdk.motion_stitch.stitch_net = capture_stitch
        try:
            source, driving = sdk.motion_stitch(info, motion(index, lip), **controls)
        finally:
            sdk.motion_stitch.stitch_net = stitch
            if hook is not None:
                hook.remove()
        stitch_delta = None
        if args.stitch_ablation:
            if unstitched_lips is None or shared_translation is None:
                raise RuntimeError("The baseline did not invoke stitching")
            unstitched_lips[..., :2] += shared_translation[:, None, :]
            stitch_delta = {
                "local_xyz": (driving[:, LIPS] - unstitched_lips)[0].tolist(),
                "shared_xy": shared_translation[0].tolist(),
            }
            if bypass_lip_stitch:
                driving = driving.copy()
                driving[:, LIPS] = unstitched_lips
        trace = None
        if alternate_track:
            trace = {"expression_xyz": lip.tolist(),
                     "before_stitch_xyz": unstitched_lips[0].tolist(),
                     "after_stitch_xyz": driving[0, LIPS].tolist(),
                     "pause_weight": weights[index] if normalize else 0}
        if normalize:
            driving[:, LIPS] = (1-weights[index])*driving[:, LIPS] + weights[index]*(source[:, LIPS]+closure)
        if trace is not None:
            trace["warp_input_xyz"] = driving[0, LIPS].tolist()
        with torch.inference_mode():
            warped = warp(features, torch.from_numpy(source).to("mps"), torch.from_numpy(driving).to("mps")).float()
            decoded = decoder(warped)[0].float().cpu().numpy().transpose(1,2,0).clip(0,1)*255
        return decoded.astype(np.uint8), stitch_delta, trace

    sdk.motion_stitch(info, motion(0, lips[0]))
    response = json.loads(args.response.read_text()) if args.response else None
    results = {}
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as mesh:
        for index, frame in enumerate(args.frames):
            variants = [("baseline", lips[frame])]
            if args.stitch_ablation:
                variants.append(("without-lip-stitch", lips[frame]))
            elif args.lip_swap:
                donor = args.lip_swap[index]
                variants.append((f"lip-donor-{donor}", lips[donor]))
            elif args.trace_track:
                variants.extend([("upstream", upstream_lips[frame]),
                                 ("without-pause-normalization", lips[frame])])
            elif args.candidate_track:
                variants = [("geometry", upstream_lips[frame])]
            elif response is None:
                for point in range(6):
                    for axis in range(3):
                        for sign in (-1, 1):
                            lip = lips[frame].copy()
                            lip[point, axis] += sign * .004
                            variants.append((f"{LIPS[point]}-{axis}-{sign:+d}", lip))
            else:
                rows = response[str(frame)]
                names = ("gap", "width", "depth_proxy", "chin_y")
                jacobian = np.array([
                    [(rows[f"{point}-{axis}-+1"][name] - rows[f"{point}-{axis}--1"][name]) / .008
                     for point in LIPS for axis in range(3)]
                    for name in names
                ])
                # Minimum-norm local changes, with other measured coordinates
                # held fixed in the linear approximation.
                directions = {}
                for name, target in (("depth", [0, 0, .010, 0]), ("jaw", [.014, 0, 0, .010])):
                    delta = np.linalg.lstsq(jacobian, target, rcond=.05)[0]
                    delta *= min(1, .004 / np.max(np.abs(delta)))
                    directions[name] = delta.reshape(6, 3).astype(np.float32)
                    variants.append((name, lips[frame] + directions[name]))
                np.savez(args.output / f"{frame}-directions.npz", **directions)
            rows = {}
            for name, lip in variants:
                rgb, stitch_delta, trace = render(frame, lip, bypass_lip_stitch=name == "without-lip-stitch",
                                                  normalize=name != "without-pause-normalization")
                found = mesh.process(rgb).multi_face_landmarks
                if not found:
                    raise ValueError(f"Missing face at {frame}/{name}")
                points = np.array([(p.x,p.y,p.z) for p in found[0].landmark])
                anchor = points[1]
                eye_width = np.linalg.norm(points[33,:2]-points[263,:2])
                rows[name] = {
                    "gap":float((points[14,1]-points[13,1])/eye_width),
                    "width":float(np.linalg.norm(points[61,:2]-points[291,:2])/eye_width),
                    "depth_proxy":float((points[[0,17],2].mean()-anchor[2])/eye_width),
                    "chin_y":float((points[152,1]-anchor[1])/eye_width),
                    "corners":((points[[61,291],:2]-anchor[:2])/eye_width).tolist(),
                }
                if stitch_delta is not None:
                    rows[name]["stitch_lip_delta"] = stitch_delta
                if trace is not None:
                    rows[name]["stages"] = trace
                cv2.imwrite(str(args.output/f"{frame}-{name}.png"), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            results[str(frame)] = rows
            print(f"Probed frame {frame}: {len(rows)} renders",flush=True)
    (args.output/"response.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    main()
