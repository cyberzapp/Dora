"""Inspect upstream lip-zero retargeting on the real source; never a video result."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", required=True, type=Path)
    parser.add_argument("--photo", required=True, type=Path)
    args = parser.parse_args()
    root = args.trial_root.resolve()
    output = root / "lip-normalization-review"
    output.mkdir(exist_ok=True)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["MPLCONFIGDIR"] = str(root / "mpl-cache")
    sys.path[:0] = [str(root / "python-deps"), str(root / "upstream")]
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from core.atomic_components.source2info import Source2Info
    from core.atomic_components.loader import load_image
    from core.atomic_components.motion_stitch import transform_keypoint
    from core.atomic_components.warp_f3d import WarpF3D
    from core.atomic_components.decode_f3d import DecodeF3D
    from core.atomic_components.cfg import parse_cfg
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    cfg = parse_cfg(str(root / "output/mps-explicit-controls.pkl"),
                    str(root / "checkpoints/ditto_pytorch"))
    image = load_image(str(args.photo.resolve()), max_dim=1920)
    source = Source2Info(**cfg[0])(image)
    source_points = transform_keypoint(source["x_s_info"])
    landmarks = source["lmk203"]
    ratio = float(np.linalg.norm(landmarks[90] - landmarks[102]) /
                  (np.linalg.norm(landmarks[48] - landmarks[66]) + 1e-6))
    repository = Path(__file__).resolve().parents[1]
    checkpoint = repository / "vendor/LivePortrait/pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "3652d5a3f95099141a56986aaddec92fadf0a73c87a20fac9a2c07c32b28b611":
        raise ValueError("Retargeting checkpoint hash mismatch")
    spec = importlib.util.spec_from_file_location("_lip_retarget_network",
        repository / "vendor/LivePortrait/src/modules/stitching_retargeting_network.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lip = module.StitchingRetargetingNetwork(65, [128, 128, 64], 63).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["retarget_mouth"]
    lip.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    lip.to("mps")
    with torch.inference_mode():
        values = np.concatenate([source_points.reshape(1, 63), np.array([[ratio, 0]], dtype=np.float32)], axis=1)
        delta = lip(torch.from_numpy(values).to("mps")).cpu().numpy().reshape(1, 21, 3)
    np.save(output / "lip-zero-delta.npy", delta)
    warp = WarpF3D(cfg[4])
    decode = DecodeF3D(cfg[5])
    inverse = cv2.invertAffineTransform(source["M_c2o"][:2, :])
    crop = cv2.warpAffine(image, inverse, (512, 512))
    tiles = [("Original crop", crop)]
    for label, change in [("Identity reconstruction", np.zeros_like(delta)), ("Upstream lip-zero delta", delta)]:
        rendered = decode(warp(source["f_s"], source_points, source_points + change))
        if not np.isfinite(rendered).all():
            raise ValueError("Non-finite neural output")
        rendered = np.clip(rendered, 0, 255).astype(np.uint8)
        filename = "identity.png" if not np.any(change) else "lip-zero.png"
        Image.fromarray(rendered).save(output / filename)
        tiles.append((label, rendered))
    contact = Image.new("RGB", (1536, 550), (24, 24, 24))
    draw = ImageDraw.Draw(contact)
    for index, (label, tile) in enumerate(tiles):
        draw.text((index * 512 + 10, 12), label, fill="white")
        contact.paste(Image.fromarray(tile), (index * 512, 38))
    contact.save(output / "comparison.png")
    report = {"purpose": "Single-frame mouth-normalization diagnostic; no new avatar video",
              "photo": str(args.photo.resolve()), "source_lip_ratio": ratio,
              "target_lip_ratio": 0, "delta_l2": float(np.linalg.norm(delta)),
              "retargeting_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "quality_accepted": False, "source_file_modified": False,
              "limits": "No speech, temporal continuity or teeth recovery established by this test"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(output / "comparison.png", flush=True)


if __name__ == "__main__":
    main()
