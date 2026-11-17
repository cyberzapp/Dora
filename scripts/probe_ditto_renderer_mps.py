"""Probe pinned Ditto rendering architectures with random weights, never media."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch


def tensor_outputs(value, prefix="output"):
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    else:
        raise TypeError(f"Unsupported output type: {type(value).__name__}")
    result = {}
    for key, child in children:
        result.update(tensor_outputs(child, f"{prefix}.{key}"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only", choices=["AppearanceFeatureExtractor", "MotionExtractor",
                        "StitchingNetwork", "WarpingNetwork", "SPADEDecoder"])
    args = parser.parse_args()
    root = args.audit_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())["ditto-talkinghead-main"]
    records = {f["path"]: f for f in manifest["files"]}
    names = ["appearance_feature_extractor", "motion_extractor", "convnextv2", "util",
             "warping_network", "dense_motion", "spade_generator", "stitching_network"]
    paths = [f"core/models/modules/{name}.py" for name in names]
    for path in paths:
        data = (root / "ditto-talkinghead-main" / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != records[path]["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
    package = types.ModuleType("_ditto_renderer_probe")
    package.__path__ = [str(root / "ditto-talkinghead-main/core/models/modules")]
    sys.modules[package.__name__] = package
    report = {"scope": "Separate FP32 random-weight stage checks; no checkpoint, avatar, or full pipeline",
              "commit": manifest["commit"], "torch": torch.__version__, "fallback": "0",
              "source_sha256": {p: records[p]["sha256"] for p in paths}, "stages": []}
    torch.manual_seed(198)
    source_kp = torch.rand(1, 21, 3) * 0.6 - 0.3
    driving_kp = source_kp + torch.randn(1, 21, 3) * 0.01
    specifications = [
        ("appearance_feature_extractor", "AppearanceFeatureExtractor", [torch.rand(1, 3, 256, 256)]),
        ("motion_extractor", "MotionExtractor", [torch.rand(1, 3, 256, 256)]),
        ("stitching_network", "StitchingNetwork", [source_kp, driving_kp]),
        ("warping_network", "WarpingNetwork", [torch.rand(1, 32, 16, 64, 64) * 0.1, source_kp, driving_kp]),
        ("spade_generator", "SPADEDecoder", [torch.rand(1, 256, 64, 64) * 0.1]),
    ]
    for module_name, class_name, inputs in specifications:
        if args.only and class_name != args.only:
            continue
        entry = {"module": class_name, "input_shapes": [list(value.shape) for value in inputs],
                 "passed": False}
        model = actual = expected = mps_inputs = None
        try:
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable in this process")
            cls = getattr(importlib.import_module(package.__name__ + "." + module_name), class_name)
            model = cls().eval()
            entry["parameter_count"] = sum(p.numel() for p in model.parameters())
            with torch.inference_mode():
                start = time.perf_counter()
                expected = model(*inputs)
                entry["cpu_forward_seconds"] = time.perf_counter() - start
                model.to("mps")
                mps_inputs = [value.to("mps") for value in inputs]
                torch.mps.synchronize()
                start = time.perf_counter()
                actual = model(*mps_inputs)
                torch.mps.synchronize()
                entry["mps_forward_seconds_first_dispatch"] = time.perf_counter() - start
                expected_dict = tensor_outputs(expected)
                actual_dict = tensor_outputs(actual)
                comparisons = []
                for key, reference in expected_dict.items():
                    value = actual_dict[key].cpu()
                    comparisons.append({"output": key, "shape": list(value.shape),
                        "finite": bool(torch.isfinite(value).all()),
                        "max_absolute_error": (value - reference).abs().max().item(),
                        "mean_absolute_error": (value - reference).abs().mean().item(),
                        "allclose_atol_0.0005_rtol_0.001": torch.allclose(value, reference, atol=5e-4, rtol=1e-3)})
                entry["outputs"] = comparisons
                entry["passed"] = all(v["finite"] and v["allclose_atol_0.0005_rtol_0.001"] for v in comparisons)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        finally:
            del model, actual, expected, mps_inputs
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        report["stages"].append(entry)
        print(json.dumps(entry), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = all(stage["passed"] for stage in report["stages"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
