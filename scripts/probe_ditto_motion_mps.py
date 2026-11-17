"""Synthetic Ditto architecture probe. Never loads a checkpoint or personal media."""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampling-steps", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.sampling_steps <= 1000:
        parser.error("--sampling-steps must be between 1 and 1000")
    audit_root = args.audit_root.resolve()
    manifest = json.loads((audit_root / "manifest.json").read_text())["ditto-talkinghead-main"]
    source_root = audit_root / "ditto-talkinghead-main"
    paths = ["core/models/modules/LMDM.py"] + [
        f"core/models/modules/lmdm_modules/{name}.py"
        for name in ["model", "utils", "rotary_embedding_torch"]
    ]
    records = {record["path"]: record for record in manifest["files"]}
    for path in paths:
        content = (source_root / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != records[path]["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
    package = types.ModuleType("_ditto_architecture_probe")
    package.__path__ = [str(source_root / "core/models/modules")]
    sys.modules[package.__name__] = package
    LMDM = importlib.import_module(package.__name__ + ".LMDM").LMDM
    report = {
        "scope": "Random weights and synthetic inputs; no checkpoint, avatar or quality evaluation",
        "commit": manifest["commit"], "source_sha256": {p: records[p]["sha256"] for p in paths},
        "torch": torch.__version__, "fallback": "0",
        "mps_available": torch.backends.mps.is_available(), "passed": False,
        "configuration": {"motion_feat_dim": 265, "audio_feat_dim": 1103, "seq_frames": 80},
    }
    try:
        if not report["mps_available"]:
            raise RuntimeError("MPS unavailable in this process")
        torch.manual_seed(735)
        cpu_model = LMDM(**report["configuration"], device="cpu").eval()
        mps_model = LMDM(**report["configuration"], device="mps").eval()
        mps_model.load_state_dict(cpu_model.state_dict(), strict=True)
        mps_model.to("mps")
        report["parameter_count"] = sum(p.numel() for p in cpu_model.parameters())
        report["parameter_bytes_fp32"] = sum(p.numel() * p.element_size() for p in cpu_model.parameters())
        inputs = (torch.randn(1, 80, 265), torch.randn(1, 265),
                  torch.randn(1, 80, 1103), torch.tensor([500], dtype=torch.long))
        with torch.inference_mode():
            cpu_result = cpu_model(*inputs)
            mps_inputs = tuple(value.to("mps") for value in inputs)
            torch.mps.synchronize()
            start = time.perf_counter()
            mps_result = mps_model(*mps_inputs)
            torch.mps.synchronize()
            report["guided_forward_seconds_first_dispatch"] = time.perf_counter() - start
            comparisons = []
            for name, expected, actual in zip(["predicted_noise", "predicted_clean_motion"], cpu_result, mps_result):
                actual = actual.cpu()
                comparisons.append({"output": name, "shape": list(actual.shape),
                    "max_absolute_error": (expected - actual).abs().max().item(),
                    "mean_absolute_error": (expected - actual).abs().mean().item(),
                    "finite": bool(torch.isfinite(actual).all()),
                    "allclose_atol_0.0005_rtol_0.001": torch.allclose(expected, actual, atol=5e-4, rtol=1e-3)})
            report["forward_comparisons"] = comparisons
            start = time.perf_counter()
            sample = mps_model.ddim_sample(mps_inputs[1], mps_inputs[2], sampling_timesteps=args.sampling_steps)
            torch.mps.synchronize()
            report["sampler"] = {"steps": args.sampling_steps, "shape": list(sample.shape),
                "finite": bool(torch.isfinite(sample).all()), "device": str(sample.device),
                "elapsed_seconds": time.perf_counter() - start,
                "cross_device_sample_comparison": False}
            report["passed"] = all(c["finite"] and c["allclose_atol_0.0005_rtol_0.001"] for c in comparisons) and report["sampler"]["finite"]
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
