"""Research probe of renderer primitives; no weights or personal media are loaded."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

# Set before importing torch; a CPU fallback must not masquerade as MPS support.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
import torch.nn.functional as F


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "scope": "FP32 forward primitives only; not full-model compatibility or quality",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": torch.backends.mps.is_available(),
        "fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "checks": [],
    }
    if report["mps_available"]:
        generator = torch.Generator().manual_seed(183)
        # Shapes follow the audited Ditto dense-motion and final-feature warps.
        for shape in [(22, 4, 16, 64, 64), (1, 32, 16, 64, 64)]:
            entry = {"operation": "grid_sample_3d", "shape": shape,
                     "dtype": "float32", "mode": "bilinear",
                     "padding_mode": "zeros", "align_corners": False}
            try:
                volume = torch.rand(shape, generator=generator)
                grid = torch.rand((shape[0], *shape[2:], 3), generator=generator) * 2.2 - 1.1
                expected = F.grid_sample(volume, grid, align_corners=False)
                mps_volume, mps_grid = volume.to("mps"), grid.to("mps")
                torch.mps.synchronize()
                start = time.perf_counter()
                actual_device = F.grid_sample(mps_volume, mps_grid, align_corners=False)
                torch.mps.synchronize()
                entry["elapsed_seconds_including_first_dispatch"] = time.perf_counter() - start
                actual = actual_device.cpu()
                entry["max_absolute_error"] = (actual - expected).abs().max().item()
                entry["mean_absolute_error"] = (actual - expected).abs().mean().item()
                entry["finite"] = bool(torch.isfinite(actual).all())
                entry["passed"] = entry["finite"] and torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)
                del volume, grid, expected, actual, actual_device, mps_volume, mps_grid
            except Exception as error:
                entry.update(passed=False, error=f"{type(error).__name__}: {error}")
            report["checks"].append(entry)
        torch.mps.empty_cache()
    report["passed"] = report["mps_available"] and bool(report["checks"]) and all(
        check["passed"] for check in report["checks"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
