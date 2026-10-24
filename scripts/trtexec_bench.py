"""Day-one TensorRT check: build an FP16 engine for each downloaded ONNX model
with NVIDIA's ``trtexec`` and print its latency.

    python scripts/trtexec_bench.py            # all downloaded models
    python scripts/trtexec_bench.py --dry-run  # just print the commands

Needs TensorRT installed with ``trtexec`` on PATH. The engine itself does not
need these files: ONNX Runtime's TensorRT provider builds and caches the same
FP16 engines automatically. This script is for checking the numbers in the
design plan on your own GPU before committing to 60 Hz.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionplay.vision import models  # noqa: E402
from motionplay.vision.runtime import read_input  # noqa: E402


def command(onnx: Path, name: str, max_batch: int) -> list[str]:
    info = read_input(onnx)
    cmd = ["trtexec", f"--onnx={onnx}", "--fp16", f"--saveEngine={onnx.with_suffix('.engine')}",
           "--warmUp=500", "--iterations=200", "--avgRuns=50"]
    if info.dynamic_batch and info.hw:
        h, w = info.hw
        cmd += [f"--minShapes={info.name}:1x3x{h}x{w}", f"--optShapes={info.name}:2x3x{h}x{w}",
                f"--maxShapes={info.name}:{max_batch}x3x{h}x{w}", f"--shapes={info.name}:2x3x{h}x{w}"]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-batch", type=int, default=4)
    args = ap.parse_args()
    onnx_models = [(n, models.local_path(s)) for n, s in models.REGISTRY.items()
                   if s.url.endswith(".zip") and models.local_path(s).exists()]
    if not onnx_models:
        print("no ONNX models downloaded yet: run `motionplay models download` first")
        return 1
    if not args.dry_run and shutil.which("trtexec") is None:
        print("trtexec not found on PATH (it ships with TensorRT); use --dry-run to see the commands")
        return 1
    for name, path in onnx_models:
        cmd = command(path, name, args.max_batch)
        print(f"\n== {name}\n" + " ".join(cmd))
        if args.dry_run:
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in out.stdout.splitlines() if "Latency:" in ln or "Throughput:" in ln]
        print("\n".join(lines) or out.stdout[-2000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
