"""
Phase 1 — Environment Verification
GPU-Accelerated Computer Vision Engine

Run this on your WSL2 box (the one with the NVIDIA GPU), same pattern as
verify_setup.py in the fraud-detection project. It checks every dependency
FR-01 onward assumes is present, and prints a clear PASS/FAIL summary so we
know Phase 1 is actually done before writing a single kernel.

Usage:
    python3 python/verify_setup.py
    (run from the project root, or anywhere — paths below are absolute checks,
     not relative file reads)
"""

import shutil
import subprocess
import sys
from pathlib import Path

CHECK = "\u2713"
CROSS = "\u2717"

results = []


def run(cmd):
    """Run a command, return (ok, stdout+stderr)."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        combined = (out.stdout or "") + (out.stderr or "")
        return out.returncode == 0, combined.strip()
    except Exception as e:
        return False, str(e)


def report(name, ok, detail=""):
    mark = CHECK if ok else CROSS
    print(f"[{mark}] {name}")
    if detail:
        for line in detail.splitlines()[:3]:
            print(f"      {line}")
    results.append((name, ok))


def main():
    print("=" * 60)
    print("GPU-Accelerated Computer Vision Engine — Phase 1 Verification")
    print("=" * 60)

    # 1. NVIDIA driver / GPU visible
    ok, out = run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    report("NVIDIA GPU detected (nvidia-smi)", ok, out)

    # 2. CUDA toolkit / nvcc
    ok, out = run("nvcc --version")
    report("CUDA compiler (nvcc) available", ok, out)

    # 3. CUDA version consistency check (driver vs toolkit is informational only)
    if ok:
        for line in out.splitlines():
            if "release" in line.lower():
                print(f"      -> Toolkit: {line.strip()}")

    # 4. g++ / C++ compiler
    ok, out = run("g++ --version")
    report("C++ compiler (g++) available", ok, out)

    # 5. CMake
    ok, out = run("cmake --version")
    report("CMake available", ok, out)

    # 6. OpenCV (C++ dev headers, via pkg-config)
    ok, out = run("pkg-config --modversion opencv4")
    if not ok:
        ok, out = run("pkg-config --modversion opencv")
    report("OpenCV C++ dev package (pkg-config)", ok, out)

    # 7. Python OpenCV binding
    try:
        import cv2  # noqa: F401
        report("Python OpenCV (cv2) importable", True, cv2.__version__)
    except Exception as e:
        report("Python OpenCV (cv2) importable", False, str(e))

    # 8. NumPy
    try:
        import numpy  # noqa: F401
        report("NumPy importable", True, numpy.__version__)
    except Exception as e:
        report("NumPy importable", False, str(e))

    # 9. Docker (optional but in scope per SRS)
    ok, out = run("docker --version")
    report("Docker available", ok, out)

    # 10. Directory structure sanity check
    root = Path(__file__).resolve().parent.parent
    expected = [
        "cuda/kernels", "cuda/optimized", "cuda/utils",
        "cpp/cpu", "cpp/benchmark",
        "python/preprocessing", "python/benchmarking", "python/visualization",
        "backend/django_app", "tests", "data/input", "data/output", "reports",
    ]
    missing = [d for d in expected if not (root / d).is_dir()]
    report("Project directory structure intact", not missing, "\n".join(missing))

    # Summary
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")

    critical = ["NVIDIA GPU detected (nvidia-smi)", "CUDA compiler (nvcc) available"]
    critical_fail = [name for name, ok in results if name in critical and not ok]
    if critical_fail:
        print("\nCRITICAL: the following must pass before Phase 3 (CUDA kernels) can start:")
        for name in critical_fail:
            print(f"  - {name}")
        print("\nInstall the NVIDIA driver + CUDA Toolkit for WSL2, then re-run this script.")
        print("https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
        sys.exit(1)

    if passed < total:
        print("\nSome non-critical checks failed — install the missing pieces before Phase 2/8.")
        sys.exit(0)

    print("\nAll checks passed. Ready to start Phase 2 (CPU baseline implementation).")
    sys.exit(0)


if __name__ == "__main__":
    main()
