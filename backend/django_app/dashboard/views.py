"""
backend/django_app/dashboard/views.py

Calls the process_image CLI (cpp/cli/process_image.cpp) via subprocess
rather than any Python/CUDA binding — see that file's own header comment
for why. This view's job is entirely: save the upload, build the right
command line, parse the JSON result, and render it. No image-processing
logic lives here.
"""

import json
import subprocess
import uuid
from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from .forms import ImageProcessForm

# Algorithms that take --kernel-size/--sigma vs --preset vs neither — used
# to decide which extra CLI flags to pass, so we don't hand the CLI
# irrelevant flags for algorithms that ignore them.
BLUR_ALGORITHMS = {"gaussian_blur"}
CONVOLUTION_ALGORITHMS = {"convolution_naive", "convolution_tiled"}

SUBPROCESS_TIMEOUT_SECONDS = 60


def index(request):
    if request.method == "POST":
        form = ImageProcessForm(request.POST, request.FILES)
        if form.is_valid():
            context = _process(request, form.cleaned_data)
            context["form"] = form
            return render(request, "dashboard/index.html", context)
    else:
        form = ImageProcessForm()

    return render(request, "dashboard/index.html", {"form": form})


def _process(request, data):
    """Saves the upload, runs the CLI for each requested device, returns
    the template context. Never raises — every failure mode (missing
    binary, CLI crash, bad JSON) is caught and surfaced as a per-result
    error message instead, since a stack trace isn't a useful result for
    someone testing image processing through a browser."""

    upload_id = uuid.uuid4().hex[:12]
    uploads_dir = Path(settings.MEDIA_ROOT) / "uploads"
    outputs_dir = Path(settings.MEDIA_ROOT) / "outputs"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    image_file = data["image"]
    suffix = Path(image_file.name).suffix or ".jpg"
    input_path = uploads_dir / f"{upload_id}{suffix}"
    with open(input_path, "wb") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    algorithm = data["algorithm"]
    execution = data["execution"]
    devices = ["cpu", "cuda"] if execution == "both" else [execution]

    results = []
    for device in devices:
        output_path = outputs_dir / f"{upload_id}_{device}.png"
        results.append(_run_cli(algorithm, device, input_path, output_path, data))

    speedup = None
    cpu_result = next((r for r in results if r["device"] == "cpu" and r["success"]), None)
    cuda_result = next((r for r in results if r["device"] == "cuda" and r["success"]), None)
    if cpu_result and cuda_result and cuda_result["time_ms"] > 0:
        speedup = cpu_result["time_ms"] / cuda_result["time_ms"]

    return {
        "original_url": settings.MEDIA_URL + f"uploads/{input_path.name}",
        "algorithm": algorithm,
        "results": results,
        "speedup": speedup,
    }


def _run_cli(algorithm, device, input_path, output_path, data):
    """Runs process_image for one device, returns a result dict for the
    template: {device, success, error, time_ms, output_url}."""

    binary = Path(settings.PROCESS_IMAGE_BINARY)
    result = {"device": device, "success": False, "error": None,
              "time_ms": None, "output_url": None}

    if not binary.exists():
        result["error"] = (
            f"process_image binary not found at {binary}. Build it first — "
            "see cpp/cli/process_image.cpp's header comment for the nvcc command."
        )
        return result

    cmd = [
        str(binary),
        "--input", str(input_path),
        "--output", str(output_path),
        "--algorithm", algorithm,
        "--device", device,
    ]
    if algorithm in BLUR_ALGORITHMS:
        if data.get("kernel_size"):
            cmd += ["--kernel-size", str(data["kernel_size"])]
        if data.get("sigma"):
            cmd += ["--sigma", str(data["sigma"])]
    elif algorithm in CONVOLUTION_ALGORITHMS:
        if data.get("preset"):
            cmd += ["--preset", data["preset"]]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"Timed out after {SUBPROCESS_TIMEOUT_SECONDS}s"
        return result
    except OSError as e:
        result["error"] = f"Failed to launch process_image: {e}"
        return result

    # The CLI always prints one line of JSON on stdout, success or failure
    # (see its own header comment) — but guard against a crash before any
    # JSON was printed at all (e.g. a segfault), which would leave stdout
    # empty rather than containing valid JSON.
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        stderr_tail = proc.stderr.strip()[-500:] if proc.stderr else "(no stderr)"
        result["error"] = (
            f"process_image produced no valid JSON output (exit code {proc.returncode}). "
            f"stderr: {stderr_tail}"
        )
        return result

    if not payload.get("success"):
        result["error"] = payload.get("error", "Unknown error from process_image")
        return result

    result["success"] = True
    result["time_ms"] = payload["time_ms"]
    result["output_url"] = settings.MEDIA_URL + f"outputs/{output_path.name}"
    return result
