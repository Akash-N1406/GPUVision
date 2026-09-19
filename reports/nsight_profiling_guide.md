# Phase 9 — Nsight Profiling Guide

This is guidance, not code — Nsight Systems and Nsight Compute are interactive
tools you run and read yourself, so unlike everything else in this project I
can't build or verify this part for you. What I can do is point you at the
*specific* commands and metrics that would actually answer the open questions
this project already surfaced, rather than a generic "how to use Nsight"
tutorial.

## 1. Check the tools are installed

```bash
nsys --version   # Nsight Systems — timeline-level profiling
ncu --version    # Nsight Compute — per-kernel deep-dive metrics
```

Both ship with the CUDA Toolkit. If missing, they're a separate install on
some distros: https://developer.nvidia.com/nsight-systems and
https://developer.nvidia.com/nsight-compute. WSL2 support exists but has
some GUI/permission quirks — command-line profiling (below) works fine even
if the GUI doesn't.

## 2. Nsight Systems (`nsys`) — the timeline view

Use this first, on the benchmark harness itself, to see the actual shape of
a full run: are H2D/kernel/D2H phases overlapping or serialized, are there
unexpected gaps between kernel launches, is the CPU idle while waiting on
the GPU or doing real work.

```bash
nsys profile --stats=true -o reports/nsys_benchmark \
    ./build_benchmark data/input/bird.jpg reports/benchmark_results.csv
```

This produces `reports/nsys_benchmark.nsys-rep` (open in the Nsight Systems
GUI if you have display access) and prints a summary table to stdout
(`--stats=true`) even without a GUI — look at the "CUDA API Summary" and
"CUDA Kernel Summary" tables it prints.

**What to actually check**: the `matmul_cuda_naive`/`matmul_cuda_tiled`
functions still allocate and free device memory on every call (documented
limitation in `performance_analysis.md` §6) — this profile should make that
visible directly as repeated `cudaMalloc`/`cudaFree` entries with real time
cost, rather than something inferred from noisy timing.

## 3. Nsight Compute (`ncu`) — per-kernel deep dive

This is the one that can actually settle the convolution investigation with
hard numbers instead of inference. Run it against the naive and tiled
kernels specifically:

```bash
ncu --set full -o reports/ncu_convolution_naive \
    --kernel-name convolution_kernel_naive \
    ./build_convolution_detailed data/input/bird.jpg

ncu --set full -o reports/ncu_convolution_tiled \
    --kernel-name convolution_tiled_kernel \
    ./build_convolution_detailed data/input/bird.jpg
```

(`build_convolution_detailed` runs both naive and tiled at every resolution
in one invocation — `--kernel-name` filters to just the one you're
targeting per run.)

**Metrics that would confirm or update §4 of the performance report**
(the finding that tiled is still marginally slower than naive at 3x3):

| Metric (in the ncu report) | What it tells you |
|---|---|
| `l1tex__t_sector_hit_rate` / L2 cache hit rate | If naive's hit rate is already high (>80-90%), that's the smoking gun for the report's hypothesis — the GPU's automatic caching was already capturing most of the reuse shared memory manually provides, which is why tiling barely helped for a 3x3 kernel |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved occupancy) | Compare naive vs tiled — if tiled's shared memory usage is limiting how many blocks fit per SM, that's a second, independent explanation worth adding to the report |
| `smsp__average_warp_latency_per_inst_issued` or the stall-reason breakdown | Shows whether tiled kernels are stalling on `__syncthreads()` specifically — would directly confirm or rule out the synchronization-overhead half of the original diagnosis |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Global memory bandwidth utilization — low utilization on naive would mean it was never bandwidth-bound in the first place, which is exactly the condition under which shared memory tiling has the least to offer |

If you get access to a GUI (even via X11 forwarding or VcXsrv on Windows for
WSL2), `ncu-ui reports/ncu_convolution_naive.ncu-rep` gives a much easier
visual read of all of this than parsing text output.

### Troubleshooting: ERR_NVGPUCTRPERM

If `ncu` reports `ERR_NVGPUCTRPERM`, it means the GPU performance counters
are locked down and **no metrics were collected at all** — but the program
being profiled still runs to completion, which produces a subtle trap: the
timing numbers it prints are not clean baseline numbers either.

In testing, running `ncu --kernel-name convolution_kernel_naive ...` and
`ncu --kernel-name convolution_tiled_kernel ...` as two separate invocations
produced wildly different (and inconsistent between the two runs)
naive-vs-tiled ratios — because `ncu`'s attach/interception mechanism adds a
real, roughly-fixed per-invocation overhead to whichever kernel matches
`--kernel-name`, even though the actual counter read then fails on the
permission error. That overhead dominates a small kernel's true time (huge
distortion at 640x480) and is a smaller fraction of a larger kernel's time
(less distortion, but still present, at 4K).

**Do not use timing output from a `ncu` run that hit this error for any
comparison.** Fix the permission issue first (try `sudo ncu ...`; if that
doesn't resolve it, follow the link in the error message itself — it's
maintained by NVIDIA and will have current platform-specific, including
WSL2-specific, instructions) and confirm a clean metrics file is actually
produced before trusting anything from that run. The unprofiled
`demo_convolution_detailed` numbers (no `ncu` involved) remain the correct
source for naive-vs-tiled timing comparisons; `ncu` is for the *metrics*
(cache hit rate, occupancy, stalls) that timing alone can't show, not a
second timing measurement.

## 4. The follow-up experiment this project already flagged

`performance_analysis.md` section 6 explicitly leaves open whether a larger
kernel radius (5x5, 7x7) would flip the tiled-vs-naive result. That's
directly testable with the existing code — `convolution_cuda_naive`/`_tiled`
already take `kernel_size` as a parameter:

```bash
# Extend cuda/demo_convolution_detailed.cu's kernel to accept a
# --kernel-size argument (currently hardcoded to 3), generate a 5x5 or 7x7
# kernel (e.g. reuse make_gaussian_kernel's pattern, or a flat box kernel),
# and re-run the same naive-vs-tiled comparison.
```

If you want this built out as actual code rather than left as a manual
extension, say so and I'll write it — it's a small, contained change to
`demo_convolution_detailed.cu` plus a quick kernel-size parameter, not a new
architecture.

## 5. What to do with the results

Whatever `ncu` shows, it's worth folding back into `performance_analysis.md`
section 4 as a follow-up — turning the current "here's my hypothesis for why
tiled is still slower" into "here's the hypothesis, and here's the Nsight
Compute data that confirms/refutes it" is a meaningfully stronger claim for
an interview, and it's genuinely low additional effort from where this
project already stands.