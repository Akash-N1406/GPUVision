"""
python/visualization/plot_benchmark_results.py

Phase 7: turns reports/benchmark_results.csv (from Phase 6's benchmark.cpp)
into the charts the SRS asks for (section 26/27):
  - Execution time vs image size, per algorithm
  - Speedup vs image size, per algorithm
  - Naive vs optimized CUDA comparison (convolution, matmul)
  - CPU vs GPU summary across all algorithms

Usage:
    python3 python/visualization/plot_benchmark_results.py \
        reports/benchmark_results.csv reports/charts/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just writing PNG files
import matplotlib.pyplot as plt
import pandas as pd


# Resolutions/sizes in pixel/element count order, so charts read left-to-right
# as "bigger problem" rather than alphabetical string order (which would put
# "1024x1024" before "128x128").
def size_key(label: str) -> int:
    w, h = label.split("x")
    return int(w) * int(h)


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df["size_sort_key"] = df["resolution"].apply(size_key)
    df = df.sort_values("size_sort_key")

    is_matmul = df["algorithm"].str.startswith("matmul")
    image_df = df[~is_matmul].copy()
    matmul_df = df[is_matmul].copy()
    return image_df, matmul_df


def plot_execution_time_vs_size(df: pd.DataFrame, out_dir: Path) -> None:
    """One subplot per algorithm: CPU vs GPU mean time across resolutions."""
    algorithms = sorted(df["algorithm"].unique())
    n = len(algorithms)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, algo in enumerate(algorithms):
        ax = axes[i]
        sub = df[df["algorithm"] == algo]
        labels = sub["resolution"]
        ax.plot(labels, sub["cpu_mean_ms"], marker="o", label="CPU")
        ax.plot(labels, sub["gpu_mean_ms"], marker="o", label="GPU")
        ax.set_yscale("log")
        ax.set_title(algo)
        ax.set_ylabel("Time (ms, log scale)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("CPU vs GPU Execution Time by Resolution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "execution_time_vs_resolution.png", dpi=150)
    plt.close(fig)


def plot_speedup_vs_size(df: pd.DataFrame, out_dir: Path) -> None:
    """One line per algorithm, all on the same axes: speedup across resolutions."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for algo in sorted(df["algorithm"].unique()):
        sub = df[df["algorithm"] == algo]
        ax.plot(sub["resolution"], sub["speedup"], marker="o", label=algo)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="CPU = GPU (1x)")
    ax.set_ylabel("Speedup (CPU time / GPU time)")
    ax.set_xlabel("Resolution")
    ax.set_title("GPU Speedup vs Resolution\n(above the dashed line = GPU faster)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "speedup_vs_resolution.png", dpi=150)
    plt.close(fig)


def plot_naive_vs_tiled(df: pd.DataFrame, out_dir: Path, base_name: str,
                         naive_label: str, tiled_label: str, x_col: str,
                         title: str, xlabel: str) -> None:
    """Bar chart comparing naive vs tiled GPU time at each size."""
    naive = df[df["algorithm"] == naive_label].sort_values("size_sort_key")
    tiled = df[df["algorithm"] == tiled_label].sort_values("size_sort_key")

    if naive.empty or tiled.empty:
        return

    labels = naive[x_col].tolist()
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], naive["gpu_mean_ms"], width, label="Naive CUDA")
    ax.bar([i + width / 2 for i in x], tiled["gpu_mean_ms"], width, label="Tiled CUDA")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("GPU time (ms, log scale)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / f"{base_name}.png", dpi=150)
    plt.close(fig)


def plot_matmul_speedup(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted(df["algorithm"].unique()):
        sub = df[df["algorithm"] == algo].sort_values("size_sort_key")
        ax.plot(sub["resolution"], sub["speedup"], marker="o", label=algo)

    ax.set_yscale("log")
    ax.set_ylabel("Speedup (CPU time / GPU time, log scale)")
    ax.set_xlabel("Matrix size")
    ax.set_title("Matrix Multiplication: GPU Speedup vs Matrix Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "matmul_speedup_vs_size.png", dpi=150)
    plt.close(fig)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <benchmark_results.csv> <output_dir>", file=sys.stderr)
        return 1

    csv_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    image_df, matmul_df = load_data(csv_path)

    if not image_df.empty:
        plot_execution_time_vs_size(image_df, out_dir)
        plot_speedup_vs_size(image_df, out_dir)
        plot_naive_vs_tiled(
            image_df, out_dir, "convolution_naive_vs_tiled",
            "convolution_naive", "convolution_tiled", "resolution",
            "Convolution: Naive vs Tiled CUDA GPU Time", "Resolution",
        )
        print(f"Wrote {out_dir / 'execution_time_vs_resolution.png'}")
        print(f"Wrote {out_dir / 'speedup_vs_resolution.png'}")
        print(f"Wrote {out_dir / 'convolution_naive_vs_tiled.png'}")

    if not matmul_df.empty:
        plot_naive_vs_tiled(
            matmul_df, out_dir, "matmul_naive_vs_tiled",
            "matmul_naive", "matmul_tiled", "resolution",
            "Matrix Multiply: Naive vs Tiled CUDA GPU Time", "Matrix size",
        )
        plot_matmul_speedup(matmul_df, out_dir)
        print(f"Wrote {out_dir / 'matmul_naive_vs_tiled.png'}")
        print(f"Wrote {out_dir / 'matmul_speedup_vs_size.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())