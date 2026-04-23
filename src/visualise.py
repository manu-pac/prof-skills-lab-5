"""
visualise.py

Loads the full distance arrays from data/pipeline/distances.npz and the
summary metrics from metrics/distances.json, then produces:

  Fig 1  --  KDE: intra vs inter distance distributions, one panel per precision
  Fig 2  --  Barplot: mean intra vs mean inter per precision
  Fig 3  --  Line plot: intra/inter ratio per precision
  Fig 4  --  Combo chart: bars for disk space + line for compute time (efficiency)
  Fig 5  --  KDE difference plots: deviation of each precision from float64 reference
  Text   --  Word rankings (ordering check + intra/inter ranking per precision)
             -> metrics/word_rankings.txt

Outputs:
  figures/fig1_kde_distributions.png
  figures/fig2_means_barplot.png
  figures/fig3_ratio.png
  figures/fig4_efficiency.png
  figures/fig5_kde_differences.png
  metrics/word_rankings.txt
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Paths
DISTANCES_PATH   = os.path.join("data", "pipeline", "distances.npz")
METRICS_PATH     = os.path.join("metrics", "distances.json")
FIGURES_DIR      = "figures"
RANKINGS_PATH    = os.path.join("metrics", "word_rankings.txt")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("metrics",   exist_ok=True)

PRECISIONS  = ["float64", "float32", "float16", "int8"]
PREC_LABELS = ["float64 (ref.)", "float32", "float16", "int8"]
INTRA_COLOR = "#e08214"
INTER_COLOR = "#542788"

# Load data
data    = np.load(DISTANCES_PATH, allow_pickle=True)
metrics = json.load(open(METRICS_PATH))


# Fig 1 - KDE: intra vs inter distributions, one panel per precision
def fig1_kde_distributions():
    # Shared x-range across all panels so they are directly comparable
    all_vals = np.concatenate([
        data[f"intra_{p}"].astype(np.float64) for p in PRECISIONS
    ] + [
        data[f"inter_{p}"].astype(np.float64) for p in PRECISIONS
    ])
    x_min, x_max = all_vals.min(), all_vals.max()
    x_grid = np.linspace(x_min - 0.01, x_max + 0.01, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    fig.suptitle(
        "Distribution of Cosine Distances — Intra- vs Inter-speaker\nby Precision Level",
        fontsize=13, fontweight="bold"
    )

    for ax, prec, label in zip(axes.flat, PRECISIONS, PREC_LABELS):
        intra = data[f"intra_{prec}"].astype(np.float64)
        inter = data[f"inter_{prec}"].astype(np.float64)

        kde_intra = gaussian_kde(intra)
        kde_inter = gaussian_kde(inter)

        ax.fill_between(x_grid, kde_intra(x_grid), alpha=0.35, color=INTRA_COLOR)
        ax.plot(x_grid, kde_intra(x_grid), color=INTRA_COLOR, linewidth=1.8,
                label=f"Intra  (μ={intra.mean():.4f})")

        ax.fill_between(x_grid, kde_inter(x_grid), alpha=0.35, color=INTER_COLOR)
        ax.plot(x_grid, kde_inter(x_grid), color=INTER_COLOR, linewidth=1.8,
                label=f"Inter  (μ={inter.mean():.4f})")

        ax.axvline(intra.mean(), color=INTRA_COLOR, linewidth=1.2, linestyle="--")
        ax.axvline(inter.mean(), color=INTER_COLOR,  linewidth=1.2, linestyle="--")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Cosine Distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, framealpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_kde_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# Fig 2 - Barplot: mean intra vs mean inter per precision
def fig2_means_barplot():
    intra_means = [metrics[p]["intra_speaker_mean"] for p in PRECISIONS]
    inter_means = [metrics[p]["inter_speaker_mean"] for p in PRECISIONS]

    n = len(PRECISIONS)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_intra = ax.bar(x - w / 2, intra_means, width=w,
                        color=INTRA_COLOR, alpha=0.85, label="Intra-speaker mean")
    bars_inter = ax.bar(x + w / 2, inter_means, width=w,
                        color=INTER_COLOR, alpha=0.85, label="Inter-speaker mean")

    # Annotate bars
    for bars, vals in [(bars_intra, intra_means), (bars_inter, inter_means)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # Zoom y-axis to make differences between precisions legible
    all_vals = intra_means + inter_means
    ax.set_ylim(min(all_vals) * 0.99, max(all_vals) * 1.015)

    ax.set_xticks(x)
    ax.set_xticklabels(PREC_LABELS, fontsize=11)
    ax.set_ylabel("Mean Cosine Distance")
    ax.set_title("Mean Intra- and Inter-speaker Cosine Distance by Precision Level",
                 fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_means_barplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# Fig 3 - Line plot: intra/inter ratio per precision
def fig3_ratio():
    ratios = [metrics[p]["ratio"] for p in PRECISIONS]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(PREC_LABELS, ratios, marker="o", linewidth=2,
            markersize=8, color="#1a9641")

    # Annotate points — placed below the marker to avoid clashing with the title
    for x, y in zip(PREC_LABELS, ratios):
        ax.text(x, y - 0.0004, f"{y:.5f}", ha="center", va="top", fontsize=9)

    ax.set_ylabel("Intra / Inter Ratio")
    ax.set_title("Intra/Inter Distance Ratio by Precision Level",
                 fontsize=12, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add top margin so the title has breathing room above the highest point
    y_min, y_max = min(ratios), max(ratios)
    margin = (y_max - y_min) * 0.25 if y_max != y_min else 0.001
    ax.set_ylim(y_min - margin * 2, y_max + margin * 3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# Fig 4 - Efficiency: bars for disk space, line for compute time
def fig4_efficiency():
    disk_mb      = [metrics[p]["disk_space_mb"]          for p in PRECISIONS]
    compute_secs = [metrics[p]["compute_time_seconds"]   for p in PRECISIONS]

    x = np.arange(len(PRECISIONS))
    w = 0.5

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bars: disk space (left axis)
    bars = ax1.bar(x, disk_mb, width=w, color="#4393c3", alpha=0.85,
                   label="Disk space (MB)")
    for bar, v in zip(bars, disk_mb):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="#4393c3")

    ax1.set_ylabel("Disk Space (MB)", color="#4393c3")
    ax1.tick_params(axis="y", labelcolor="#4393c3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(PREC_LABELS, fontsize=11)
    ax1.set_ylim(0, max(disk_mb) * 1.18)
    ax1.spines["top"].set_visible(False)

    # Line: compute time (right axis)
    ax2 = ax1.twinx()
    ax2.plot(x, compute_secs, marker="o", linewidth=2, markersize=8,
             color="#d6604d", label="Compute time (s)")
    for xi, y in zip(x, compute_secs):
        ax2.text(xi, y + max(compute_secs) * 0.02, f"{y:.2f}s",
                 ha="center", va="bottom", fontsize=9, color="#d6604d")

    ax2.set_ylabel("Compute Time (s)", color="#d6604d")
    ax2.tick_params(axis="y", labelcolor="#d6604d")
    ax2.set_ylim(0, max(compute_secs) * 1.25)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper right", framealpha=0.5)

    ax1.set_title("Computational Efficiency by Precision Level",
                  fontsize=12, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

# Fig 5 - KDE difference plots: deviation from float64 reference
def fig5_kde_differences():
    """
    For each reduced precision (float32, float16, int8), plots:
      KDE(intra_precXX)(x) - KDE(intra_float64)(x)
      KDE(inter_precXX)(x) - KDE(inter_float64)(x)
    on a shared x-grid. A flat y=0 line marks the float64 reference.
    The y-axis is scaled tightly to the actual deviation range to make
    even tiny differences visible.
    """
    REDUCED = ["float32", "float16", "int8"]
    REDUCED_LABELS = ["float32", "float16", "int8"]

    # Shared x-grid across all data
    all_vals = np.concatenate([
        data[f"intra_{p}"].astype(np.float64) for p in PRECISIONS
    ] + [
        data[f"inter_{p}"].astype(np.float64) for p in PRECISIONS
    ])
    x_min, x_max = all_vals.min(), all_vals.max()
    x_grid = np.linspace(x_min - 0.01, x_max + 0.01, 500)

    # Reference KDEs (float64)
    ref_intra = data["intra_float64"].astype(np.float64)
    ref_inter = data["inter_float64"].astype(np.float64)
    kde_ref_intra = gaussian_kde(ref_intra)(x_grid)
    kde_ref_inter = gaussian_kde(ref_inter)(x_grid)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(
        "KDE Deviation from float64 Reference — Intra- and Inter-speaker",
        fontsize=13, fontweight="bold"
    )

    for ax, prec, label in zip(axes, REDUCED, REDUCED_LABELS):
        intra = data[f"intra_{prec}"].astype(np.float64)
        inter = data[f"inter_{prec}"].astype(np.float64)

        diff_intra = gaussian_kde(intra)(x_grid) - kde_ref_intra
        diff_inter = gaussian_kde(inter)(x_grid) - kde_ref_inter

        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.7,
                   label="float64 reference (Δ=0)")

        ax.fill_between(x_grid, diff_intra, alpha=0.25, color=INTRA_COLOR)
        ax.plot(x_grid, diff_intra, color=INTRA_COLOR, linewidth=1.6,
                label=f"Intra  (max|Δ|={np.abs(diff_intra).max():.5f})")

        ax.fill_between(x_grid, diff_inter, alpha=0.25, color=INTER_COLOR)
        ax.plot(x_grid, diff_inter, color=INTER_COLOR, linewidth=1.6,
                label=f"Inter  (max|Δ|={np.abs(diff_inter).max():.5f})")

        # Tight y-axis scaled to actual deviation
        all_diffs = np.concatenate([diff_intra, diff_inter])
        y_abs_max = np.abs(all_diffs).max()
        ax.set_ylim(-y_abs_max * 1.3, y_abs_max * 1.3)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Cosine Distance")
        ax.set_ylabel("ΔKDE (precision − float64)")
        ax.legend(fontsize=8, framealpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_kde_differences.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)



def word_rankings():
    words = [str(w) for w in data["words"]]
    lines = []

    for prec, label in zip(PRECISIONS, PREC_LABELS):
        intra_means = data[f"word_intra_means_{prec}"]
        inter_means = data[f"word_inter_means_{prec}"]

        # Per-word ordering check: intra < inter ?
        ordering = {w: bool(intra_means[i] < inter_means[i]) for i, w in enumerate(words)}

        # Rankings: sorted by mean distance, closest first
        intra_ranked = sorted(enumerate(words), key=lambda t: intra_means[t[0]])
        inter_ranked  = sorted(enumerate(words), key=lambda t: inter_means[t[0]])

        sep  = "═" * 56
        sep2 = "─" * 56
        lines.append(sep)
        lines.append(f"  {label}")
        lines.append(sep)

        # Ordering check
        lines.append("  Ordering check (intra < inter per word):")
        all_ok = all(ordering.values())
        for w, ok in ordering.items():
            mark = "✓" if ok else "✗"
            lines.append(f"    {mark}  {w}")
        lines.append(f"  → Global ordering preserved: {all_ok}")
        lines.append(sep2)

        # Intra-speaker ranking
        lines.append("  Intra-speaker ranking (closest → furthest):")
        for rank, (i, w) in enumerate(intra_ranked, 1):
            lines.append(f"    {rank:>2}. {w:<20}  mean={intra_means[i]:.6f}")
        lines.append(sep2)

        # Inter-speaker ranking
        lines.append("  Inter-speaker ranking (closest → furthest):")
        for rank, (i, w) in enumerate(inter_ranked, 1):
            lines.append(f"    {rank:>2}. {w:<20}  mean={inter_means[i]:.6f}")
        lines.append("")

    output = "\n".join(lines)
    print("\n" + output)
    with open(RANKINGS_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"Saved {RANKINGS_PATH}")

# Main
if __name__ == "__main__":
    print("-- Fig 1: KDE distributions --")
    fig1_kde_distributions()

    print("-- Fig 2: means barplot --")
    fig2_means_barplot()

    print("-- Fig 3: ratio line plot --")
    fig3_ratio()

    print("-- Fig 4: efficiency plot --")
    fig4_efficiency()

    print("-- Fig 5: KDE difference plots --")
    fig5_kde_differences()

    print("-- Word rankings --")
    word_rankings()

    print("\nDone. All figures saved to figures/")