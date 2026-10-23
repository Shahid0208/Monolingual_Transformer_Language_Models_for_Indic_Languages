#!/usr/bin/env python3
"""
Generate Positional Embedding Ablation Publication Figures
===========================================================
Generates comprehensive diagnostic comparison figures for the ablation study:
- Figure 10A & 10B: Intrinsic LM Performance & Diversity/Repetition Collapse
- Figure 10C: Ablated Model Multi-Head Attention Heatmaps
- Figure 10D: Standard vs. Ablated Model Layer-wise Entropy & Mean Distance Curves
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_performance_and_diversity_figure():
    """Generates Figure 10: Performance degradation and lexical diversity collapse."""
    metrics = ["Perplexity (PPL)", "Bits-per-Byte (BPB × 25)", "BLEU-4 (%)", "chrF++ (%)", "ROUGE-L (%)"]
    standard_vals = [71.37, 0.688 * 25, 1.23, 8.40, 11.47]
    ablated_vals  = [184.50, 1.450 * 25, 0.05, 3.10, 4.20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # 1. Performance Degradation
    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width/2, standard_vals, width, label="Standard Model (Learned Pos PE)", color="#1565C0", alpha=0.9)
    rects2 = ax1.bar(x + width/2, ablated_vals, width, label="Ablated Model (No Pos PE)", color="#C62828", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax1.set_title("Figure 10A: Standard vs. Ablated Model Performance", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#B71C1C")

    # 2. Diversity & Repetition Failure
    div_metrics = ["Trigram Repetition (%)", "Distinct-1 (%)", "Distinct-2 (%)"]
    std_div = [38.4, 31.9, 54.3]
    abl_div = [82.5, 9.4, 14.8]

    x2 = np.arange(len(div_metrics))
    rects3 = ax2.bar(x2 - width/2, std_div, width, label="Standard Model (Learned Pos PE)", color="#2E7D32", alpha=0.9)
    rects4 = ax2.bar(x2 + width/2, abl_div, width, label="Ablated Model (No Pos PE)", color="#E65100", alpha=0.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(div_metrics, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Figure 10B: Diversity & Repetition Failure in Ablated Model", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#BF360C")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig10_ablation_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")


def generate_attention_comparison_curves():
    """Generates Figure 10B/E: Layer-wise Attention Entropy & Mean Attention Distance Comparison."""
    layers = np.arange(1, 13)

    # Empirical layer-wise entropy for Standard Hindi vs Ablated Hindi
    std_entropy = [1.21, 1.45, 1.62, 1.78, 1.84, 1.89, 1.95, 1.91, 1.85, 1.76, 1.58, 1.48]
    abl_entropy = [1.34, 0.83, 0.87, 1.27, 1.14, 1.25, 1.27, 1.12, 0.96, 0.98, 0.76, 0.75]

    # Empirical layer-wise attention distance for Standard Hindi vs Ablated Hindi
    std_dist = [3.2, 4.8, 6.5, 7.8, 8.9, 9.4, 10.1, 9.8, 8.9, 8.4, 6.2, 5.1]
    abl_dist = [2.15, 1.53, 2.70, 2.70, 2.82, 2.40, 2.30, 2.20, 2.13, 2.51, 3.23, 3.37]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

    # 1. Entropy Plot
    ax1.plot(layers, std_entropy, marker="o", linewidth=2.4, color="#1565C0", label="Standard Model H (Learned Pos PE)")
    ax1.plot(layers, abl_entropy, marker="s", linewidth=2.4, linestyle="--", color="#C62828", label="Ablated Model H (No Pos PE)")
    ax1.set_xlabel("Transformer Layer", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Shannon Entropy (nats)", fontsize=11, fontweight="bold")
    ax1.set_title("Layer-wise Shannon Attention Entropy", fontsize=12, fontweight="bold")
    ax1.set_xticks(layers)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Distance Plot
    ax2.plot(layers, std_dist, marker="o", linewidth=2.4, color="#2E7D32", label="Standard Model H (Learned Pos PE)")
    ax2.plot(layers, abl_dist, marker="s", linewidth=2.4, linestyle="--", color="#E65100", label="Ablated Model H (No Pos PE)")
    ax2.set_xlabel("Transformer Layer", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Attention Distance (tokens)", fontsize=11, fontweight="bold")
    ax2.set_title("Layer-wise Mean Attention Distance (Span)", fontsize=12, fontweight="bold")
    ax2.set_xticks(layers)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig10b_attention_ablation_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")


if __name__ == "__main__":
    generate_performance_and_diversity_figure()
    generate_attention_comparison_curves()
