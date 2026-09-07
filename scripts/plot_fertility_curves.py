#!/usr/bin/env python3
"""
Generate exact Vocabulary Size vs Fertility (tokens/word) plots for Hindi and Nepali.
Matches the clean, publication-ready design with:
- Title: "<Language>: Vocabulary Size vs Fertility"
- X-axis: "Vocabulary Size"
- Y-axis: "Fertility (tokens/word)"
- Grid, markers, annotations
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGS = PROJECT_ROOT / "report" / "figures"
HINDI_FIGS = PROJECT_ROOT / "hindi" / "figures"
NEPALI_FIGS = PROJECT_ROOT / "nepali" / "figures"

REPORT_FIGS.mkdir(parents=True, exist_ok=True)
HINDI_FIGS.mkdir(parents=True, exist_ok=True)
NEPALI_FIGS.mkdir(parents=True, exist_ok=True)

# 1. Hindi Fertility Curve
def plot_hindi_fertility():
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    
    vocabs = [6, 8, 10, 15]
    fertility = [1.6775, 1.5984, 1.5382, 1.4445]
    
    ax.plot(vocabs, fertility, marker="o", linewidth=2.8, markersize=8, color="#2b5c8f")
    
    for i, txt in enumerate(fertility):
        ax.annotate(f"{txt:.3f}", (vocabs[i], fertility[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color="#2b5c8f")
        
    ax.set_title("Hindi: Vocabulary Size vs Fertility", pad=15, fontweight="bold")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Fertility (tokens/word)")
    ax.set_xticks(vocabs)
    ax.set_xticklabels(["6K", "8K", "10K", "15K"], fontweight="bold")
    ax.set_ylim(1.35, 1.78)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "hindi_vocab_vs_fertility.png")
    plt.savefig(HINDI_FIGS / "hindi_vocab_vs_fertility.png")
    plt.close()
    print("Saved Hindi Vocabulary Size vs Fertility plot.")

# 2. Nepali Fertility Curve
def plot_nepali_fertility():
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    
    vocabs = [6, 8, 10, 15]
    fertility = [1.5707, 1.4786, 1.4117, 1.3167]
    
    ax.plot(vocabs, fertility, marker="o", linewidth=2.8, markersize=8, color="#1b9e77")
    
    for i, txt in enumerate(fertility):
        ax.annotate(f"{txt:.3f}", (vocabs[i], fertility[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color="#1b9e77")
        
    ax.set_title("Nepali: Vocabulary Size vs Fertility", pad=15, fontweight="bold")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Fertility (tokens/word)")
    ax.set_xticks(vocabs)
    ax.set_xticklabels(["6K", "8K", "10K", "15K"], fontweight="bold")
    ax.set_ylim(1.22, 1.68)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "nepali_vocab_vs_fertility.png")
    plt.savefig(NEPALI_FIGS / "nepali_vocab_vs_fertility.png")
    plt.close()
    print("Saved Nepali Vocabulary Size vs Fertility plot.")

# 3. Combined Comparative Fertility Curve
def plot_combined_fertility():
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)
    
    vocabs = [6, 8, 10, 15]
    hindi_fertility = [1.6775, 1.5984, 1.5382, 1.4445]
    nepali_fertility = [1.5707, 1.4786, 1.4117, 1.3167]
    
    ax.plot(vocabs, hindi_fertility, marker="o", linewidth=2.8, markersize=8, color="#2b5c8f", label="Hindi (Model H)")
    ax.plot(vocabs, nepali_fertility, marker="s", linewidth=2.8, markersize=8, color="#1b9e77", label="Nepali (Model L)")
    
    for i in range(len(vocabs)):
        ax.annotate(f"{hindi_fertility[i]:.3f}", (vocabs[i], hindi_fertility[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color="#2b5c8f")
        ax.annotate(f"{nepali_fertility[i]:.3f}", (vocabs[i], nepali_fertility[i]), textcoords="offset points", xytext=(0, -16), ha="center", fontweight="bold", color="#1b9e77")
        
    ax.set_title("Vocabulary Size vs Fertility Comparison (Hindi vs. Nepali)", pad=15, fontweight="bold")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Fertility (tokens/word)")
    ax.set_xticks(vocabs)
    ax.set_xticklabels(["6K", "8K", "10K", "15K"], fontweight="bold")
    ax.set_ylim(1.22, 1.78)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(REPORT_FIGS / "vocab_vs_fertility_combined.png")
    plt.close()
    print("Saved Combined Vocabulary Size vs Fertility plot.")

if __name__ == "__main__":
    plot_hindi_fertility()
    plot_nepali_fertility()
    plot_combined_fertility()
