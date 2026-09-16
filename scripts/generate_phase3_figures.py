#!/usr/bin/env python3
"""
Phase 3 Publication Figures Generator
=====================================
Generates high-resolution publication-quality figures for Phase 3:
- Figure 12: Hindi Attention Shift & Head Specialization (Pretrained vs Finetuned)
- Figure 13: Nepali Attention Shift & Head Specialization (Pretrained vs Finetuned)
- Figure 14: Reasoning Benchmark Performance (Pretrained vs Finetuned Model H & Model L)
- Figure 15: Supervised Reasoning Finetuning Loss & Convergence Dynamics
- Figure 16: Attention Entropy & Mean Distance Quantitative Diagnostics
"""

import os
import sys
import json
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use clean modern styling
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def copy_attention_heatmaps():
    """Copies the empirically generated attention comparison heatmaps into report/figures/."""
    hi_src = PROJECT_ROOT / "hindi" / "results-finetune-hindi" / "hindi_attention_comparison.png"
    ne_src = PROJECT_ROOT / "nepali" / "results-finetune-nepali" / "nepali_attention_comparison.png"
    
    hi_dst = FIGURES_DIR / "fig12_hindi_reasoning_attention.png"
    ne_dst = FIGURES_DIR / "fig13_nepali_reasoning_attention.png"
    
    if hi_src.exists():
        shutil.copy(hi_src, hi_dst)
        print(f"✓ Copied: {hi_dst}")
    if ne_src.exists():
        shutil.copy(ne_src, ne_dst)
        print(f"✓ Copied: {ne_dst}")


def plot_reasoning_accuracy_comparison():
    """Plots a multi-group bar chart of reasoning accuracy across tasks."""
    tasks = ["Direct Comparison", "Transitive Chaining", "Extremes Ordering", "Overall Exact Match"]
    
    # Empirical Metrics
    pretrain_acc = [0.0, 0.0, 0.0, 0.0]
    hindi_ft_acc = [24.03, 43.35, 13.25, 26.86]
    nepali_ft_acc = [30.90, 19.74, 0.85, 17.14]
    
    x = np.arange(len(tasks))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    rects1 = ax.bar(x - width, pretrain_acc, width, label="Pretrained Zero-Shot (Base)", color="#888888", edgecolor="#555555", alpha=0.85)
    rects2 = ax.bar(x, hindi_ft_acc, width, label="Hindi Finetuned (Model H — High Resource)", color="#2b5c8f", edgecolor="#1b3b5f")
    rects3 = ax.bar(x + width, nepali_ft_acc, width, label="Nepali Finetuned (Model L — Low Resource)", color="#d95f02", edgecolor="#a63e00")
    
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 3 Synthetic Reasoning Benchmark Evaluation (Held-Out Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 52)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10, loc="upper right")
    
    # Add value annotations on top of bars
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1b3b5f")
    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#a63e00")
        
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_path = FIGURES_DIR / "fig14_reasoning_accuracy_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_finetuning_loss_curves():
    """Plots the empirical loss decay trajectories during 5 epochs of supervised finetuning."""
    # Empirical Step-by-Step Training Logs from Phase 3 SFT GPU Runs
    hi_steps = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875]
    hi_losses = [0.6386, 0.1651, 0.1306, 0.0914, 0.0772, 0.0760, 0.0885, 0.0379, 0.0339, 0.0321, 0.0412, 0.0260, 0.0351, 0.0299, 0.0274, 0.0095, 0.0109, 0.0194, 0.0110, 0.0104, 0.0109, 0.0210, 0.0029, 0.0020, 0.0136, 0.0062, 0.0105, 0.0068, 0.0039, 0.0038, 0.0030, 0.0055, 0.0053, 0.0008, 0.0058]
    hi_val_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]
    hi_val_losses = [0.3140, 0.3058, 0.3093, 0.2718, 0.2874, 0.3033, 0.2638, 0.2269, 0.2383, 0.2707, 0.2287, 0.2343, 0.2371, 0.2439, 0.2283, 0.2355, 0.2337]

    ne_steps = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875]
    ne_losses = [0.6330, 0.1562, 0.0962, 0.0599, 0.0819, 0.0710, 0.0637, 0.0579, 0.0404, 0.0414, 0.0534, 0.0373, 0.0435, 0.0381, 0.0385, 0.0346, 0.0291, 0.0283, 0.0341, 0.0262, 0.0283, 0.0253, 0.0233, 0.0278, 0.0168, 0.0114, 0.0213, 0.0202, 0.0082, 0.0101, 0.0134, 0.0084, 0.0109, 0.0062, 0.0077]
    ne_val_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]
    ne_val_losses = [0.3458, 0.3628, 0.4432, 0.4375, 0.5171, 0.4730, 0.5324, 0.5011, 0.5809, 0.5811, 0.5895, 0.6534, 0.6069, 0.5644, 0.5590, 0.5587, 0.5634]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)
    
    # 1. Full training loss curve
    ax1.plot(hi_steps, hi_losses, label="Hindi Model H (Train Loss)", color="#1b4d3e", lw=2.2, marker="o", markersize=4)
    ax1.plot(hi_val_steps, hi_val_losses, label="Hindi Model H (Val Loss)", color="#52b788", lw=1.8, linestyle="--", marker="^", markersize=4)
    ax1.plot(ne_steps, ne_losses, label="Nepali Model L (Train Loss)", color="#c1121f", lw=2.2, marker="s", markersize=4)
    ax1.plot(ne_val_steps, ne_val_losses, label="Nepali Model L (Val Loss)", color="#e07a5f", lw=1.8, linestyle="--", marker="v", markersize=4)
        
    ax1.set_xlabel("Optimization Step (5 Epochs)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11, fontweight="bold")
    ax1.set_title("Supervised Finetuning Loss Decay (Linear)", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(0, 900)
    ax1.set_ylim(0, 0.70)
    
    # 2. Log-scale zoom-in on convergence
    ax2.plot(hi_steps, hi_losses, label="Hindi Model H (Train)", color="#1b4d3e", lw=2.2, marker="o", markersize=3.5)
    ax2.plot(ne_steps, ne_losses, label="Nepali Model L (Train)", color="#c1121f", lw=2.2, marker="s", markersize=3.5)
        
    ax2.set_yscale("log")
    ax2.set_xlabel("Optimization Step (5 Epochs)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Log Cross-Entropy Loss (nats)", fontsize=11, fontweight="bold")
    ax2.set_title("Convergence Asymptote (Log Scale)", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5, which="both")
    ax2.set_xlim(0, 900)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig15_finetuning_loss_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_attention_shift_summary():
    """Plots Shannon attention entropy and mean attention distance shifts post-finetuning."""
    models = ["Hindi (Model H)", "Nepali (Model L)"]
    
    # Empirical Metrics
    pre_entropy = [1.7773, 1.9272]
    post_entropy = [1.5290, 1.8035]
    
    pre_dist = [8.1312, 9.1402]
    post_dist = [7.5282, 8.3475]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    
    # Entropy
    rects1 = ax1.bar(x - width/2, pre_entropy, width, label="Pretrained Base", color="#888888", edgecolor="#555555")
    rects2 = ax1.bar(x + width/2, post_entropy, width, label="Finetuned SFT", color="#2ca02c", edgecolor="#1e6b1e")
    ax1.set_ylabel("Shannon Entropy (nats)", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Attention Entropy Shift", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 2.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#1e6b1e")
        
    # Distance
    rects3 = ax2.bar(x - width/2, pre_dist, width, label="Pretrained Base", color="#888888", edgecolor="#555555")
    rects4 = ax2.bar(x + width/2, post_dist, width, label="Finetuned SFT", color="#7570b3", edgecolor="#4d478a")
    ax2.set_ylabel("Mean Attention Distance (Tokens)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Attention Span Shift", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#4d478a")
        
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig16_attention_shift_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    print("Generating Phase 3 publication figures...")
    copy_attention_heatmaps()
    plot_reasoning_accuracy_comparison()
    plot_finetuning_loss_curves()
    plot_attention_shift_summary()
    print("All Phase 3 figures generated successfully!")


if __name__ == "__main__":
    main()
