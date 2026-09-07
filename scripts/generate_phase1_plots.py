#!/usr/bin/env python3
"""
Generate the 10 recommended publication-quality figures for Phase 1 Technical Report.
Covers both Hindi and Nepali:
1. Raw vs Cleaned records
2. Raw vs Cleaned corpus size (MB)
3. Source distribution (Manual vs Downloaded)
4. Train/Validation/Test split distribution (Rows & Percentages)
5. Tokens by split (Train/Val/Test Tokens)
6. Cleaning/filtering & deduplication statistics
7. Hindi vs Nepali overall corpus comparison
8. Tokenizer vocabulary comparison (6K, 8K, 10K, 15K Model & Vocab Sizes)
9. Tokenizer fertility & compression (Characters-per-token & Tokens-per-char)
10. UNK rate & coverage comparison across vocabularies
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Styling configuration
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Figure 1: Raw vs Cleaned Records (Rows)
# ------------------------------------------------------------
def plot_raw_vs_cleaned_records():
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    
    categories = ["Hindi (Model H)", "Nepali (Model L)"]
    raw_rows = [8716094 / 1e6, 11927379 / 1e6]
    clean_rows = [8513165 / 1e6, 11053584 / 1e6]
    
    x = np.arange(len(categories))
    width = 0.35
    
    b1 = ax.bar(x - width/2, raw_rows, width, label="Raw Records", color="#7293CB", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, clean_rows, width, label="Cleaned Records", color="#E1974C", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.15, f"{h:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.15, f"{h:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    ax.set_title("Figure 1: Raw vs. Cleaned Document Records", pad=15, fontweight="bold")
    ax.set_xlabel("Language Corpus")
    ax.set_ylabel("Records / Documents (Millions)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 14.5)
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig1_raw_vs_clean_records.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 2. Figure 2: Raw vs Cleaned Corpus Size (MB)
# ------------------------------------------------------------
def plot_raw_vs_cleaned_size():
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    
    categories = ["Hindi (Model H)", "Nepali (Model L)"]
    raw_mb = [2075.36, 2721.18]
    clean_mb = [2062.93, 2585.83]
    
    x = np.arange(len(categories))
    width = 0.35
    
    b1 = ax.bar(x - width/2, raw_mb, width, label="Raw Parquet Size (MB)", color="#84BA5B", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, clean_mb, width, label="Cleaned Parquet Size (MB)", color="#D35E60", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 30, f"{h:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 30, f"{h:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    ax.set_title("Figure 2: Raw vs. Cleaned Corpus Storage Size (MB)", pad=15, fontweight="bold")
    ax.set_xlabel("Language Corpus")
    ax.set_ylabel("Parquet File Size (Megabytes)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 3200)
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig2_raw_vs_clean_size.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 3. Figure 3: Source Distribution (Manual vs Downloaded)
# ------------------------------------------------------------
def plot_source_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    
    # Hindi
    labels = ["Manual Sourcing (23.09%)", "Public / Downloaded (76.91%)"]
    hindi_tokens = [126.35, 420.76]
    colors = ["#2b5c8f", "#5da5da"]
    
    ax1.pie(hindi_tokens, labels=labels, autopct="%1.2f%%", startangle=140, colors=colors, explode=(0.06, 0), shadow=True, textprops=dict(fontweight="bold"))
    ax1.set_title("Hindi Source Share (Model H)\nManual: 126.35M / Total: 547.12M Tokens\n[Requirement ≥ 20%: PASS]", fontweight="bold")
    
    # Nepali
    nepali_labels = ["Manual Sourcing (23.16%)", "Public / Downloaded (76.84%)"]
    nepali_tokens = [119.34, 395.90]
    colors_nepali = ["#807dba", "#bcbddc"]
    
    ax2.pie(nepali_tokens, labels=nepali_labels, autopct="%1.2f%%", startangle=140, colors=colors_nepali, explode=(0.06, 0), shadow=True, textprops=dict(fontweight="bold"))
    ax2.set_title("Nepali Source Share (Model L)\nManual: 119.34M / Total: 515.24M Tokens\n[Requirement ≥ 20%: PASS]", fontweight="bold")
    
    fig.suptitle("Figure 3: Token Share Source Distribution (Manual Scraping vs. Public Downloaded)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig3_source_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 4. Figure 4: Train / Validation / Test Split Distribution (Rows)
# ------------------------------------------------------------
def plot_partition_distribution():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    
    splits = ["Train Partition (92%)", "Validation Partition (4%)", "Test Partition (4%)"]
    x = np.arange(len(splits))
    width = 0.35
    
    hindi_split_rows = [7831689 / 1e6, 341528 / 1e6, 339948 / 1e6]
    nepali_split_rows = [10167950 / 1e6, 443871 / 1e6, 441763 / 1e6]
    
    b1 = ax.bar(x - width/2, hindi_split_rows, width, label="Hindi (Model H)", color="#d95f02", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, nepali_split_rows, width, label="Nepali (Model L)", color="#7570b3", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.15, f"{h:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.15, f"{h:.2f}M", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
        
    ax.set_title("Figure 4: Deterministic 92/4/4 Partition Records (Train / Val / Test)", pad=15, fontweight="bold")
    ax.set_xlabel("Partition Split")
    ax.set_ylabel("Records (Millions)")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold")
    ax.set_ylim(0, 12.0)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig4_train_val_test_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 5. Figure 5: Tokens by Split
# ------------------------------------------------------------
def plot_tokens_by_split():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    
    splits = ["Train Split", "Validation Split", "Test Split"]
    x = np.arange(len(splits))
    width = 0.35
    
    hindi_tokens = [499.72, 21.58, 21.74]
    nepali_tokens = [469.13, 20.39, 20.44]
    
    b1 = ax.bar(x - width/2, hindi_tokens, width, label="Hindi Tokens (Total: 547.12M)", color="#1b9e77", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, nepali_tokens, width, label="Nepali Tokens (Total: 515.24M)", color="#e7298a", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 6, f"{h:.1f}M", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 6, f"{h:.1f}M", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
        
    ax.set_title("Figure 5: Estimated Subword Tokens by Split Partition", pad=15, fontweight="bold")
    ax.set_xlabel("Partition Split")
    ax.set_ylabel("Subword Tokens (Millions)")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold")
    ax.set_ylim(0, 560)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig5_tokens_by_split.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 6. Figure 6: Cleaning & Deduplication Statistics
# ------------------------------------------------------------
def plot_cleaning_filtering_stats():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    
    filter_stages = ["Exact SHA-256 Duplicates", "Near-Duplicate Anchors", "Noise / Short / Purity Filters", "Total Filtered Records"]
    x = np.arange(len(filter_stages))
    width = 0.35
    
    hindi_filtered = [56322, 20932, 125675, 202929]
    nepali_filtered = [480693, 122252, 270850, 873795]
    
    b1 = ax.bar(x - width/2, [v/1e3 for v in hindi_filtered], width, label="Hindi (Total Removed: 202.9K)", color="#386cb0", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, [v/1e3 for v in nepali_filtered], width, label="Nepali (Total Removed: 873.8K)", color="#f0027f", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 10, f"{h:.1f}K", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 10, f"{h:.1f}K", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
        
    ax.set_title("Figure 6: Data Cleaning, Deduplication & Quality Filtering Breakdown", pad=15, fontweight="bold")
    ax.set_xlabel("Deduplication & Quality Filtering Stage")
    ax.set_ylabel("Records Removed (Thousands)")
    ax.set_xticks(x)
    ax.set_xticklabels(filter_stages, fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, 1000)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig6_cleaning_filtering_stats.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 7. Figure 7: Hindi vs Nepali Overall Corpus Comparison
# ------------------------------------------------------------
def plot_hindi_vs_nepali_corpus():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    
    metrics = ["Clean Documents (M)", "Clean Words (M)", "Total Tokens (M)", "Storage Size (x100 MB)"]
    x = np.arange(len(metrics))
    width = 0.35
    
    hindi_metrics = [8.51, 405.27, 547.12, 20.63]
    nepali_metrics = [11.05, 381.66, 515.24, 25.86]
    
    b1 = ax.bar(x - width/2, hindi_metrics, width, label="Hindi (Model H — High Resource)", color="#ff7f00", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, nepali_metrics, width, label="Nepali (Model L — Low Resource)", color="#984ea3", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 8, f"{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 8, f"{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
        
    ax.set_title("Figure 7: Comprehensive Hindi vs. Nepali Clean Corpus Comparison", pad=15, fontweight="bold")
    ax.set_xlabel("Corpus Quantitative Metric")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 620)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig7_hindi_vs_nepali_corpus.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 8. Figure 8: Tokenizer Vocabulary Comparison (Model & Vocab Size)
# ------------------------------------------------------------
def plot_tokenizer_vocab_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    
    vocabs = ["6K", "8K", "10K", "15K"]
    x = np.arange(len(vocabs))
    width = 0.35
    
    hindi_model_kb = [155.3, 214.5, 273.9, 429.4]
    nepali_model_kb = [159.9, 220.1, 280.7, 438.5]
    
    b1 = ax.bar(x - width/2, hindi_model_kb, width, label="Hindi Tokenizer Model Size (KB)", color="#4daf4a", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, nepali_model_kb, width, label="Nepali Tokenizer Model Size (KB)", color="#377eb8", edgecolor="black", linewidth=0.8)
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 6, f"{h:.1f}K", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 6, f"{h:.1f}K", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
        
    ax.set_title("Figure 8: Tokenizer Model Binary Size Comparison across Vocabularies", pad=15, fontweight="bold")
    ax.set_xlabel("Target Vocabulary Size")
    ax.set_ylabel("SentencePiece Checkpoint Size (KB)")
    ax.set_xticks(x)
    ax.set_xticklabels(vocabs, fontweight="bold")
    ax.set_ylim(0, 500)
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig8_tokenizer_vocab_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 9. Figure 9: Tokenizer Fertility & Characters-per-Token Comparison
# ------------------------------------------------------------
def plot_fertility_and_compression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    
    vocabs = [6, 8, 10, 15]
    hindi_cpt = [3.7143, 4.0857, 4.3333, 4.7667]
    nepali_cpt = [4.6545, 5.1200, 5.1200, 5.3333]
    
    hindi_tpc = [1/v for v in hindi_cpt]  # Tokens per character
    nepali_tpc = [1/v for v in nepali_cpt]
    
    # Subplot 1: Characters Per Token (Compression Ratio)
    ax1.plot(vocabs, hindi_cpt, marker="o", linewidth=2.5, markersize=8, color="#e41a1c", label="Hindi (Model H)")
    ax1.plot(vocabs, nepali_cpt, marker="s", linewidth=2.5, markersize=8, color="#377eb8", label="Nepali (Model L)")
    for i, txt in enumerate(hindi_cpt):
        ax1.annotate(f"{txt:.2f}", (vocabs[i], hindi_cpt[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#e41a1c")
    for i, txt in enumerate(nepali_cpt):
        ax1.annotate(f"{txt:.2f}", (vocabs[i], nepali_cpt[i]), textcoords="offset points", xytext=(0, -15), ha="center", fontweight="bold", color="#377eb8")
    ax1.set_title("Characters per Token (Compression Ratio)\n(Higher = More Efficient Encoding)", fontweight="bold")
    ax1.set_xlabel("Vocabulary Size (Thousands)")
    ax1.set_ylabel("Characters per Subword Token")
    ax1.set_xticks(vocabs)
    ax1.set_xticklabels(["6K", "8K", "10K", "15K"], fontweight="bold")
    ax1.set_ylim(3.2, 5.8)
    ax1.legend(loc="lower right", frameon=True)
    
    # Subplot 2: Tokens per Character (Inverse Fertility)
    ax2.plot(vocabs, hindi_tpc, marker="^", linewidth=2.5, markersize=8, color="#e41a1c", linestyle="--", label="Hindi Tokens / Char")
    ax2.plot(vocabs, nepali_tpc, marker="D", linewidth=2.5, markersize=8, color="#377eb8", linestyle="--", label="Nepali Tokens / Char")
    for i, txt in enumerate(hindi_tpc):
        ax2.annotate(f"{txt:.3f}", (vocabs[i], hindi_tpc[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#e41a1c")
    for i, txt in enumerate(nepali_tpc):
        ax2.annotate(f"{txt:.3f}", (vocabs[i], nepali_tpc[i]), textcoords="offset points", xytext=(0, -15), ha="center", fontweight="bold", color="#377eb8")
    ax2.set_title("Tokens per Character (Fertility)\n(Lower = Shorter Sequence Lengths)", fontweight="bold")
    ax2.set_xlabel("Vocabulary Size (Thousands)")
    ax2.set_ylabel("Tokens per Character")
    ax2.set_xticks(vocabs)
    ax2.set_xticklabels(["6K", "8K", "10K", "15K"], fontweight="bold")
    ax2.set_ylim(0.15, 0.30)
    ax2.legend(loc="upper right", frameon=True)
    
    fig.suptitle("Figure 9: Subword Tokenizer Compression & Fertility Dynamics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = FIGURES_DIR / "fig9_fertility_and_compression.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating recommended figures for Phase 1 report...")
    plot_raw_vs_cleaned_records()
    plot_raw_vs_cleaned_size()
    plot_source_distribution()
    plot_partition_distribution()
    plot_tokens_by_split()
    plot_cleaning_filtering_stats()
    plot_hindi_vs_nepali_corpus()
    plot_tokenizer_vocab_comparison()
    plot_fertility_and_compression()
    print("Figures generated successfully!")
