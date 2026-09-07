"""
Attention Analysis Toolkit for Nepali Transformer LM (Model L)
=============================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid (Monsoon 2026)
Language: Nepali (Model L)

Implements:
1. Multi-head attention extraction across all 12 layers and 6 heads
2. Attention Heatmaps for Early Layers (Layer 1, 2) and Late Layers (Layer 11, 12)
3. Attention Entropy computation per head and per layer
4. Mean Attention Distance computation (local vs long-range specialization)
5. Head categorization (local/positional vs content/semantic)
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sentencepiece as spm
import torch

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans Devanagari",
    "Lohit Devanagari",
    "Gargi",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "sans-serif",
]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nepali.model.config import TransformerConfig
from nepali.model.transformer import TransformerLM


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    L, h, T, _ = attn_weights.shape
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    mean_entropy = torch.mean(entropy[:, :, 1:], dim=-1)
    return mean_entropy.detach().cpu().numpy()


def compute_mean_attention_distance(attn_weights: torch.Tensor) -> np.ndarray:
    L, h, T, _ = attn_weights.shape
    device = attn_weights.device
    
    q_pos = torch.arange(T, device=device).unsqueeze(1)
    k_pos = torch.arange(T, device=device).unsqueeze(0)
    dist_matrix = (q_pos - k_pos).clamp(min=0).float()
    
    weighted_dist = attn_weights * dist_matrix.view(1, 1, T, T)
    sum_dist = torch.sum(weighted_dist, dim=-1)
    
    mean_dist = torch.mean(sum_dist[:, :, 1:], dim=-1)
    return mean_dist.detach().cpu().numpy()


def plot_attention_heatmaps(
    attn_weights: torch.Tensor,
    tokens: List[str],
    save_path: Path,
    layers_to_plot: List[int] = [0, 1, 10, 11],
    heads_to_plot: List[int] = [0, 1, 2],
    title_prefix: str = "Nepali Model L",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_layers = len(layers_to_plot)
    num_heads = len(heads_to_plot)

    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(5 * num_heads, 4.5 * num_layers), dpi=300
    )

    clean_tokens = [t.replace("\u2581", "").replace(" ", "").replace("▁", "") for t in tokens]
    clean_tokens = [t if t else "<s>" for t in clean_tokens]
    T = len(clean_tokens)

    for r_idx, layer_idx in enumerate(layers_to_plot):
        for c_idx, head_idx in enumerate(heads_to_plot):
            ax = axes[r_idx, c_idx] if num_layers > 1 else axes[c_idx]
            attn_matrix = attn_weights[layer_idx, head_idx, :T, :T].detach().cpu().numpy()

            sns.heatmap(
                attn_matrix,
                ax=ax,
                cmap="viridis",
                vmin=0.0,
                vmax=np.max(attn_matrix),
                cbar=(c_idx == num_heads - 1),
                square=True,
                xticklabels=clean_tokens if r_idx == num_layers - 1 else False,
                yticklabels=clean_tokens if c_idx == 0 else False,
            )

            layer_label = f"Layer {layer_idx + 1}"
            head_label = f"Head {head_idx + 1}"
            ax.set_title(f"{layer_label} — {head_label}", fontsize=11, fontweight="bold")
            if r_idx == num_layers - 1:
                ax.set_xlabel("Key Token", fontsize=10)
            if c_idx == 0:
                ax.set_ylabel("Query Token", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)

    plt.suptitle(f"{title_prefix}: Multi-Head Causal Attention Heatmaps", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmaps to: {save_path}")


def plot_entropy_and_distance(
    entropy_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    save_path: Path,
    title_prefix: str = "Nepali Model L",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_layers, n_heads = entropy_matrix.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    for h in range(n_heads):
        ax1.plot(range(1, n_layers + 1), entropy_matrix[:, h], marker="o", label=f"Head {h+1}", linewidth=1.8)
    ax1.plot(range(1, n_layers + 1), entropy_matrix.mean(axis=1), color="black", linestyle="--", linewidth=2.5, label="Layer Mean")
    ax1.set_title(f"{title_prefix}: Attention Entropy per Layer", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Transformer Layer", fontsize=11)
    ax1.set_ylabel("Shannon Entropy (nats)", fontsize=11)
    ax1.set_xticks(range(1, n_layers + 1))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    for h in range(n_heads):
        ax2.plot(range(1, n_layers + 1), distance_matrix[:, h], marker="s", label=f"Head {h+1}", linewidth=1.8)
    ax2.plot(range(1, n_layers + 1), distance_matrix.mean(axis=1), color="black", linestyle="--", linewidth=2.5, label="Layer Mean")
    ax2.set_title(f"{title_prefix}: Mean Attention Distance per Layer", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Transformer Layer", fontsize=11)
    ax2.set_ylabel("Attention Distance (tokens)", fontsize=11)
    ax2.set_xticks(range(1, n_layers + 1))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved entropy and distance plots to: {save_path}")


def analyze_attention(
    model: TransformerLM,
    sp_tokenizer: spm.SentencePieceProcessor,
    test_sentence: str,
    output_dir: Path,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    token_ids = [sp_tokenizer.bos_id()] + sp_tokenizer.encode_as_ids(test_sentence) + [sp_tokenizer.eos_id()]
    tokens = [sp_tokenizer.id_to_piece(t) for t in token_ids]
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_tensor, return_attention_weights=True)

    attn_weights = torch.cat(outputs["attention_weights"], dim=0)

    entropy_matrix = compute_attention_entropy(attn_weights)
    distance_matrix = compute_mean_attention_distance(attn_weights)

    heatmap_path = output_dir / "nepali_attention_heatmaps.png"
    plot_attention_heatmaps(
        attn_weights,
        tokens,
        heatmap_path,
        layers_to_plot=[0, 1, 10, 11],
        heads_to_plot=[0, 1, 2],
        title_prefix="Nepali Model L",
    )

    entropy_path = output_dir / "nepali_attention_entropy_distance.png"
    plot_entropy_and_distance(
        entropy_matrix,
        distance_matrix,
        entropy_path,
        title_prefix="Nepali Model L",
    )

    summary = {
        "language": "Nepali",
        "test_sentence": test_sentence,
        "token_count": len(tokens),
        "tokens": tokens,
        "mean_layer_entropy": entropy_matrix.mean(axis=1).tolist(),
        "mean_layer_distance": distance_matrix.mean(axis=1).tolist(),
        "per_head_entropy": entropy_matrix.tolist(),
        "per_head_distance": distance_matrix.tolist(),
        "heatmap_figure": str(heatmap_path),
        "entropy_figure": str(entropy_path),
    }

    with open(output_dir / "attention_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Nepali Transformer Attention")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config_file", type=str, default=str(PROJECT_ROOT / "nepali/configs/model_config.json"))
    parser.add_argument("--tokenizer_model", type=str, default=str(PROJECT_ROOT / "nepali/tokenizer/unigram_10k/tokenizer.model"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "nepali/eval/attention"))
    parser.add_argument("--sentence", type=str, default="नेपाली भाषा नेपालको राष्ट्रिय भाषा हो।")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "config" in ckpt:
        config = TransformerConfig.from_dict(ckpt["config"])
    else:
        config = TransformerConfig.load_json(Path(args.config_file))

    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Analyzing Attention for Nepali Sentence: '{args.sentence}'")
    summary = analyze_attention(model, sp, args.sentence, Path(args.output_dir), device=device)
    print("Attention analysis complete.")


if __name__ == "__main__":
    main()
