#!/usr/bin/env python3
"""
Pretrained vs Finetuned Attention Comparison Diagnostics (Phase 3)
==================================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 3: Post-Finetune Attention Analysis

Functions:
- Extracts attention maps from Pretrained vs Finetuned models on reasoning prompts
- Plots publication-grade side-by-side attention heatmaps (Early vs Late layers, Heads 0-5)
- Computes Layer-wise and Head-wise Shannon Attention Entropy & Mean Attention Distance
- Saves diagnostic plots to `report/figures/`
"""

import os
import sys
import math
import argparse
import glob
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import sentencepiece as spm

# Suppress harmless font and Devanagari layout warnings
warnings.filterwarnings("ignore", message=".*Devanagari.*")
warnings.filterwarnings("ignore", message=".*missing from font.*")
warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")


def setup_devanagari_font():
    """Configures matplotlib with fonts supporting both Devanagari and Latin (English) characters."""
    font_patterns = [
        "/usr/share/fonts/**/FreeSans*.ttf",
        "/usr/share/fonts/**/FreeSerif*.ttf",
        "/usr/share/fonts/**/NotoSansDevanagari*.ttf",
        "/usr/share/fonts/**/NotoSansDevanagari*.otf",
        "/usr/share/fonts/**/NotoSerifDevanagari*.ttf",
        "/usr/share/fonts/**/Lohit-Devanagari.ttf",
        "/usr/share/fonts/**/lohit_hi.ttf",
        "/usr/share/fonts/**/gargi.ttf",
        "/usr/share/fonts/**/*Devanagari*.ttf",
        "/usr/share/fonts/**/*Devanagari*.otf",
    ]
    for pattern in font_patterns:
        for font_path in glob.glob(pattern, recursive=True):
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass

    local_font_path = Path("/tmp/NotoSansDevanagari-Regular.ttf")
    if not local_font_path.exists():
        urls = [
            "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
            "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari-Regular.ttf"
        ]
        for url in urls:
            try:
                urllib.request.urlretrieve(url, str(local_font_path))
                break
            except Exception:
                pass

    if local_font_path.exists():
        try:
            fm.fontManager.addfont(str(local_font_path))
        except Exception:
            pass

    matplotlib.rcParams["font.sans-serif"] = [
        "FreeSans",
        "Noto Sans Devanagari",
        "Noto Serif Devanagari",
        "Lohit Devanagari",
        "Lohit Hindi",
        "Gargi",
        "Samyak Devanagari",
        "DejaVu Sans",
        "Liberation Sans",
        "Arial",
        "sans-serif",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


# Initialize font configuration
setup_devanagari_font()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device
) -> TransformerLM:

    config = TransformerConfig.load_json(Path(config_path))
    model = TransformerLM(config).to(device)

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


@torch.no_grad()
def extract_attention_weights(
    model: TransformerLM,
    tokens: List[int],
    device: torch.device
) -> List[np.ndarray]:
    """Returns list of (n_heads, T, T) numpy arrays for each layer."""

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output = model(
        input_ids,
        return_attention_weights=True
    )

    attn_list = output["attention_weights"]
    return [
        layer_attn[0].cpu().numpy()
        for layer_attn in attn_list
    ]


def compute_entropy_and_distance(
    layer_attns: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Shannon Entropy and Mean Attention Distance
    per layer and per head.

    Entropy = - sum(p * log(p))
    Distance = sum(p_ij * |i - j|)
    """
    n_layers = len(layer_attns)
    n_heads = layer_attns[0].shape[0]
    T = layer_attns[0].shape[1]

    entropies = np.zeros((n_layers, n_heads))
    distances = np.zeros((n_layers, n_heads))

    pos_matrix = np.abs(
        np.arange(T)[:, None] - np.arange(T)[None, :]
    )

    for l_idx, layer_attn in enumerate(layer_attns):
        for h_idx in range(n_heads):
            attn = layer_attn[h_idx]

            # Shannon Entropy along key dimension
            entropy_t = []
            for t in range(T):
                row = attn[t, :t + 1]
                row = row[row > 1e-12]
                ent = -np.sum(row * np.log(row)) if len(row) > 0 else 0.0
                entropy_t.append(ent)

            entropies[l_idx, h_idx] = np.mean(entropy_t)

            # Mean Attention Distance
            dist_t = []
            for t in range(T):
                dist = np.sum(attn[t, :t + 1] * pos_matrix[t, :t + 1])
                dist_t.append(dist)

            distances[l_idx, h_idx] = np.mean(dist_t)

    return entropies, distances


def plot_attention_comparison_heatmap(
    pretrained_attn: List[np.ndarray],
    finetuned_attn: List[np.ndarray],
    tokens_str: List[str],
    language: str,
    output_path: str,
    early_layer: int = 0,
    late_layer: int = 11,
    head_idx: int = 0
):
    """Plots side-by-side comparison heatmaps for Pretrained vs Finetuned."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    # Clean SentencePiece subword character (\u2581) to remove leading rectangles
    display_tokens = [
        t.replace("\u2581", "").replace(" ", "").strip()
        for t in tokens_str
    ]
    display_tokens = [t if t else "<sp>" for t in display_tokens]
    T = len(display_tokens)

    # ------------------------------------------------------------
    # Early Layer — Pretrained
    # ------------------------------------------------------------
    sns.heatmap(
        pretrained_attn[early_layer][head_idx][:T, :T],
        ax=axes[0, 0],
        cmap="Blues",
        cbar=True,
        vmin=0,
        vmax=1,
        xticklabels=display_tokens,
        yticklabels=display_tokens
    )
    axes[0, 0].set_title(
        f"Pretrained Base (Layer {early_layer}, Head {head_idx})",
        fontsize=12,
        fontweight="bold"
    )
    axes[0, 0].set_ylabel("Query Tokens")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # ------------------------------------------------------------
    # Early Layer — Finetuned
    # ------------------------------------------------------------
    sns.heatmap(
        finetuned_attn[early_layer][head_idx][:T, :T],
        ax=axes[0, 1],
        cmap="Purples",
        cbar=True,
        vmin=0,
        vmax=1,
        xticklabels=display_tokens,
        yticklabels=display_tokens
    )
    axes[0, 1].set_title(
        f"Reasoning Finetuned (Layer {early_layer}, Head {head_idx})",
        fontsize=12,
        fontweight="bold"
    )
    axes[0, 1].tick_params(axis="x", rotation=45)

    # ------------------------------------------------------------
    # Late Layer — Pretrained
    # ------------------------------------------------------------
    sns.heatmap(
        pretrained_attn[late_layer][head_idx][:T, :T],
        ax=axes[1, 0],
        cmap="Blues",
        cbar=True,
        vmin=0,
        vmax=1,
        xticklabels=display_tokens,
        yticklabels=display_tokens
    )
    axes[1, 0].set_title(
        f"Pretrained Base (Layer {late_layer}, Head {head_idx})",
        fontsize=12,
        fontweight="bold"
    )
    axes[1, 0].set_ylabel("Query Tokens")
    axes[1, 0].set_xlabel("Key Tokens")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # ------------------------------------------------------------
    # Late Layer — Finetuned
    # ------------------------------------------------------------
    sns.heatmap(
        finetuned_attn[late_layer][head_idx][:T, :T],
        ax=axes[1, 1],
        cmap="Purples",
        cbar=True,
        vmin=0,
        vmax=1,
        xticklabels=display_tokens,
        yticklabels=display_tokens
    )
    axes[1, 1].set_title(
        f"Reasoning Finetuned (Layer {late_layer}, Head {head_idx})",
        fontsize=12,
        fontweight="bold"
    )
    axes[1, 1].set_xlabel("Key Tokens")
    axes[1, 1].tick_params(axis="x", rotation=45)

    # ------------------------------------------------------------
    # Overall figure
    # ------------------------------------------------------------
    plt.suptitle(
        f"{language} Attention Shift: Pretrained Base vs. Reasoning Finetuned",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"✓ Saved attention comparison heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Pretrained vs Finetuned Attention."
    )
    parser.add_argument("--language", type=str, default="Hindi")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--finetuned_checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_model", type=str, required=True)
    parser.add_argument(
        "--output_figure",
        type=str,
        default="report/figures/fig13_attention_shift_comparison.png"
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_model)

    # ------------------------------------------------------------
    # Representative reasoning prompt
    # ------------------------------------------------------------
    if args.language.lower() == "hindi":
        sample_prompt = (
            "प्रश्न: यदि राहुल, अमित से लंबा है "
            "और अमित, रोहन से लंबा है, "
            "तो सबसे लंबा कौन है? "
            "तर्क: राहुल > अमित > रोहन। "
            "उत्तर: राहुल"
        )
    else:
        sample_prompt = (
            "प्रश्न: यदि सुवास, बिपिन भन्दा अग्लो छ "
            "र बिपिन, प्रकाश भन्दा अग्लो छ भने, "
            "सबैभन्दा अग्लो को हो? "
            "तर्क: सुवास > बिपिन > प्रकाश। "
            "उत्तर: सुवास"
        )

    token_ids = sp.encode(sample_prompt, out_type=int)
    token_pieces = sp.encode(sample_prompt, out_type=str)

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------
    print(f"Loading Pretrained model from: {args.pretrained_checkpoint}")
    pre_model = load_model(args.config_file, args.pretrained_checkpoint, device)

    print(f"Loading Finetuned model from: {args.finetuned_checkpoint}")
    ft_model = load_model(args.config_file, args.finetuned_checkpoint, device)

    # ------------------------------------------------------------
    # Extract attentions
    # ------------------------------------------------------------
    print("\nExtracting pretrained attention...")
    pre_attns = extract_attention_weights(pre_model, token_ids, device)

    print("Extracting fine-tuned attention...")
    ft_attns = extract_attention_weights(ft_model, token_ids, device)

    # ------------------------------------------------------------
    # Compute diagnostics
    # ------------------------------------------------------------
    pre_entropy, pre_distance = compute_entropy_and_distance(pre_attns)
    ft_entropy, ft_distance = compute_entropy_and_distance(ft_attns)

    print("\n" + "=" * 80)
    print("ATTENTION ENTROPY")
    print("=" * 80)
    print(f"Pretrained mean entropy: {pre_entropy.mean():.6f}")
    print(f"Fine-tuned mean entropy: {ft_entropy.mean():.6f}")

    print("\n" + "=" * 80)
    print("ATTENTION DISTANCE")
    print("=" * 80)
    print(f"Pretrained mean distance: {pre_distance.mean():.6f}")
    print(f"Fine-tuned mean distance: {ft_distance.mean():.6f}")

    # ------------------------------------------------------------
    # Plot comparison heatmap
    # ------------------------------------------------------------
    plot_attention_comparison_heatmap(
        pre_attns,
        ft_attns,
        token_pieces,
        args.language,
        args.output_figure
    )


if __name__ == "__main__":
    main()
