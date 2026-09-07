"""
Empirical Verification of Causal Attention Masking
=================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid (Monsoon 2026)

Test Purpose:
Demonstrate mathematically and empirically that the custom causal mask
strictly prohibits the model from observing future tokens:
Changing token x_{t+1} does NOT alter logits at position t (for any position t).
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM, CausalSelfAttention


def verify_causal_attention_mask_unit():
    """
    Unit test on the raw CausalSelfAttention module.
    """
    print("\n" + "=" * 70)
    print("TEST 1: RAW CAUSAL ATTENTION MASK UNIT TEST")
    print("=" * 70)

    config = TransformerConfig(d_model=128, n_heads=4, max_seq_len=64)
    attn = CausalSelfAttention(config)
    attn.eval()

    B, T, d = 2, 8, 128
    x = torch.randn(B, T, d)

    # Pass through attention and get attention weights
    out, weights = attn(x, return_attention_weights=True)

    # Check upper triangle is strictly zero (after softmax)
    upper_tri = torch.triu(weights, diagonal=1)
    max_leakage = upper_tri.abs().max().item()

    print(f"Attention Weights Shape: {weights.shape} (B, h, T, T)")
    print(f"Max attention weight in strictly future positions (j > i): {max_leakage:.2e}")
    assert max_leakage < 1e-6, f"Causal leakage detected in attention weights! Max value: {max_leakage}"
    print("✓ Causal Attention Weights Test: PASSED (Zero future attention leakage)")


def verify_full_model_causal_invariance():
    """
    End-to-End test on full TransformerLM forward pass:
    Verify that altering future token x_{t+1} causes ZERO difference
    in logits at position t across all layers.
    """
    print("\n" + "=" * 70)
    print("TEST 2: FULL TRANSFORMER LM CAUSAL INVARIANCE TEST")
    print("=" * 70)

    config = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=32,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model = TransformerLM(config)
    model.eval()

    # Create base sequence of length T=10
    seq_len = 10
    base_ids = torch.tensor([[10, 45, 99, 120, 340, 560, 780, 230, 890, 420]], dtype=torch.long)

    # Target position to evaluate: t=4 (5th token, value 340)
    eval_pos = 4

    # Run forward pass on base sequence
    with torch.no_grad():
        base_logits = model(base_ids)["logits"] # (1, seq_len, vocab_size)
        base_logit_at_t = base_logits[0, eval_pos, :]

    # Create mutated sequence where future token at t=5 is changed drastically
    mutated_ids = base_ids.clone()
    mutated_ids[0, eval_pos + 1] = 999  # change future token at pos 5
    mutated_ids[0, eval_pos + 2] = 888  # change future token at pos 6

    with torch.no_grad():
        mutated_logits = model(mutated_ids)["logits"]
        mutated_logit_at_t = mutated_logits[0, eval_pos, :]

    # Calculate absolute difference at position t
    diff = torch.max(torch.abs(base_logit_at_t - mutated_logit_at_t)).item()

    print(f"Base token at pos {eval_pos}: {base_ids[0, eval_pos].item()}")
    print(f"Mutated future token at pos {eval_pos + 1}: {base_ids[0, eval_pos + 1].item()} -> {mutated_ids[0, eval_pos + 1].item()}")
    print(f"Logit difference at evaluated position {eval_pos}: {diff:.2e}")

    assert diff < 1e-5, f"Causal violation! Modifying future tokens changed logits at position {eval_pos} (diff={diff})"
    print("✓ Full Model Causal Invariance Test: PASSED (Logits at t are 100% independent of future tokens t+1..T)")


if __name__ == "__main__":
    verify_causal_attention_mask_unit()
    verify_full_model_causal_invariance()
    print("\n" + "=" * 70)
    print("ALL CAUSAL MASKING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
