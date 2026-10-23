"""
Bonus Ablation Experiment: Transformer LM Without Positional Embeddings
======================================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid0208 (Monsoon 2026)
Language: Hindi (Model H Ablation)

Purpose:
Pretrains and evaluates a Decoder-only Transformer model with positional embeddings removed:
X_emb = TokenEmbeddings(X)  (no learned/sinusoidal position injected)

Investigates:
1. Impact on Causal Language Modeling (Perplexity and Bits-Per-Byte degradation)
2. Impact on Autoregressive Generation (BLEU-4, chrF++, ROUGE-L, word-order collapse)
3. Attention head behavior when position information is absent
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Any

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM
from hindi.train.dataset import create_dataloader
from hindi.train.train import configure_optimizers, get_cosine_schedule_with_warmup, save_checkpoint
from hindi.eval.evaluate import evaluate_intrinsic_lm, evaluate_generation
from hindi.eval.attention_analysis import analyze_attention


def run_ablation(args):
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("BONUS ABLATION: HINDI TRANSFORMER LM WITHOUT POSITIONAL EMBEDDINGS")
    print("=" * 75)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Config with pos_embedding_type = 'none'
    config = TransformerConfig(
        vocab_size=10000,
        d_model=384,
        n_layers=12,
        n_heads=6,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1,
        pos_embedding_type="none",  # ABLATION TRIGGER
        language="Hindi_Ablation_NoPos",
        model_name="Model_H_Ablation_NoPos_25M",
    )
    config.save_json(output_dir / "model_config.json")

    model = TransformerLM(config).to(device)
    print(f"Ablated Model Parameters: {model.count_parameters()['trainable_parameters']:,} (Positional Embeddings Removed)")

    # DataLoaders
    train_loader = create_dataloader(
        parquet_path=Path(args.train_parquet),
        tokenizer_model_path=Path(args.tokenizer_model),
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shuffle=True,
    )
    val_loader = create_dataloader(
        parquet_path=Path(args.val_parquet),
        tokenizer_model_path=Path(args.tokenizer_model),
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shuffle=False,
    )

    optimizer = configure_optimizers(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.use_amp))

    # Pretraining Loop
    model.train()
    data_iter = iter(train_loader)
    step = 0
    running_loss = 0.0
    start_time = time.time()

    print(f"Training ablated model for {args.max_steps} steps...")
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.use_amp)):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        if device.type == "cuda" and args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        step += 1
        running_loss += loss.item()

        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            print(f"Step [{step:5d}/{args.max_steps:5d}] | Ablation Loss: {avg_loss:.4f} | PPL: {math.exp(min(20.0, avg_loss)):.2f}")
            running_loss = 0.0

    # Save final checkpoint
    ckpt_path = checkpoints_dir / "ablation_no_pos_final.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler if device.type == "cuda" else None, step, loss.item(), None, config)

    # Run Full Evaluation Suite
    print("\n" + "=" * 70)
    print("EVALUATING ABLATED MODEL (NO POSITIONAL EMBEDDINGS)")
    print("=" * 70)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)

    print("\n[1/3] Intrinsic Language Modeling (Loss, PPL, BPB)...")
    intrinsic = evaluate_intrinsic_lm(model, Path(args.test_parquet), sp, max_eval_docs=100, device=device)
    print(f"  Ablation CE Loss: {intrinsic['cross_entropy_loss']:.4f}")
    print(f"  Ablation PPL:     {intrinsic['perplexity']:.2f}")
    print(f"  Ablation BPB:     {intrinsic['bits_per_byte']:.4f}")

    print("\n[2/3] Generation Quality (BLEU, chrF++, ROUGE-L, Diversity)...")
    gen_results = evaluate_generation(model, Path(args.test_parquet), sp, device=device)
    for temp, m in gen_results["metrics_by_temperature"].items():
        print(f"  [{temp}] BLEU-4: {m['bleu_4']:.2f} | chrF++: {m['chrf_plus_plus']:.2f} | ROUGE-L: {m['rouge_l']:.2f} | Dist-1: {m['distinct_1']:.3f}")

    print("\n[3/3] Attention Analysis...")
    attn_summary = analyze_attention(
        model,
        sp,
        "हिंदी भारत की प्रमुख भाषाओं में से एक है।",
        output_dir / "attention",
        device=device,
    )

    # Save complete ablation report
    full_results = {
        "experiment": "Bonus Ablation: No Positional Embeddings",
        "language": "Hindi",
        "intrinsic_metrics": intrinsic,
        "generation_metrics": gen_results["metrics_by_temperature"],
        "sample_generations": gen_results["sample_generations"],
        "attention_summary": attn_summary,
        "timestamp": time.time(),
    }
    with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nAblation evaluation saved to: {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run No-Positional-Embedding Ablation Study")
    parser.add_argument("--train_parquet", type=str, default=str(PROJECT_ROOT / "hindi/data/splits/train.parquet"))
    parser.add_argument("--val_parquet", type=str, default=str(PROJECT_ROOT / "hindi/data/splits/val.parquet"))
    parser.add_argument("--test_parquet", type=str, default=str(PROJECT_ROOT / "hindi/data/splits/test.parquet"))
    parser.add_argument("--tokenizer_model", type=str, default=str(PROJECT_ROOT / "hindi/tokenizer/unigram_10k/tokenizer.model"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "hindi/eval/ablation_no_pos"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    run_ablation(args)
