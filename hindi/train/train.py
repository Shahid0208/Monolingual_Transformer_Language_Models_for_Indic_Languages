"""
Pretraining Engine with Checkpoint Recovery for Hindi Transformer LM
===================================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid (Monsoon 2026)
Language: Hindi (Model H)

Features:
- AdamW optimizer with decoupled weight decay (excluding biases & LayerNorms)
- Cosine Annealing learning rate schedule with linear warmup
- Automatic mixed precision (AMP fp16/bf16) with GradScaler on CUDA GPUs
- Gradient clipping for stable training dynamics
- Resilient checkpoint save & recovery (weights, optimizer, scheduler, step, config)
- Live validation loss monitoring and JSONL metric logging
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM
from hindi.train.dataset import create_dataloader


# ============================================================
# OPTIMIZER & LR SCHEDULER
# ============================================================

def configure_optimizers(
    model: TransformerLM,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-8,
) -> AdamW:
    """
    Separates parameters into decay (2D matrix weights) and no-decay (biases, 1D LayerNorms).
    Robustly handles weight tying (tied token embeddings and lm_head).
    """
    decay = []
    no_decay = []

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1D tensors (biases, layernorm weights) get 0.0 weight decay
        if pn.endswith("bias") or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)

    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup followed by cosine annealing decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
    save_path: Path,
    model: TransformerLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    config: TransformerConfig,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config.to_dict(),
        "timestamp": time.time(),
    }
    torch.save(state, save_path)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved checkpoint to: {save_path} (step {step:,})")


def load_checkpoint(
    checkpoint_path: Path,
    model: TransformerLM,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    assert checkpoint_path.exists(), f"Checkpoint file not found: {checkpoint_path}"
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
        
    step = state.get("step", 0)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Successfully resumed from checkpoint: {checkpoint_path} at step {step:,}")
    return step


# ============================================================
# EVALUATION HELPER
# ============================================================

@torch.no_grad()
def evaluate_validation_loss(
    model: TransformerLM,
    val_dataloader,
    max_eval_batches: int = 50,
    device: torch.device = torch.device("cpu"),
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0

    for batch in val_dataloader:
        if batches >= max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            batches += 1

    model.train()
    return total_loss / max(1, batches)


# ============================================================
# MAIN TRAINING ROUTINE
# ============================================================

def train(args):
    # Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("PRETRAINING HINDI TRANSFORMER LM (MODEL H)")
    print("=" * 75)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # Paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train_log.jsonl"

    train_parquet = Path(args.train_parquet)
    val_parquet = Path(args.val_parquet)
    tokenizer_model = Path(args.tokenizer_model)

    # Config & Model Initialization
    if args.config_file and Path(args.config_file).exists():
        config = TransformerConfig.load_json(Path(args.config_file))
    else:
        config = TransformerConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            language="Hindi",
            model_name="Model_H_Hindi_25M",
        )
        config.save_json(output_dir / "model_config.json")

    model = TransformerLM(config).to(device)
    param_counts = model.count_parameters()
    print(f"\nModel Parameters Audit:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    # Optimizer & Scheduler
    optimizer = configure_optimizers(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        min_lr_ratio=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.use_amp))

    # Checkpoint Resume
    start_step = 0
    latest_ckpt = checkpoints_dir / "checkpoint_latest.pt"
    if args.resume and latest_ckpt.exists():
        start_step = load_checkpoint(latest_ckpt, model, optimizer, scheduler, scaler, device)
    elif args.checkpoint_path and Path(args.checkpoint_path).exists():
        start_step = load_checkpoint(Path(args.checkpoint_path), model, optimizer, scheduler, scaler, device)

    # DataLoaders
    train_loader = create_dataloader(
        parquet_path=train_parquet,
        tokenizer_model_path=tokenizer_model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        buffer_size=5000,
        shuffle=True,
    )
    val_loader = create_dataloader(
        parquet_path=val_parquet,
        tokenizer_model_path=tokenizer_model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        buffer_size=1000,
        shuffle=False,
    )

    print(f"\nTraining Settings:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Seq Len: {args.max_seq_len}")
    print(f"  Tokens / Step: {args.batch_size * args.max_seq_len:,}")
    print(f"  Max Steps: {args.max_steps:,}")
    print(f"  Starting Step: {start_step:,}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Warmup Steps: {args.warmup_steps:,}")
    print(f"  Mixed Precision (AMP): {device.type == 'cuda' and args.use_amp}")
    print()

    # Training Loop
    model.train()
    step = start_step
    best_val_loss = float("inf")
    running_loss = 0.0
    running_steps = 0
    start_time = time.time()

    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass with AMP
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and args.use_amp)):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        # Backward Pass with GradScaler
        if device.type == "cuda" and args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        scheduler.step()

        step += 1
        running_loss += loss.item()
        running_steps += 1

        # Logging
        if step % args.log_interval == 0 or step == args.max_steps:
            avg_loss = running_loss / running_steps
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            tokens_per_sec = (running_steps * args.batch_size * args.max_seq_len) / max(1e-5, elapsed)
            ppl = math.exp(min(20.0, avg_loss))

            print(
                f"Step [{step:6d}/{args.max_steps:6d}] | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train PPL: {ppl:7.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Tok/s: {tokens_per_sec:7.1f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Log to JSONL
            log_entry = {
                "step": step,
                "train_loss": avg_loss,
                "train_ppl": ppl,
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "elapsed_time": elapsed,
                "timestamp": time.time(),
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            running_loss = 0.0
            running_steps = 0
            start_time = time.time()

        # Validation & Checkpointing
        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n---> Running validation at step {step:,}...")
            val_loss = evaluate_validation_loss(model, val_loader, max_eval_batches=args.eval_batches, device=device)
            val_ppl = math.exp(min(20.0, val_loss))
            print(f"---> Step {step:,} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}\n")

            # Update log
            val_entry = {
                "step": step,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "timestamp": time.time(),
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_entry) + "\n")

            # Save Latest Checkpoint
            save_checkpoint(
                latest_ckpt,
                model,
                optimizer,
                scheduler,
                scaler if (device.type == "cuda" and args.use_amp) else None,
                step,
                avg_loss if "avg_loss" in locals() else loss.item(),
                val_loss,
                config,
            )

            # Save Best Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = checkpoints_dir / "checkpoint_best.pt"
                save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    scaler if (device.type == "cuda" and args.use_amp) else None,
                    step,
                    avg_loss if "avg_loss" in locals() else loss.item(),
                    val_loss,
                    config,
                )

        # Periodic Step Checkpoint
        if step % args.save_interval == 0 and step != args.max_steps:
            step_ckpt = checkpoints_dir / f"step_{step}.pt"
            save_checkpoint(
                step_ckpt,
                model,
                optimizer,
                scheduler,
                scaler if (device.type == "cuda" and args.use_amp) else None,
                step,
                loss.item(),
                None,
                config,
            )

    print("\n" + "=" * 75)
    print("HINDI PRETRAINING COMPLETE")
    print(f"Final Step: {step:,}")
    print(f"Best Val Loss: {best_val_loss:.4f} (PPL: {math.exp(min(20.0, best_val_loss)):.2f})")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Hindi Transformer LM")
    
    # Model Config
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data Paths
    parser.add_argument("--train_parquet", type=str, default=str(PROJECT_ROOT / "hindi/data/splits/train.parquet"))
    parser.add_argument("--val_parquet", type=str, default=str(PROJECT_ROOT / "hindi/data/splits/val.parquet"))
    parser.add_argument("--tokenizer_model", type=str, default=str(PROJECT_ROOT / "hindi/tokenizer/unigram_10k/tokenizer.model"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "hindi/checkpoints"))

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Logging & Checkpointing
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    train(args)
