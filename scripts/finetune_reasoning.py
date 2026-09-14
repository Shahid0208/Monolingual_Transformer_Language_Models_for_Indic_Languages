#!/usr/bin/env python3
"""
Supervised Reasoning Finetuning Engine for Monolingual Indic Transformers (Phase 3)
==================================================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 3: Reasoning Finetuning & Evaluation

Supports:
- Model H (Hindi) and Model L (Nepali)
- Loading from Phase 2 Pretrained Checkpoint (`checkpoint_best.pt`)
- Prompt-Target Loss Masking (loss only on reasoning steps + answer)
- Resilient Checkpoint Saving and Resume (weights, optimizer, scheduler, step, config)
- Live Step & Epoch Loss Logging to `finetune_log.jsonl`
- Automatic Mixed Precision (AMP) acceleration on CUDA GPUs
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import sentencepiece as spm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM


# ==============================================================================
# 1. SUPERVISED REASONING DATASET WITH TARGET LOSS MASKING
# ==============================================================================

class ReasoningSFTDataset(Dataset):
    """
    Dataset for Supervised Finetuning on Reasoning Tasks.
    Encodes full text [Prompt + Reasoning + Answer] and masks prompt tokens with pad_token_id
    so loss is computed purely on the reasoning and answer generation.
    """
    def __init__(
        self,
        parquet_path: str,
        tokenizer_path: str,
        max_seq_len: int = 512,
        mask_prompt_loss: bool = True
    ):
        self.max_seq_len = max_seq_len
        self.mask_prompt_loss = mask_prompt_loss
        
        assert os.path.exists(parquet_path), f"File not found: {parquet_path}"
        assert os.path.exists(tokenizer_path), f"Tokenizer not found: {tokenizer_path}"
        
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 3
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        
        df = pd.read_parquet(parquet_path)
        self.records = df.to_dict(orient="records")
        print(f"Loaded {len(self.records):,} reasoning samples from: {parquet_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]
        prompt = row["prompt"] + "\n"
        full_text = row["full_text"] + "\n"

        prompt_ids = self.sp.encode(prompt, out_type=int)
        full_ids = self.sp.encode(full_text, out_type=int)
        
        # Append EOS if not present
        if not full_ids or full_ids[-1] != self.eos_id:
            full_ids.append(self.eos_id)

        # Truncate to max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mask prompt tokens if requested
        if self.mask_prompt_loss:
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels[:prompt_len] = self.pad_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": torch.tensor(len(full_ids), dtype=torch.long)
        }


def collate_reasoning_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int = 3) -> Dict[str, torch.Tensor]:
    """Dynamically pads input_ids and labels to the batch max sequence length."""
    max_len = max(item["length"].item() for item in batch)
    
    batch_input_ids = []
    batch_labels = []
    
    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - len(inp)
        
        if pad_len > 0:
            padded_inp = F.pad(inp, (0, pad_len), value=pad_id)
            padded_lbl = F.pad(lbl, (0, pad_len), value=pad_id)
        else:
            padded_inp = inp
            padded_lbl = lbl
            
        batch_input_ids.append(padded_inp)
        batch_labels.append(padded_lbl)
        
    return {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
    }


# ==============================================================================
# 2. OPTIMIZER & LR SCHEDULING
# ==============================================================================

def configure_finetune_optimizer(
    model: TransformerLM,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    decay = []
    no_decay = []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn.endswith("bias") or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.98), eps=1e-8)


def get_finetune_cosine_schedule(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 3. CHECKPOINT LOADING & SAVING (100% COMPATIBLE WITH PHASE 2)
# ==============================================================================

def load_pretrained_weights(
    checkpoint_path: str,
    model: TransformerLM,
    device: torch.device
) -> Dict[str, Any]:
    """Loads weights from Phase 2 pretrained checkpoint."""
    print(f"Loading pretrained checkpoint from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
        
    print("✓ Successfully loaded pretrained model weights.")
    return state


def save_finetuned_checkpoint(
    save_path: Path,
    model: TransformerLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TransformerConfig,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config.to_dict(),
        "timestamp": time.time(),
        "finetune_phase": 3,
    }
    torch.save(state, save_path)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved finetuned checkpoint: {save_path} (step {step:,}, val_loss: {val_loss:.4f})")


# ==============================================================================
# 4. VALIDATION EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate_validation(
    model: TransformerLM,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            count += 1
            
    model.train()
    return total_loss / max(1, count)


# ==============================================================================
# 5. MAIN FINETUNING LOOP
# ==============================================================================

def train_finetune(args):
    # Setup Device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 75)
    print(f"STARTING PHASE 3 REASONING FINETUNING ({args.language.upper()})")
    print("=" * 75)
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU Device    : {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "finetune_log.jsonl"

    # Load Model Configuration
    if args.config_file and os.path.exists(args.config_file):
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
            language=args.language,
            model_name=f"Model_{args.language}_25M_Finetuned",
        )
    config.save_json(output_dir / "model_config.json")

    # Instantiate Model
    model = TransformerLM(config).to(device)

    # Load Pretrained Weights
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        load_pretrained_weights(args.pretrained_checkpoint, model, device)
    else:
        print("WARNING: No pretrained checkpoint provided! Finetuning from scratch / random init.")

    # Datasets & Dataloaders
    train_dataset = ReasoningSFTDataset(
        parquet_path=args.train_parquet,
        tokenizer_path=args.tokenizer_model,
        max_seq_len=config.max_seq_len,
        mask_prompt_loss=args.mask_prompt_loss
    )
    val_dataset = ReasoningSFTDataset(
        parquet_path=args.val_parquet,
        tokenizer_path=args.tokenizer_model,
        max_seq_len=config.max_seq_len,
        mask_prompt_loss=args.mask_prompt_loss
    )

    pad_id = train_dataset.pad_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_id=pad_id),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_id=pad_id)
    )

    # Total Steps & Optimizer
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = configure_finetune_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = get_finetune_cosine_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.use_amp))

    print("-" * 75)
    print(f"Finetuning Schedule:")
    print(f"  Train Samples    : {len(train_dataset):,}")
    print(f"  Val Samples      : {len(val_dataset):,}")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Steps / Epoch    : {steps_per_epoch:,}")
    print(f"  Total Steps      : {total_steps:,}")
    print(f"  Warmup Steps     : {warmup_steps:,}")
    print(f"  Peak Learning Rate: {args.learning_rate}")
    print("-" * 75)

    # Initial Pretrained Validation Loss
    init_val_loss = evaluate_validation(model, val_loader, device)
    print(f"Initial Zero-Shot Val Loss on Reasoning: {init_val_loss:.4f} (PPL: {math.exp(init_val_loss):.2f})")

    best_val_loss = init_val_loss
    global_step = 0
    start_time = time.time()

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            step_count += 1
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.use_amp)):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            # Logging
            if global_step % args.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch}/{args.epochs}] | Step [{global_step:4d}/{total_steps}] | "
                    f"Loss: {loss_val:.4f} | PPL: {math.exp(min(20, loss_val)):.2f} | "
                    f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
                )

                # Log to JSONL
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": loss_val,
                        "lr": current_lr,
                        "timestamp": time.time()
                    }) + "\n")

            # Periodic Validation Evaluation
            if global_step % args.eval_interval == 0:
                val_loss = evaluate_validation(model, val_loader, device)
                val_ppl = math.exp(min(20, val_loss))
                print(f"---> Step {global_step} Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")

                # Save latest
                save_finetuned_checkpoint(
                    checkpoints_dir / "checkpoint_latest.pt",
                    model, optimizer, scheduler, scaler,
                    global_step, epoch, loss_val, val_loss, config
                )

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_finetuned_checkpoint(
                        checkpoints_dir / "checkpoint_best.pt",
                        model, optimizer, scheduler, scaler,
                        global_step, epoch, loss_val, val_loss, config
                    )
                    print(f"★ New best model saved! (Val Loss: {best_val_loss:.4f})")

        # End of Epoch Summary & Evaluation
        avg_train_loss = epoch_loss / max(1, step_count)
        epoch_val_loss = evaluate_validation(model, val_loader, device)
        print("=" * 75)
        print(
            f"EPOCH {epoch}/{args.epochs} COMPLETE | "
            f"Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
        )
        print("=" * 75)

        # Save milestone checkpoint
        save_finetuned_checkpoint(
            checkpoints_dir / f"checkpoint_epoch_{epoch}.pt",
            model, optimizer, scheduler, scaler,
            global_step, epoch, avg_train_loss, epoch_val_loss, config
        )

    print("\n" + "=" * 75)
    print("REASONING FINETUNING COMPLETED SUCCESSFULLY!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Optimal Checkpoint  : {checkpoints_dir / 'checkpoint_best.pt'}")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning Finetuning for Indic Transformer LM (Phase 3)")
    parser.add_argument("--language", type=str, default="Hindi", help="Target Language (Hindi / Nepali)")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Path to Phase 2 pretrained checkpoint_best.pt")
    parser.add_argument("--train_parquet", type=str, required=True, help="Path to reasoning train.parquet")
    parser.add_argument("--val_parquet", type=str, required=True, help="Path to reasoning val.parquet")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--config_file", type=str, default=None, help="Path to model_config.json")
    parser.add_argument("--output_dir", type=str, default="finetuned_reasoning_checkpoints", help="Output directory")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Total training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate for finetuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup fraction of total steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--mask_prompt_loss", action="store_true", default=True, help="Mask prompt tokens from loss calculation")
    parser.add_argument("--log_interval", type=int, default=25, help="Steps between training logs")
    parser.add_argument("--eval_interval", type=int, default=50, help="Steps between validation evaluations")

    # Hardware
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    parser.add_argument("--device", type=str, default=None, help="Hardware device ('cuda' or 'cpu')")

    # Architecture fallbacks
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    train_finetune(parse_args())
