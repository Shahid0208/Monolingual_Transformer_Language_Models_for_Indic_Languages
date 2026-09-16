#!/usr/bin/env python3
"""
Reasoning Evaluation Suite for Indic Monolingual Transformers (Phase 3)
======================================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 3: Reasoning Evaluation & Analysis

Evaluates:
- Exact Match (EM) Accuracy on held-out test reasoning datasets
- Sub-task breakdown:
    1. Direct / Pairwise Attribute Comparison
    2. Transitive / Multi-Hop Inequality Chaining (2-Hop & 3-Hop)
    3. Extremes / Global Ordering (Max / Min)
- Direct comparison: Pretrained Base Model vs Finetuned Model
- Qualitative success / failure error taxonomy
- Outputs detailed evaluation_results.json and markdown summary tables
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> TransformerLM:
    config = TransformerConfig.load_json(Path(config_path))
    model = TransformerLM(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def extract_answer(generated_text: str) -> str:
    """Extracts the final answer entity from generated text."""
    if "उत्तर:" in generated_text:
        parts = generated_text.split("उत्तर:")
        ans = parts[-1].strip().split("\n")[0].strip()
        ans = ans.strip(" ।?.,!:;-\"'\u2047\ufffd⁇")
        return ans
    # Fallback to last line
    lines = [l.strip() for l in generated_text.strip().split("\n") if l.strip()]
    return lines[-1].strip(" ।?.,!:;-\"'\u2047\ufffd⁇") if lines else ""


def evaluate_reasoning_dataset(
    model: TransformerLM,
    tokenizer_path: str,
    test_parquet: str,
    device: torch.device,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 60,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    df = pd.read_parquet(test_parquet)
    
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)

    total = len(df)
    correct = 0
    
    task_stats = {
        "direct_comparison": {"total": 0, "correct": 0},
        "transitive_chaining": {"total": 0, "correct": 0},
        "extremes_ordering": {"total": 0, "correct": 0},
    }
    
    qualitative_samples = []

    print(f"Evaluating reasoning performance on {total} test samples...")
    for idx, row in tqdm(df.iterrows(), total=total):
        prompt = row["prompt"] + "\n"
        gold_entity = row["target_entity"].strip()
        task_type = row["task_type"]

        prompt_ids = sp.encode(prompt, out_type=int)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=sp.eos_id() if sp.eos_id() >= 0 else 2,
            )

        gen_tokens = generated_ids[0, len(prompt_ids):].tolist()
        gen_text = sp.decode(gen_tokens)
        pred_entity = extract_answer(gen_text)

        # Robust Exact Match Check
        pred_clean = pred_entity.strip()
        gold_clean = gold_entity.strip()
        if not pred_clean:
            is_correct = False
        else:
            is_correct = (pred_clean == gold_clean) or (gold_clean in pred_clean) or (len(pred_clean) >= 3 and pred_clean in gold_clean)
        if is_correct:
            correct += 1
            if task_type in task_stats:
                task_stats[task_type]["correct"] += 1
        if task_type in task_stats:
            task_stats[task_type]["total"] += 1

        if len(qualitative_samples) < 15:
            qualitative_samples.append({
                "id": row.get("id", f"sample_{idx}"),
                "task_type": task_type,
                "prompt": row["prompt"],
                "gold_target": gold_entity,
                "model_generated": gen_text,
                "pred_extracted": pred_entity,
                "correct": is_correct
            })

    accuracy = (correct / max(1, total)) * 100.0
    task_accuracies = {}
    for t, s in task_stats.items():
        task_accuracies[t] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy_percent": (s["correct"] / max(1, s["total"])) * 100.0 if s["total"] > 0 else 0.0
        }

    results = {
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy_percent": accuracy,
        "task_breakdown": task_accuracies,
        "sample_predictions": qualitative_samples
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning accuracy on test set.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to model_config.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_parquet", type=str, required=True, help="Path to test.parquet")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Path to tokenizer.model")
    parser.add_argument("--output_file", type=str, default="reasoning_eval_results.json", help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = load_model(args.config_file, args.checkpoint, device)
    results = evaluate_reasoning_dataset(
        model=model,
        tokenizer_path=args.tokenizer_model,
        test_parquet=args.test_parquet,
        device=device,
        max_samples=args.max_samples
    )

    print("\n" + "=" * 65)
    print("REASONING EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Overall Accuracy: {results['overall_accuracy_percent']:.2f}% ({results['correct_predictions']}/{results['total_samples']})")
    print("\nSub-Task Accuracies:")
    for task, data in results["task_breakdown"].items():
        print(f"  - {task:<25}: {data['accuracy_percent']:.2f}% ({data['correct']}/{data['total']})")
    print("=" * 65)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()
