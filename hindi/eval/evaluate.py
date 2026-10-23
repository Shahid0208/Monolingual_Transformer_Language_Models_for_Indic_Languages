"""
Evaluation Suite for Hindi Transformer Language Model (Model H)
==============================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid0208 (Monsoon 2026)
Language: Hindi (Model H)

Evaluates:
1. Intrinsic Language Modeling:
   - Validation Cross-Entropy Loss
   - Perplexity (PPL = exp(loss))
   - Bits-per-Byte (BPB = total_loss_nats / (ln(2) * total_utf8_bytes))
2. Generation Quality across Decoding Temperatures:
   - Greedy (T=0.0), T=0.5, T=1.0, T=1.5
   - Corpus-level BLEU-4
   - chrF / chrF++
   - ROUGE-L
3. Fluency and Diversity Diagnostics:
   - Distinct-1 and Distinct-2
   - N-gram Repetition Rate
"""

import argparse
import collections
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any

import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM


# ============================================================
# METRICS COMPUTATION (SELF-CONTAINED)
# ============================================================

def compute_bleu_ngram_precision(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> Tuple[float, List[float], float]:
    """
    Computes corpus-level BLEU with Brevity Penalty.
    """
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_lengths = 0
    hyp_lengths = 0

    for ref, hyp in zip(references, hypotheses):
        ref_lengths += len(ref)
        hyp_lengths += len(hyp)

        ref_ngrams_all = collections.Counter()
        for n in range(1, max_n + 1):
            ref_ngrams = collections.Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)])
            ref_ngrams_all.update(ref_ngrams)

        for n in range(1, max_n + 1):
            hyp_ngrams = collections.Counter([tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)])
            ref_n = collections.Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)])
            for ngram, count in hyp_ngrams.items():
                clipped_counts[n-1] += min(count, ref_n[ngram])
                total_counts[n-1] += count

    precisions = []
    for c, t in zip(clipped_counts, total_counts):
        precisions.append((c / t) if t > 0 else 0.0)

    # Brevity Penalty
    if hyp_lengths == 0:
        return 0.0, precisions, 0.0
    if hyp_lengths > ref_lengths:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_lengths / max(1, hyp_lengths)))

    # Geometric mean of precisions
    log_precisions = [math.log(p) if p > 0 else -999.0 for p in precisions]
    if any(p <= 0 for p in precisions):
        bleu = 0.0
    else:
        bleu = bp * math.exp(sum(log_precisions) / max_n)

    return bleu * 100.0, precisions, bp


def compute_chrf(references: List[str], hypotheses: List[str], char_order: int = 6, beta: float = 2.0) -> float:
    """
    Computes chrF++ (Character n-gram F-score with beta=2 for Indic morphology).
    """
    total_f_score = 0.0
    valid_pairs = 0

    for ref_str, hyp_str in zip(references, hypotheses):
        ref_chars = list(ref_str.replace(" ", ""))
        hyp_chars = list(hyp_str.replace(" ", ""))

        if not ref_chars or not hyp_chars:
            continue

        precisions = []
        recalls = []

        for n in range(1, char_order + 1):
            ref_ngrams = collections.Counter([tuple(ref_chars[i:i+n]) for i in range(len(ref_chars) - n + 1)])
            hyp_ngrams = collections.Counter([tuple(hyp_chars[i:i+n]) for i in range(len(hyp_chars) - n + 1)])

            total_hyp = sum(hyp_ngrams.values())
            total_ref = sum(ref_ngrams.values())

            matching = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())

            p = (matching / total_hyp) if total_hyp > 0 else 0.0
            r = (matching / total_ref) if total_ref > 0 else 0.0

            precisions.append(p)
            recalls.append(r)

        avg_p = sum(precisions) / len(precisions) if precisions else 0.0
        avg_r = sum(recalls) / len(recalls) if recalls else 0.0

        if (beta**2 * avg_p + avg_r) > 0:
            f_score = (1 + beta**2) * (avg_p * avg_r) / (beta**2 * avg_p + avg_r)
        else:
            f_score = 0.0

        total_f_score += f_score
        valid_pairs += 1

    return (total_f_score / max(1, valid_pairs)) * 100.0


def compute_rouge_l(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """
    Computes ROUGE-L F1 based on Longest Common Subsequence (LCS).
    """
    def lcs(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    total_f1 = 0.0
    valid = 0

    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            continue
        lcs_len = lcs(ref, hyp)
        prec = lcs_len / len(hyp)
        rec = lcs_len / len(ref)
        if (prec + rec) > 0:
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        total_f1 += f1
        valid += 1

    return (total_f1 / max(1, valid)) * 100.0


def compute_diversity_metrics(token_sequences: List[List[str]]) -> Dict[str, float]:
    """
    Computes Distinct-1, Distinct-2, and Repetition Rate.
    """
    all_unigrams = []
    all_bigrams = []
    all_trigrams = []
    repeated_trigram_count = 0
    total_trigrams = 0

    for seq in token_sequences:
        all_unigrams.extend(seq)
        for i in range(len(seq) - 1):
            all_bigrams.append((seq[i], seq[i+1]))
        for i in range(len(seq) - 2):
            tri = (seq[i], seq[i+1], seq[i+2])
            all_trigrams.append(tri)

    distinct_1 = (len(set(all_unigrams)) / len(all_unigrams)) if all_unigrams else 0.0
    distinct_2 = (len(set(all_bigrams)) / len(all_bigrams)) if all_bigrams else 0.0

    tri_counts = collections.Counter(all_trigrams)
    repeated_tri = sum(count - 1 for count in tri_counts.values() if count > 1)
    rep_rate = (repeated_tri / len(all_trigrams)) if all_trigrams else 0.0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repetition_rate_trigram": rep_rate,
    }


# ============================================================
# EVALUATION RUNNER
# ============================================================

@torch.no_grad()
def evaluate_intrinsic_lm(
    model: TransformerLM,
    parquet_path: Path,
    sp_tokenizer: spm.SentencePieceProcessor,
    max_eval_docs: int = 200,
    max_seq_len: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluates Validation Loss, Perplexity, and Bits-per-Byte (BPB).
    """
    model.eval()
    effective_max_seq_len = min(max_seq_len, model.config.max_seq_len)
    parquet_file = pq.ParquetFile(parquet_path)
    table = parquet_file.read_row_group(0, columns=["text"])
    texts = table.column("text").to_pylist()[:max_eval_docs]

    total_loss_sum = 0.0
    total_tokens = 0
    total_utf8_bytes = 0

    for text in texts:
        if not text or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        raw_bytes = len(text.encode("utf-8"))
        token_ids = [sp_tokenizer.bos_id()] + sp_tokenizer.encode_as_ids(text) + [sp_tokenizer.eos_id()]
        if len(token_ids) < 2:
            continue

        # Chunk into effective_max_seq_len
        for i in range(0, len(token_ids), effective_max_seq_len):
            chunk = token_ids[i : i + effective_max_seq_len]
            if len(chunk) < 2:
                continue

            input_tensor = torch.tensor([chunk], dtype=torch.long, device=device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs["loss"].item()
            num_targets = len(chunk) - 1

            total_loss_sum += loss * num_targets
            total_tokens += num_targets
            total_utf8_bytes += raw_bytes

    avg_loss = total_loss_sum / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    
    # Bits-per-byte (BPB) calculation:
    # BPB = total_loss_nats / (ln(2) * total_utf8_bytes)
    # where total_loss_nats = avg_loss * total_tokens
    bpb = (total_loss_sum) / (math.log(2.0) * max(1, total_utf8_bytes))

    return {
        "cross_entropy_loss": avg_loss,
        "perplexity": ppl,
        "bits_per_byte": bpb,
        "total_eval_tokens": total_tokens,
        "total_eval_bytes": total_utf8_bytes,
    }


@torch.no_grad()
def evaluate_generation(
    model: TransformerLM,
    parquet_path: Path,
    sp_tokenizer: spm.SentencePieceProcessor,
    prompt_tokens: int = 15,
    max_new_tokens: int = 35,
    num_samples: int = 100,
    temperatures: List[float] = [0.0, 0.5, 1.0, 1.5],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluates generation across temperatures: BLEU-4, chrF++, ROUGE-L, Distinct-1/2.
    """
    model.eval()
    max_new_tokens = min(max_new_tokens, max(1, model.config.max_seq_len - prompt_tokens - 2))
    parquet_file = pq.ParquetFile(parquet_path)
    table = parquet_file.read_row_group(0, columns=["text"])
    texts = [t for t in table.column("text").to_pylist() if t and len(t.split()) > 10][:num_samples]

    results_by_temp = {}
    sample_generations = []

    for temp in temperatures:
        temp_name = f"temp_{temp:.1f}" if temp > 0 else "greedy"
        refs_str: List[str] = []
        hyps_str: List[str] = []
        refs_tokens: List[List[str]] = []
        hyps_tokens: List[List[str]] = []

        for idx, text in enumerate(texts):
            token_ids = sp_tokenizer.encode_as_ids(text)
            if len(token_ids) <= prompt_tokens:
                continue

            prompt_ids = token_ids[:prompt_tokens]
            ref_target_ids = token_ids[prompt_tokens : prompt_tokens + max_new_tokens]

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            gen_tensor = model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=50 if temp > 0 else None,
                top_p=0.9 if temp > 0 else None,
            )

            gen_ids = gen_tensor[0, prompt_tokens:].tolist()

            ref_text = sp_tokenizer.decode_ids(ref_target_ids)
            hyp_text = sp_tokenizer.decode_ids(gen_ids)

            refs_str.append(ref_text)
            hyps_str.append(hyp_text)

            refs_tokens.append([sp_tokenizer.id_to_piece(t) for t in ref_target_ids])
            hyps_tokens.append([sp_tokenizer.id_to_piece(t) for t in gen_ids])

            if temp == 0.5 and len(sample_generations) < 5:
                prompt_text = sp_tokenizer.decode_ids(prompt_ids)
                sample_generations.append({
                    "sample_id": idx + 1,
                    "prompt": prompt_text,
                    "reference": ref_text,
                    "generated_temp_0.5": hyp_text,
                })

        bleu, precisions, bp = compute_bleu_ngram_precision(refs_tokens, hyps_tokens, max_n=4)
        chrf = compute_chrf(refs_str, hyps_str)
        rouge_l = compute_rouge_l(refs_tokens, hyps_tokens)
        diversity = compute_diversity_metrics(hyps_tokens)

        results_by_temp[temp_name] = {
            "bleu_4": bleu,
            "bleu_precisions": precisions,
            "brevity_penalty": bp,
            "chrf_plus_plus": chrf,
            "rouge_l": rouge_l,
            **diversity,
        }

    return {
        "metrics_by_temperature": results_by_temp,
        "sample_generations": sample_generations,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hindi Transformer LM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_file", type=str, default=str(PROJECT_ROOT / "hindi/configs/model_config.json"))
    parser.add_argument("--test_parquet", type=str, default=str(PROJECT_ROOT / "hindi/data/splits/test.parquet"))
    parser.add_argument("--tokenizer_model", type=str, default=str(PROJECT_ROOT / "hindi/tokenizer/unigram_10k/tokenizer.model"))
    parser.add_argument("--output_json", type=str, default=str(PROJECT_ROOT / "hindi/eval/evaluation_results.json"))
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on device: {device}")

    # Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "config" in ckpt:
        config = TransformerConfig.from_dict(ckpt["config"])
    else:
        config = TransformerConfig.load_json(Path(args.config_file))

    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from step: {ckpt.get('step', 'N/A')} (Vocab: {config.vocab_size}, SeqLen: {config.max_seq_len})")

    # 1. Intrinsic Language Modeling
    print("\n[1/2] Computing Intrinsic LM Metrics (Loss, PPL, BPB)...")
    intrinsic = evaluate_intrinsic_lm(model, Path(args.test_parquet), sp, max_eval_docs=100, device=device)
    print(f"  CE Loss: {intrinsic['cross_entropy_loss']:.4f}")
    print(f"  PPL:     {intrinsic['perplexity']:.2f}")
    print(f"  BPB:     {intrinsic['bits_per_byte']:.4f}")

    # 2. Generation Quality across Temperatures
    print("\n[2/2] Evaluating Generation Quality across Temperatures (0.0, 0.5, 1.0, 1.5)...")
    generation = evaluate_generation(model, Path(args.test_parquet), sp, device=device)
    for temp, m in generation["metrics_by_temperature"].items():
        print(f"  [{temp}] BLEU-4: {m['bleu_4']:.2f} | chrF++: {m['chrf_plus_plus']:.2f} | ROUGE-L: {m['rouge_l']:.2f} | Dist-1: {m['distinct_1']:.3f} | Dist-2: {m['distinct_2']:.3f}")

    # Save Results
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        "language": "Hindi",
        "checkpoint": args.checkpoint,
        "intrinsic_metrics": intrinsic,
        "generation_metrics": generation["metrics_by_temperature"],
        "sample_generations": generation["sample_generations"],
        "timestamp": time.time(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation metrics to: {out_path}")


if __name__ == "__main__":
    main()
