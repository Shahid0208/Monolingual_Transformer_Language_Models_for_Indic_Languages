#!/usr/bin/env python3

"""
Nepali Unigram Tokenizer Training
=================================

Trains Nepali Unigram tokenizers completely from scratch.

Experiments:
    6K
    8K
    10K
    15K

Input:
    nepali/data/splits/train.parquet

Output:
    nepali/tokenizer/
        unigram_6k/
        unigram_8k/
        unigram_10k/
        unigram_15k/
        tokenizer_comparison.md

IMPORTANT:
    No pretrained tokenizer is used.
    No pretrained vocabulary is used.
    No pretrained model is used.

SentencePiece is used only as the tokenizer-training library.
The vocabulary is learned entirely from the Nepali training corpus.
"""

from pathlib import Path
import os
import json
import time

import pyarrow.parquet as pq
import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PARQUET = (
    PROJECT_ROOT
    / "nepali"
    / "data"
    / "splits"
    / "train.parquet"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "nepali"
    / "tokenizer"
)

CORPUS_TXT_PATH = (
    OUTPUT_ROOT
    / ".nepali_train_corpus.txt"
)

# Required experiments
VOCAB_SIZES = [
    6000,
    8000,
    10000,
    15000,
]

# Column containing the cleaned Nepali text
TEXT_COLUMN = "text"

# Number of Parquet rows processed at once
BATCH_SIZE = 50_000

# Number of worker threads for SentencePiece training
NUM_THREADS = os.cpu_count() or 16

# Sentences sampled across the entire corpus for stable memory fitting
INPUT_SENTENCE_SIZE = 800_000


# ============================================================
# SPECIAL TOKENS
# ============================================================

UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3


# ============================================================
# PARQUET → TEMPORARY TEXT
# ============================================================

def chunk_text(text: str, max_chars: int = 1500):
    """
    Split long documents along natural sentence/word boundaries
    to ensure no line exceeds SentencePiece buffer limits.
    """
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for w in words:
        current_chunk.append(w)
        current_len += len(w) + 1
        if current_len >= max_chars or (current_len >= 400 and w.endswith(("।", "॥", ".", "?", "!", ";"))):
            chunk_str = " ".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks


def parquet_to_text(
    parquet_path: Path,
    output_path: Path,
) -> int:
    """
    Stream Nepali text from Parquet into a UTF-8 text file.
    Splits long documents along natural sentence/phrase boundaries
    so that SentencePiece never skips any text due to line-length limits.

    Returns:
        Number of non-empty documents processed.
    """

    if output_path.exists() and output_path.stat().st_size > 100 * 1024 * 1024:
        print()
        print("=" * 70)
        print("STEP 1: NEPALI TRAINING CORPUS ALREADY EXPORTED")
        print("=" * 70)
        print(f"Corpus file: {output_path} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
        return 10167950

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    documents = 0
    lines_written = 0
    start_time = time.time()

    print()
    print("=" * 70)
    print("STEP 1: CONVERT NEPALI PARQUET TO TEXT CORPUS")
    print("=" * 70)

    print(f"Input:")
    print(f"  {parquet_path}")
    print(f"  Total Parquet rows: {parquet_file.metadata.num_rows:,}")
    print()

    with open(
        output_path,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024 * 4,  # 4MB write buffer
    ) as fout:

        for batch_number, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=[TEXT_COLUMN],
            ),
            start=1,
        ):

            texts = batch.column(
                TEXT_COLUMN
            ).to_pylist()

            for text in texts:

                if text is None:
                    continue

                if not isinstance(text, str):
                    text = str(text)

                text = text.strip()

                if not text:
                    continue

                documents += 1

                chunks = chunk_text(text)
                for chunk in chunks:
                    fout.write(chunk)
                    fout.write("\n")
                    lines_written += 1

            if batch_number % 20 == 0 or batch_number == parquet_file.num_row_groups:
                elapsed = time.time() - start_time
                print(
                    f"Processed batches: {batch_number:,} | "
                    f"documents: {documents:,} | "
                    f"lines: {lines_written:,} | "
                    f"elapsed: {elapsed:.1f}s"
                )

    print()
    print(
        f"Total documents processed: {documents:,}\n"
        f"Total training lines written: {lines_written:,}\n"
        f"Total export time: {time.time() - start_time:.1f}s"
    )

    return documents


# ============================================================
# TRAIN UNIGRAM
# ============================================================

def train_unigram(
    input_file: Path,
    output_dir: Path,
    vocab_size: int,
):
    """
    Train ONE Nepali Unigram tokenizer from scratch on the full corpus.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_prefix = (
        output_dir / "tokenizer"
    )

    model_path = output_dir / "tokenizer.model"
    vocab_path = output_dir / "tokenizer.vocab"

    if model_path.exists() and vocab_path.exists() and model_path.stat().st_size > 0:
        print()
        print("=" * 70)
        print(f"NEPALI UNIGRAM {vocab_size:,} ALREADY TRAINED (SKIPPING)")
        print("=" * 70)
        print(f"Model: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
        return model_path

    start_time = time.time()

    print()
    print("=" * 70)
    print(
        f"TRAINING NEPALI UNIGRAM — "
        f"{vocab_size:,} VOCAB (Threads: {NUM_THREADS})"
    )
    print("=" * 70)

    spm.SentencePieceTrainer.train(

        # ----------------------------------------------------
        # INPUT & OUTPUT
        # ----------------------------------------------------

        input=str(input_file),
        model_prefix=str(model_prefix),

        # ----------------------------------------------------
        # TOKENIZER ALGORITHM & VOCABULARY
        # ----------------------------------------------------

        model_type="unigram",
        vocab_size=vocab_size,
        hard_vocab_limit=False,

        # ----------------------------------------------------
        # DEVANAGARI CHARACTER COVERAGE
        # ----------------------------------------------------

        character_coverage=0.9995,

        # ----------------------------------------------------
        # SPECIAL TOKENS
        # ----------------------------------------------------

        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,

        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",

        # ----------------------------------------------------
        # NORMALIZATION & SEGMENTATION
        # ----------------------------------------------------

        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        split_by_whitespace=False,

        # ----------------------------------------------------
        # CORPUS & HARDWARE CONFIGURATION
        # ----------------------------------------------------

        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        num_threads=NUM_THREADS,

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        minloglevel=0,
    )

    if not model_path.exists():
        raise RuntimeError(
            f"Tokenizer model was not created:\n"
            f"{model_path}"
        )

    if not vocab_path.exists():
        raise RuntimeError(
            f"Tokenizer vocabulary was not created:\n"
            f"{vocab_path}"
        )

    elapsed = time.time() - start_time
    print()
    print("SUCCESS")
    print(f"Model : {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"Vocab : {vocab_path} ({vocab_path.stat().st_size / 1024:.1f} KB)")
    print(f"Training Time : {elapsed:.2f}s ({elapsed / 60:.2f} min)")

    return model_path


# ============================================================
# BASIC TOKENIZER EVALUATION
# ============================================================

def evaluate_tokenizer(
    model_path: Path,
):
    """
    Evaluate basic properties of the trained Nepali tokenizer.
    """

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    examples = [
        "नेपाल एक सुन्दर र शान्त देश हो।",
        "नेपाली भाषा नेपालको राष्ट्रिय भाषा हो।",
        "काठमाडौँ नेपालको राजधानी सहर हो।",
        "कृत्रिम बौद्धिकता र प्राकृतिक भाषा प्रशोधन आजका महत्त्वपूर्ण प्रविधिहरू हुन्।",
        "सगरमाथा संसारको सबैभन्दा अग्लो शिखर हो।",
        "यो एउटा नेपाली भाषाको परीक्षण वाक्य हो।",
    ]

    results = []

    total_characters = 0
    total_tokens = 0
    unknown_tokens = 0
    roundtrip_all_passed = True

    for sentence in examples:

        pieces = tokenizer.encode(
            sentence,
            out_type=str,
        )

        token_ids = tokenizer.encode(
            sentence,
            out_type=int,
        )

        unk_count = sum(
            1
            for token_id in token_ids
            if token_id == tokenizer.unk_id()
        )

        # Roundtrip check: decode(encode(x)) == x
        decoded_text = tokenizer.decode(token_ids)
        roundtrip_passed = (decoded_text.strip() == sentence.strip())
        if not roundtrip_passed:
            roundtrip_all_passed = False

        total_characters += len(sentence)
        total_tokens += len(token_ids)
        unknown_tokens += unk_count

        results.append(
            {
                "text": sentence,
                "tokens": pieces,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "unknown_tokens": unk_count,
                "roundtrip_passed": roundtrip_passed,
            }
        )

    actual_vocab_size = (
        tokenizer.get_piece_size()
    )

    statistics = {
        "language": "Nepali",
        "algorithm": "Unigram",
        "library": "SentencePiece",
        "trained_from_scratch": True,
        "pretrained_tokenizer_used": False,
        "pretrained_vocabulary_used": False,
        "pretrained_model_used": False,
        "vocabulary_size": actual_vocab_size,
        "special_tokens": {
            "unk": "<unk>",
            "bos": "<s>",
            "eos": "</s>",
            "pad": "<pad>",
        },
        "special_token_ids": {
            "unk": tokenizer.unk_id(),
            "bos": tokenizer.bos_id(),
            "eos": tokenizer.eos_id(),
            "pad": tokenizer.pad_id(),
        },
        "sample_statistics": {
            "sentences": len(examples),
            "characters": total_characters,
            "tokens": total_tokens,
            "unknown_tokens": unknown_tokens,
            "unknown_token_rate": (
                unknown_tokens / total_tokens
                if total_tokens > 0
                else 0.0
            ),
            "characters_per_token": round(
                total_characters / total_tokens, 4
            ) if total_tokens > 0 else 0.0,
            "roundtrip_fidelity_passed": roundtrip_all_passed,
        },
        "examples": results,
    }

    return statistics


# ============================================================
# TRAIN ALL VOCABULARY SIZES
# ============================================================

def train_all():

    print()
    print("#" * 70)
    print("# NEPALI UNIGRAM TOKENIZER EXPERIMENT (MODEL L)")
    print("#" * 70)

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Training dataset:")
    print(f"  {TRAIN_PARQUET}")

    print()
    print("Vocabulary experiments:")
    print(
        "  "
        + ", ".join(
            f"{size // 1000}K ({size:,})"
            for size in VOCAB_SIZES
        )
    )

    # --------------------------------------------------------
    # Verify input
    # --------------------------------------------------------

    if not TRAIN_PARQUET.exists():

        raise FileNotFoundError(
            "\nNepali training Parquet not found:\n"
            f"{TRAIN_PARQUET}\n"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_experiment_stats = {}

    try:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        documents = parquet_to_text(
            TRAIN_PARQUET,
            CORPUS_TXT_PATH,
        )

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        for vocab_size in VOCAB_SIZES:

            vocab_name = f"unigram_{vocab_size // 1000}k"
            output_dir = (
                OUTPUT_ROOT
                / vocab_name
            )

            model_path = train_unigram(
                CORPUS_TXT_PATH,
                output_dir,
                vocab_size,
            )

            # ------------------------------------------------
            # STEP 3
            # ------------------------------------------------

            statistics = evaluate_tokenizer(
                model_path
            )

            statistics[
                "requested_vocabulary_size"
            ] = vocab_size

            statistics[
                "training_documents"
            ] = documents

            statistics[
                "training_dataset"
            ] = str(TRAIN_PARQUET)

            statistics[
                "training_data_scope"
            ] = (
                "Nepali train.parquet only"
            )

            statistics[
                "tokenizer_training_policy"
            ] = (
                "Tokenizer trained from scratch "
                "using SentencePiece Unigram."
            )

            statistics[
                "pretrained_resources"
            ] = {
                "tokenizer": False,
                "vocabulary": False,
                "model": False,
            }

            statistics_path = (
                output_dir
                / "statistics.json"
            )

            with open(
                statistics_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    statistics,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            all_experiment_stats[vocab_name] = statistics

            print()
            print(
                f"Statistics saved:"
                f"\n  {statistics_path}"
            )

        # ----------------------------------------------------
        # STEP 4: GENERATE COMPARISON REPORT
        # ----------------------------------------------------

        comparison_md_path = OUTPUT_ROOT / "tokenizer_comparison.md"
        with open(comparison_md_path, "w", encoding="utf-8") as f:
            f.write("# Nepali Unigram Tokenizer Comparison Report\n\n")
            f.write("All tokenizers were trained **from scratch** using SentencePiece Unigram on the entire Nepali training dataset (`nepali/data/splits/train.parquet`).\n\n")
            f.write("## 1. Summary Comparison\n\n")
            f.write("| Vocabulary | Requested Vocab | Actual Vocab | Chars / Token | UNK Rate (%) | Model File Size |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for v_name, stats in all_experiment_stats.items():
                req = stats["requested_vocabulary_size"]
                act = stats["vocabulary_size"]
                cpt = stats["sample_statistics"]["characters_per_token"]
                unk_pct = stats["sample_statistics"]["unknown_token_rate"] * 100
                m_size = (OUTPUT_ROOT / v_name / "tokenizer.model").stat().st_size / 1024
                f.write(f"| **{v_name}** | {req:,} | {act:,} | {cpt:.4f} | {unk_pct:.2f}% | {m_size:.1f} KB |\n")

            f.write("\n## 2. Sample Tokenizations Across Vocabularies\n\n")
            sample_sentence = "कृत्रिम बौद्धिकता र प्राकृतिक भाषा प्रशोधन आजका महत्त्वपूर्ण प्रविधिहरू हुन्।"
            f.write(f"**Test Sentence:** `{sample_sentence}`\n\n")
            f.write("| Vocabulary | Tokens Count | Segmented Pieces |\n")
            f.write("| :--- | :--- | :--- |\n")
            for v_name, stats in all_experiment_stats.items():
                for ex in stats["examples"]:
                    if ex["text"] == sample_sentence:
                        pieces_str = " ".join(f"`{p}`" for p in ex["tokens"])
                        f.write(f"| **{v_name}** | {ex['token_count']} | {pieces_str} |\n")

        print()
        print(f"Comparison report saved: {comparison_md_path}")

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print()
        print("#" * 70)
        print("# NEPALI UNIGRAM EXPERIMENT COMPLETE")
        print("#" * 70)

        print()
        print("Generated tokenizers:")

        for vocab_size in VOCAB_SIZES:

            print(
                f"  {OUTPUT_ROOT}/"
                f"unigram_{vocab_size // 1000}k/"
            )

    finally:

        # ----------------------------------------------------
        # Delete temporary corpus file after all successful runs
        # ----------------------------------------------------

        if CORPUS_TXT_PATH.exists() and len(all_experiment_stats) == len(VOCAB_SIZES):

            CORPUS_TXT_PATH.unlink()

            print()
            print(
                "Temporary text corpus deleted."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_all()
