#!/usr/bin/env python3
"""High-Performance 16-Core Parquet-Native Pipeline for Indic Language Models.

Stages:
  1. RAW PARQUET ENRICHMENT
     Ensures 5 schema columns:
       - document_id (string, unique)
       - text (string)
       - language (string)
       - source_type (string: manual | downloaded)
       - source (string)
  2. MULTIPROCESSING CLEANING & QUALITY FILTERING
     - Unicode NFC Normalization & Invisible/Control Char Stripping
     - HTML/XML, URLs, Email, Whitespace Removal
     - Punctuation Normalization (Preserving standard punctuation, । and ॥)
     - Quality Filtering (Min 15 chars, Min 4 words, Symbol Noise <= 30%)
     - Language / Script Filtering (>= 45% Devanagari Unicode: \\u0900-\\u097F)
  3. STREAMING DEDUPLICATION
     - Exact Deduplication: SHA-256 deterministic 64-bit hashing across sources
     - Near-Duplicate Removal: Punctuation-invariant hash + 8-word span matching
  4. CLEAN PARQUET OUTPUT
     - data/cleaned/*_clean.parquet (5 columns)
  5. CORPUS STATISTICS
     - data/cleaned/dataset_statistics.json & dataset_statistics.md
  6. TRAIN / VAL / TEST PARQUET SPLITTING
     - Deterministic 92.0% Train / 4.0% Val / 4.0% Test (Seed 42)
     - data/splits/{train.parquet, val.parquet, test.parquet} (5 columns)
"""

import os
import re
import sys
import time
import json
import random
import hashlib
import unicodedata
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional, Set, Tuple, List

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: PyArrow is not installed. Please install pyarrow.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ParquetPipeline")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 42
VAL_RATIO = 0.04
TEST_RATIO = 0.04
TRAIN_RATIO = 0.92
TOKEN_MULTIPLIER = 1.35
NUM_WORKERS = min(16, os.cpu_count() or 4)

# 5-Column Parquet Schema
PARQUET_SCHEMA = pa.schema([
    pa.field("document_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("language", pa.string()),
    pa.field("source_type", pa.string()),
    pa.field("source", pa.string()),
])

# Precompiled regex patterns
RE_HTML = re.compile(r"<[^>]+>", re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_INVISIBLE_CHARS = re.compile(r"[\u200B-\u200D\uFEFF\u00A0\u200E\u200F\u202A-\u202E]")
RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
RE_EXTRA_SPACES = re.compile(r"[ \t\r\f\v]+")
RE_CONSEC_PUNCT = re.compile(r"([!?,.:;।॥\-\'\"])\1{2,}")
RE_DEV_CHAR = re.compile(r"[\u0900-\u097F]")
RE_SYMBOLS = re.compile(r"[^\w\s\u0900-\u097F!?,.:;।॥\-\'\"]")
RE_PUNCT_SPACES = re.compile(r"[!?,.:;।॥\-\'\"\s]+")


def compute_sha256_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_near_hash(text: str) -> int:
    norm = RE_PUNCT_SPACES.sub("", text.lower())
    digest = hashlib.md5(norm.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_article_span_hash(words: List[str]) -> Optional[int]:
    if len(words) < 20:
        return None
    span_str = " ".join(words[:8]) + " " + " ".join(words[-8:])
    digest = hashlib.md5(span_str.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def clean_line_worker(text: str) -> Optional[str]:
    """Clean a single text line in worker process."""
    if not text:
        return None

    # Step 2: Unicode Normalization (NFC) & Invisible / Control Character Removal
    text = unicodedata.normalize("NFC", text)
    text = RE_INVISIBLE_CHARS.sub(" ", text)
    text = RE_CONTROL_CHARS.sub("", text)

    # Step 3: Noise & Boilerplate Removal
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)

    # Step 4: Punctuation Normalization (preserve ।, ॥, normal punct; collapse runs)
    text = RE_CONSEC_PUNCT.sub(r"\1\1", text)
    text = RE_EXTRA_SPACES.sub(" ", text).strip()

    # Step 5: Quality Filtering
    if len(text) < 15:
        return None

    words = text.split()
    if len(words) < 4:
        return None

    if len(RE_SYMBOLS.findall(text)) / len(text) > 0.30:
        return None

    # Step 6: Language / Script Filter (>= 45% Devanagari chars)
    if len(RE_DEV_CHAR.findall(text)) / len(text) < 0.45:
        return None

    return text


def clean_batch_worker(batch: List[str]) -> List[Optional[str]]:
    """Clean a batch of text lines in worker process."""
    return [clean_line_worker(t) for t in batch]


def enrich_raw_parquet_if_needed(
    raw_parquet_path: str,
    lang_name: str,
    source_type: str,
    source_name: str,
    id_prefix: str,
):
    """Enrich raw parquet with document_id, language, source_type, source if single-column."""
    schema = pq.read_schema(raw_parquet_path)
    if "document_id" in schema.names:
        logger.info(f"[{lang_name.upper()}] Raw parquet already has 5 columns: {os.path.basename(raw_parquet_path)}")
        return

    logger.info(f"[{lang_name.upper()}] Enriching raw parquet with 5 metadata columns: {os.path.basename(raw_parquet_path)}...")
    temp_out = raw_parquet_path + ".tmp.parquet"
    parquet_file = pq.ParquetFile(raw_parquet_path)
    row_count = 0

    with pq.ParquetWriter(temp_out, PARQUET_SCHEMA, compression="snappy") as writer:
        for batch in parquet_file.iter_batches(batch_size=100000):
            pydict = batch.to_pydict()
            texts = pydict["text"]
            batch_len = len(texts)
            doc_ids = [f"{id_prefix}_{row_count + j + 1:08d}" for j in range(batch_len)]
            row_count += batch_len
            langs = [lang_name] * batch_len
            stypes = [source_type] * batch_len
            sources = [source_name] * batch_len

            table = pa.Table.from_arrays(
                [pa.array(doc_ids), pa.array(texts), pa.array(langs), pa.array(stypes), pa.array(sources)],
                schema=PARQUET_SCHEMA,
            )
            writer.write_table(table)

    os.replace(temp_out, raw_parquet_path)
    logger.info(f"[{lang_name.upper()}] Raw parquet enriched: {os.path.basename(raw_parquet_path)} ({row_count:,} rows)")


def process_language_pipeline(lang_name: str, config: Dict[str, Any], pool: ProcessPoolExecutor) -> Dict[str, Any]:
    """Execute complete high-speed cleaning, deduplication, clean parquet emission, and 92/4/4 splitting."""
    logger.info("=" * 80)
    logger.info(f"[{lang_name.upper()}] STARTING FAST PARQUET CLEANING & 92/4/4 SPLITTING PIPELINE")
    logger.info("=" * 80)
    start_total = time.time()
    random.seed(SEED)

    # 1. Enrich Raw Parquets with 5 columns if needed
    enrich_raw_parquet_if_needed(
        config["manual_raw_parquet"], lang_name, "manual", config["manual_source_name"], f"{config['prefix']}_m"
    )
    enrich_raw_parquet_if_needed(
        config["down_raw_parquet"], lang_name, "downloaded", config["down_source_name"], f"{config['prefix']}_d"
    )

    # 2. Setup Clean & Split Parquet Writers
    os.makedirs(config["cleaned_dir"], exist_ok=True)
    os.makedirs(config["splits_dir"], exist_ok=True)

    manual_clean_writer = pq.ParquetWriter(config["manual_clean_parquet"], PARQUET_SCHEMA, compression="snappy")
    down_clean_writer = pq.ParquetWriter(config["down_clean_parquet"], PARQUET_SCHEMA, compression="snappy")

    train_writer = pq.ParquetWriter(config["train_parquet"], PARQUET_SCHEMA, compression="snappy")
    val_writer = pq.ParquetWriter(config["val_parquet"], PARQUET_SCHEMA, compression="snappy")
    test_writer = pq.ParquetWriter(config["test_parquet"], PARQUET_SCHEMA, compression="snappy")

    global_exact_hashes: Set[int] = set()
    global_near_hashes: Set[int] = set()
    global_article_hashes: Set[int] = set()

    FLUSH_SIZE = 50000

    def create_batch_dict():
        return {"id": [], "text": [], "lang": [], "stype": [], "src": []}

    buf_manual_clean = create_batch_dict()
    buf_down_clean = create_batch_dict()
    buf_train = create_batch_dict()
    buf_val = create_batch_dict()
    buf_test = create_batch_dict()

    def flush_buffer(buf: dict, writer: pq.ParquetWriter):
        if buf["text"]:
            table = pa.Table.from_arrays(
                [pa.array(buf["id"]), pa.array(buf["text"]), pa.array(buf["lang"]), pa.array(buf["stype"]), pa.array(buf["src"])],
                schema=PARQUET_SCHEMA,
            )
            writer.write_table(table)
            buf["id"].clear()
            buf["text"].clear()
            buf["lang"].clear()
            buf["stype"].clear()
            buf["src"].clear()

    stats = {
        "manual": {"raw_rows": 0, "raw_words": 0, "clean_rows": 0, "clean_words": 0, "exact_dups": 0, "near_dups": 0},
        "downloaded": {"raw_rows": 0, "raw_words": 0, "clean_rows": 0, "clean_words": 0, "exact_dups": 0, "near_dups": 0},
        "splits": {
            "train": {"rows": 0, "words": 0, "tokens": 0},
            "val": {"rows": 0, "words": 0, "tokens": 0},
            "test": {"rows": 0, "words": 0, "tokens": 0},
        }
    }

    # Process Raw Parquets
    for src_key, raw_pq_path, is_manual in [
        ("manual", config["manual_raw_parquet"], True),
        ("downloaded", config["down_raw_parquet"], False),
    ]:
        logger.info(f"[{lang_name.upper()}] Multiprocess streaming & cleaning {os.path.basename(raw_pq_path)}...")
        parquet_file = pq.ParquetFile(raw_pq_path)
        cur_stats = stats[src_key]
        src_start = time.time()

        CHUNK_SIZE = 10000
        for batch in parquet_file.iter_batches(batch_size=80000):
            pydict = batch.to_pydict()
            b_ids = pydict["document_id"]
            b_texts = pydict["text"]
            b_langs = pydict["language"]
            b_stypes = pydict["source_type"]
            b_srcs = pydict["source"]
            n_items = len(b_texts)

            # Split batch into sub-chunks for workers
            sub_chunks = [b_texts[j:j+CHUNK_SIZE] for j in range(0, n_items, CHUNK_SIZE)]
            cleaned_sub_chunks = list(pool.map(clean_batch_worker, sub_chunks))

            # Flatten and process in main thread for dedup & partitioning
            idx = 0
            for chunk_res in cleaned_sub_chunks:
                for cleaned in chunk_res:
                    cur_stats["raw_rows"] += 1
                    raw_w = len(b_texts[idx].split())
                    cur_stats["raw_words"] += raw_w

                    if cleaned is not None:
                        # Exact Dedup
                        exact_h = compute_sha256_hash(cleaned)
                        if exact_h in global_exact_hashes:
                            cur_stats["exact_dups"] += 1
                            idx += 1
                            continue
                        global_exact_hashes.add(exact_h)

                        # Near Dedup
                        near_h = compute_near_hash(cleaned)
                        if near_h in global_near_hashes:
                            cur_stats["near_dups"] += 1
                            idx += 1
                            continue
                        global_near_hashes.add(near_h)

                        words = cleaned.split()
                        art_h = compute_article_span_hash(words)
                        if art_h is not None:
                            if art_h in global_article_hashes:
                                cur_stats["near_dups"] += 1
                                idx += 1
                                continue
                            global_article_hashes.add(art_h)

                        # Valid Clean Document
                        cur_stats["clean_rows"] += 1
                        n_words = len(words)
                        cur_stats["clean_words"] += n_words
                        doc_id = b_ids[idx]
                        lang_val = b_langs[idx]
                        stype_val = b_stypes[idx]
                        src_val = b_srcs[idx]

                        # Clean Parquet Buffer
                        c_buf = buf_manual_clean if is_manual else buf_down_clean
                        c_writer = manual_clean_writer if is_manual else down_clean_writer
                        c_buf["id"].append(doc_id)
                        c_buf["text"].append(cleaned)
                        c_buf["lang"].append(lang_val)
                        c_buf["stype"].append(stype_val)
                        c_buf["src"].append(src_val)
                        if len(c_buf["text"]) >= FLUSH_SIZE:
                            flush_buffer(c_buf, c_writer)

                        # 92 / 4 / 4 Deterministic Partitioning
                        r = random.random()
                        if r < TEST_RATIO:
                            s_buf = buf_test
                            s_writer = test_writer
                            s_stat = stats["splits"]["test"]
                        elif r < (TEST_RATIO + VAL_RATIO):
                            s_buf = buf_val
                            s_writer = val_writer
                            s_stat = stats["splits"]["val"]
                        else:
                            s_buf = buf_train
                            s_writer = train_writer
                            s_stat = stats["splits"]["train"]

                        s_stat["rows"] += 1
                        s_stat["words"] += n_words
                        s_stat["tokens"] += int(n_words * TOKEN_MULTIPLIER)

                        s_buf["id"].append(doc_id)
                        s_buf["text"].append(cleaned)
                        s_buf["lang"].append(lang_val)
                        s_buf["stype"].append(stype_val)
                        s_buf["src"].append(src_val)
                        if len(s_buf["text"]) >= FLUSH_SIZE:
                            flush_buffer(s_buf, s_writer)

                    idx += 1

            now_elapsed = time.time() - src_start
            rate = cur_stats["raw_rows"] / now_elapsed if now_elapsed > 0 else 0
            logger.info(
                f"[{lang_name.upper()}] {src_key.upper()}: {cur_stats['raw_rows']:,} lines | "
                f"Clean: {cur_stats['clean_rows']:,} | Rate: {rate:,.0f} lines/sec"
            )

    # Flush all remaining buffers
    flush_buffer(buf_manual_clean, manual_clean_writer)
    flush_buffer(buf_down_clean, down_clean_writer)
    flush_buffer(buf_train, train_writer)
    flush_buffer(buf_val, val_writer)
    flush_buffer(buf_test, test_writer)

    # Close all Parquet writers
    manual_clean_writer.close()
    down_clean_writer.close()
    train_writer.close()
    val_writer.close()
    test_writer.close()

    # Calculate Totals & Shares
    total_clean_rows = stats["manual"]["clean_rows"] + stats["downloaded"]["clean_rows"]
    total_clean_words = stats["manual"]["clean_words"] + stats["downloaded"]["clean_words"]
    total_clean_tokens = int(total_clean_words * TOKEN_MULTIPLIER)

    manual_clean_words = stats["manual"]["clean_words"]
    manual_pct = round((manual_clean_words / total_clean_words * 100) if total_clean_words > 0 else 0, 2)
    downloaded_pct = round(100.0 - manual_pct, 2)

    total_exact_dups = stats["manual"]["exact_dups"] + stats["downloaded"]["exact_dups"]
    total_near_dups = stats["manual"]["near_dups"] + stats["downloaded"]["near_dups"]
    total_dups = total_exact_dups + total_near_dups

    train_mb = round(os.path.getsize(config["train_parquet"]) / (1024 * 1024), 2)
    val_mb = round(os.path.getsize(config["val_parquet"]) / (1024 * 1024), 2)
    test_mb = round(os.path.getsize(config["test_parquet"]) / (1024 * 1024), 2)
    manual_clean_mb = round(os.path.getsize(config["manual_clean_parquet"]) / (1024 * 1024), 2)
    down_clean_mb = round(os.path.getsize(config["down_clean_parquet"]) / (1024 * 1024), 2)
    total_clean_mb = round(manual_clean_mb + down_clean_mb, 2)

    # Update Split Percentages
    for s_name in ["train", "val", "test"]:
        s_dict = stats["splits"][s_name]
        s_dict["percentage"] = round((s_dict["words"] / total_clean_words * 100) if total_clean_words > 0 else 0, 2)

    stats["splits"]["train"]["size_mb"] = train_mb
    stats["splits"]["val"]["size_mb"] = val_mb
    stats["splits"]["test"]["size_mb"] = test_mb

    elapsed_total = round(time.time() - start_total, 2)

    summary_data = {
        "language": lang_name.capitalize(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "Apache Parquet (Snappy)",
        "schema_columns": ["document_id", "text", "language", "source_type", "source"],
        "token_multiplier": TOKEN_MULTIPLIER,
        "manual_token_share_percent": manual_pct,
        "downloaded_token_share_percent": downloaded_pct,
        "meets_manual_20pct_requirement": manual_pct >= 20.0,
        "total_tokens_target_500m_met": total_clean_tokens >= 500_000_000,
        "manual": {
            "file_name": os.path.basename(config["manual_clean_parquet"]),
            "raw_rows": stats["manual"]["raw_rows"],
            "cleaned_rows": stats["manual"]["clean_rows"],
            "cleaned_words": stats["manual"]["clean_words"],
            "estimated_tokens": int(stats["manual"]["clean_words"] * TOKEN_MULTIPLIER),
            "exact_duplicates_removed": stats["manual"]["exact_dups"],
            "near_duplicates_removed": stats["manual"]["near_dups"],
            "file_size_mb": manual_clean_mb,
        },
        "downloaded": {
            "file_name": os.path.basename(config["down_clean_parquet"]),
            "raw_rows": stats["downloaded"]["raw_rows"],
            "cleaned_rows": stats["downloaded"]["clean_rows"],
            "cleaned_words": stats["downloaded"]["clean_words"],
            "estimated_tokens": int(stats["downloaded"]["clean_words"] * TOKEN_MULTIPLIER),
            "exact_duplicates_removed": stats["downloaded"]["exact_dups"],
            "near_duplicates_removed": stats["downloaded"]["near_dups"],
            "file_size_mb": down_clean_mb,
        },
        "total_clean_corpus": {
            "rows": total_clean_rows,
            "words": total_clean_words,
            "estimated_tokens": total_clean_tokens,
            "exact_duplicates_removed": total_exact_dups,
            "near_duplicates_removed": total_near_dups,
            "total_duplicates_removed": total_dups,
            "file_size_mb": total_clean_mb,
            "processing_time_seconds": elapsed_total,
        },
        "splits": {
            "train": {
                "file_name": "train.parquet",
                "rows": stats["splits"]["train"]["rows"],
                "words": stats["splits"]["train"]["words"],
                "estimated_tokens": stats["splits"]["train"]["tokens"],
                "size_mb": train_mb,
                "percentage": stats["splits"]["train"]["percentage"],
            },
            "val": {
                "file_name": "val.parquet",
                "rows": stats["splits"]["val"]["rows"],
                "words": stats["splits"]["val"]["words"],
                "estimated_tokens": stats["splits"]["val"]["tokens"],
                "size_mb": val_mb,
                "percentage": stats["splits"]["val"]["percentage"],
            },
            "test": {
                "file_name": "test.parquet",
                "rows": stats["splits"]["test"]["rows"],
                "words": stats["splits"]["test"]["words"],
                "estimated_tokens": stats["splits"]["test"]["tokens"],
                "size_mb": test_mb,
                "percentage": stats["splits"]["test"]["percentage"],
            },
        },
    }

    # Save JSON statistics
    with open(config["stats_json"], "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info(f"[{lang_name.upper()}] Saved JSON statistics to: {config['stats_json']}")

    # Save Markdown report
    tr_tokens_m = stats["splits"]["train"]["tokens"] / 1e6
    va_tokens_m = stats["splits"]["val"]["tokens"] / 1e6
    te_tokens_m = stats["splits"]["test"]["tokens"] / 1e6
    total_tokens_m = total_clean_tokens / 1e6

    report_md = f"""# {lang_name.capitalize()} Parquet Dataset Cleaning & 92/4/4 Partition Report

**Language:** {lang_name.capitalize()} (Devanagari script)  
**Format:** Apache Parquet (Snappy Compression)  
**Schema:** `[document_id, text, language, source_type, source]`  
**Generated:** {summary_data['timestamp']}  
**Deterministic Seed:** {SEED}  
**Token Multiplier:** {TOKEN_MULTIPLIER}x  

---

## 1. Corpus Summary & Compliance Overview

| Source Category | Parquet File | Rows | Clean Words | Est. Subword Tokens | Size | Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manual Cleaned** | `{os.path.basename(config['manual_clean_parquet'])}` | {stats['manual']['clean_rows']:,} | {stats['manual']['clean_words']:,} | {int(stats['manual']['clean_words']*TOKEN_MULTIPLIER):,} | {manual_clean_mb:.2f} MB | **{manual_pct:.2f}%** |
| **Downloaded Cleaned** | `{os.path.basename(config['down_clean_parquet'])}` | {stats['downloaded']['clean_rows']:,} | {stats['downloaded']['clean_words']:,} | {int(stats['downloaded']['clean_words']*TOKEN_MULTIPLIER):,} | {down_clean_mb:.2f} MB | **{downloaded_pct:.2f}%** |
| **Total Clean Corpus** | **Combined** | **{total_clean_rows:,}** | **{total_clean_words:,}** | **{total_clean_tokens:,} ({total_tokens_m:.2f}M)** | **{total_clean_mb:.2f} MB** | **100.00%** |

- **Assignment Requirement (Manual $\\ge 20\%$):** **{manual_pct:.2f}% ({'PASS' if manual_pct >= 20 else 'FAIL'})**
- **Assignment Target Tokens ($\\sim 500\\text{{M}}$):** **{total_tokens_m:.2f}M Tokens ({'PASS' if total_clean_tokens >= 500e6 else 'FAIL'})**
- **Exact Duplicates Removed:** **{total_exact_dups:,}** (SHA-256 deterministic global hash)
- **Near Duplicates Removed:** **{total_near_dups:,}** (Punctuation-invariant + 8-word span matching)

---

## 2. Partition Splits (92% Train / 4% Val / 4% Test)

| Partition File | Target Ratio | Actual Ratio | Rows | Words | Estimated Tokens | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train.parquet`** | **92.0%** | **{stats['splits']['train']['percentage']:.2f}%** | {stats['splits']['train']['rows']:,} | {stats['splits']['train']['words']:,} | {stats['splits']['train']['tokens']:,} ({tr_tokens_m:.2f}M) | {train_mb:.2f} MB |
| **`val.parquet`** | **4.0%** | **{stats['splits']['val']['percentage']:.2f}%** | {stats['splits']['val']['rows']:,} | {stats['splits']['val']['words']:,} | {stats['splits']['val']['tokens']:,} ({va_tokens_m:.2f}M) | {val_mb:.2f} MB |
| **`test.parquet`** | **4.0%** | **{stats['splits']['test']['percentage']:.2f}%** | {stats['splits']['test']['rows']:,} | {stats['splits']['test']['words']:,} | {stats['splits']['test']['tokens']:,} ({te_tokens_m:.2f}M) | {test_mb:.2f} MB |
| **Total** | **100.0%** | **100.00%** | **{total_clean_rows:,}** | **{total_clean_words:,}** | **{total_clean_tokens:,} ({total_tokens_m:.2f}M)** | **{(train_mb + val_mb + test_mb):.2f} MB** |

---

## 3. Data Integrity & Schema Verification
- **Schema Verified:** 5-column structure (`document_id`, `text`, `language`, `source_type`, `source`).
- **Unicode NFC Normalization:** Applied canonical composition and removed zero-width control chars.
- **Deduplication:** SHA-256 exact hash + 8-word span anchor near-duplicate elimination across all source files.
"""
    with open(config["stats_md"], "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"[{lang_name.upper()}] Saved Markdown report to: {config['stats_md']}")

    logger.info(
        f"[{lang_name.upper()}] Partitioning complete in {elapsed_total:.1f}s | "
        f"Train: {tr_tokens_m:.2f}M ({stats['splits']['train']['percentage']}%) | "
        f"Val: {va_tokens_m:.2f}M ({stats['splits']['val']['percentage']}%) | "
        f"Test: {te_tokens_m:.2f}M ({stats['splits']['test']['percentage']}%)"
    )
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="High-Performance Parquet Pipeline for Indic LM Corpora")
    parser.add_argument(
        "--lang",
        choices=["hindi", "nepali", "all"],
        default="all",
        help="Language pipeline to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"RUNNING PARQUET PIPELINE ON {NUM_WORKERS} CORES (Target: {args.lang.upper()})")
    logger.info("=" * 80)
    pipeline_start = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        if args.lang in ["hindi", "all"]:
            # 1. Hindi Configuration
            hindi_config = {
                "prefix": "hi",
                "manual_raw_parquet": os.path.join(BASE_DIR, "hindi/data/raw/hindi_manual_raw.parquet"),
                "down_raw_parquet": os.path.join(BASE_DIR, "hindi/data/raw/hindi_downloaded_raw.parquet"),
                "manual_source_name": "wikipedia_wikisource_news_manual",
                "down_source_name": "indiccorp_v2_downloaded",
                "cleaned_dir": os.path.join(BASE_DIR, "hindi/data/cleaned"),
                "manual_clean_parquet": os.path.join(BASE_DIR, "hindi/data/cleaned/hindi_manual_clean.parquet"),
                "down_clean_parquet": os.path.join(BASE_DIR, "hindi/data/cleaned/hindi_downloaded_clean.parquet"),
                "stats_json": os.path.join(BASE_DIR, "hindi/data/cleaned/dataset_statistics.json"),
                "stats_md": os.path.join(BASE_DIR, "hindi/data/cleaned/dataset_statistics.md"),
                "splits_dir": os.path.join(BASE_DIR, "hindi/data/splits"),
                "train_parquet": os.path.join(BASE_DIR, "hindi/data/splits/train.parquet"),
                "val_parquet": os.path.join(BASE_DIR, "hindi/data/splits/val.parquet"),
                "test_parquet": os.path.join(BASE_DIR, "hindi/data/splits/test.parquet"),
            }
            process_language_pipeline("hindi", hindi_config, pool)

        if args.lang in ["nepali", "all"]:
            # 2. Nepali Configuration
            nepali_config = {
                "prefix": "ne",
                "manual_raw_parquet": os.path.join(BASE_DIR, "nepali/data/raw/nepali_manual_raw.parquet"),
                "down_raw_parquet": os.path.join(BASE_DIR, "nepali/data/raw/nepali_downloaded_raw.parquet"),
                "manual_source_name": "kantipur_setopati_news_manual",
                "down_source_name": "indiccorp_v2_oscar_nepali",
                "cleaned_dir": os.path.join(BASE_DIR, "nepali/data/cleaned"),
                "manual_clean_parquet": os.path.join(BASE_DIR, "nepali/data/cleaned/nepali_manual_clean.parquet"),
                "down_clean_parquet": os.path.join(BASE_DIR, "nepali/data/cleaned/nepali_downloaded_clean.parquet"),
                "stats_json": os.path.join(BASE_DIR, "nepali/data/cleaned/dataset_statistics.json"),
                "stats_md": os.path.join(BASE_DIR, "nepali/data/cleaned/dataset_statistics.md"),
                "splits_dir": os.path.join(BASE_DIR, "nepali/data/splits"),
                "train_parquet": os.path.join(BASE_DIR, "nepali/data/splits/train.parquet"),
                "val_parquet": os.path.join(BASE_DIR, "nepali/data/splits/val.parquet"),
                "test_parquet": os.path.join(BASE_DIR, "nepali/data/splits/test.parquet"),
            }
            process_language_pipeline("nepali", nepali_config, pool)

    total_pipeline_time = round(time.time() - pipeline_start, 2)
    logger.info(f"PARQUET DATA PIPELINE COMPLETED SUCCESSFULLY IN {total_pipeline_time}s!")


if __name__ == "__main__":
    main()
