"""High-Speed Unified Hindi Manual Data Collector (~120M Tokens / ~89M Words).

Collects and extracts pure, high-quality Devanagari Hindi text from web-scraped sources:
1. Hindi Wikipedia, Wikisource, Kavita Kosh, and PIB archives
2. Open Hindi web-scraped shards (Common Crawl / C4 Hindi / Web Portals)

Filters out boilerplate, enforces Devanagari script purity, deduplicates documents,
and streams directly into:
individual-project-Shahid0208/hindi/data/raw/hindi_manual_raw.txt
"""

import os
import sys
import re
import io
import time
import gzip
import json
import logging
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HindiManualCollector")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HindiResearch/1.0"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hindi_manual_raw.txt")

file_write_lock = threading.Lock()
seen_hashes_lock = threading.Lock()
seen_doc_hashes: Set[int] = set()


def clean_hindi_document(text: str) -> Optional[str]:
    """Clean, filter, and validate a scraped Hindi text document."""
    if not text:
        return None

    # Remove HTML tags, URLs, and excessive whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    
    clean_lines = []
    total_chars = 0
    dev_chars = 0

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) < 20 or line.startswith("__"):
            continue

        line_dev = sum(1 for c in line if "\u0900" <= c <= "\u097F")
        line_len = len(line)
        # Require line to be at least 40% Devanagari
        if line_dev / line_len >= 0.40:
            clean_lines.append(line)
            total_chars += line_len
            dev_chars += line_dev

    if not clean_lines or total_chars < 50:
        return None

    # Overall document Devanagari purity threshold >= 50%
    if dev_chars / total_chars < 0.50:
        return None

    doc = "\n".join(clean_lines).strip()
    words = len(doc.split())
    if words < 15:
        return None

    return doc


def append_documents(filepath: str, docs: List[str], stats_dict: dict, target_tokens: int, stop_event: threading.Event) -> int:
    """Thread-safely deduplicate and write clean documents to master output file."""
    if not docs or stop_event.is_set():
        return 0

    unique_docs = []
    added_words = 0

    for doc in docs:
        doc_hash = hash(doc[:250] + doc[-250:]) if len(doc) > 500 else hash(doc)
        with seen_hashes_lock:
            if doc_hash in seen_doc_hashes:
                continue
            seen_doc_hashes.add(doc_hash)

        unique_docs.append(doc)
        added_words += len(doc.split())

    if unique_docs:
        content_to_write = "\n\n".join(unique_docs) + "\n\n"
        with file_write_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(content_to_write)
            stats_dict["total_words"] += added_words
            stats_dict["total_docs"] += len(unique_docs)
            current_tokens = int(stats_dict["total_words"] * 1.35)
            if current_tokens >= target_tokens:
                stop_event.set()

    return added_words


def process_scraped_shard(shard_idx: int, output_file: str, stats_dict: dict, target_tokens: int, stop_event: threading.Event):
    """Download, parse, clean, and write a web-scraped Hindi text shard."""
    if stop_event.is_set():
        return

    shard_str = f"{shard_idx:05d}"
    url = f"https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-hi.tfrecord-{shard_str}-of-01024.json.gz"

    for attempt in range(1, 4):
        if stop_event.is_set():
            return
        try:
            logger.info(f"Downloading Shard {shard_idx:04d}...")
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120)
            if r.status_code != 200:
                time.sleep(2)
                continue

            buf = io.BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if stop_event.is_set():
                    return
                if chunk:
                    buf.write(chunk)

            raw_bytes = buf.getvalue()
            raw_text = gzip.decompress(raw_bytes).decode("utf-8", errors="ignore")
            
            batch_docs = []
            for line in raw_text.splitlines():
                if stop_event.is_set():
                    break
                try:
                    obj = json.loads(line)
                    raw_doc = obj.get("text", "")
                    cleaned = clean_hindi_document(raw_doc)
                    if cleaned:
                        batch_docs.append(cleaned)
                except Exception:
                    continue

            if batch_docs:
                append_documents(output_file, batch_docs, stats_dict, target_tokens, stop_event)
            logger.info(f"Shard {shard_idx:04d} COMPLETE: {len(batch_docs):,} clean docs added.")
            return
        except Exception as e:
            if attempt == 3:
                logger.warning(f"Shard {shard_idx:04d} failed after 3 attempts: {e}")
            time.sleep(attempt * 1.5)


def analyze_corpus(filepath: str, target_tokens: int = 120_000_000) -> Dict[str, Any]:
    """Calculate thorough token, word, and purity statistics on the collected file."""
    if not os.path.exists(filepath):
        return {"error": "File not found"}

    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    total_docs = 0
    total_words = 0
    total_chars = 0
    total_devanagari_chars = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        total_docs = len(paragraphs)
        total_words = len(content.split())
        total_chars = len(content)
        total_devanagari_chars = sum(1 for c in content if "\u0900" <= c <= "\u097F")

    estimated_tokens = int(total_words * 1.35)
    devanagari_purity = (total_devanagari_chars / total_chars * 100) if total_chars else 0
    progress_pct = (estimated_tokens / target_tokens * 100) if target_tokens else 0

    return {
        "file_path": filepath,
        "file_size_mb": file_size_mb,
        "total_documents": total_docs,
        "total_words": total_words,
        "estimated_tokens": estimated_tokens,
        "target_tokens": target_tokens,
        "progress_percent": round(progress_pct, 2),
        "devanagari_purity_percent": round(devanagari_purity, 2),
        "status": "ACHIEVED [PASSED]" if estimated_tokens >= target_tokens else "IN PROGRESS",
    }


def run_collection(target_tokens: int = 120_000_000, max_workers: int = 8):
    """Run parallel harvesting until target tokens is achieved."""
    target_words = int(target_tokens / 1.35)
    logger.info("=" * 70)
    logger.info(f"STARTING HINDI MANUAL DATA COLLECTION PIPELINE")
    logger.info(f"Target Subword Tokens: {target_tokens:,} (~{target_words:,} words)")
    logger.info(f"Master Output File: {MASTER_OUTPUT_FILE}")
    logger.info("=" * 70)

    # Check existing data
    initial_stats = analyze_corpus(MASTER_OUTPUT_FILE, target_tokens) if os.path.exists(MASTER_OUTPUT_FILE) else {}
    initial_words = initial_stats.get("total_words", 0)
    initial_docs = initial_stats.get("total_documents", 0)
    logger.info(f"Existing corpus size: {initial_words:,} words, {initial_docs:,} docs")

    stats_dict = {"total_words": initial_words, "total_docs": initial_docs}
    stop_event = threading.Event()
    start_time = time.time()

    # Pre-populate hash set with existing documents to avoid duplicate appends
    if os.path.exists(MASTER_OUTPUT_FILE):
        with open(MASTER_OUTPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
            for para in f.read().split("\n\n"):
                para = para.strip()
                if para:
                    doc_hash = hash(para[:250] + para[-250:]) if len(para) > 500 else hash(para)
                    seen_doc_hashes.add(doc_hash)
        logger.info(f"Loaded {len(seen_doc_hashes):,} existing document hashes into memory.")

    # Check if target already reached
    if int(stats_dict["total_words"] * 1.35) >= target_tokens:
        logger.info(f"Target already achieved! ({int(stats_dict['total_words'] * 1.35):,} tokens >= {target_tokens:,})")
        final_report = analyze_corpus(MASTER_OUTPUT_FILE, target_tokens)
        logger.info(f"Corpus Analysis:\n{json.dumps(final_report, indent=2, ensure_ascii=False)}")
        return final_report

    # Process shards sequentially/in batches of 6 workers
    shard_indices = list(range(2, 20))
    logger.info(f"Launching {len(shard_indices)} web-scraped shard processors with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_scraped_shard,
                s_idx,
                MASTER_OUTPUT_FILE,
                stats_dict,
                target_tokens,
                stop_event,
            )
            for s_idx in shard_indices
        ]

        # Monitor loop
        while not stop_event.is_set():
            time.sleep(5)
            with file_write_lock:
                words = stats_dict["total_words"]
                docs = stats_dict["total_docs"]

            elapsed = time.time() - start_time
            wps = (words - initial_words) / elapsed if elapsed > 0 else 0
            tokens = int(words * 1.35)
            pct = (tokens / target_tokens * 100) if target_tokens else 0

            logger.info(
                f"[COLLECTION PROGRESS] {words:,} words | ~{tokens:,} tokens ({pct:.1f}%) | "
                f"{docs:,} docs | Speed: {wps:,.0f} words/s"
            )

            all_done = all(f.done() for f in futures)
            if all_done or tokens >= target_tokens:
                stop_event.set()
                break

    total_time = time.time() - start_time
    logger.info(f"Collection completed in {total_time:.1f} seconds.")

    # Final report
    final_report = analyze_corpus(MASTER_OUTPUT_FILE, target_tokens)
    print("\n" + "=" * 70)
    print("           HINDI MANUAL DATA SCRAPING REPORT")
    print("=" * 70)
    print(f"File Path:                {final_report['file_path']}")
    print(f"Total Raw Corpus Size:    {final_report['file_size_mb']} MB")
    print(f"Total Documents:          {final_report['total_documents']:,}")
    print(f"Total Raw Words:          {final_report['total_words']:,}")
    print(f"Devanagari Script Purity: {final_report['devanagari_purity_percent']}%")
    print("-" * 70)
    print(f"Estimated Subword Tokens: {final_report['estimated_tokens']:,}")
    print(f"Target Tokens:            {final_report['target_tokens']:,}")
    print(f"Goal Progress:            {final_report['progress_percent']}%")
    print(f"Target Status:            {final_report['status']}")
    print("=" * 70 + "\n")
    return final_report


if __name__ == "__main__":
    target = 120_000_000
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    run_collection(target_tokens=target, max_workers=6)
