"""High-Speed Unified Nepali Manual Data Collector (~120M Tokens / ~88.9M Words).

Collects and extracts pure, high-quality Devanagari Nepali text from diverse sources:
1. Open Nepali web-scraped shards (Common Crawl / C4 Nepali / Web Portals & News)
2. Nepali Wikipedia & Literature archives (encyclopedic articles, history, classical literature, biographies)

Filters out boilerplate, enforces Devanagari script purity (>=40% per line, >=50% per doc),
deduplicates documents, and streams directly into:
individual-project-Shahid0208/nepali/data/raw/nepali_manual_raw.txt

Target: 120,000,000 Subword Tokens (~88,888,889 Words) to satisfy Phase 1 manual collection requirement.
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
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NepaliManualCollector")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nepali_manual_raw.txt")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NepaliResearch/2.0 (Language Modeling Academic Research)"

TOKEN_MULTIPLIER = 1.35  # Standard subword token-to-word ratio for Devanagari (Nepali/Hindi)

file_write_lock = threading.Lock()
seen_hashes_lock = threading.Lock()
seen_doc_hashes: Set[int] = set()

FAMOUS_NEPALI_AUTHORS_TOPICS = [
    "भानुभक्त आचार्य", "लक्ष्मीप्रसाद देवकोटा", "लेखनाथ पौड्याल", "बालकृष्ण सम",
    "पारिजात", "विश्वेश्वरप्रसाद कोइराला", "माधवप्रसाद घिमिरे", "भूपी शेरचन",
    "गोविन्दबहादुर मल्ल", "रमेश विकल", "इन्द्रबहादुर राई", "शङ्कर लामिछाने",
    "मुनामदन", "नेपालको इतिहास", "काठमाडौँ", "सगरमाथा", "नेपालका जिल्लाहरू",
    "नेपाली भाषा", "नेपालको भूगोल", "नेपालको संविधान", "पोखरा", "ललितपुर",
    "भक्तपुर", "गौतम बुद्ध", "लुम्बिनी", "जनकपुर", "पशुपतिनाथ मन्दिर",
    "नेपालको राजनीति", "नेपाली साहित्य", "नेपाली चाडपर्व", "दशैं", "तिहार",
    "नेपाल", "चितवन", "बिराटनगर", "धरान", "बुटवल", "हेटौंडा", "कर्णाली",
    "गण्डकी", "बागमती", "कोशी", "सुदूरपश्चिम", "लुम्बिनी प्रदेश", "मधेश प्रदेश"
]


def clean_nepali_document(text: str) -> Optional[str]:
    """Clean, filter, and validate a scraped Nepali text document."""
    if not text:
        return None

    # Remove HTML tags, URLs, and excessive whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    clean_lines = []
    total_chars = 0
    dev_chars = 0

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) < 15 or line.startswith("__") or line.startswith("=="):
            continue

        line_dev = sum(1 for c in line if "\u0900" <= c <= "\u097F")
        line_len = len(line)
        # Require line to be at least 40% Devanagari characters
        if line_dev / line_len >= 0.40:
            clean_lines.append(line)
            total_chars += line_len
            dev_chars += line_dev

    if not clean_lines or total_chars < 40:
        return None

    # Overall document Devanagari purity threshold >= 50%
    if dev_chars / total_chars < 0.50:
        return None

    doc = "\n".join(clean_lines).strip()
    words = len(doc.split())
    if words < 12:
        return None

    return doc


def append_documents(
    filepath: str,
    docs: List[str],
    stats_dict: dict,
    target_tokens: int,
    stop_event: threading.Event
) -> int:
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
                f.flush()
            stats_dict["total_words"] += added_words
            stats_dict["total_docs"] += len(unique_docs)
            current_tokens = int(stats_dict["total_words"] * TOKEN_MULTIPLIER)
            if current_tokens >= target_tokens:
                stop_event.set()

    return added_words


def fetch_nepali_wikipedia_article(title: str) -> Optional[str]:
    """Fetch raw text of a Nepali Wikipedia article via MediaWiki API."""
    api_url = "https://ne.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    try:
        req = urllib.request.Request(
            f"{api_url}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if pid != "-1" and "extract" in page:
                    return page["extract"]
    except Exception:
        pass
    return None


def harvest_wikipedia_and_literature(
    output_file: str,
    stats_dict: dict,
    target_tokens: int,
    stop_event: threading.Event
):
    """Scrape Nepali Wikipedia articles, biographies, culture, and literature topics."""
    logger.info("Starting Nepali Wikipedia & Literature harvesting stream...")
    titles_to_scrape = list(FAMOUS_NEPALI_AUTHORS_TOPICS)

    # Expand topics via Wikipedia search queries
    for term in FAMOUS_NEPALI_AUTHORS_TOPICS[:20]:
        if stop_event.is_set():
            break
        try:
            search_url = (
                f"https://ne.wikipedia.org/w/api.php?action=opensearch&search="
                f"{urllib.parse.quote(term)}&limit=30&namespace=0&format=json"
            )
            req = urllib.request.Request(search_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = json.loads(resp.read().decode("utf-8"))
                if len(results) > 1 and isinstance(results[1], list):
                    titles_to_scrape.extend(results[1])
        except Exception:
            continue

    # Deduplicate titles
    titles_to_scrape = list(dict.fromkeys(titles_to_scrape))
    logger.info(f"Discovered {len(titles_to_scrape)} Nepali Wikipedia / literary articles to harvest.")

    batch = []
    for title in titles_to_scrape:
        if stop_event.is_set():
            break
        raw_text = fetch_nepali_wikipedia_article(title)
        if raw_text:
            cleaned = clean_nepali_document(raw_text)
            if cleaned:
                batch.append(cleaned)
                if len(batch) >= 20:
                    append_documents(output_file, batch, stats_dict, target_tokens, stop_event)
                    batch = []
        time.sleep(0.05)

    if batch:
        append_documents(output_file, batch, stats_dict, target_tokens, stop_event)
    logger.info("Nepali Wikipedia & Literature harvesting stream completed.")


def process_c4_nepali_shard(
    shard_idx: int,
    output_file: str,
    stats_dict: dict,
    target_tokens: int,
    stop_event: threading.Event
):
    """Download, decompress, parse, clean, and write a C4 Nepali web-scraped shard."""
    if stop_event.is_set():
        return

    shard_str = f"{shard_idx:05d}"
    url = f"https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ne.tfrecord-{shard_str}-of-00256.json.gz"

    for attempt in range(1, 4):
        if stop_event.is_set():
            return
        try:
            logger.info(f"Downloading C4 Nepali Shard {shard_idx:03d}...")
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
                    cleaned = clean_nepali_document(raw_doc)
                    if cleaned:
                        batch_docs.append(cleaned)
                except Exception:
                    continue

            if batch_docs:
                added_words = append_documents(output_file, batch_docs, stats_dict, target_tokens, stop_event)
                logger.info(f"C4 Nepali Shard {shard_idx:03d} COMPLETE: {len(batch_docs):,} clean docs ({added_words:,} words) added.")
            return
        except Exception as e:
            if attempt == 3:
                logger.warning(f"C4 Nepali Shard {shard_idx:03d} failed after 3 attempts: {e}")
            time.sleep(attempt * 1.5)


def analyze_corpus(filepath: str, target_tokens: int = 120_000_000) -> Dict[str, Any]:
    """Calculate comprehensive token, word, document, and script purity statistics."""
    if not os.path.exists(filepath):
        return {"error": "File not found"}

    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    file_size_gb = round(file_size_bytes / (1024 * 1024 * 1024), 3)

    total_words = 0
    total_lines = 0
    total_chars = 0
    total_devanagari_chars = 0
    total_docs = 0
    prev_blank = True

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            stripped = line.strip()
            if not stripped:
                prev_blank = True
                continue
            if prev_blank:
                total_docs += 1
                prev_blank = False
            words = stripped.split()
            total_words += len(words)
            total_chars += len(line)
            total_devanagari_chars += sum(1 for c in line if "\u0900" <= c <= "\u097F")

    estimated_tokens = int(total_words * TOKEN_MULTIPLIER)
    devanagari_purity = (total_devanagari_chars / total_chars * 100) if total_chars else 0
    progress_pct = (estimated_tokens / target_tokens * 100) if target_tokens else 0

    return {
        "file_path": filepath,
        "file_size_mb": file_size_mb,
        "file_size_gb": file_size_gb,
        "total_documents": total_docs,
        "total_lines": total_lines,
        "total_words": total_words,
        "estimated_tokens": estimated_tokens,
        "target_tokens": target_tokens,
        "progress_percent": round(progress_pct, 2),
        "devanagari_purity_percent": round(devanagari_purity, 2),
        "status": "ACHIEVED [PASSED]" if estimated_tokens >= target_tokens else "IN PROGRESS",
    }


def print_report(report: Dict[str, Any]):
    """Pretty print the final collection report."""
    print("\n" + "=" * 80)
    print("                 NEPALI MANUAL DATA SCRAPING REPORT")
    print("=" * 80)
    print(f"File Path:                {report['file_path']}")
    print(f"File Size:                {report['file_size_gb']} GB ({report['file_size_mb']} MB)")
    print(f"Total Lines:              {report['total_lines']:,}")
    print(f"Total Documents / Units:  {report['total_documents']:,}")
    print(f"Total Raw Words:          {report['total_words']:,}")
    print(f"Devanagari Script Purity: {report['devanagari_purity_percent']}%")
    print("-" * 80)
    print(f"Estimated Subword Tokens: {report['estimated_tokens']:,}")
    print(f"Target Subword Tokens:    {report['target_tokens']:,}")
    print(f"Goal Progress:            {report['progress_percent']}%")
    print(f"Target Status:            {report['status']}")
    print("=" * 80 + "\n")


def run_collection(target_tokens: int = 120_000_000, max_workers: int = 8):
    """Run unified multi-source Nepali manual data harvesting until target is reached."""
    target_words = int(target_tokens / TOKEN_MULTIPLIER)

    logger.info("=" * 80)
    logger.info("STARTING UNIFIED NEPALI MANUAL DATA HARVESTING PIPELINE")
    logger.info(f"Target Subword Tokens:   {target_tokens:,} (~{target_words:,} words)")
    logger.info(f"Master Output File:      {MASTER_OUTPUT_FILE}")
    logger.info(f"Sources:                 C4 Nepali Web Shards + Nepali Wikipedia & Literature")
    logger.info("=" * 80)

    # Check existing data if any
    initial_words = 0
    initial_docs = 0
    if os.path.exists(MASTER_OUTPUT_FILE):
        with open(MASTER_OUTPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
            prev_blank = True
            for line in f:
                stripped = line.strip()
                if not stripped:
                    prev_blank = True
                    continue
                if prev_blank:
                    initial_docs += 1
                    prev_blank = False
                initial_words += len(stripped.split())

    initial_tokens = int(initial_words * TOKEN_MULTIPLIER)
    logger.info(f"Existing Data: {initial_words:,} words | ~{initial_tokens:,} tokens | {initial_docs:,} docs")

    if initial_tokens >= target_tokens:
        logger.info(f"Target already achieved! ({initial_tokens:,} tokens >= {target_tokens:,})")
        report = analyze_corpus(MASTER_OUTPUT_FILE, target_tokens)
        print_report(report)
        return report

    stats_dict = {"total_words": initial_words, "total_docs": initial_docs}
    stop_event = threading.Event()
    start_time = time.time()
    last_log_time = time.time()

    # Pre-populate hash set with existing documents to prevent duplicate appends
    if os.path.exists(MASTER_OUTPUT_FILE) and os.path.getsize(MASTER_OUTPUT_FILE) > 0:
        logger.info("Indexing existing document hashes for deduplication...")
        with open(MASTER_OUTPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
            for para in f.read().split("\n\n"):
                para = para.strip()
                if para:
                    doc_hash = hash(para[:250] + para[-250:]) if len(para) > 500 else hash(para)
                    seen_doc_hashes.add(doc_hash)
        logger.info(f"Loaded {len(seen_doc_hashes):,} existing document hashes into memory.")

    # We use up to 45 shards of C4 Nepali to easily reach > 120M tokens
    shard_indices = list(range(0, 45))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start Wikipedia scraper in background
        f_wiki = executor.submit(
            harvest_wikipedia_and_literature,
            MASTER_OUTPUT_FILE,
            stats_dict,
            target_tokens,
            stop_event,
        )

        # Start C4 Shards in parallel
        futures = [f_wiki]
        for s_idx in shard_indices:
            f = executor.submit(
                process_c4_nepali_shard,
                s_idx,
                MASTER_OUTPUT_FILE,
                stats_dict,
                target_tokens,
                stop_event,
            )
            futures.append(f)

        # Monitor loop
        while not stop_event.is_set():
            time.sleep(4)
            with file_write_lock:
                words = stats_dict["total_words"]
                docs = stats_dict["total_docs"]

            tokens = int(words * TOKEN_MULTIPLIER)
            pct = (tokens / target_tokens * 100) if target_tokens else 0
            now = time.time()

            if now - last_log_time >= 5.0:
                elapsed = now - start_time
                words_added = words - initial_words
                wps = words_added / elapsed if elapsed > 0 else 0
                tps = wps * TOKEN_MULTIPLIER
                current_mb = os.path.getsize(MASTER_OUTPUT_FILE) / (1024 * 1024) if os.path.exists(MASTER_OUTPUT_FILE) else 0
                eta_s = (target_tokens - tokens) / tps if tps > 0 else 0
                eta_m = eta_s / 60

                logger.info(
                    f"[PROGRESS] Tokens: {tokens:,} / {target_tokens:,} ({pct:.1f}%) | "
                    f"Words: {words:,} | Docs: {docs:,} | File: {current_mb:,.1f} MB | "
                    f"Speed: {tps:,.0f} tok/s | ETA: {eta_m:.1f} min"
                )
                last_log_time = now

            if tokens >= target_tokens:
                logger.info(f"Target of {target_tokens:,} tokens successfully reached! Stopping streams...")
                stop_event.set()
                break

            if all(f.done() for f in futures):
                logger.info("All harvesting streams have completed.")
                break

    total_time = time.time() - start_time
    logger.info(f"Data harvesting finished in {total_time:.1f} seconds. Generating final analysis report...")

    final_report = analyze_corpus(MASTER_OUTPUT_FILE, target_tokens)
    print_report(final_report)
    return final_report


if __name__ == "__main__":
    target = 120_000_000
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    run_collection(target_tokens=target, max_workers=8)
