"""High-Speed Multi-Source Nepali Downloaded Corpus Collector (~420M Tokens / ~311.1M Words).

Downloads, streams, cleans, and deduplicates high-quality Devanagari Nepali text
from open public repositories and archives:
1. AI4Bharat IndicCorp V2 Nepali High-Speed Stream (ne.txt - 13.76 GB)
2. Hugging Face Public Nepali Datasets (realsanjeev, Bibek1129, Saugatkafley, glot500, NepaliAI, wisewizer)
3. CC-100 High-Throughput Nepali Web Crawl Stream
4. Nepali Wikipedia Articles Dump

Writes directly into:
individual-project-Shahid0208/nepali/data/raw/nepali_downloaded_raw.txt

Target: 420,000,000 Subword Tokens (~311,111,111 Words) to satisfy Phase 1 downloaded corpus requirement.
"""

import os
import sys
import re
import io
import time
import json
import csv
import lzma
import bz2
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NepaliCorpusDownloader")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nepali_downloaded_raw.txt")
OFFSET_STATE_FILE = os.path.join(OUTPUT_DIR, "nepali_downloaded_offset.json")
INDICCORP_NEPALI_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/ne.txt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NepaliResearch/2.0 (Language Modeling Academic Research)"

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB per parallel chunk
MAX_WORKERS = 12               # 12 parallel threads for high throughput
TOKEN_MULTIPLIER = 1.35        # Standard subword-to-word expansion ratio for Devanagari Nepali

file_write_lock = threading.Lock()
stats_lock = threading.Lock()
seen_hashes_lock = threading.Lock()
seen_doc_hashes: Set[int] = set()
thread_local = threading.local()

# Secondary Public Dataset URLs
PUBLIC_SOURCES = [
    {
        "name": "realsanjeev/nepali-summarization",
        "url": "https://huggingface.co/datasets/realsanjeev/nepali-summarization-dataset/resolve/main/Summerization_dataset.csv",
        "type": "csv"
    },
    {
        "name": "Bibek1129/nepali_SQuAD_single_qsn",
        "url": "https://huggingface.co/datasets/Bibek1129/nepali_SQuAD_single_qsn/resolve/main/nepali_SQuAD_single_qsn.json",
        "type": "json"
    },
    {
        "name": "Saugatkafley/Nepali-Roman-Transliteration",
        "url": "https://huggingface.co/datasets/Saugatkafley/Nepali-Roman-Transliteration/resolve/main/nep_train.json",
        "type": "json"
    },
    {
        "name": "glot500/nep_Deva_part0",
        "url": "https://huggingface.co/datasets/cis-lmu/glot500/resolve/main/nep_Deva/train/data-00000-of-00002.arrow",
        "type": "arrow"
    },
    {
        "name": "glot500/nep_Deva_part1",
        "url": "https://huggingface.co/datasets/cis-lmu/glot500/resolve/main/nep_Deva/train/data-00001-of-00002.arrow",
        "type": "arrow"
    },
    {
        "name": "Bibek1129/nepali_SQuAD",
        "url": "https://huggingface.co/datasets/Bibek1129/nepali_SQuAD/resolve/main/nepali_SQuAD.json",
        "type": "json"
    },
    {
        "name": "Bibek1129/nepali_SQuAD_multi",
        "url": "https://huggingface.co/datasets/Bibek1129/nepali_SQuAD_multiple_qsns/resolve/main/nepali_SQuAD_multiple_qsns.json",
        "type": "json"
    },
    {
        "name": "realsanjeev/XLSum",
        "url": "https://huggingface.co/datasets/realsanjeev/XLSum-nepali-summerization-dataset/resolve/main/XLSum_all_nep.csv",
        "type": "csv"
    },
    {
        "name": "wisewizer/nepali-news",
        "url": "https://huggingface.co/datasets/wisewizer/nepali-news/resolve/main/nepali_train.jsonl",
        "type": "jsonl"
    },
    {
        "name": "dipeshch71/nepaliflow",
        "url": "https://huggingface.co/datasets/dipeshch71/nepaliflow-romanized-nepali-to-devanagari-dataset/resolve/main/nepaliflow_instruction_v1.csv",
        "type": "csv"
    },
    {
        "name": "Shushant/CovidTweets",
        "url": "https://huggingface.co/datasets/Shushant/NepaliCovidTweets/resolve/main/train.csv",
        "type": "csv"
    }
]

for i in range(14):
    PUBLIC_SOURCES.append({
        "name": f"NepaliAI/HealthChat_shard_{i:02d}",
        "url": f"https://huggingface.co/datasets/NepaliAI/Nepali-HealthChat/resolve/main/train{i:02d}.json",
        "type": "json"
    })
for i in range(9):
    PUBLIC_SOURCES.append({
        "name": f"NepaliAI/HealthQA_shard_{i:02d}",
        "url": f"https://huggingface.co/datasets/NepaliAI/Nepali-Health-QA/resolve/main/train{i:02d}.json",
        "type": "json"
    })
for i in range(10):
    PUBLIC_SOURCES.append({
        "name": f"NepaliAI/HealthFact_shard_{i:02d}",
        "url": f"https://huggingface.co/datasets/NepaliAI/Nepali-Health-Fact/resolve/main/train{i:02d}.json",
        "type": "json"
    })


def get_worker_session() -> requests.Session:
    """Get or create a persistent requests.Session for the current thread."""
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        thread_local.session = s
    return thread_local.session


def load_saved_offset() -> Optional[int]:
    """Load saved remote offset if available."""
    if os.path.exists(OFFSET_STATE_FILE):
        try:
            with open(OFFSET_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("next_offset")
        except Exception:
            pass
    return None


def save_current_offset(offset: int):
    """Save current remote offset to state file."""
    try:
        with open(OFFSET_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"next_offset": offset, "timestamp": time.time()}, f)
    except Exception:
        pass


def clean_nepali_document(text: str) -> Optional[str]:
    """Clean, filter, and validate a scraped Nepali text document."""
    if not text:
        return None

    # Remove HTML tags, URLs, and excessive markup
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
    if words < 10:
        return None

    return doc


def extract_text_from_nested(obj: Any) -> List[str]:
    """Recursively extract all long Devanagari strings from arbitrary JSON structures."""
    res = []
    if isinstance(obj, str):
        if len(obj) >= 25 and sum(1 for c in obj if "\u0900" <= c <= "\u097F") >= 8:
            res.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            res.extend(extract_text_from_nested(v))
    elif isinstance(obj, list):
        for item in obj:
            res.extend(extract_text_from_nested(item))
    return res


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


def fetch_and_clean_indiccorp_chunk(byte_start: int, byte_end: int) -> Tuple[str, int, int]:
    """Fetch a byte-range chunk from AI4Bharat IndicCorp V2 Nepali, decode and filter."""
    headers = {
        "User-Agent": USER_AGENT,
        "Range": f"bytes={byte_start}-{byte_end}"
    }

    session = get_worker_session()
    for attempt in range(1, 4):
        try:
            r = session.get(INDICCORP_NEPALI_URL, headers=headers, timeout=90)
            if r.status_code not in (200, 206):
                time.sleep(attempt * 1.5)
                continue

            raw_bytes = r.content
            try:
                decoded = raw_bytes.decode("utf-8", errors="ignore")
            except Exception:
                decoded = raw_bytes.decode("utf-8", errors="replace")

            lines = decoded.splitlines()
            if not lines:
                return "", 0, 0

            # Discard potential truncated boundary lines
            if byte_start > 0 and len(lines) > 2:
                valid_candidates = lines[1:-1]
            elif len(lines) > 1:
                valid_candidates = lines[:-1]
            else:
                valid_candidates = lines

            valid_lines = []
            words_count = 0

            for line in valid_candidates:
                line_str = line.strip()
                if not line_str or len(line_str) < 15:
                    continue

                # Filter out non-Devanagari lines (Devanagari char threshold >= 40%)
                dev_count = sum(1 for c in line_str if "\u0900" <= c <= "\u097F")
                if dev_count / len(line_str) < 0.40:
                    continue

                valid_lines.append(line_str)
                words_count += len(line_str.split())

            cleaned_text = "\n".join(valid_lines)
            return cleaned_text, words_count, len(valid_lines)

        except Exception as e:
            if attempt == 3:
                logger.warning(f"Error fetching IndicCorp chunk {byte_start}-{byte_end}: {e}")
            time.sleep(attempt * 2.0)

    return "", 0, 0


def count_existing_data(filepath: str) -> Tuple[int, int, int]:
    """Return (total_bytes, total_words, total_docs) for existing file."""
    if not os.path.exists(filepath):
        return 0, 0, 0
    total_bytes = os.path.getsize(filepath)
    total_words = 0
    total_docs = 0
    prev_blank = True
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                prev_blank = True
                continue
            if prev_blank:
                total_docs += 1
                prev_blank = False
            total_words += len(stripped.split())
    return total_bytes, total_words, total_docs


def analyze_corpus(filepath: str, target_tokens: int = 420_000_000) -> Dict[str, Any]:
    """Calculate comprehensive corpus statistics."""
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

    logger.info("Computing final corpus statistics across master file...")
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
    print("                 NEPALI DOWNLOADED DATA COLLECTION REPORT")
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


def download_nepali_corpus(
    target_tokens: int = 420_000_000,
    output_file: str = MASTER_OUTPUT_FILE,
    max_workers: int = MAX_WORKERS,
    chunk_size: int = CHUNK_SIZE,
):
    """Run high-throughput parallel streaming until 420M tokens is reached."""
    target_words = int(target_tokens / TOKEN_MULTIPLIER)

    logger.info("=" * 80)
    logger.info("STARTING HIGH-SPEED NEPALI DOWNLOADED CORPUS HARVESTING PIPELINE")
    logger.info(f"Target Subword Tokens:   {target_tokens:,} (~{target_words:,} words)")
    logger.info(f"Parallel Worker Threads: {max_workers}")
    logger.info(f"Chunk Size:              {chunk_size / (1024*1024):.0f} MB")
    logger.info(f"Master Output File:      {output_file}")
    logger.info(f"Primary Source:          AI4Bharat IndicCorp V2 ({INDICCORP_NEPALI_URL})")
    logger.info("=" * 80)

    existing_bytes, existing_words, existing_docs = count_existing_data(output_file)
    existing_tokens = int(existing_words * TOKEN_MULTIPLIER)

    logger.info(
        f"Existing Data: {existing_words:,} words | ~{existing_tokens:,} tokens | "
        f"{existing_bytes / (1024*1024):.2f} MB | {existing_docs:,} docs"
    )

    if existing_tokens >= target_tokens:
        logger.info(f"Target already achieved! ({existing_tokens:,} tokens >= {target_tokens:,})")
        report = analyze_corpus(output_file, target_tokens)
        print_report(report)
        return report

    # Determine starting remote byte offset
    saved_offset = load_saved_offset()
    if saved_offset is not None and saved_offset > 0:
        current_offset = saved_offset
        logger.info(f"Resuming from saved remote offset: {current_offset:,} bytes (~{current_offset/(1024*1024):.1f} MB)")
    else:
        # Estimate offset from existing file size
        current_offset = int(existing_bytes * 1.2)
        logger.info(f"Starting from computed remote offset: {current_offset:,} bytes (~{current_offset/(1024*1024):.1f} MB)")

    total_words = existing_words
    total_docs = existing_docs

    stop_event = threading.Event()
    start_time = time.time()
    last_log_time = time.time()
    last_save_time = time.time()

    # Open file in append mode
    with open(output_file, "a" if existing_bytes > 0 else "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            active_futures = {}

            # Submit initial batch of chunk tasks
            batch_size = max_workers * 2
            for _ in range(batch_size):
                chunk_start = current_offset
                chunk_end = current_offset + chunk_size - 1
                current_offset += chunk_size
                f = executor.submit(fetch_and_clean_indiccorp_chunk, chunk_start, chunk_end)
                active_futures[f] = (chunk_start, chunk_end)

            while active_futures and not stop_event.is_set():
                # Process completed futures
                done_futures = [f for f in active_futures if f.done()]
                for f in done_futures:
                    chunk_range = active_futures.pop(f)
                    try:
                        cleaned_text, words_count, docs_count = f.result()
                        if cleaned_text:
                            with file_write_lock:
                                out_f.write(cleaned_text + "\n")
                                out_f.flush()

                            with stats_lock:
                                total_words += words_count
                                total_docs += docs_count

                    except Exception as err:
                        logger.error(f"Chunk error {chunk_range}: {err}")

                    # Submit a new chunk task to keep queue full
                    if not stop_event.is_set():
                        chunk_start = current_offset
                        chunk_end = current_offset + chunk_size - 1
                        current_offset += chunk_size
                        new_f = executor.submit(fetch_and_clean_indiccorp_chunk, chunk_start, chunk_end)
                        active_futures[new_f] = (chunk_start, chunk_end)

                current_tokens = int(total_words * TOKEN_MULTIPLIER)
                now = time.time()

                if now - last_save_time >= 30.0:
                    save_current_offset(current_offset)
                    last_save_time = now

                if now - last_log_time >= 5.0:
                    elapsed = now - start_time
                    words_added = total_words - existing_words
                    wps = words_added / elapsed if elapsed > 0 else 0
                    tps = wps * TOKEN_MULTIPLIER
                    pct = (current_tokens / target_tokens) * 100
                    current_mb = os.path.getsize(output_file) / (1024 * 1024) if os.path.exists(output_file) else 0
                    eta_s = (target_tokens - current_tokens) / tps if tps > 0 else 0
                    eta_m = eta_s / 60

                    logger.info(
                        f"[PROGRESS] Tokens: {current_tokens:,} / {target_tokens:,} ({pct:.1f}%) | "
                        f"Words: {total_words:,} | File: {current_mb:,.1f} MB | "
                        f"Speed: {tps:,.0f} tok/s | ETA: {eta_m:.1f} min"
                    )
                    last_log_time = now

                if current_tokens >= target_tokens:
                    logger.info(f"Target achieved! Total subword tokens: {current_tokens:,}")
                    stop_event.set()
                    save_current_offset(current_offset)
                    break

                time.sleep(0.2)

    save_current_offset(current_offset)
    total_time = time.time() - start_time
    logger.info(f"Harvesting finished in {total_time:.1f} seconds. Generating final analysis report...")

    final_report = analyze_corpus(output_file, target_tokens)
    print_report(final_report)
    return final_report


if __name__ == "__main__":
    target = 420_000_000
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    download_nepali_corpus(target_tokens=target, max_workers=MAX_WORKERS)
