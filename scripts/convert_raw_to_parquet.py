"""Stream convert raw Hindi and Nepali text files into Parquet format with Snappy compression.

Preserves original .txt files and creates:
  - hindi/data/raw/hindi_manual_raw.parquet
  - hindi/data/raw/hindi_downloaded_raw.parquet
  - nepali/data/raw/nepali_manual_raw.parquet
  - nepali/data/raw/nepali_downloaded_raw.parquet
"""

import os
import sys
import time
import logging

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: PyArrow is not installed. Please install pyarrow first.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ParquetConverter")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FILES_TO_CONVERT = [
    {
        "name": "Hindi Manual Raw",
        "txt": os.path.join(BASE_DIR, "hindi/data/raw/hindi_manual_raw.txt"),
        "parquet": os.path.join(BASE_DIR, "hindi/data/raw/hindi_manual_raw.parquet"),
    },
    {
        "name": "Hindi Downloaded Raw",
        "txt": os.path.join(BASE_DIR, "hindi/data/raw/hindi_downloaded_raw.txt"),
        "parquet": os.path.join(BASE_DIR, "hindi/data/raw/hindi_downloaded_raw.parquet"),
    },
    {
        "name": "Nepali Manual Raw",
        "txt": os.path.join(BASE_DIR, "nepali/data/raw/nepali_manual_raw.txt"),
        "parquet": os.path.join(BASE_DIR, "nepali/data/raw/nepali_manual_raw.parquet"),
    },
    {
        "name": "Nepali Downloaded Raw",
        "txt": os.path.join(BASE_DIR, "nepali/data/raw/nepali_downloaded_raw.txt"),
        "parquet": os.path.join(BASE_DIR, "nepali/data/raw/nepali_downloaded_raw.parquet"),
    },
]


def convert_txt_to_parquet(txt_path: str, parquet_path: str, batch_size: int = 100000):
    if not os.path.exists(txt_path):
        logger.warning(f"File not found: {txt_path}. Skipping.")
        return

    logger.info(f"Converting: {os.path.basename(txt_path)} -> {os.path.basename(parquet_path)}")
    start_time = time.time()

    schema = pa.schema([pa.field("text", pa.string())])
    total_lines = 0
    total_chars = 0

    with open(txt_path, "r", encoding="utf-8", errors="replace", buffering=32*1024*1024) as in_f, \
         pq.ParquetWriter(parquet_path, schema, compression="snappy") as writer:

        batch = []
        for line in in_f:
            line_str = line.rstrip("\r\n")
            if line_str:
                batch.append(line_str)
                total_lines += 1
                total_chars += len(line_str)

                if len(batch) >= batch_size:
                    table = pa.Table.from_arrays([pa.array(batch)], schema=schema)
                    writer.write_table(table)
                    batch = []

                    if total_lines % 500000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"  Processed {total_lines:,} lines ({total_lines/elapsed:,.0f} lines/sec)...")

        if batch:
            table = pa.Table.from_arrays([pa.array(batch)], schema=schema)
            writer.write_table(table)

    elapsed = round(time.time() - start_time, 2)
    txt_mb = round(os.path.getsize(txt_path) / (1024 * 1024), 2)
    parquet_mb = round(os.path.getsize(parquet_path) / (1024 * 1024), 2)
    compression_ratio = round((1 - parquet_mb / txt_mb) * 100, 1) if txt_mb > 0 else 0

    logger.info(
        f"Completed {os.path.basename(parquet_path)} in {elapsed}s: "
        f"{total_lines:,} rows | TXT: {txt_mb} MB -> Parquet: {parquet_mb} MB (Saved {compression_ratio}%)"
    )


def main():
    logger.info("=" * 80)
    logger.info("STARTING RAW TEXT TO PARQUET CONVERSION PIPELINE")
    logger.info("=" * 80)
    start_all = time.time()

    for item in FILES_TO_CONVERT:
        convert_txt_to_parquet(item["txt"], item["parquet"])

    total_time = round(time.time() - start_all, 2)
    logger.info(f"All Parquet conversions completed in {total_time}s!")


if __name__ == "__main__":
    main()
