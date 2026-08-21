# Nepali Parquet Dataset Cleaning & 92/4/4 Partition Report

**Language:** Nepali (Devanagari script)  
**Format:** Apache Parquet (Snappy Compression)  
**Schema:** `[document_id, text, language, source_type, source]`  
**Generated:** 2026-08-21 19:41:08  
**Deterministic Seed:** 42  
**Token Multiplier:** 1.35x  

---

## 1. Corpus Summary & Compliance Overview

| Source Category | Parquet File | Rows | Clean Words | Est. Subword Tokens | Size | Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manual Cleaned** | `nepali_manual_clean.parquet` | 2,746,254 | 88,399,397 | 119,339,185 | 555.36 MB | **23.16%** |
| **Downloaded Cleaned** | `nepali_downloaded_clean.parquet` | 8,307,330 | 293,259,552 | 395,900,395 | 2030.47 MB | **76.84%** |
| **Total Clean Corpus** | **Combined** | **11,053,584** | **381,658,949** | **515,239,581 (515.24M)** | **2585.83 MB** | **100.00%** |

- **Assignment Requirement (Manual $\ge 20\%$):** **23.16% (PASS)**
- **Assignment Target Tokens ($\sim 500\text{M}$):** **515.24M Tokens (PASS)**
- **Exact Duplicates Removed:** **480,693** (SHA-256 deterministic global hash)
- **Near Duplicates Removed:** **122,252** (Punctuation-invariant + 8-word span matching)

---

## 2. Partition Splits (92% Train / 4% Val / 4% Test)

| Partition File | Target Ratio | Actual Ratio | Rows | Words | Estimated Tokens | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train.parquet`** | **92.0%** | **91.99%** | 10,167,950 | 351,106,424 | 469,134,008 (469.13M) | 2381.59 MB |
| **`val.parquet`** | **4.0%** | **4.00%** | 443,871 | 15,258,557 | 20,386,984 (20.39M) | 106.32 MB |
| **`test.parquet`** | **4.0%** | **4.01%** | 441,763 | 15,293,968 | 20,435,624 (20.44M) | 106.56 MB |
| **Total** | **100.0%** | **100.00%** | **11,053,584** | **381,658,949** | **515,239,581 (515.24M)** | **2594.47 MB** |

---

## 3. Data Integrity & Schema Verification
- **Schema Verified:** 5-column structure (`document_id`, `text`, `language`, `source_type`, `source`).
- **Unicode NFC Normalization:** Applied canonical composition and removed zero-width control chars.
- **Deduplication:** SHA-256 exact hash + 8-word span anchor near-duplicate elimination across all source files.
