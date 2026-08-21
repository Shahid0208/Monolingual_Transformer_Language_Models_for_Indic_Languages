# Hindi Parquet Dataset Cleaning & 92/4/4 Partition Report

**Language:** Hindi (Devanagari script)  
**Format:** Apache Parquet (Snappy Compression)  
**Schema:** `[document_id, text, language, source_type, source]`  
**Generated:** 2026-08-21 19:21:31  
**Deterministic Seed:** 42  
**Token Multiplier:** 1.35x  

---

## 1. Corpus Summary & Compliance Overview

| Source Category | Parquet File | Rows | Clean Words | Est. Subword Tokens | Size | Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manual Cleaned** | `hindi_manual_clean.parquet` | 3,147,018 | 93,595,709 | 126,354,207 | 455.18 MB | **23.09%** |
| **Downloaded Cleaned** | `hindi_downloaded_clean.parquet` | 5,366,147 | 311,677,182 | 420,764,195 | 1607.75 MB | **76.91%** |
| **Total Clean Corpus** | **Combined** | **8,513,165** | **405,272,891** | **547,118,402 (547.12M)** | **2062.93 MB** | **100.00%** |

- **Assignment Requirement (Manual $\ge 20\%$):** **23.09% (PASS)**
- **Assignment Target Tokens ($\sim 500\text{M}$):** **547.12M Tokens (PASS)**
- **Exact Duplicates Removed:** **56,322** (SHA-256 deterministic global hash)
- **Near Duplicates Removed:** **20,932** (Punctuation-invariant + 8-word span matching)

---

## 2. Partition Splits (92% Train / 4% Val / 4% Test)

| Partition File | Target Ratio | Actual Ratio | Rows | Words | Estimated Tokens | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train.parquet`** | **92.0%** | **92.02%** | 7,831,689 | 372,946,150 | 499,715,727 (499.72M) | 1900.10 MB |
| **`val.parquet`** | **4.0%** | **3.97%** | 341,528 | 16,104,546 | 21,577,262 (21.58M) | 84.24 MB |
| **`test.parquet`** | **4.0%** | **4.00%** | 339,948 | 16,222,195 | 21,736,667 (21.74M) | 84.81 MB |
| **Total** | **100.0%** | **100.00%** | **8,513,165** | **405,272,891** | **547,118,402 (547.12M)** | **2069.15 MB** |

---

## 3. Data Integrity & Schema Verification
- **Schema Verified:** 5-column structure (`document_id`, `text`, `language`, `source_type`, `source`).
- **Unicode NFC Normalization:** Applied canonical composition and removed zero-width control chars.
- **Deduplication:** SHA-256 exact hash + 8-word span anchor near-duplicate elimination across all source files.
