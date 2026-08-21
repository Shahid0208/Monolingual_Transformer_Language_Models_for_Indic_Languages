[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Monolingual Transformer Language Models for Indic Languages
**Course:** Language Models and Agents (Monsoon 2026)  
**Assignment:** Individual Project — Building & Analyzing Monolingual Transformer LMs in Two Indian Languages  
**Branch:** `phase-1`

---

## Google Drive Links for Datasets & Model Checkpoints

Per assignment instructions, large binary datasets (Parquet files) and trained tokenizer checkpoints (`.model` and `.vocab` files) are organized with strict language separation:

### A. Hindi (Model H — High-Resource)
* **Hindi Clean Datasets & Splits (Parquet):** [Google Drive — Hindi Clean Parquet & Splits](https://drive.google.com/drive/folders/placeholder_hindi_datasets)
  * `hindi_downloaded_clean.parquet` (1,607.75 MB, 5,366,147 rows)
  * `hindi_manual_clean.parquet` (455.18 MB, 3,147,018 rows)
  * `splits/train.parquet` (1,900.10 MB, 7,831,689 rows)
  * `splits/val.parquet` (84.24 MB, 341,528 rows)
  * `splits/test.parquet` (84.81 MB, 339,948 rows)
* **Hindi Tokenizer Checkpoints (SentencePiece Unigram):** [Google Drive — Hindi Tokenizer Checkpoints](https://drive.google.com/drive/folders/placeholder_hindi_tokenizers)
  * `unigram_6k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)
  * `unigram_8k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)
  * `unigram_10k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)
  * `unigram_15k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)

### B. Nepali (Model L — Low-Resource)
* **Nepali Clean Datasets & Splits (Parquet):** [Google Drive — Nepali Clean Parquet & Splits](https://drive.google.com/drive/folders/placeholder_nepali_datasets)
  * `nepali_downloaded_clean.parquet` (2,030.47 MB, 8,307,330 rows)
  * `nepali_manual_clean.parquet` (555.36 MB, 2,746,254 rows)
  * `splits/train.parquet` (2,381.59 MB, 10,167,950 rows)
  * `splits/val.parquet` (106.32 MB, 443,871 rows)
  * `splits/test.parquet` (106.56 MB, 441,763 rows)
* **Nepali Tokenizer Checkpoints (SentencePiece Unigram):** [Google Drive — Nepali Tokenizer Checkpoints](https://drive.google.com/drive/folders/placeholder_nepali_tokenizers)
  * `unigram_6k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)
  * `unigram_8k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)
  * `unigram_10k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)
  * `unigram_15k/` (`tokenizer.model`, `tokenizer.vocab`, `statistics.json`)

---

## 1. Project Overview & Compliance Summary

This repository implements the end-to-end data engineering, preprocessing, deduplication, 92/4/4 partitioning, and from-scratch Unigram subword tokenizer training for two completely independent monolingual language models:
- **Model H (High-Resource):** **Hindi** (`hi`) — Devanagari script
- **Model L (Low-Resource):** **Nepali** (`ne`) — Devanagari script (Selected from the approved lower-resource language list)

### Core Constraints Verification Matrix
| Requirement | Hindi (Model H) | Nepali (Model L) | Status |
| :--- | :--- | :--- | :--- |
| **Total Volume Target** ($\ge 500\text{M}$ tokens) | **547.12M Tokens** | **515.24M Tokens** | **PASS** |
| **Manual Sourcing Share** ($\ge 20\%$ tokens) | **23.09%** (126.35M tokens) | **23.16%** (119.34M tokens) | **PASS** |
| **Data Independence** | 100% Isolated Pipeline & Corpora | 100% Isolated Pipeline & Corpora | **PASS** |
| **Deterministic Splits (92/4/4)** | 92.02% Train / 3.97% Val / 4.00% Test | 91.99% Train / 4.00% Val / 4.01% Test | **PASS** |
| **Trained from Scratch (No Pretrained)** | SentencePiece Unigram from scratch | SentencePiece Unigram from scratch | **PASS** |
| **Special Token Schema** | `<unk>`: 0, `<s>`: 1, `</s>`: 2, `<pad>`: 3 | `<unk>`: 0, `<s>`: 1, `</s>`: 2, `<pad>`: 3 | **PASS** |
| **Lossless Roundtrip Reconstruction** | `decode(encode(x)) == x` (100%) | `decode(encode(x)) == x` (100%) | **PASS** |
| **Unknown Token Rate (Devanagari)** | **0.00%** | **0.00%** | **PASS** |

---

## 2. Data Collection, Conversion & Pipeline Lifecycle

The data ingestion and engineering workflow was executed through the following structured stages:

1. **Manual Scraping & Collection ($\ge 20\%$ of Total Tokens):**
   - Manually scraped and collected raw text datasets accounting for **>20%** of the total volume for both Hindi (23.09%) and Nepali (23.16%).
   - Sources include national and regional news portals, literary repositories, Wikipedia dumps, blogs, and government/cultural publications.
2. **Downloaded Corpora Collection (~80% of Total Tokens):**
   - Downloaded the remaining **~80%** of raw data in raw text (`.txt`) format from approved large-scale public Indic corpora (e.g., IndicCorp v2 / OSCAR / CC100).
3. **Conversion from Raw Text (`.txt`) to Standardized Parquet:**
   - Transformed all raw text documents into a unified, high-performance 5-column Apache Parquet format (`document_id`, `text`, `language`, `source_type`, `source`) with Snappy compression using `scripts/convert_raw_to_parquet.py`.
4. **Multiprocessing Cleaning & Multi-Level Deduplication:**
   - Filtered noise, stripped control/zero-width characters, applied Unicode NFC normalization, enforced Devanagari script purity ($\ge 45\%$), and performed 64-bit integer exact hashing and near-duplicate span filtering via `scripts/run_full_parquet_pipeline.py`.
5. **Deterministic 92/4/4 Splitting:**
   - Partitioned clean Parquet datasets deterministically with fixed random seed `42` into `train.parquet` (92%), `val.parquet` (4%), and `test.parquet` (4%).
6. **Unified Training & Evaluation Lifecycle:**
   - The resulting `train.parquet` dataset was used directly to train the SentencePiece Unigram tokenizers from scratch.
   - The exact same `train.parquet`, `val.parquet`, and `test.parquet` partitions are used for model pretraining, validation monitoring, and perplexity evaluation.

---

## 3. Repository Structure

```
individual-project-Shahid0208/
├── README.md                                   # Comprehensive project documentation, dataset & tokenizer reports
├── report.md                                   # Phase 1 Technical Grading Report
├── scripts/
│   ├── run_full_parquet_pipeline.py            # 16-core multiprocessing cleaning, dedup & 92/4/4 splitting pipeline
│   └── convert_raw_to_parquet.py               # Raw text to 5-column Parquet converter
├── hindi/                                      # Hindi (Model H) pipeline, datasets & tokenizers
│   ├── scrapers/
│   │   ├── manual_hindi_collector.py           # Manual collector (news, literature, portals)
│   │   └── download_indiccorp.py               # IndicCorp v2 / public corpus stream downloader
│   ├── data/
│   │   ├── raw/                                # Enriched 5-column raw Parquet files
│   │   │   ├── hindi_manual_raw.parquet        # 3,299,937 rows (460.80 MB)
│   │   │   └── hindi_downloaded_raw.parquet    # 5,416,157 rows (1,614.56 MB)
│   │   ├── cleaned/                            # Quality filtered & deduplicated Parquet files
│   │   │   ├── hindi_manual_clean.parquet      # 3,147,018 rows (455.18 MB)
│   │   │   ├── hindi_downloaded_clean.parquet  # 5,366,147 rows (1,607.75 MB)
│   │   │   ├── dataset_statistics.json         # Machine-readable corpus statistics
│   │   │   └── dataset_statistics.md           # Markdown corpus audit report
│   │   └── splits/                             # Deterministic 92/4/4 train/val/test splits
│   │       ├── train.parquet                   # 7,831,689 rows (92.02%, 1,900.10 MB, ~499.72M tokens)
│   │       ├── val.parquet                     # 341,528 rows (3.97%, 84.24 MB, ~21.58M tokens)
│   │       └── test.parquet                    # 339,948 rows (4.00%, 84.81 MB, ~21.74M tokens)
│   └── tokenizer/                              # Hindi Unigram tokenizers (6K, 8K, 10K, 15K)
│       ├── train_unigram.py                    # Unigram training script from scratch
│       ├── tokenizer_comparison.md             # Detailed comparison report across vocabulary sizes
│       ├── unigram_6k/                         # tokenizer.model, tokenizer.vocab, statistics.json
│       ├── unigram_8k/                         # tokenizer.model, tokenizer.vocab, statistics.json
│       ├── unigram_10k/                        # tokenizer.model, tokenizer.vocab, statistics.json
│       └── unigram_15k/                        # tokenizer.model, tokenizer.vocab, statistics.json
└── nepali/                                     # Nepali (Model L) pipeline, datasets & tokenizers
    ├── scrapers/
    │   ├── manual_nepali_collector.py          # Manual collector (Kantipur, Setopati, archives)
    │   └── download_nepali_corpus.py           # IndicCorp v2 / OSCAR Nepali downloader
    ├── data/
    │   ├── raw/                                # Enriched 5-column raw Parquet files
    │   │   ├── nepali_manual_raw.parquet       # 2,949,145 rows (566.51 MB)
    │   │   └── nepali_downloaded_raw.parquet   # 8,978,234 rows (2,154.67 MB)
    │   ├── cleaned/                            # Quality filtered & deduplicated Parquet files
    │   │   ├── nepali_manual_clean.parquet     # 2,746,254 rows (555.36 MB)
    │   │   ├── nepali_downloaded_clean.parquet # 8,307,330 rows (2,030.47 MB)
    │   │   ├── dataset_statistics.json         # Machine-readable corpus statistics
    │   │   └── dataset_statistics.md           # Markdown corpus audit report
    │   └── splits/                             # Deterministic 92/4/4 train/val/test splits
    │       ├── train.parquet                   # 10,167,950 rows (91.99%, 2,381.59 MB, ~469.13M tokens)
    │       ├── val.parquet                     # 443,871 rows (4.00%, 106.32 MB, ~20.39M tokens)
    │       └── test.parquet                    # 441,763 rows (4.01%, 106.56 MB, ~20.44M tokens)
    └── tokenizer/                              # Nepali Unigram tokenizers (6K, 8K, 10K, 15K)
        ├── train_unigram.py                    # Unigram training script from scratch
        ├── tokenizer_comparison.md             # Detailed comparison report across vocabulary sizes
        ├── unigram_6k/                         # tokenizer.model, tokenizer.vocab, statistics.json
        ├── unigram_8k/                         # tokenizer.model, tokenizer.vocab, statistics.json
        ├── unigram_10k/                        # tokenizer.model, tokenizer.vocab, statistics.json
        └── unigram_15k/                        # tokenizer.model, tokenizer.vocab, statistics.json
```

---

## 4. Standardized 5-Column Parquet Schema

All datasets (raw, cleaned, and splits) strictly adhere to an Apache Parquet schema compressed with Snappy:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `document_id` | `pa.string()` | Unique deterministic identifier (e.g. `hi_m_00000001`, `ne_d_00000001`) |
| `text` | `pa.string()` | Document text payload |
| `language` | `pa.string()` | Language identifier (`hindi` / `nepali`) |
| `source_type` | `pa.string()` | Origin category (`manual` / `downloaded`) |
| `source` | `pa.string()` | Explicit source identifier (e.g., `wikipedia_news_manual`, `indiccorp_v2`) |

---

## 5. Cleaning, Filtering & Deduplication Pipeline

The pipeline (`scripts/run_full_parquet_pipeline.py`) applies a high-speed multi-stage filter across 16 worker processes:

1. **Unicode NFC Normalization & Control Character Stripping**:
   * Applies `unicodedata.normalize("NFC", text)`.
   * Strips zero-width invisible and directionality characters (`\u200B-\u200D`, `\uFEFF`, `\u00A0`, `\u200E\u200F`, `\u202A-\u202E`).
   * Eliminates ASCII/control codes (`\x00-\x08`, `\x0B\x0C`, `\x0E-\x1F`, `\x7F`).
2. **Noise, HTML, and Boilerplate Removal**:
   * Removes HTML/XML tags (`<[^>]+>`), web URLs (`https?://\S+`), and email patterns.
   * Collapses redundant spaces, tabs, and newlines.
3. **Punctuation Normalization**:
   * Collapses repetitive punctuation runs while explicitly preserving Indic danda (`।`) and double danda (`॥`).
4. **Length & Quality Filtering**:
   * Discards lines with character length $< 15$ or word count $< 4$.
   * Filters out documents where symbol/noise density $> 30\%$.
5. **Language Purity & Script Verification**:
   * Enforces Devanagari character purity $\ge 45\%$ (`[\u0900-\u097F]`).
6. **Streaming Multi-Level Deduplication**:
   * **Exact Deduplication:** Deterministic 64-bit integer hashing (truncated SHA-256) tracked globally across manual and downloaded sources.
   * **Near-Duplicate Removal:** Punctuation-invariant normalization hash + 8-word prefix/suffix anchor span matching.
7. **Deterministic 92/4/4 Partitioning**:
   * Partitions clean documents into `train.parquet` (92%), `val.parquet` (4%), and `test.parquet` (4%) with deterministic seed `42`.

---

## 6. Dataset Statistics & Splits

### A. Hindi (Model H — High-Resource)

#### Corpus Composition & Compliance
| Category | File | Raw Rows | Clean Rows | Clean Words | Est. Subword Tokens | Size | Token Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manual** | `hindi_manual_clean.parquet` | 3,299,937 | 3,147,018 | 93,595,709 | 126,354,207 | 455.18 MB | **23.09%** |
| **Downloaded** | `hindi_downloaded_clean.parquet` | 5,416,157 | 5,366,147 | 311,677,182 | 420,764,195 | 1,607.75 MB | **76.91%** |
| **Total Clean** | **Combined** | **8,716,094** | **8,513,165** | **405,272,891** | **547,118,402 (547.12M)** | **2,062.93 MB** | **100.00%** |

* **Manual Token Share Requirement ($\ge 20\%$):** **23.09% (PASS )**
* **Target Volume ($\sim 500\text{M}$ tokens):** **547.12M Tokens (PASS )**
* **Duplicates Removed:** **77,254** (56,322 exact SHA-256 + 20,932 near-duplicate)

#### Deterministic Splits (92% Train / 4% Val / 4% Test)
| Split Partition | File | Rows | Words | Estimated Tokens | File Size | Ratio (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `hindi/data/splits/train.parquet` | 7,831,689 | 372,946,150 | 499,715,727 (499.72M) | 1,900.10 MB | **92.02%** |
| **Validation** | `hindi/data/splits/val.parquet` | 341,528 | 16,104,546 | 21,577,262 (21.58M) | 84.24 MB | **3.97%** |
| **Test** | `hindi/data/splits/test.parquet` | 339,948 | 16,222,195 | 21,736,667 (21.74M) | 84.81 MB | **4.00%** |

---

### B. Nepali (Model L — Low-Resource)

#### Corpus Composition & Compliance
| Category | File | Raw Rows | Clean Rows | Clean Words | Est. Subword Tokens | Size | Token Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manual** | `nepali_manual_clean.parquet` | 2,949,145 | 2,746,254 | 88,399,397 | 119,339,185 | 555.36 MB | **23.16%** |
| **Downloaded** | `nepali_downloaded_clean.parquet` | 8,978,234 | 8,307,330 | 293,259,552 | 395,900,395 | 2,030.47 MB | **76.84%** |
| **Total Clean** | **Combined** | **11,927,379** | **11,053,584** | **381,658,949** | **515,239,581 (515.24M)** | **2,585.83 MB** | **100.00%** |

* **Manual Token Share Requirement ($\ge 20\%$):** **23.16% (PASS )**
* **Target Volume ($\sim 500\text{M}$ tokens):** **515.24M Tokens (PASS )**
* **Duplicates Removed:** **602,945** (480,693 exact SHA-256 + 122,252 near-duplicate)

#### Deterministic Splits (92% Train / 4% Val / 4% Test)
| Split Partition | File | Rows | Words | Estimated Tokens | File Size | Ratio (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `nepali/data/splits/train.parquet` | 10,167,950 | 351,106,424 | 469,134,008 (469.13M) | 2,381.59 MB | **91.99%** |
| **Validation** | `nepali/data/splits/val.parquet` | 443,871 | 15,258,557 | 20,386,984 (20.39M) | 106.32 MB | **4.00%** |
| **Test** | `nepali/data/splits/test.parquet` | 441,763 | 15,293,968 | 20,435,624 (20.44M) | 106.56 MB | **4.01%** |

---

## 7. Tokenizer Training, Evaluation & Comparative Analysis

All tokenizers were trained **completely from scratch** using the SentencePiece Unigram algorithm on the respective full training corpora (`train.parquet`) with standardized special token schema:
- `<unk>`: `0` (Unknown Token)
- `<s>`: `1` (Beginning of Sentence)
- `</s>`: `2` (End of Sentence)
- `<pad>`: `3` (Padding Token)

---

### A. Hindi (Model H — High-Resource) Tokenizer Report

#### 1. Summary Metrics Across Vocabulary Sizes
| Vocabulary | Requested Vocab | Actual Vocab | Chars / Token (Compression) | Avg Tokens / Sentence | UNK Rate (%) | Model File Size | Vocab File Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`unigram_6k`** | 6,000 | 6,000 | **3.7143** | 12.83 | **0.00%** | 155.3 KB | 160.3 KB |
| **`unigram_8k`** | 8,000 | 8,000 | **4.0857** | 11.67 | **0.00%** | 214.5 KB | 221.2 KB |
| **`unigram_10k`** | 10,000 | 10,000 | **4.3333** | 11.00 | **0.00%** | 273.9 KB | 282.3 KB |
| **`unigram_15k`** | 15,000 | 15,000 | **4.7667** | 10.00 | **0.00%** | 429.4 KB | 442.0 KB |

#### 2. Token-Frequency & Subword Composition Statistics
| Vocabulary | Avg Piece Length | Single-Char Base Pieces | Multi-Char Subwords | Top 5 Most Frequent Subwords (Unigram Log-Likelihood) |
| :--- | :--- | :--- | :--- | :--- |
| **6K** | 5.91 chars | 148 | 5,848 | `।` (-3.51), `,` (-4.03), `▁में` (-4.15), `▁के` (-4.29), `▁की` (-4.46) |
| **8K** | 6.23 chars | 149 | 7,847 | `।` (-3.45), `,` (-3.97), `▁में` (-4.14), `▁के` (-4.33), `▁की` (-4.46) |
| **10K** | 6.43 chars | 149 | 9,847 | `।` (-3.41), `,` (-3.93), `▁में` (-4.16), `▁के` (-4.35), `▁की` (-4.50) |
| **15K** | 6.87 chars | 149 | 14,847 | `।` (-3.34), `,` (-3.86), `▁में` (-4.23), `▁के` (-4.39), `▁और` (-4.53) |

#### 3. Tokenization Examples Across Vocabularies
* **Sentence 1:** `भारत एक विविधताओं वाला देश है।`
  * **6K (9 tokens):** `▁भारत` `▁एक` `▁वि` `विध` `ताओं` `▁वाला` `▁देश` `▁है` `।`
  * **8K (9 tokens):** `▁भारत` `▁एक` `▁वि` `विध` `ताओं` `▁वाला` `▁देश` `▁है` `।`
  * **10K (8 tokens):** `▁भारत` `▁एक` `▁विविध` `ताओं` `▁वाला` `▁देश` `▁है` `।`
  * **15K (8 tokens):** `▁भारत` `▁एक` `▁विविध` `ताओं` `▁वाला` `▁देश` `▁है` `।`
* **Sentence 2:** `कृत्रिम बुद्धिमत्ता और मशीन लर्निंग आज महत्वपूर्ण तकनीकें हैं।`
  * **6K (18 tokens):** `▁` `कृत` `्रि` `म` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।`
  * **8K (18 tokens):** `▁` `कृत` `्रि` `म` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।`
  * **10K (15 tokens):** `▁कृत्रिम` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।`
  * **15K (15 tokens):** `▁कृत्रिम` `▁बुद्धि` `म` `त्ता` `▁और` `▁मशीन` `▁ल` `र्न` `िंग` `▁आज` `▁महत्वपूर्ण` `▁तकनीक` `ें` `▁हैं` `।`

---

### B. Nepali (Model L — Low-Resource) Tokenizer Report

#### 1. Summary Metrics Across Vocabulary Sizes
| Vocabulary | Requested Vocab | Actual Vocab | Chars / Token (Compression) | Avg Tokens / Sentence | UNK Rate (%) | Model File Size | Vocab File Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`unigram_6k`** | 6,000 | 6,000 | **4.6545** | 9.17 | **0.00%** | 159.9 KB | 164.9 KB |
| **`unigram_8k`** | 8,000 | 8,000 | **5.1200** | 8.33 | **0.00%** | 220.1 KB | 226.7 KB |
| **`unigram_10k`** | 10,000 | 10,000 | **5.1200** | 8.33 | **0.00%** | 280.7 KB | 289.1 KB |
| **`unigram_15k`** | 15,000 | 15,000 | **5.3333** | 8.00 | **0.00%** | 438.5 KB | 451.2 KB |

#### 2. Token-Frequency & Subword Composition Statistics
| Vocabulary | Avg Piece Length | Single-Char Base Pieces | Multi-Char Subwords | Top 5 Most Frequent Subwords (Unigram Log-Likelihood) |
| :--- | :--- | :--- | :--- | :--- |
| **6K** | 6.11 chars | 136 | 5,860 | `▁।` (-3.77), `को` (-3.78), `मा` (-3.80), `,` (-4.04), `ले` (-4.25) |
| **8K** | 6.41 chars | 136 | 7,860 | `▁।` (-3.71), `को` (-3.80), `मा` (-3.84), `,` (-3.99), `ले` (-4.28) |
| **10K** | 6.61 chars | 136 | 9,860 | `▁।` (-3.67), `को` (-3.85), `मा` (-3.87), `,` (-3.95), `।` (-4.28) |
| **15K** | 7.01 chars | 136 | 14,860 | `▁।` (-3.60), `,` (-3.88), `को` (-3.94), `मा` (-3.98), `।` (-4.22) |

#### 3. Tokenization Examples Across Vocabularies
* **Sentence 1:** `नेपाल एक सुन्दर र शान्त देश हो।`
  * **6K (8 tokens):** `▁नेपाल` `▁एक` `▁सुन्दर` `▁र` `▁शान्त` `▁देश` `▁हो` `।`
  * **8K (8 tokens):** `▁नेपाल` `▁एक` `▁सुन्दर` `▁र` `▁शान्त` `▁देश` `▁हो` `।`
  * **10K (8 tokens):** `▁नेपाल` `▁एक` `▁सुन्दर` `▁र` `▁शान्त` `▁देश` `▁हो` `।`
  * **15K (8 tokens):** `▁नेपाल` `▁एक` `▁सुन्दर` `▁र` `▁शान्त` `▁देश` `▁हो` `।`
* **Sentence 2:** `कृत्रिम बौद्धिकता र प्राकृतिक भाषा प्रशोधन आजका महत्त्वपूर्ण प्रविधिहरू हुन्।`
  * **6K (15 tokens):** `▁कृत्रिम` `▁बौद्ध` `िक` `ता` `▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।`
  * **8K (14 tokens):** `▁कृत्रिम` `▁बौद्धिक` `ता` `▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।`
  * **10K (14 tokens):** `▁कृत्रिम` `▁बौद्धिक` `ता` `▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।`
  * **15K (13 tokens):** `▁कृत्रिम` `▁बौद्धिक` `ता▁र` `▁प्राकृतिक` `▁भाषा` `▁प्रशोधन` `▁आज` `का` `▁महत्त्वपूर्ण` `▁प्रविधि` `हरू` `▁हुन्` `।`

---

## 8. Pipeline Reproduction Instructions

### 1. Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate
pip install pyarrow sentencepiece requests beautifulsoup4 aiohttp tqdm
```

### 2. High-Performance Parquet Cleaning, Deduplication & Splitting
```bash
# Run both Hindi & Nepali data pipelines in parallel (16 cores)
python3 scripts/run_full_parquet_pipeline.py --lang all
```

### 3. Tokenizer Training (Completely From Scratch)
```bash
# Train Hindi Unigram tokenizers (6K, 8K, 10K, 15K)
python3 hindi/tokenizer/train_unigram.py

# Train Nepali Unigram tokenizers (6K, 8K, 10K, 15K)
python3 nepali/tokenizer/train_unigram.py
```
