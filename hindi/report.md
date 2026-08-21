# Hindi (Model H — High-Resource) Phase 1 Technical Report
## Corpus Engineering, Partitioning & Subword Tokenizer Evaluation

**Language:** **Hindi** (`hi`)  
**Script:** Devanagari  
**Course:** Language Models and Agents (Monsoon 2026)  
**Assignment:** Individual Project — Building & Analyzing Monolingual Transformer LMs in Two Indian Languages  
**Branch:** `phase-1`

---

## 1. Corpus Engineering & Compliance Audit

The Hindi corpus was curated using a combination of manual scraping and filtered public corpora.

### A. Corpus Volume & Sourcing Verification
| Category | File | Raw Records | Clean Records | Clean Words | Est. Subword Tokens | Size (MB) | Token Share (%) | Compliance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Manual Scraping** | `data/cleaned/hindi_manual_clean.parquet` | 3,299,937 | 3,147,018 | 93,595,709 | 126,354,207 | 455.18 MB | **23.09%** | **PASS (≥ 20%)** |
| **Public / Downloaded** | `data/cleaned/hindi_downloaded_clean.parquet` | 5,416,157 | 5,366,147 | 311,677,182 | 420,764,195 | 1,607.75 MB | **76.91%** | Supporting Corpus |
| **Total Clean Corpus** | **Combined** | **8,716,094** | **8,513,165** | **405,272,891** | **547,118,402 (547.12M)** | **2,062.93 MB** | **100.00%** | **PASS (≥ 500M)** |

* **Total Clean Tokens:** **547.12M Tokens** (Target $\ge 500\text{M}$ tokens: **PASS**)
* **Manual Token Share:** **23.09%** (Requirement $\ge 20\%$: **PASS**)
* **Duplicates Filtered:** **77,254** (56,322 exact SHA-256 + 20,932 near-duplicate)
* **Noise / Boilerplate Filtered:** **125,675** low-quality lines

### B. Deterministic 92% / 4% / 4% Partition Splits
| Partition Split | File | Records (Rows) | Words | Estimated Subword Tokens | File Size | Partition Ratio (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Split** | `data/splits/train.parquet` | 7,831,689 | 372,946,150 | 499,715,727 (499.72M) | 1,900.10 MB | **92.02%** |
| **Validation Split** | `data/splits/val.parquet` | 341,528 | 16,104,546 | 21,577,262 (21.58M) | 84.24 MB | **3.97%** |
| **Test Split** | `data/splits/test.parquet` | 339,948 | 16,222,195 | 21,736,667 (21.74M) | 84.81 MB | **4.00%** |

---

## 2. Subword Tokenizer Training & Quantitative Evaluation

All tokenizers were trained **completely from scratch** on `data/splits/train.parquet` (7.83M rows / ~500M tokens) using SentencePiece Unigram.

### A. Special Token Schema
Standardized across all vocabulary sizes:
* `<unk>`: ID `0` (Unknown Token)
* `<s>`: ID `1` (Beginning of Sequence)
* `</s>`: ID `2` (End of Sequence)
* `<pad>`: ID `3` (Padding Token)

### B. Comparative Metrics Across Vocabulary Sizes (6K, 8K, 10K, 15K)

| Vocabulary Folder | Target Vocab | Actual Vocab | Fertility (Tokens / Word) | Chars / Token (Compression) | Tokens / Sentence | UNK Rate (%) | Model File Size | Vocab File Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [unigram_6k](file:///home/shahid/LMA_A1/individual-project-Shahid0208/hindi/tokenizer/unigram_6k) | 6,000 | 6,000 | **1.6775** | **3.7143** | 12.83 | **0.00%** | 155.3 KB | 160.3 KB |
| [unigram_8k](file:///home/shahid/LMA_A1/individual-project-Shahid0208/hindi/tokenizer/unigram_8k) | 8,000 | 8,000 | **1.5984** | **4.0857** | 11.67 | **0.00%** | 214.5 KB | 221.2 KB |
| [unigram_10k](file:///home/shahid/LMA_A1/individual-project-Shahid0208/hindi/tokenizer/unigram_10k) | 10,000 | 10,000 | **1.5382** | **4.3333** | 11.00 | **0.00%** | 273.9 KB | 282.3 KB |
| [unigram_15k](file:///home/shahid/LMA_A1/individual-project-Shahid0208/hindi/tokenizer/unigram_15k) | 15,000 | 15,000 | **1.4445** | **4.7667** | 10.00 | **0.00%** | 429.4 KB | 442.0 KB |

---

## 3. Hindi: Vocabulary Size vs Fertility (tokens/word)

![Hindi: Vocabulary Size vs Fertility](figures/hindi_vocab_vs_fertility.png)

### Fertility Analysis
1. **Monotonic Decrease in Fertility:** As vocabulary size scales from 6K to 15K, the average subword fertility decreases from **1.678 tokens/word** down to **1.445 tokens/word** (and down to **1.144 tokens/word** on large domain samples).
2. **Context Window Efficiency:** Lower fertility translates directly to fewer tokens per document, allowing the Transformer model to process longer context spans within a fixed context window length (e.g., 512 or 1,024 tokens) and reducing quadratic self-attention memory overhead ($O(N^2)$).
3. **Subword Granularity Trade-off:** While 15K provides the lowest fertility (1.445 tokens/word) and highest compression (4.77 chars/token), 8K and 10K provide strong morphological preservation of Devanagari inflections with minimal embedding table parameter overhead.

---

## 4. Token-Frequency & Subword Composition Statistics

### A. Subword Piece Length & Structural Distribution
| Vocabulary Size | Mean Piece Length (Chars) | Single-Character Base Pieces | Multi-Character Subwords | Character Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **6K** | 5.91 chars | 148 base pieces | 5,848 subwords | 99.95% + Fallback (0% UNK) |
| **8K** | 6.23 chars | 149 base pieces | 7,847 subwords | 99.95% + Fallback (0% UNK) |
| **10K** | 6.43 chars | 149 base pieces | 9,847 subwords | 99.95% + Fallback (0% UNK) |
| **15K** | 6.87 chars | 149 base pieces | 14,847 subwords | 99.95% + Fallback (0% UNK) |

### B. Top 10 Most Frequent Subwords & Unigram Log-Likelihoods
| Rank | `unigram_6k` Piece (Score) | `unigram_8k` Piece (Score) | `unigram_10k` Piece (Score) | `unigram_15k` Piece (Score) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `।` (-3.51) | `।` (-3.45) | `।` (-3.41) | `।` (-3.34) |
| **2** | `,` (-4.03) | `,` (-3.97) | `,` (-3.93) | `,` (-3.86) |
| **3** | `▁में` (-4.15) | `▁में` (-4.14) | `▁में` (-4.16) | `▁में` (-4.23) |
| **4** | `▁के` (-4.29) | `▁के` (-4.33) | `▁के` (-4.35) | `▁के` (-4.39) |
| **5** | `▁की` (-4.46) | `▁की` (-4.46) | `▁की` (-4.50) | `▁और` (-4.53) |
| **6** | `▁` (-4.55) | `▁` (-4.59) | `▁` (-4.61) | `▁की` (-4.55) |
| **7** | `▁को` (-4.59) | `▁को` (-4.61) | `▁और` (-4.62) | `▁` (-4.64) |
| **8** | `▁से` (-4.64) | `▁से` (-4.63) | `▁को` (-4.65) | `▁को` (-4.66) |
| **9** | `▁और` (-4.65) | `▁और` (-4.63) | `▁से` (-4.67) | `▁से` (-4.69) |
| **10** | `▁का` (-4.74) | `▁का` (-4.75) | `▁का` (-4.78) | `▁का` (-4.81) |

---

## 5. Qualitative Tokenization Examples Across Vocabulary Sizes

### Example 1: `भारत एक विविधताओं वाला देश है।`
* **6K Tokenizer (9 tokens):**
  `['▁भारत', '▁एक', '▁वि', 'विध', 'ताओं', '▁वाला', '▁देश', '▁है', '।']`
* **8K Tokenizer (9 tokens):**
  `['▁भारत', '▁एक', '▁वि', 'विध', 'ताओं', '▁वाला', '▁देश', '▁है', '।']`
* **10K Tokenizer (8 tokens):**
  `['▁भारत', '▁एक', '▁विविध', 'ताओं', '▁वाला', '▁देश', '▁है', '।']`
* **15K Tokenizer (8 tokens):**
  `['▁भारत', '▁एक', '▁विविध', 'ताओं', '▁वाला', '▁देश', '▁है', '।']`

### Example 2: `हिंदी भारत की प्रमुख भाषाओं में से एक है।`
* **6K Tokenizer (8 tokens):**
  `['▁हिंदी', '▁भारत▁की', '▁प्रमुख', '▁भाषा', 'ओं', '▁में▁से▁एक', '▁है', '।']`
* **8K Tokenizer (8 tokens):**
  `['▁हिंदी', '▁भारत▁की', '▁प्रमुख', '▁भाषा', 'ओं', '▁में▁से▁एक', '▁है', '।']`
* **10K Tokenizer (8 tokens):**
  `['▁हिंदी', '▁भारत▁की', '▁प्रमुख', '▁भाषा', 'ओं', '▁में▁से▁एक', '▁है', '।']`
* **15K Tokenizer (8 tokens):**
  `['▁हिंदी', '▁भारत▁की', '▁प्रमुख', '▁भाषा', 'ओं', '▁में▁से▁एक', '▁है', '।']`

### Example 3: `कृत्रिम बुद्धिमत्ता और मशीन लर्निंग आज महत्वपूर्ण तकनीकें हैं।`
* **6K Tokenizer (18 tokens):**
  `['▁', 'कृत', '्रि', 'म', '▁बुद्धि', 'म', 'त्ता', '▁और', '▁मशीन', '▁ल', 'र्न', 'िंग', '▁आज', '▁महत्वपूर्ण', '▁तकनीक', 'ें', '▁हैं', '।']`
* **8K Tokenizer (18 tokens):**
  `['▁', 'कृत', '्रि', 'म', '▁बुद्धि', 'म', 'त्ता', '▁और', '▁मशीन', '▁ल', 'र्न', 'िंग', '▁आज', '▁महत्वपूर्ण', '▁तकनीक', 'ें', '▁हैं', '।']`
* **10K Tokenizer (15 tokens):**
  `['▁कृत्रिम', '▁बुद्धि', 'म', 'त्ता', '▁और', '▁मशीन', '▁ल', 'र्न', 'िंग', '▁आज', '▁महत्वपूर्ण', '▁तकनीक', 'ें', '▁हैं', '।']`
* **15K Tokenizer (15 tokens):**
  `['▁कृत्रिम', '▁बुद्धि', 'म', 'त्ता', '▁और', '▁मशीन', '▁ल', 'र्न', 'िंग', '▁आज', '▁महत्वपूर्ण', '▁तकनीक', 'ें', '▁हैं', '।']`

---

## 6. Unknown Token (UNK) & Roundtrip Lossless Fidelity Analysis

* **Roundtrip Reconstruction:** Evaluated across all test sequences via `decode(encode(text)) == text`.
  * **Result: 100.00% PASS** for `unigram_6k`, `unigram_8k`, `unigram_10k`, and `unigram_15k`.
* **Unknown Token Rate:**
  * **0.00% UNK** rate across all evaluation samples.
  * Byte-fallback and Devanagari Unicode character pieces ensure zero out-of-vocabulary information loss.
