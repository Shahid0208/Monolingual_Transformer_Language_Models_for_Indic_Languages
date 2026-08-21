# Phase 1 Technical Reports: Data Engineering, Partitioning & Tokenizer Training

This directory contains the complete technical reports and figures for Phase 1 of the Indic Monolingual Transformer Language Models project.

---

## Reports Index

1.  **Consolidated Technical Report:** [phase_1_report.md](phase_1_report.md)  
   *Consolidated cross-language report covering both Hindi and Nepali pipelines, compliance audits, comparative analyses, and embedded visualizations.*
2.  **Hindi (Model H — High-Resource) Report:** [hindi/report.md](../hindi/report.md)  
   *Dedicated report for Hindi: 547M token corpus, 23.09% manual share, 92/4/4 splits, and Unigram tokenizer evaluations (6K, 8K, 10K, 15K).*
3.  **Nepali (Model L — Low-Resource) Report:** [nepali/report.md](../nepali/report.md)  
   *Dedicated report for Nepali: 515M token corpus, 23.16% manual share, 92/4/4 splits, and Unigram tokenizer evaluations (6K, 8K, 10K, 15K).*

---

## Embedded Figures Index (`report/figures/`)

1. **Figure 1: Raw vs Cleaned Document Records:** `figures/fig1_raw_vs_clean_records.png`
2. **Figure 2: Raw vs Cleaned Corpus Storage Size (MB):** `figures/fig2_raw_vs_clean_size.png`
3. **Figure 3: Token Share Source Distribution (Manual vs Downloaded):** `figures/fig3_source_distribution.png`
4. **Figure 4: Train / Validation / Test Partition Distribution (92/4/4 Records):** `figures/fig4_train_val_test_distribution.png`
5. **Figure 5: Estimated Subword Tokens by Split:** `figures/fig5_tokens_by_split.png`
6. **Figure 6: Data Cleaning, Deduplication & Quality Filtering Breakdown:** `figures/fig6_cleaning_filtering_stats.png`
7. **Figure 7: Comprehensive Hindi vs Nepali Clean Corpus Comparison:** `figures/fig7_hindi_vs_nepali_corpus.png`
8. **Figure 8: Tokenizer Checkpoint Model Binary Size Comparison across Vocabularies:** `figures/fig8_tokenizer_vocab_comparison.png`
9. **Figure 9: Subword Tokenizer Compression Ratio & Fertility Dynamics:** `figures/fig9_fertility_and_compression.png`
