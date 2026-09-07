# Project Technical Reports Index

This directory contains the complete technical reports and figures for the Indic Monolingual Transformer Language Models project.

---

## Reports Index

1. **Phase 2 Technical Report (Pretraining & Evaluation):** [phase_2_report.md](phase_2_report.md)  
   *Comprehensive report for Phase 2: custom Transformer architecture, 30,518 pretraining steps on 500M tokens, intrinsic evaluation (Loss, PPL, BPB), multi-temperature generation suite (BLEU-4, chrF++, ROUGE-L, Distinct-1/2, Repetition), multi-head attention diagnostics (heatmaps, entropy, distance), and positional embedding ablation study.*

2. **Phase 1 Consolidated Technical Report:** [phase_1_report.md](phase_1_report.md)  
   *Consolidated cross-language report covering both Hindi and Nepali pipelines, compliance audits, comparative analyses, and embedded visualizations.*

3. **Hindi (Model H — High-Resource) Phase 1 Report:** [hindi/report.md](../hindi/report.md)  
   *Dedicated report for Hindi: 547M token corpus, 23.09% manual share, 92/4/4 splits, and Unigram tokenizer evaluations (6K, 8K, 10K, 15K).*

4. **Nepali (Model L — Low-Resource) Phase 1 Report:** [nepali/report.md](../nepali/report.md)  
   *Dedicated report for Nepali: 515M token corpus, 23.16% manual share, 92/4/4 splits, and Unigram tokenizer evaluations (6K, 8K, 10K, 15K).*

---

## Phase 2 Embedded Figures Index (`report/figures/`)

1. **Figure 1: Pretraining Loss & Perplexity Convergence (30,518 Steps):** `figures/fig1_pretraining_loss_ppl.png`
2. **Figure 2: Cosine Learning Rate Schedule & Training Throughput:** `figures/fig2_lr_and_throughput.png`
3. **Figure 3: Intrinsic Evaluation — Held-Out Test Perplexity & Bits-per-Byte:** `figures/fig3_intrinsic_evaluation.png`
4. **Figure 4: Generation Quality Metrics Across Temperatures (BLEU-4, chrF++, ROUGE-L):** `figures/fig4_generation_metrics.png`
5. **Figure 5: Lexical Diversity & Trigram Repetition Rate Diagnostics:** `figures/fig5_diversity_repetition.png`
6. **Figure 6: Multi-Head Attention Entropy Across Transformer Depth:** `figures/fig6_attention_entropy.png`
7. **Figure 7: Mean Attention Distance Profile (Local vs Global Heads):** `figures/fig7_mean_attention_distance.png`
8. **Figure 8: Hindi Model H Causal Attention Heatmaps:** `figures/fig8_hindi_attention_heatmaps.png`
9. **Figure 9: Nepali Model L Causal Attention Heatmaps:** `figures/fig9_nepali_attention_heatmaps.png`
10. **Figure 10: Bonus Positional Embedding Ablation Study:** `figures/fig10_ablation_comparison.png`
11. **Figure 11: Transformer Parameter Allocation Distribution (~25.33M Params):** `figures/fig11_parameter_allocation.png`

---

## Phase 1 Embedded Figures Index (`report/figures/`)

1. **Figure 1: Raw vs Cleaned Document Records:** `figures/fig1_raw_vs_clean_records.png`
2. **Figure 2: Raw vs Cleaned Corpus Storage Size (MB):** `figures/fig2_raw_vs_clean_size.png`
3. **Figure 3: Token Share Source Distribution (Manual vs Downloaded):** `figures/fig3_source_distribution.png`
4. **Figure 4: Train / Validation / Test Partition Distribution (92/4/4 Records):** `figures/fig4_train_val_test_distribution.png`
5. **Figure 5: Estimated Subword Tokens by Split:** `figures/fig5_tokens_by_split.png`
6. **Figure 6: Data Cleaning, Deduplication & Quality Filtering Breakdown:** `figures/fig6_cleaning_filtering_stats.png`
7. **Figure 7: Comprehensive Hindi vs Nepali Clean Corpus Comparison:** `figures/fig7_hindi_vs_nepali_corpus.png`
8. **Figure 8: Tokenizer Checkpoint Model Binary Size Comparison:** `figures/fig8_tokenizer_vocab_comparison.png`
9. **Figure 9: Subword Tokenizer Compression Ratio & Fertility Dynamics:** `figures/fig9_fertility_and_compression.png`
