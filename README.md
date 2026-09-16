# Bonus Task: Positional Embedding Ablation Study
## Architectural Collapse & Diagnostic Analysis of Transformers Without Positional Embeddings

**Course:** Language Models and Agents (Monsoon 2026)  
**Assignment:** Individual Project — Bonus Task: Positional Embedding Ablation  
**Branch:** `bonus-ablation-no-pos`  
**Language Evaluated:** **Hindi** (`hi` — Model H Baseline vs. Ablated Model)  
**Model Scale:** $\approx 25.33\text{M}$ Parameters per Model  
**Script:** Devanagari  

---

## Executive Summary

As an optional bonus investigation in accordance with the assignment specification, we trained and evaluated a decoder-only Transformer language model from scratch with **positional embeddings completely removed** ($X_{\text{emb}} = \text{TokenEmbeddings}(X)$ with $\text{PE} = 0$).

We ran the **full Phase 2 evaluation suite** on the ablated model and compared it directly against the standard Hindi Model H (which uses learned absolute positional embeddings) across:
1. **Intrinsic Language Modeling:** Cross-entropy loss, perplexity (PPL), and bits-per-byte (BPB).
2. **Generation Quality:** Multi-temperature BLEU-4, chrF++, and ROUGE-L.
3. **Lexical Diversity & Repetition:** Distinct-1, Distinct-2, and trigram repetition rates.
4. **Attention Diagnostics:** Layer-wise Shannon attention entropy, mean attention distance, and attention heatmaps.

### Model Checkpoints & Pretrained Artifacts
* **Ablated Model Checkpoint (`ablation_no_pos_final.pt`):** [Google Drive — Ablated Hindi Model H](https://drive.google.com/file/d/19mjFnqzE9yLC6ML7U7srrU99zEbaB5Ug/view?usp=drive_link)
* **Standard Model H Baseline Checkpoint:** [Google Drive — Standard Hindi Model H](https://drive.google.com/file/d/1zqv-qrALlxFdNHmKJejUZZH0ZVjodgbr/view?usp=sharing)
* **Pre-computed Evaluation Results:** [bonus-ablation-no-pos/hindi_ablation_no_pos/ablation_results.json](bonus-ablation-no-pos/hindi_ablation_no_pos/ablation_results.json)

---

## 1. Key Comparative Results

| Evaluation Metric | Standard Model (Learned Pos PE) | Ablated Model (No Pos PE) | Impact / Degradation |
| :--- | :---: | :---: | :---: |
| **Held-Out Test Loss** | **4.268 nats** | **5.218 nats** | **$+0.950$ nats** |
| **Held-Out Test Perplexity ($\text{PPL} = e^{\mathcal{L}}$)** | **71.37** | **184.50** | **$\mathbf{2.58\times}$ Degradation** |
| **Bits-per-Byte (BPB — Normalized)** | **0.688 BPB** | **1.450 BPB** | **$+110.8\%$ Compression Loss** |
| **BLEU-4 Score ($T=0.5$)** | **1.23%** | **0.05%** | **Severe Generation Collapse** |
| **chrF++ Score ($T=0.5$)** | **8.40** | **3.10** | **$-63.1\%$ Drop in Recall** |
| **ROUGE-L Score ($T=0.5$)** | **11.47%** | **4.20%** | **$-63.4\%$ Drop in Subsequences** |
| **Trigram Repetition Rate ($T=1.0$)** | **38.4%** | **82.5%** | **Extreme Repetition Loops** |
| **Distinct-1 (Unigram Diversity)** | **0.319** | **0.094** | **$-70.5\%$ Vocabulary Starvation** |
| **Distinct-2 (Bigram Diversity)** | **0.543** | **0.148** | **$-72.7\%$ Loss of Bigram Diversity** |
| **Mean Attention Entropy** | **1.777 nats** | **2.341 nats** | **$+31.7\%$ (Diffuse / Unfocused)** |
| **Mean Attention Distance** | **8.131 tokens** | **3.412 tokens** | **Collapsed to immediate past token** |

---

## 2. Comparative Publication Figures & Attention Diagnostics

### A. Performance Degradation & Diversity Collapse
![Figure 10: Standard vs. Ablated Model Performance](report/figures/fig10_ablation_comparison.png)

---

### B. Layer-wise Attention Entropy & Attention Span Comparison
![Figure 10B: Attention Entropy and Distance Comparison](report/figures/fig10b_attention_ablation_comparison.png)

---

### C. Multi-Head Attention Heatmaps (Ablated vs. Standard Model)

#### 1. Ablated Model H Attention Heatmaps (No Positional Embeddings)
![Figure 10C: Ablated Model Attention Heatmaps](report/figures/fig10c_ablation_attention_heatmaps.png)

#### 2. Standard Baseline Model H Attention Heatmaps (Learned Pos PE)
![Figure 8: Standard Hindi Attention Heatmaps](report/figures/fig8_hindi_attention_heatmaps.png)

---

## 3. Scientific Explanation: What Breaks Without Position Information?

1. **Permutation Invariance of Self-Attention:**
   - Standard Scaled Dot-Product Attention $\operatorname{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ is permutation-equivariant. Without position encodings, the model cannot distinguish between sequences with permuted token orders (e.g., "राम ने मोहन को देखा" vs. "मोहन ने राम को देखा").
2. **Loss of Syntactic Head Specialization:**
   - In standard transformers, early heads learn diagonal positional bands to bind adjacent Devanagari subwords and case markers. Without positional coordinates, attention heads cannot track relative distance, causing heads to diffuse uniformly across all past tokens (entropy jumps from $1.78 \to 2.34$ nats).
3. **Autoregressive Fixed-Point Attractors (Degenerate Repetition Loops):**
   - Because generation cannot differentiate temporal distance, the most recently emitted token acts as a static bag-of-words attractor, causing the decoder to repeat the same unigram in infinite loops (repetition surges to **82.5%**).

---

## 4. Full Reproduction Guide

### A. Run via CLI Script
```bash
# 1. Run Pretraining & Full Phase 2 Evaluation Suite
python scripts/run_ablation_no_pos.py \
    --train_parquet hindi/data/splits/train.parquet \
    --val_parquet hindi/data/splits/val.parquet \
    --test_parquet hindi/data/splits/test.parquet \
    --tokenizer_model hindi/tokenizer/unigram_10k/tokenizer.model \
    --output_dir hindi/eval/ablation_no_pos \
    --batch_size 16 \
    --max_steps 1000 \
    --use_amp

# 2. Generate Comparison Diagnostic Figures
python scripts/generate_ablation_figures.py
```

### B. Run via Interactive Kaggle / Colab Notebook
* Open [notebooks/kaggle-ablation-no-pos.ipynb](notebooks/kaggle-ablation-no-pos.ipynb) in Kaggle GPU / Google Colab for end-to-end execution and visualization.
