# Monolingual Transformer Language Models for Indic Languages
## Phase 2: Model Implementation, Pretraining, and Evaluation

**Course:** Language Models and Agents (Monsoon 2026)  
**Assignment:** Individual Project — Building & Analyzing Monolingual Transformer LMs in Two Indian Languages  
**Branch:** `phase-2`  
**Languages:** **Hindi** (`hi` — Model H, High-Resource) & **Nepali** (`ne` — Model L, Low-Resource)  
**Model Scale:** $\approx 25.33\text{M}$ Parameters per Model ($\approx 25.27\text{M}$ effective with weight tying)

---

## Google Drive Checkpoint Links

Per project submission guidelines, full model checkpoint snapshots are stored on Google Drive with strict language separation:

### A. Hindi (Model H — High-Resource)
* **Pretrained Transformer Checkpoints:** [Google Drive — Hindi Model H Checkpoints](https://drive.google.com/file/d/1zqv-qrALlxFdNHmKJejUZZH0ZVjodgbr/view?usp=sharing)
  * `checkpoint_best.pt` (Optimal checkpoint: Step 30,518)
  * `checkpoint_latest.pt` (Resumable training snapshot)
  * `step_2000.pt` through `step_30000.pt` (Periodic weights)
* **Dataset Splits (Parquet):** [Google Drive — Hindi Parquet Splits](https://drive.google.com/drive/folders/17mJ-Z-ZV0Z4JJwsiUeF_vFyrdrOFH_j0?usp=drive_link)

### B. Nepali (Model L — Low-Resource)
* **Pretrained Transformer Checkpoints:** [Google Drive — Nepali Model L Checkpoints](https://drive.google.com/file/d/1ocqIcYfPMbqKulJKXM90mGxmSBglzlmU/view?usp=sharing)
  * `checkpoint_best.pt` (Optimal checkpoint: Step 30,518)
  * `checkpoint_latest.pt` (Resumable training snapshot)
  * `step_2000.pt` through `step_30000.pt` (Periodic weights)
* **Dataset Splits (Parquet):** [Google Drive — Nepali Parquet Splits](https://drive.google.com/drive/folders/1RpUp7q1akokeMXfAdlaxQTO-MHqfoxX4?usp=drive_link)

---

## 1. Phase 2 Architecture & Parameter Budget

Both models are decoder-only Transformer language models implemented from fundamental PyTorch primitives (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`) without any pre-built attention blocks or Hugging Face model classes.

| Hyperparameter | Symbol | Value | Architectural Justification |
| :--- | :---: | :---: | :--- |
| **Model Dimension** | $d_{\text{model}}$ | **384** | Balances representational width with quadratic attention compute |
| **Layers** | $N$ | **12** | Sufficient depth for hierarchical Devanagari syntax and semantics |
| **Attention Heads** | $h$ | **6** | Head dimension $d_k = 384/6 = 64$ (optimal for tensor operations) |
| **Feed-Forward Dimension** | $d_{\text{ff}}$ | **1,536** | Standard $4 \times d_{\text{model}}$ non-linear expansion (GELU) |
| **Context Length** | $T_{\text{max}}$ | **512** | Covers long-span document sequences |
| **Vocabulary Size** | $V$ | **10,000** | Monolingual SentencePiece Unigram from Phase 1 |
| **Weight Tying** | — | **True** | Ties LM head with token embeddings (saves $3.84\text{M}$ params) |
| **Total Parameters** | — | **25,330,944** | Strictly satisfies the $\sim 25\text{M}$ parameter budget |

---

## 2. Repository Layout

```
individual-project-Shahid0208/
├── README.md                                   # Root project documentation
├── readme_phase2.md                            # Phase 2 Technical & Reproduction Guide
├── report/
│   ├── phase_2_report.md                       # Comprehensive Phase 2 Technical Report
│   ├── README.md                               # Reports and Figures Index
│   └── figures/                                # Publication figures (loss, metrics, heatmaps)
├── notebooks/
│   ├── kaggle-pretrain-hindi.ipynb             # Hindi GPU pretraining & evaluation notebook
│   └── kaggle-pretrain-nepali.ipynb            # Nepali GPU pretraining & evaluation notebook
├── scripts/
│   ├── generate_phase2_figures.py              # Generates publication figures from empirical logs
│   ├── verify_causal_mask.py                   # Causal future-invariance verification
│   └── run_ablation_no_pos.py                  # Bonus: Positional embedding ablation
├── hindi/                                      # Hindi (Model H) Module
│   ├── model/                                  # Pure PyTorch Decoder Transformer (config, model)
│   ├── train/                                  # Pretraining loop, dataset & checkpointing
│   ├── eval/                                   # Evaluation suite & attention diagnostics
│   │   ├── evaluation_results.json             # Empirical test metrics
│   │   └── attention/                          # Attention heatmaps & entropy summary
│   ├── configs/                                # model_config.json & train_config.json
│   ├── checkpoints/                            # train_log.jsonl & hindi_loss_curve.png
│   └── tokenizer/                              # Unigram 10K tokenizer model & vocabulary
└── nepali/                                     # Nepali (Model L) Module
    ├── model/                                  # Pure PyTorch Decoder Transformer (config, model)
    ├── train/                                  # Pretraining loop, dataset & checkpointing
    ├── eval/                                   # Evaluation suite & attention diagnostics
    │   ├── evaluation_results.json             # Empirical test metrics
    │   └── attention/                          # Attention heatmaps & entropy summary
    ├── configs/                                # model_config.json & train_config.json
    ├── checkpoints/                            # train_log.jsonl & nepali_loss_curve.png
    └── tokenizer/                              # Unigram 10K tokenizer model & vocabulary
```

---

## 3. Phase 2 Key Empirical Results Summary

Both models were pretrained independently from scratch for **30,518 optimization steps** (~500M tokens, 1 epoch) using AdamW with cosine learning rate schedule and automatic mixed precision (AMP).

| Evaluation Metric | Hindi (Model H — High Resource) | Nepali (Model L — Low Resource) |
| :--- | :---: | :---: |
| **Final Pretraining Train Loss / PPL** | **3.851 nats** / **47.03 PPL** | **3.945 nats** / **51.68 PPL** |
| **Final Pretraining Val Loss / PPL** | **4.354 nats** / **77.76 PPL** | **3.867 nats** / **47.81 PPL** |
| **Held-Out Test Cross-Entropy Loss** | **4.268 nats** | **3.880 nats** |
| **Held-Out Test Perplexity ($\text{PPL} = e^{\mathcal{L}}$)** | **71.37** | **48.43** |
| **Bits-per-Byte (BPB — Normalized)** | **0.688 BPB** | **0.483 BPB** |
| **Unknown Token Rate (Devanagari UNK)** | **0.00%** | **0.00%** |
| **Generation BLEU-4 ($T=0.5$)** | **1.23%** | **0.57%** |
| **Generation chrF++ ($T=1.5$)** | **11.50%** | **15.60%** |
| **Generation ROUGE-L ($T=0.5$)** | **11.47%** | **11.21%** |
| **Trigram Repetition Rate ($T=1.0$)** | **4.75%** | **2.44%** |
| **Distinct-2 Lexical Diversity ($T=1.0$)** | **0.860** | **0.903** |

*For complete layer-wise attention heatmaps, Shannon entropy graphs, mean attention distance profiles, and qualitative continuation samples, see the full [Phase 2 Technical Report](report/phase_2_report.md).*

---

## 4. Reproduction Instructions

### 1. Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate
pip install torch pyarrow sentencepiece matplotlib seaborn numpy pandas
```

### 2. Verify Causal Mask (Future-Invariance Test)
```bash
python3 scripts/verify_causal_mask.py
```

### 3. Pretraining from Scratch (30,518 Steps with Checkpoint Resume)
```bash
# Hindi Model H Pretraining
python3 hindi/train/train.py \
    --train_parquet hindi/data/splits/train.parquet \
    --val_parquet hindi/data/splits/val.parquet \
    --tokenizer_model hindi/tokenizer/unigram_10k/tokenizer.model \
    --output_dir hindi/checkpoints \
    --batch_size 32 \
    --max_seq_len 512 \
    --max_steps 30518 \
    --learning_rate 5e-4 \
    --warmup_steps 1000 \
    --save_interval 2000 \
    --eval_interval 500 \
    --resume \
    --use_amp

# Nepali Model L Pretraining
python3 nepali/train/train.py \
    --train_parquet nepali/data/splits/train.parquet \
    --val_parquet nepali/data/splits/val.parquet \
    --tokenizer_model nepali/tokenizer/unigram_10k/tokenizer.model \
    --output_dir nepali/checkpoints \
    --batch_size 32 \
    --max_seq_len 512 \
    --max_steps 30518 \
    --learning_rate 5e-4 \
    --warmup_steps 1000 \
    --save_interval 2000 \
    --eval_interval 500 \
    --resume \
    --use_amp
```

### 4. Language Modeling & Generation Evaluation Suite
```bash
# Hindi Model H Evaluation (Loss, PPL, BPB, BLEU, chrF++, ROUGE-L, Diversity)
python3 hindi/eval/evaluate.py \
    --checkpoint hindi/checkpoints/checkpoints/checkpoint_best.pt \
    --config_file hindi/configs/model_config.json \
    --test_parquet hindi/data/splits/test.parquet \
    --tokenizer_model hindi/tokenizer/unigram_10k/tokenizer.model \
    --output_json hindi/eval/evaluation_results.json

# Nepali Model L Evaluation
python3 nepali/eval/evaluate.py \
    --checkpoint nepali/checkpoints/checkpoints/checkpoint_best.pt \
    --config_file nepali/configs/model_config.json \
    --test_parquet nepali/data/splits/test.parquet \
    --tokenizer_model nepali/tokenizer/unigram_10k/tokenizer.model \
    --output_json nepali/eval/evaluation_results.json
```

### 5. Multi-Head Attention Diagnostics (Heatmaps & Entropy)
```bash
# Hindi Attention Diagnostics
python3 hindi/eval/attention_analysis.py \
    --checkpoint hindi/checkpoints/checkpoints/checkpoint_best.pt \
    --config_file hindi/configs/model_config.json \
    --tokenizer_model hindi/tokenizer/unigram_10k/tokenizer.model \
    --output_dir hindi/eval/attention

# Nepali Attention Diagnostics
python3 nepali/eval/attention_analysis.py \
    --checkpoint nepali/checkpoints/checkpoints/checkpoint_best.pt \
    --config_file nepali/configs/model_config.json \
    --tokenizer_model nepali/tokenizer/unigram_10k/tokenizer.model \
    --output_dir nepali/eval/attention
```

### 6. Interactive GPU Pretraining Notebooks
- **Hindi Pretraining Notebook:** [notebooks/kaggle-pretrain-hindi.ipynb](notebooks/kaggle-pretrain-hindi.ipynb)
- **Nepali Pretraining Notebook:** [notebooks/kaggle-pretrain-nepali.ipynb](notebooks/kaggle-pretrain-nepali.ipynb)

### 7. Bonus Task: Positional Embedding Ablation
```bash
python3 scripts/run_ablation_no_pos.py
```
