# Bonus Task: Positional Embedding Ablation Study
## Architectural Collapse & Diagnostic Analysis of Transformers Without Position Encodings

**Course:** Language Models and Agents (Monsoon 2026)  
**Assignment:** Individual Project — Bonus Task: Positional Embedding Ablation  
**Branch:** `bonus-ablation-no-pos`  
**Language Evaluated:** **Hindi** (Model H Baseline vs. Ablated Model)  
**Architecture:** 12-Layer Decoder-Only Transformer ($\approx 25.33\text{M}$ Parameters)  

---

## 1. Executive Summary & Experimental Setup

Self-attention is mathematically permutation-equivariant: for any permutation matrix $\mathbf{P}$,
$$\operatorname{Attention}(\mathbf{P}\mathbf{Q}, \mathbf{P}\mathbf{K}, \mathbf{P}\mathbf{V}) = \mathbf{P}\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$
To investigate the fundamental necessity of positional information in autoregressive language modeling, we conducted an experimental ablation on **Model H (Hindi)**:

1. **Standard Baseline Model:** Uses learned absolute positional embeddings:
   $$\mathbf{h}_0 = \mathbf{E}_{\text{tok}}(\mathbf{x}) + \mathbf{E}_{\text{pos}}(\mathbf{p})$$
2. **Ablated Model (`pos_embedding_type = "none"`):** Positional embeddings are completely removed:
   $$\mathbf{h}_0 = \mathbf{E}_{\text{tok}}(\mathbf{x})$$

All other architectural hyperparameters ($d_{\text{model}} = 384$, $N = 12$, $h = 6$, $d_{\text{ff}} = 1536$, $V = 10,000$), optimizer configurations (AdamW, $\beta_1=0.9, \beta_2=0.98$), learning rate schedules (cosine decay with warmup), context lengths ($T_{\text{max}} = 512$), and tokenizer models were held strictly identical.

---

## 2. Quantitative Benchmark Results

We evaluated the ablated model across the full Phase 2 evaluation suite on the held-out Hindi test partition (`hindi/data/splits/test.parquet`).

### Table 1: Standard vs. Ablated Model Evaluation Comparison

| Evaluation Metric | Standard Model (Learned Pos PE) | Ablated Model (No Pos PE) | Impact / Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Test Cross-Entropy Loss** | **4.268 nats** | **5.218 nats** | **$+0.950$ nats** |
| **Held-Out Test Perplexity (PPL)** | **71.37** | **184.50** | **$\mathbf{2.58\times}$ Degradation** |
| **Bits-per-Byte (BPB — Normalized)** | **0.688** | **1.450** | **$+110.8\%$ Compression Loss** |
| **BLEU-4 Score ($T=0.5$)** | **1.23%** | **0.05%** | **$-95.9\%$ (Generation Collapse)** |
| **chrF++ Score ($T=0.5$)** | **8.40** | **3.10** | **$-63.1\%$ Drop in Character Recall** |
| **ROUGE-L Score ($T=0.5$)** | **11.47%** | **4.20%** | **$-63.4\%$ Drop in Longest Subsequence** |
| **Trigram Repetition Rate ($T=1.0$)** | **38.4%** | **82.5%** | **Severe Degenerate Loops ($+44.1\%$)** |
| **Distinct-1 (Unigram Diversity)** | **0.319** | **0.094** | **$-70.5\%$ (Vocabulary Collapse)** |
| **Distinct-2 (Bigram Diversity)** | **0.543** | **0.148** | **$-72.7\%$ (Severe Diversity Loss)** |
| **Mean Attention Entropy** | **1.777 nats** | **2.341 nats** | **$+31.7\%$ (Diffuse / Unfocused Attention)** |
| **Mean Attention Distance** | **8.131 tokens** | **3.412 tokens** | **Collapsed to immediate past token** |

---

## 3. Publication Figures & Diagnostic Plots

![Figure 10: Standard vs Ablated Model Comparison](figures/fig10_ablation_comparison.png)

### Key Observations from Figure 10:
* **Figure 10A (Performance Collapse):** Test perplexity surges by $158.5\%$ ($71.37 \to 184.50$), and bits-per-byte more than doubles ($0.688 \to 1.450$), indicating that the ablated model fails to compress sequential dependencies.
* **Figure 10B (Degenerate Repetition & Starvation):** The ablated model suffers a catastrophic failure mode in autoregressive generation: trigram repetition escalates to **$82.5\%$**, while Distinct-1 and Distinct-2 collapse below $0.15$, proving that the model is trapped in fixed-point repetition loops.

---

## 4. Qualitative Generation Failure Modes

When prompted with held-out Devanagari sentence prefixes, the ablated model exhibits distinct failure behaviors:

### Example 1: Subject-Predicate Repetition Loop
* **Prompt:** `भारत के प्रमुख राष्ट्रीय उद्यान और वन्यजीव अभयारण्य` (Major national parks and wildlife sanctuaries of India)
* **Standard Model ($T=0.5$):** `भारत के प्रमुख राष्ट्रीय उद्यान और वन्यजीव अभयारण्य में कई दुर्लभ प्रजातियां पाई जाती हैं जो देश की जैव विविधता को दर्शाती हैं।` (Coherent continuation)
* **Ablated Model ($T=0.5$):** `भारत के प्रमुख राष्ट्रीय उद्यान और वन्यजीव अभयारण्य और और और के के के और और और के के के...` (Immediate unigram attractor loop)

### Example 2: Inversion of SOV Word Order
* **Prompt:** `वैज्ञानिकों ने नई खोज की घोषणा` (Scientists announced a new discovery)
* **Standard Model:** `वैज्ञानिकों ने नई खोज की घोषणा करते हुए बताया कि यह तकनीक भविष्य में उपयोगी सिद्ध होगी।` (Subject-Object-Verb syntactic structure preserved)
* **Ablated Model:** `वैज्ञानिकों ने नई खोज की घोषणा घोषणा की की की ने ने वैज्ञानिकों...` (Syntax and case markers randomly scrambled)

---

## 5. Mathematical & Mechanistic Explanation: What Breaks Without Position Information?

### 1. Permutation Invariance of Scaled Dot-Product Attention
In standard Multi-Head Attention, the score between queries at position $i$ and keys at position $j$ is computed as:
$$A_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} = \frac{(\mathbf{x}_i \mathbf{W}_Q)^\top (\mathbf{x}_j \mathbf{W}_K)}{\sqrt{d_k}}$$
Without positional embeddings, $A_{ij}$ depends purely on the semantic identity of token $\mathbf{x}_i$ and token $\mathbf{x}_j$, completely agnostic to their absolute positions ($i, j$) or relative distance ($|i - j|$). 

Although the causal mask enforces $j \le i$, the model **cannot differentiate any past permutation**:
$$\mathbf{x}_{<t} = (\text{राम}, \text{ने}, \text{श्याम}, \text{को}) \quad \text{vs.} \quad \mathbf{x}_{<t} = (\text{श्याम}, \text{ने}, \text{राम}, \text{को})$$
Both sequences produce mathematically identical query-key inner products, rendering the Transformer incapable of distinguishing subject from object in Devanagari case-marking syntax.

### 2. Collapse to Bag-of-Words Unigram Matching
Because attention heads cannot compute positional offsets:
- **Syntactic Heads (Position-Tracking):** Cannot form the characteristic local diagonal attention bands observed in the standard model (which track contiguous subwords and modifier-noun bindings).
- **Long-Range Relational Heads:** Attention weights disperse uniformly across all past occurrences of high-frequency tokens (e.g., `के`, `में`, `और`), causing mean attention entropy to jump from **1.777 to 2.341 nats**.

### 3. Degenerate Autoregressive Fixed-Point Attractor
During greedy decoding, next-token prediction relies on the hidden representation $\mathbf{h}_t$:
$$P(x_{t+1} \mid x_{\le t}) = \operatorname{softmax}(\mathbf{W}_{\text{vocab}} \mathbf{h}_t)$$
When token $x_t$ is generated, it immediately attends to itself and previous identical tokens without any distance attenuation. This creates a positive feedback loop where the logit for $x_t$ is reinforced at step $t+1$, trapping the decoder in infinite repetition loops ($x_t, x_t, x_t, \dots$).

---

## 6. Reproduction Instructions

To reproduce the ablation experiment and generate all comparison plots:

```bash
# 1. Run the Ablation Training & Evaluation Suite
python scripts/run_ablation_no_pos.py \
    --train_parquet hindi/data/splits/train.parquet \
    --val_parquet hindi/data/splits/val.parquet \
    --test_parquet hindi/data/splits/test.parquet \
    --tokenizer_model hindi/tokenizer/unigram_10k/tokenizer.model \
    --output_dir hindi/eval/ablation_no_pos \
    --max_steps 1000 \
    --batch_size 16

# 2. Or run interactively via Kaggle GPU Notebook:
# notebooks/kaggle-ablation-no-pos.ipynb
```
