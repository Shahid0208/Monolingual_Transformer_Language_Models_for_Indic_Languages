"""
Decoder-Only Transformer Language Model from First Principles
============================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid0208 (Monsoon 2026)
Language: Hindi (Model H)

IMPORTANT ARCHITECTURAL CONSTRAINTS:
-----------------------------------
1. Built strictly from primitive PyTorch modules:
   - nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout
   - No nn.Transformer*, no HuggingFace Transformer classes, no pre-built attention blocks.
2. Multi-Head Causal Self-Attention implemented from scratch:
   - Explicit reshape and transpose: (B, T, d) -> (B, h, T, dk) and (B, h, T, dk) -> (B, T, d)
   - Scaled dot-product attention with 1/sqrt(dk) variance stabilization
   - Additive causal mask M where M_ij = 0 for j <= i and -inf for j > i
3. Pre-LayerNorm residual blocks for stable deep training dynamics
4. Positional Encodings: Learned Absolute Positional Embeddings with sinusoidal and ablation support
5. Weight Tying: Option to tie output projection head with token embedding table
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


# ============================================================
# 1. POSITIONAL EMBEDDINGS
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding (Vaswani et al., 2017).
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class PositionalEmbedding(nn.Module):
    """
    Modular Positional Embedding layer.
    Supports:
    - 'learned': Learned absolute positional embedding matrix (T_max x d_model)
    - 'sinusoidal': Fixed deterministic trigonometric frequencies
    - 'none': Zero-embedding for ablation study (Bonus Task)
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pos_type = config.pos_embedding_type
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model

        if self.pos_type == "learned":
            self.emb = nn.Embedding(config.max_seq_len, config.d_model)
        elif self.pos_type == "sinusoidal":
            self.emb = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        elif self.pos_type == "none":
            self.emb = None
        else:
            raise ValueError(f"Unknown positional embedding type: {self.pos_type}")

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, T) tensor of integer token indices [0, 1, ..., T-1]
            x: (B, T, d_model) token embedding tensor
        Returns:
            Positional embedding tensor of shape (B, T, d_model) or zeros
        """
        if self.pos_type == "learned":
            return self.emb(positions)
        elif self.pos_type == "sinusoidal":
            return self.emb(x)
        elif self.pos_type == "none":
            return torch.zeros_like(x)
        raise RuntimeError("Invalid positional embedding branch")


# ============================================================
# 2. MULTI-HEAD CAUSAL SELF-ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention implemented from first principles.

    Given sequence X in R^{B x T x d}:
    1. Linear projections to Queries, Keys, Values:
       Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
    2. Reshape and transpose across h heads:
       (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k) where d_k = d / h
    3. Scaled dot-product attention per head:
       Scores = (Q @ K^T) / sqrt(d_k)
       Masking: Scores = Scores + Causal_Mask (where upper triangle is -inf)
       Weights = Softmax(Scores, dim=-1)
       Context = Weights @ V  # (B, h, T, d_k)
    4. Reshape & project back:
       (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d)
       Out = Context @ W_O
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.attention_dropout
        self.max_seq_len = config.max_seq_len

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: upper triangular matrix filled with -inf
        # register_buffer ensures mask moves automatically with model to GPU/device
        # without being treated as a trainable parameter
        mask = torch.triu(torch.full((config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Multi-Head Causal Self-Attention.

        Args:
            x: Input tensor of shape (Batch_Size B, Seq_Len T, d_model d)
            attention_mask: Optional padding mask (B, 1, 1, T) or (B, 1, T, T)
            return_attention_weights: If True, returns attention weight matrix for analysis

        Returns:
            out: Projected context tensor of shape (B, T, d_model)
            attn_weights: Optional attention weights of shape (B, n_heads, T, T)
        """
        B, T, d = x.shape

        # Step 1: Project Q, K, V
        # Shape: (B, T, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Reshape & Transpose for Multi-Head decomposition
        # (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 3: Scaled Dot-Product Attention
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        # Scaling by 1/sqrt(head_dim) is mathematically necessary to prevent dot products
        # from growing with dimension, which would push softmax into zero-gradient regions.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 4: Apply Causal Mask
        # Causal mask ensures position t attends only to positions <= t
        causal_slice = self.causal_mask[:, :, :T, :T]
        scores = scores + causal_slice

        # Optional external padding mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Step 5: Softmax & Attention Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Step 6: Attention context aggregation
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        context = torch.matmul(attn_weights_dropped, v)

        # Step 7: Concatenate heads and project out
        # (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, d)

        # Step 8: Output linear transformation & residual dropout
        out = self.resid_dropout(self.out_proj(context))

        if return_attention_weights:
            return out, attn_weights
        return out, None


# ============================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK (FFN)
# ============================================================

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2
    where W_1 in R^{d_model x d_ff}, W_2 in R^{d_ff x d_model}.
    Standard expansion factor: d_ff = 4 * d_model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation.lower() == "gelu":
            self.act = nn.GELU()
        elif config.activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ============================================================
# 4. TRANSFORMER BLOCK (PRE-LAYERNORM)
# ============================================================

class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Decoder Block:
    1. x = x + SelfAttention(LayerNorm_1(x))
    2. x = x + FFN(LayerNorm_2(x))

    Why Pre-LN over Post-LN?
    - Pre-LN keeps an unobstructed identity residual path from input to final layer.
    - Gradients propagate smoothly without vanishing/exploding, eliminating the need
      for fragile learning rate warmups and ensuring superior training stability.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = PositionwiseFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN Self-Attention sublayer
        normed_1 = self.ln_1(x)
        attn_out, attn_weights = self.attn(
            normed_1,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )
        x = x + attn_out

        # Pre-LN Feed-Forward sublayer
        normed_2 = self.ln_2(x)
        ffn_out = self.mlp(normed_2)
        x = x + ffn_out

        return x, attn_weights


# ============================================================
# 5. DECODER-ONLY TRANSFORMER LANGUAGE MODEL
# ============================================================

class TransformerLM(nn.Module):
    """
    Complete Decoder-Only Monolingual Transformer Language Model.

    Pipeline:
    1. Input Tokens (B, T) -> Token Embeddings (B, T, d_model)
    2. Positional Indices (B, T) -> Positional Embeddings (B, T, d_model)
    3. Sum + Embedding Dropout -> Stack of N Transformer Blocks
    4. Final Layer Normalization
    5. Output Linear Projection (d_model -> vocab_size) to produce next-token Logits
    6. Causal Cross-Entropy Loss computation against shifted target tokens
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 1. Token Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Positional Embeddings
        self.position_embeddings = PositionalEmbedding(config)

        # 3. Embedding Dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # 4. Stack of N Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 5. Final Layer Normalization
        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # 6. Output LM Head (Projection to Vocab)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 7. Weight Tying (Parameter Saving Optimization)
        # Tying the LM head weights to the token embedding matrix saves (vocab_size x d_model)
        # parameters (~3.84M parameters) while improving representation alignment.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Parameter Initialization
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections (per GPT-2 / Radford et al.)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculates exact parameter counts across sub-modules.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        emb_params = sum(p.numel() for p in self.token_embeddings.parameters())
        if self.position_embeddings.emb is not None and isinstance(self.position_embeddings.emb, nn.Embedding):
            emb_params += sum(p.numel() for p in self.position_embeddings.emb.parameters())
        
        block_params = sum(sum(p.numel() for p in b.parameters()) for b in self.blocks)
        head_params = sum(p.numel() for p in self.lm_head.parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "embedding_parameters": emb_params,
            "transformer_block_parameters": block_params,
            "lm_head_parameters": head_params,
            "weight_tying_applied": self.config.tie_word_embeddings,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass for Language Modeling.

        Args:
            input_ids: Tensor of shape (Batch_Size B, Seq_Len T) with token IDs
            labels: Optional Tensor of shape (B, T) for cross-entropy loss computation
            attention_mask: Optional attention mask (B, 1, 1, T)
            return_attention_weights: If True, collects attention maps across all layers

        Returns:
            dict containing:
                - 'logits': (B, T, vocab_size)
                - 'loss': Scalar cross-entropy loss (if labels provided)
                - 'attention_weights': List of (B, h, T, T) per layer (if requested)
        """
        device = input_ids.device
        B, T = input_ids.shape

        assert T <= self.config.max_seq_len, (
            f"Input sequence length ({T}) exceeds max_seq_len ({self.config.max_seq_len})"
        )

        # 1. Generate position index tensor: (1, T) -> (B, T)
        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

        # 2. Token + Positional Embeddings
        tok_emb = self.token_embeddings(input_ids)  # (B, T, d_model)
        pos_emb = self.position_embeddings(positions, tok_emb)  # (B, T, d_model)
        
        x = self.emb_dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # 3. Pass through Transformer Blocks
        all_attentions = [] if return_attention_weights else None
        for block in self.blocks:
            x, attn_w = block(
                x,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights and attn_w is not None:
                all_attentions.append(attn_w)

        # 4. Final LayerNorm & LM Head Projection
        x = self.final_ln(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 5. Causal LM Loss Computation
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict token n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten across batch and sequence
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        output = {"logits": logits, "loss": loss}
        if return_attention_weights:
            output["attention_weights"] = all_attentions
        return output

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive next-token text generation.

        Args:
            prompt_ids: (B, T_prompt) token ID tensor
            max_new_tokens: Maximum tokens to generate
            temperature: Softmax scaling factor (1.0 = standard, 0.0 = greedy)
            top_k: Top-K truncation filter
            top_p: Nucleus top-P cumulative probability filter
            eos_token_id: End of sequence token ID to terminate generation early

        Returns:
            (B, T_prompt + generated_tokens) completed sequence tensor
        """
        self.eval()
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds max_seq_len
            x_cond = generated if generated.size(1) <= self.config.max_seq_len else generated[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self(x_cond)
            # Logits for the next token at the final sequence position: (B, vocab_size)
            next_token_logits = outputs["logits"][:, -1, :]

            # Greedy decoding
            if temperature == 0.0 or temperature < 1e-6:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Temperature scaling
                scaled_logits = next_token_logits / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                # Top-P (Nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to include the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = -float("Inf")

                # Sample from softmax distribution
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Check if all batches generated EOS
            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated
