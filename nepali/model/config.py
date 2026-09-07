"""
Model Configuration for Model L (Nepali) Transformer LM
=======================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Defines hyperparameters targeting ~25M trainable parameters:
- d_model: 384
- n_layers: 12
- n_heads: 6 (head_dim = 64)
- d_ff: 1536 (4 * d_model)
- max_seq_len: 512
- vocab_size: 10,000 (Nepali Unigram Tokenizer)
- tie_word_embeddings: True
Total parameters: ~25.27M
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class TransformerConfig:
    """
    Hyperparameter configuration for Decoder-Only Transformer LM.
    """
    # Vocabulary & Sequence Length
    vocab_size: int = 10000
    max_seq_len: int = 512
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0

    # Architecture Dimensions
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 1536  # 4 * d_model
    
    # Regularization & Stability
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Structural Flags
    bias: bool = True
    pos_embedding_type: str = "learned"  # 'learned', 'sinusoidal', 'none'
    tie_word_embeddings: bool = True
    activation: str = "gelu"  # 'gelu' or 'relu'
    
    # Metadata
    language: str = "Nepali"
    model_name: str = "Model_L_Nepali_25M"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.pos_embedding_type in ["learned", "sinusoidal", "none"], (
            f"Invalid pos_embedding_type: {self.pos_embedding_type}"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerConfig":
        return cls(**data)

    def save_json(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "TransformerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
