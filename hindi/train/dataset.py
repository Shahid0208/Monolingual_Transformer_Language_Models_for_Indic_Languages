"""
Streaming Parquet Dataset & DataLoader for Hindi Language Modeling
=================================================================
Course: Language Models and Agents (Monsoon 2026)
Assignment: Individual Project — Phase 2: Model Implementation, Pretraining, and Evaluation

Author: Shahid (Monsoon 2026)
Language: Hindi (Model H)

Features:
- Memory-efficient streaming directly from Apache Parquet shards using PyArrow
- On-the-fly subword tokenization via trained SentencePiece Unigram model
- Token packing / chunking into fixed context length sequences (default: 512)
- Shuffled ring buffer for robust training stochasticity
- Zero memory blowup on massive 500M+ token datasets
"""

import math
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset, DataLoader


class StreamingParquetDataset(IterableDataset):
    """
    Streaming IterableDataset for training Transformer LMs on large Parquet files.
    """
    def __init__(
        self,
        parquet_path: Path,
        tokenizer_model_path: Path,
        max_seq_len: int = 512,
        batch_size: int = 32,
        buffer_size: int = 10000,
        shuffle: bool = True,
        text_column: str = "text",
        add_bos: bool = True,
        add_eos: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.tokenizer_model_path = Path(tokenizer_model_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.text_column = text_column
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.seed = seed

        assert self.parquet_path.exists(), f"Parquet file not found: {self.parquet_path}"
        assert self.tokenizer_model_path.exists(), f"Tokenizer not found: {self.tokenizer_model_path}"

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.tokenizer_model_path))
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 2
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 3

    def _token_generator(self) -> Iterator[int]:
        """
        Streams texts from Parquet, tokenizes them, and yields a continuous stream of token IDs.
        """
        parquet_file = pq.ParquetFile(self.parquet_path)
        num_row_groups = parquet_file.num_row_groups
        rg_indices = list(range(num_row_groups))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(rg_indices)

        for rg_idx in rg_indices:
            table = parquet_file.read_row_group(rg_idx, columns=[self.text_column])
            texts = table.column(self.text_column).to_pylist()

            if self.shuffle:
                random.shuffle(texts)

            for text in texts:
                if not text or not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                token_ids = self.sp.encode_as_ids(text)
                if not token_ids:
                    continue

                if self.add_bos:
                    yield self.bos_id
                yield from token_ids
                if self.add_eos:
                    yield self.eos_id

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Packs the continuous stream of tokens into chunks of length max_seq_len.
        Yields (input_ids, labels) tensors of shape (max_seq_len,).
        """
        token_stream = self._token_generator()
        buffer: List[int] = []
        sample_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for token in token_stream:
            buffer.append(token)

            # Once we have enough tokens for a full context window
            if len(buffer) >= self.max_seq_len:
                seq = buffer[: self.max_seq_len]
                buffer = buffer[self.max_seq_len :]

                input_tensor = torch.tensor(seq, dtype=torch.long)
                # Labels are identical to inputs; TransformerLM handles target shifting internally
                label_tensor = torch.tensor(seq, dtype=torch.long)

                if self.shuffle:
                    sample_buffer.append((input_tensor, label_tensor))
                    if len(sample_buffer) >= self.buffer_size:
                        random.shuffle(sample_buffer)
                        for item in sample_buffer:
                            yield item
                        sample_buffer.clear()
                else:
                    yield input_tensor, label_tensor

        # Flush any remaining samples in buffer
        if sample_buffer:
            if self.shuffle:
                random.shuffle(sample_buffer)
            for item in sample_buffer:
                yield item


def create_dataloader(
    parquet_path: Path,
    tokenizer_model_path: Path,
    batch_size: int = 32,
    max_seq_len: int = 512,
    buffer_size: int = 5000,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader backed by StreamingParquetDataset.
    """
    dataset = StreamingParquetDataset(
        parquet_path=parquet_path,
        tokenizer_model_path=tokenizer_model_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        buffer_size=buffer_size,
        shuffle=shuffle,
        seed=seed,
    )

    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
