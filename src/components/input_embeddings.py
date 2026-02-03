# src/components/input_embeddings.py
"""This file is just the combination of TokenEmbedding and PositionalEncoding."""

import torch
import torch.nn as nn
from loguru import logger

from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding


class InputEmbeddings(nn.Module):
    """
    Combined token embeddings + positional encoding.
    Sequentially applies token embedding then positional encoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.d_model = d_model
        
        logger.debug(
            f"InputEmbeddings: vocab={vocab_size}, dim={d_model}, "
            f"max_len={max_seq_len}, dropout={dropout}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape (batch_size, seq_len)
        Returns:
            Embeddings with positional encoding, shape (batch_size, seq_len, d_model)
        """
        # Step 1: Token embedding
        tok_emb = self.token_emb(x)
        
        # Step 2: Add positional encoding
        out = self.pos_enc(tok_emb)
        
        return out