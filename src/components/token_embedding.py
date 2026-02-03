# src/components/token_embedding.py

import math
import torch
import torch.nn as nn
from loguru import logger


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.
    Maps token IDs to dense vectors of dimension d_model.
    Scaled by sqrt(d_model) as in the original Transformer paper.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self._init_weights()
        
        logger.debug(f"TokenEmbedding: vocab={vocab_size}, dim={d_model}")
    
    def _init_weights(self):
        """Initialize with normal distribution scaled by sqrt(d_model)."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=math.sqrt(1.0 / self.d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token IDs, shape (batch_size, seq_len)
        Returns:
            Embedded tokens, shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


