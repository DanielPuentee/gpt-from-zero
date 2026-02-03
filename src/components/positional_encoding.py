# src/components/positional_encoding.py

import math
import torch
import torch.nn as nn
from loguru import logger

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from "Attention is All You Need".
    Uses sine and cosine functions of different frequencies.
    Never updated during training - purely deterministic.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding matrix (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices: 0, 1, 2, ..., max_seq_len-1
        # Shape: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Divisor term: 10000^(2i/d_model) for i in [0, d_model/2)
        # We compute in log space for numerical stability
        # Shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, persists in state_dict)
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(p=dropout)
        
        logger.debug(
            f"SinusoidalPositionalEncoding: max_len={max_seq_len}, "
            f"dim={d_model}, dropout={dropout}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Token embeddings, shape (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added, same shape
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds max {self.max_seq_len}. "
                f"Truncating to max length."
            )
            seq_len = self.max_seq_len
            x = x[:, :seq_len, :]
        
        # Add positional encoding
        # x: (batch, seq_len, d_model)
        # pe: (1, max_seq_len, d_model) -> slice to (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (trainable parameters).
    Each position has its own embedding vector learned during training.
    Used in GPT, BERT, and most modern Transformer variants.
    """
    pass

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) - used in LLaMA, Mistral.
    Encodes position by rotating query/key vectors.
    More efficient for long sequences.
    """
    pass
