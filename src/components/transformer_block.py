# src/components/transformer_block.py

import torch
import torch.nn as nn
from loguru import logger

from .attention import SelfAttention
from .feedforward import FeedForward
from .layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    Single Transformer block.
    Combines: Attention -> Add&Norm -> FFN -> Add&Norm
    Uses Pre-LN architecture (more stable): Norm -> Attention -> Add
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        # Pre-layer normalization (applied before sublayers)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
        # Sublayers
        self.attn = SelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        logger.debug(
            f"TransformerBlock: d_model={d_model}, n_heads={n_heads}, "
            f"d_ff={d_ff}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Pre-LN: LayerNorm -> Sublayer -> Residual Add
        
        # Attention block
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = residual + x  # Residual connection
        
        # Feed-forward block
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x
