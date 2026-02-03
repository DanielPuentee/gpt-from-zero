# src/components/attention.py

import math
import torch
import torch.nn as nn
from loguru import logger


class SelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    Each token can only attend to previous tokens and itself (left-to-right).
    Used in GPT-style decoder-only models.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        bias: bool = False
    ):
        super().__init__()
        
        # d_model must be divisible by n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # Dimension of each head
        self.scale = math.sqrt(self.head_dim)  # Scaling factor for attention scores
        
        # Combined Q, K, V projection in one linear layer (more efficient)
        # Input: d_model, Output: 3 * d_model (Q, K, V concatenated)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask (lower triangular matrix)
        # Shape: (1, 1, max_seq_len, max_seq_len)
        # Registered as buffer so it moves with model.to(device) and saves in state_dict
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)
        
        self._init_weights()
        
        logger.debug(
            f"CausalSelfAttention: d_model={d_model}, n_heads={n_heads}, "
            f"head_dim={self.head_dim}, dropout={dropout}"
        )
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Compute Q, K, V
        # qkv output: (batch, seq, 3*d_model)
        qkv = self.qkv(x)
        
        # Reshape to separate Q, K, V and heads
        # New shape: (batch, seq, 3, n_heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        
        # Permute to: (3, batch, n_heads, seq, head_dim)
        # This puts batch and heads together for parallel attention computation
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Unpack Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Each has shape: (batch, n_heads, seq, head_dim)
        
        # Step 2: Compute attention scores (Q @ K^T / sqrt(d_k))
        # q: (batch, n_heads, seq, head_dim)
        # k.transpose: (batch, n_heads, head_dim, seq)
        # Result: (batch, n_heads, seq, seq)
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        
        # Step 3: Apply causal mask
        # Mask out future positions (set to -inf so softmax becomes 0)
        mask = self.mask[:, :, :seq_len, :seq_len]  # Slice to current seq_len
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 4: Softmax to get attention weights
        # Each row sums to 1, represents how much to "attend" to each position
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Step 5: Apply attention weights to values
        # attn_weights: (batch, n_heads, seq, seq)
        # v: (batch, n_heads, seq, head_dim)
        # Result: (batch, n_heads, seq, head_dim)
        out = attn_weights @ v
        
        # Step 6: Reshape back
        # (batch, n_heads, seq, head_dim) -> (batch, seq, n_heads, head_dim)
        out = out.transpose(1, 2)
        
        # (batch, seq, n_heads, head_dim) -> (batch, seq, d_model)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        
        # Step 7: Final linear projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out
