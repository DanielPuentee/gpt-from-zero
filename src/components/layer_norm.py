# src/components/layer_norm.py

import torch
import torch.nn as nn
from loguru import logger


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    Normalizes across the feature dimension (d_model).
    Stabilizes training by keeping values in reasonable range.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps  # Small number to prevent division by zero
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift
        
        logger.debug(f"LayerNorm: d_model={d_model}, eps={eps}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Calculate mean and std across last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize: (x - mean) / (std + eps)
        x_norm = (x - mean) / (std + self.eps)
        
        # Scale and shift: gamma * x_norm + beta
        out = self.gamma * x_norm + self.beta
        
        return out
