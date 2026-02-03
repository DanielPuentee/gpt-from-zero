# src/components/feedforward.py

import torch
import torch.nn as nn
from loguru import logger


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Each position is processed independently with the same MLP.
    Standard Transformer: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear: expand dimension
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # Second linear: project back
        self.fc2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._init_weights()
        
        logger.debug(
            f"FeedForward: d_model={d_model}, d_ff={d_ff}, "
            f"activation={activation}, dropout={dropout}"
        )
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Expand: d_model -> d_ff
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project back: d_ff -> d_model
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x