# src/components/model.py

import torch
import torch.nn as nn
from loguru import logger

from .input_embeddings import InputEmbeddings
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    GPT-style Transformer for causal language modeling.
    Decoder-only architecture with causal self-attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embeddings (token + positional)
        self.input_emb = InputEmbeddings(
            vocab_size, d_model, max_seq_len, dropout
        )
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.ln_final = nn.LayerNorm(d_model)
        
        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and LM head
        # This reduces parameters and improves performance
        self.input_emb.token_emb.embedding.weight = self.lm_head.weight
        
        self._init_weights()
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"GPTModel initialized: "
            f"vocab={vocab_size}, d_model={d_model}, "
            f"n_heads={n_heads}, n_layers={n_layers}, "
            f"d_ff={d_ff}, max_seq_len={max_seq_len}"
        )
        logger.info(f"Total parameters: {n_params:,}")
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize LM head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape (batch_size, seq_len)
        Returns:
            Logits for each token, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds max {self.max_seq_len}. "
                f"Truncating."
            )
            x = x[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Step 1: Input embeddings
        x = self.input_emb(x)  # (batch, seq, d_model)
        
        # Step 2: Pass through transformer blocks
        for block in self.blocks:
            x = block(x)  # (batch, seq, d_model)
        
        # Step 3: Final layer norm
        x = self.ln_final(x)
        
        # Step 4: Project to vocabulary (get logits for each token)
        logits = self.lm_head(x)  # (batch, seq, vocab_size)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens, shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = default, <1.0 = conservative, >1.0 = creative)
            top_k: If set, only sample from top k tokens
        Returns:
            Generated token IDs, shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to max_seq_len if needed
                input_crop = input_ids[:, -self.max_seq_len:]
                
                # Forward pass
                logits = self(input_crop)
                
                # Get logits for last token only
                logits = logits[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                logits = logits / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Print the top 3 most probable tokens for debugging
                # top_probs, top_indices = torch.topk(probs, 3, dim=-1)
                # logger.debug(
                #     f"Next token probabilities: {top_probs.squeeze().tolist()}, "
                #     f"Indices: {top_indices.squeeze().tolist()}"
                # )
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids