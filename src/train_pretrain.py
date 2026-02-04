# src/train_pretrain.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import sys
from tqdm import tqdm
import json

from components.bpe_tokenizer import BPETokenizer
from components.model import GPTModel

import json

def load_config(config_path):
    with open(config_path, 'r') as f: return json.load(f)
    
class TextDataset(Dataset):
    """
    Dataset for causal language modeling.
    Creates sequences of fixed length from text file.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        max_seq_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load and tokenize text
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode full text
        self.tokens = tokenizer.encode(text)
        logger.info(f"Total tokens: {len(self.tokens)}")
        
        # Calculate number of sequences
        # Each sequence is max_seq_len + 1 (input + target)
        self.n_sequences = max(0, len(self.tokens) - max_seq_len)
        logger.info(f"Number of sequences: {self.n_sequences}")
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Get sequence of tokens
        start = idx
        end = idx + self.max_seq_len + 1
        
        chunk = self.tokens[start:end]
        
        # Pad if necessary (shouldn't happen with proper data)
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [0] * (self.max_seq_len + 1 - len(chunk))
        
        # Input: first max_seq_len tokens
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        
        # Target: last max_seq_len tokens (shifted by 1)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def train(
    data_path: str = "data/data.txt",
    vocab_size: int = 5000,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    max_seq_len: int = 128,
    dropout: float = 0.1,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    n_epochs: int = 10,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints",
    tokenizer_path: str = "checkpoints/tokenizer.json"
):
    """
    Pre-train GPT model on text data.
    """
    
    # Setup
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Train or load tokenizer
    # if os.path.exists(tokenizer_path):
    #     logger.info(f"Loading tokenizer from {tokenizer_path}")
    #     tokenizer = BPETokenizer(vocab_size=vocab_size)
    #     tokenizer.load(tokenizer_path)

    # We will always train a new tokenizer for pretraining
    logger.info("Training new tokenizer")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Create dataset and dataloader
    dataset = TextDataset(data_path, tokenizer, max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)  # (batch, seq, vocab_size)
            
            # Compute loss
            # Reshape for cross entropy: (batch*seq, vocab_size) vs (batch*seq)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log every 100 steps
            if global_step % 100 == 0:
                avg_loss = total_loss / n_batches
                logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
        
        # Epoch summary
        avg_epoch_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"model_epoch{epoch+1}.pt"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'config': {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'n_heads': n_heads,
                'n_layers': n_layers,
                'd_ff': d_ff,
                'max_seq_len': max_seq_len
            }
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info("Training completed!")
    return model, tokenizer


if __name__ == "__main__":

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/pretrain.log", rotation="10 MB", level="DEBUG")
    os.makedirs("logs", exist_ok=True)
    
    # Load config
    config = load_config("config/config.json")
    model_config = config['model']
    training_config = config['training']

    # Train
    model, tokenizer = train(
        **training_config,
        **model_config
    )