# src/train_finetune.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from loguru import logger
import sys
from tqdm import tqdm

from components.bpe_tokenizer import BPETokenizer
from components.model import GPTModel


class RatingDataset(Dataset):
    """
    Dataset for rating-based fine-tuning with prompt-response pairs.
    Format: prompt, response, rating
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        max_seq_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load CSV
        logger.info(f"Loading ratings from {data_path}")
        df = pd.read_csv(data_path)
        
        required_cols = ['prompt', 'response', 'rating']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV must contain column: {col}")
        
        # Filter valid ratings (1-5)
        df = df[df['rating'].between(1, 5)]
        
        self.prompts = df['prompt'].tolist()
        self.responses = df['response'].tolist()
        self.ratings = df['rating'].tolist()
        
        # Convert ratings to weights (higher rating = higher weight)
        # Rating 1 -> weight 0.2, Rating 5 -> weight 1.0
        self.weights = [r / 5.0 for r in self.ratings]
        
        logger.info(f"Loaded {len(self.prompts)} prompt-response pairs")
        logger.info(f"Rating distribution: {df['rating'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = str(self.prompts[idx])
        response = str(self.responses[idx])
        weight = self.weights[idx]
        rating = self.ratings[idx]
        
        # Format: prompt + response + end token
        # Model learns to predict response given prompt
        full_text = prompt + response
        
        # Encode
        tokens = self.tokenizer.encode(full_text)
        
        # Add end-of-text token
        eot_id = self.tokenizer.special_tokens.get('<|endoftext|>', 0)
        tokens.append(eot_id)
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[:self.max_seq_len + 1]
        
        # Pad if too short
        if len(tokens) < 2:
            tokens = tokens + [0] * (2 - len(tokens))
        
        # Create input and target (causal LM)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Create loss mask: only compute loss on response tokens, not prompt
        # Find where prompt ends in tokenized form
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)
        
        # Mask: 0 for prompt (ignore loss), 1 for response (compute loss)
        mask = torch.zeros_like(x, dtype=torch.float)
        if prompt_len < len(x):
            mask[prompt_len:] = 1.0
        
        return {
            'input_ids': x,
            'target_ids': y,
            'mask': mask,
            'weight': torch.tensor(weight, dtype=torch.float),
            'rating': rating,
            'prompt_len': prompt_len
        }


def collate_fn(batch):
    """
    Collate function to pad batches to same length.
    """
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    batch_input_ids = []
    batch_target_ids = []
    batch_masks = []
    batch_weights = []
    batch_ratings = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        # Pad sequences
        input_ids = torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        target_ids = torch.cat([item['target_ids'], torch.zeros(pad_len, dtype=torch.long)])
        mask = torch.cat([item['mask'], torch.zeros(pad_len)])
        
        batch_input_ids.append(input_ids)
        batch_target_ids.append(target_ids)
        batch_masks.append(mask)
        batch_weights.append(item['weight'])
        batch_ratings.append(item['rating'])
    
    return {
        'input_ids': torch.stack(batch_input_ids),
        'target_ids': torch.stack(batch_target_ids),
        'mask': torch.stack(batch_masks),
        'weight': torch.stack(batch_weights),
        'rating': batch_ratings
    }


def weighted_cross_entropy_loss(logits, targets, mask, weights):
    """
    Compute weighted cross-entropy loss.
    Only counts loss on response tokens (not prompt).
    Higher rated samples contribute more to loss.
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)
    
    # Compute loss per token (no reduction)
    loss_per_token = nn.functional.cross_entropy(
        logits_flat, targets_flat, reduction='none'
    )
    
    # Apply mask (ignore prompt and padding)
    loss_per_token = loss_per_token * mask_flat
    
    # Reshape to apply sample weights
    loss_per_token = loss_per_token.view(batch_size, seq_len)
    mask_reshaped = mask.view(batch_size, seq_len)
    
    # Sum loss per sample, divide by number of response tokens
    loss_per_sample = loss_per_token.sum(dim=1) / (mask_reshaped.sum(dim=1) + 1e-8)
    
    # Apply rating weights
    weighted_loss = (loss_per_sample * weights).mean()
    
    return weighted_loss


def finetune(
    data_path: str = "data/fine_tune_data.csv",
    checkpoint_dir: str = "checkpoints",
    pretrained_checkpoint: str = "model_epoch5.pt",
    output_dir: str = "checkpoints",
    max_seq_len: int = None,  # None = use pretrained model's value
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    n_epochs: int = 5,
    device: str = "cpu"
):
    """
    Fine-tune pretrained model on prompt-response-rating data.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.load(tokenizer_path)
    
    # Load pretrained model
    pretrained_path = os.path.join(checkpoint_dir, pretrained_checkpoint)
    logger.info(f"Loading pretrained model from {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    config = checkpoint['config']

    max_seq_len = config['max_seq_len']  # Always use pretrained value
    
    # Now create model with correct config
    model = GPTModel(**config, dropout=0.1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Pretrained model loaded successfully")
    
    # Create dataset and dataloader
    dataset = RatingDataset(data_path, tokenizer, max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info("Starting fine-tuning...")
    global_step = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Fine-tune Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            mask = batch['mask'].to(device)
            weights = batch['weight'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute weighted loss (only on response tokens)
            loss = weighted_cross_entropy_loss(logits, target_ids, mask, weights)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            avg_rating = sum(batch['rating']) / len(batch['rating'])
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_rating": f"{avg_rating:.1f}"
            })
            
            if global_step % 50 == 0:
                logger.info(
                    f"Step {global_step}, Loss: {loss.item():.4f}, "
                    f"Avg batch rating: {avg_rating:.1f}"
                )
        
        # Epoch summary
        avg_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            output_dir,
            f"model_finetuned_epoch{epoch+1}.pt"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info("Fine-tuning completed!")
    return model


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/finetune.log", rotation="10 MB", level="DEBUG")
    
    os.makedirs("logs", exist_ok=True)
    
    # Fine-tune
    model = finetune(
        data_path="data/fine_tune_data.csv",
        pretrained_checkpoint="model_epoch5.pt",
        max_seq_len=64,  # (same as pretraining)
        batch_size=4,
        learning_rate=1e-5,
        n_epochs=5,
        device="cpu"
    )
    
    data_path: str = "data/fine_tune_data.csv",
    checkpoint_dir: str = "checkpoints",
    pretrained_checkpoint: str = "model_epoch5.pt",
    output_dir: str = "checkpoints",
    max_seq_len: int = None,  # None = use pretrained model's value
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    n_epochs: int = 5,
    device: str = "cpu"