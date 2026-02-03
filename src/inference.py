# src/inference.py

import torch
import os
import sys
from loguru import logger

from components.bpe_tokenizer import BPETokenizer
from components.model import GPTModel


def load_model(checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
    """
    Load trained model and tokenizer.
    """
    device = torch.device(device)
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model with saved config
    model = GPTModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    tokenizer.load(tokenizer_path)
    
    logger.info(f"Model loaded! Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Config: {config}")
    
    return model, tokenizer, device


def generate_text(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cpu"
) -> str:
    """
    Generate text from prompt.
    """
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded]).to(device)
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Prompt tokens: {encoded}")
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def interactive_mode(model, tokenizer, device):
    """
    Interactive text generation loop.
    """
    print("\n" + "="*50)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        # Get prompt
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        # Get generation parameters
        try:
            max_tokens = int(input("Max new tokens (default 50): ") or "50")
            temp = float(input("Temperature (default 1.0): ") or "1.0")
        except ValueError:
            max_tokens = 50
            temp = 1.0
        
        # Generate
        print("\nGenerating...")
        result = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            device=device
        )
        
        print(f"\nResult: {result}")
        print("-"*50 + "\n")

