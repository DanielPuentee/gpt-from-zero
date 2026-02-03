# src/components/bpe_tokenizer.py

import json
import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from loguru import logger

class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer.
    Implements the original BPE algorithm with pre-tokenization.
    """
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.vocab = {}  # index -> bytes or string
        self.merges = []  # List of (pair, new_id) tuples
        
        # Special tokens (will be added after training)
        self.special_tokens = {
            '<|endoftext|>': vocab_size - 1,  # Last ID
            '<|pad|>': vocab_size - 2,        # Second to last
        }
        
        # Regex for pre-tokenization (GPT-2 style)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def _get_stats(self, tokens: List[Tuple[int, ...]]) -> Dict[Tuple[int, int], int]:
        """Count frequency of adjacent pairs."""
        pairs = defaultdict(int)
        for word in tokens:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += 1
        return pairs
    
    def _merge_vocab(self, pair: Tuple[int, int], tokens: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Merge all occurrences of pair in tokens."""
        new_tokens = []
        bigram = pair
        new_id = bigram[0] * 256 + bigram[1]  # This was the bug!
        
        for word in tokens:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_tokens.append(tuple(new_word))
        return new_tokens
    
    def train(self, text: str):
        """
        Train BPE on text.
        Start with byte-level (0-255) and merge up to vocab_size.
        """
        logger.info(f"Training BPE with target vocab size: {self.vocab_size}")
        
        # Pre-tokenize
        words = self.pat.findall(text)
        logger.info(f"Found {len(words)} word pieces")
        
        # Convert to bytes (initial vocab is 0-255)
        tokens = [tuple(bytearray(word.encode('utf-8'))) for word in words]
        
        # Initial vocab: bytes 0-255 map to themselves
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256  # Next available ID for merged tokens
        
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                logger.warning(f"No more pairs to merge at step {i}")
                break
            
            best = max(stats, key=stats.get)
            
            # Create new token by concatenating the byte sequences
            left_bytes = self.vocab[best[0]]
            right_bytes = self.vocab[best[1]]
            self.vocab[next_id] = left_bytes + right_bytes
            
            # Store the merge rule
            self.merges.append((best, next_id))
            
            # Apply merge to corpus
            tokens = self._merge_vocab(best, tokens, next_id)
            
            next_id += 1
            
            # if (i + 1) % 50 == 0 or i < 5:
            #     logger.info(f"  Merge {i+1}/{num_merges}: {best} -> {next_id-1} (freq: {stats[best]})")
        
        # Add special tokens to vocab
        for token_name, idx in self.special_tokens.items():
            self.vocab[idx] = token_name.encode('utf-8')
        
        logger.success(f"\nFinal vocab size: {len(self.vocab)}")
        # logger.info(f"Actual merges performed: {len(self.merges)}")
    
    def _merge_vocab(self, pair: Tuple[int, int], tokens: List[Tuple[int, ...]], new_id: int) -> List[Tuple[int, ...]]:
        """Merge all occurrences of pair in tokens."""
        new_tokens = []
        bigram = pair
        
        for word in tokens:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_tokens.append(tuple(new_word))
        return new_tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []
        
        words = self.pat.findall(text)
        ids = []
        
        for word in words:
            word_bytes = tuple(bytearray(word.encode('utf-8')))
            
            # Apply merges in order
            for pair, new_id in self.merges:
                new_word = []
                i = 0
                while i < len(word_bytes):
                    if i < len(word_bytes) - 1 and word_bytes[i] == pair[0] and word_bytes[i+1] == pair[1]:
                        new_word.append(new_id)
                        i += 2
                    else:
                        new_word.append(word_bytes[i])
                        i += 1
                word_bytes = tuple(new_word)
            
            ids.extend(list(word_bytes))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        bytes_list = []
        for idx in ids:
            if idx in self.vocab:
                token_bytes = self.vocab[idx]
                if isinstance(token_bytes, bytes):
                    bytes_list.append(token_bytes)
                else:
                    bytes_list.append(str(token_bytes).encode('utf-8'))
            else:
                bytes_list.append(b'<?>')
        
        try:
            return b''.join(bytes_list).decode('utf-8', errors='replace')
        except:
            return ''.join(str(b) for b in bytes_list)
    
    def encode_with_special(self, text: str, add_eot: bool = True) -> List[int]:
        """Encode with special tokens."""
        ids = self.encode(text)
        if add_eot:
            ids.append(self.special_tokens['<|endoftext|>'])
        return ids
    
    def save(self, path: str):
        """Save tokenizer to disk."""
        # Convert bytes to list for JSON serialization
        serializable_vocab = {}
        for k, v in self.vocab.items():
            if isinstance(v, bytes):
                serializable_vocab[str(k)] = list(v)
            else:
                serializable_vocab[str(k)] = v
        
        data = {
            'vocab': serializable_vocab,
            'merges': [[list(m[0]), m[1]] for m in self.merges],
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        
        # Reconstruct vocab
        self.vocab = {}
        for k, v in data['vocab'].items():
            if isinstance(v, list):
                self.vocab[int(k)] = bytes(v)
            else:
                self.vocab[int(k)] = v
        
        # Reconstruct merges
        self.merges = [(tuple(m[0]), m[1]) for m in data['merges']]
