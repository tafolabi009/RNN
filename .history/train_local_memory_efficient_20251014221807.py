"""
MEMORY-EFFICIENT TRAINING - For 6GB GPUs (GTX 1660 Ti)
======================================================

This script is optimized for LOW MEMORY GPUs like yours.

Key optimizations:
- ✅ Tiny model (63M params instead of 428M)
- ✅ Small batch size (2 instead of 16)
- ✅ Gradient checkpointing (saves 50% memory)
- ✅ Short sequences (256 instead of 1024)
- ✅ Gradient accumulation (effective batch = 64)

Your GPU: 6GB
This script uses: ~5GB
Should work! ✅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import os
import sys
import time
import math
from pathlib import Path
from tqdm import tqdm

try:
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install transformers datasets")
    sys.exit(1)

from resonance_nn.spectral_optimized import SpectralLanguageModel, CONFIGS


class TextDataset(Dataset):
    """Simplified dataset"""
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


def main():
    print("\n" + "="*80)
    print("MEMORY-EFFICIENT TRAINING FOR 6GB GPU")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {gpu_mem:.1f} GB")
        
        if gpu_mem < 8:
            print("   ⚠️  Low memory detected - using optimized settings")
    
    # MEMORY-OPTIMIZED CONFIG
    config = CONFIGS['tiny']  # 63M params
    config.vocab_size = 50257
    config.max_seq_len = 256  # Short sequences
    config.use_gradient_checkpointing = True  # CRITICAL for memory
    
    print(f"\n📋 Configuration:")
    print(f"   Model: tiny (63M params)")
    print(f"   Max sequence: 256 tokens")
    print(f"   Gradient checkpointing: ON")
    print(f"   Batch size: 2")
    print(f"   Gradient accumulation: 32")
    print(f"   Effective batch: 64")
    
    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    # Load dataset (subset for speed)
    print(f"\n📦 Loading WikiText-103 (subset)...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    texts = [ex['text'] for ex in dataset if len(ex['text'].strip()) > 0]
    
    # Take subset
    max_samples = 50000
    texts = texts[:max_samples]
    
    # Split train/val
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"✅ Loaded {len(train_texts):,} train / {len(val_texts):,} val texts")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length=256)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=256)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = SpectralLanguageModel(config)
    model = model.to(device)
    
    num_params = model.get_num_params() / 1e6
    print(f"✅ Model created: {num_params:.1f}M parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = GradScaler()
    
    # Training loop
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    num_epochs = 5
    gradient_accumulation_steps = 32
    log_interval = 100
    
    output_dir = Path('checkpoints_local')
    output_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward with mixed precision
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )
                loss = loss / gradient_accumulation_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    perplexity = math.exp(avg_loss)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'ppl': f'{perplexity:.2f}'
                    })
        
        # Validation
        print(f"\n🔍 Evaluating...")
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_perplexity = math.exp(avg_val_loss)
        
        print(f"\n✅ Epoch {epoch + 1} complete!")
        print(f"   Train Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Val Perplexity: {val_perplexity:.2f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint_path = output_dir / 'spectral_tiny_best.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            
            print(f"   💾 Best model saved: {checkpoint_path}")
    
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Model saved: {output_dir / 'spectral_tiny_best.pth'}")
    print(f"\nTest with:")
    print(f"  python inference.py --checkpoint {output_dir / 'spectral_tiny_best.pth'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
