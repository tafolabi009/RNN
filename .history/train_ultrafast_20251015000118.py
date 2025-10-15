"""
ULTRA-FAST TRAINING - Optimized for Speed
==========================================

Key speed optimizations:
1. Pre-tokenized dataset (cache to disk)
2. Multi-process data loading (4 workers)
3. Smaller dataset (10K samples for testing)
4. Compiled model (torch.compile for 2x speedup)
5. Optimized batch size and accumulation

Expected: 1 epoch in 5-10 minutes (vs 2.5 hours!)
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
import pickle
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


class PreTokenizedDataset(Dataset):
    """Pre-tokenized dataset for FAST loading"""
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def prepare_dataset(tokenizer, max_samples=10000, max_length=256, cache_file='tokenized_cache.pkl'):
    """Pre-tokenize and cache dataset"""
    
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        print(f"📦 Loading cached tokenized data from {cache_file}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data['train'])} train / {len(data['val'])} val samples")
        return data
    
    print(f"📦 Loading and tokenizing WikiText-103 (first {max_samples} samples)...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    texts = [ex['text'] for ex in dataset if len(ex['text'].strip()) > 50][:max_samples]
    
    print(f"🔄 Tokenizing {len(texts)} texts...")
    tokenized_samples = []
    
    for text in tqdm(texts, desc="Tokenizing"):
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        
        tokenized_samples.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encoded['attention_mask'].squeeze(0)
        })
    
    # Split train/val
    split_idx = int(len(tokenized_samples) * 0.9)
    train_data = tokenized_samples[:split_idx]
    val_data = tokenized_samples[split_idx:]
    
    data = {'train': train_data, 'val': val_data}
    
    # Cache to disk
    print(f"💾 Caching tokenized data to {cache_file}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Prepared {len(train_data)} train / {len(val_data)} val samples")
    return data


def main():
    print("\n" + "="*80)
    print("ULTRA-FAST TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {gpu_mem:.1f} GB")
    
    # SPEED-OPTIMIZED CONFIG
    config = CONFIGS['tiny']
    config.vocab_size = 50257
    config.max_seq_len = 256
    config.use_gradient_checkpointing = True
    
    print(f"\n📋 Configuration:")
    print(f"   Model: tiny (63M params)")
    print(f"   Max sequence: 256 tokens")
    print(f"   Dataset: 10K samples (for speed testing)")
    print(f"   Batch size: 4 (increased from 2)")
    print(f"   Gradient accumulation: 16 (decreased from 32)")
    print(f"   Data workers: 4 (parallel loading)")
    
    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset (cached)
    data = prepare_dataset(tokenizer, max_samples=10000, max_length=256)
    
    train_dataset = PreTokenizedDataset(data['train'])
    val_dataset = PreTokenizedDataset(data['val'])
    
    # FAST DATA LOADERS with multiple workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Increased from 2
        shuffle=True,
        num_workers=4,  # Parallel loading!
        pin_memory=True,
        persistent_workers=True  # Keep workers alive
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\n✅ Data loaders created:")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    print(f"   Workers: 4 (train) / 2 (val)")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = SpectralLanguageModel(config)
    model = model.to(device)
    
    # Try to compile model (PyTorch 2.0+) for 2x speedup
    try:
        print(f"⚡ Compiling model with torch.compile()...")
        model = torch.compile(model)
        print(f"✅ Model compiled! (expect 2x speedup)")
    except Exception as e:
        print(f"⚠️  Could not compile model: {e}")
        print(f"   (torch.compile requires PyTorch 2.0+)")
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Model ready: {num_params:.1f}M parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = GradScaler()
    
    # Training loop
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    num_epochs = 3  # Reduced for speed testing
    gradient_accumulation_steps = 16  # Reduced from 32
    log_interval = 50
    
    output_dir = Path('checkpoints_fast')
    output_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            step_start = time.time()
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
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
            
            step_time = time.time() - step_start
            samples_per_sec = 4 / step_time  # batch_size / time
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                perplexity = math.exp(avg_loss)
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'samples/s': f'{samples_per_sec:.1f}'
                })
        
        epoch_time = time.time() - epoch_start
        print(f"\n⏱️  Epoch time: {epoch_time/60:.1f} minutes")
        
        # Validation
        print(f"🔍 Evaluating...")
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                
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
            
            checkpoint_path = output_dir / 'spectral_ultrafast_best.pth'
            
            # Save without torch.compile wrapper if present
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
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
    print(f"Model saved: {output_dir / 'spectral_ultrafast_best.pth'}")
    print(f"\nTest with:")
    print(f"  python inference.py --checkpoint {output_dir / 'spectral_ultrafast_best.pth'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
