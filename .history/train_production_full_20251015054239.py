"""
PRODUCTION TRAINING - Full WikiText-103
=======================================

This will train on the FULL dataset for better quality:
- ✅ Full WikiText-103 (1.8M articles, not 10K)
- ✅ 10 epochs (not 3)
- ✅ Longer sequences (512 tokens, not 256)
- ✅ Better optimization

Expected results:
- Perplexity: <20 (vs current 22.22)
- No repetition
- Coherent paragraphs
- Time: ~8 hours on your GPU

If that's too long, use Colab T4 (~3 hours)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast as torch_autocast, GradScaler as TorchGradScaler

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


def prepare_full_dataset(tokenizer, max_length=512, cache_file='tokenized_full_cache.pkl'):
    """Pre-tokenize and cache FULL WikiText-103 dataset"""
    
    cache_path = Path(cache_file)
    
    if cache_path.exists():
        print(f"📦 Loading cached tokenized data from {cache_file}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data['train'])} train / {len(data['val'])} val samples")
        return data
    
    print(f"📦 Loading FULL WikiText-103 dataset...")
    print(f"   This will take ~10-15 minutes (one-time cost)")
    
    dataset_train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    dataset_val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    
    # Filter out empty texts
    train_texts = [ex['text'] for ex in dataset_train if len(ex['text'].strip()) > 50]
    val_texts = [ex['text'] for ex in dataset_val if len(ex['text'].strip()) > 50]
    
    print(f"📝 Training texts: {len(train_texts):,}")
    print(f"📝 Validation texts: {len(val_texts):,}")
    
    print(f"\n🔄 Tokenizing training data...")
    train_tokenized = []
    for text in tqdm(train_texts, desc="Train"):
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
        
        train_tokenized.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encoded['attention_mask'].squeeze(0)
        })
    
    print(f"\n🔄 Tokenizing validation data...")
    val_tokenized = []
    for text in tqdm(val_texts, desc="Val"):
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
        
        val_tokenized.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': encoded['attention_mask'].squeeze(0)
        })
    
    data = {'train': train_tokenized, 'val': val_tokenized}
    
    # Cache to disk
    print(f"\n💾 Caching tokenized data to {cache_file}...")
    print(f"   This may take a few minutes...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Cached! Future runs will be instant.")
    return data


def main():
    print("\n" + "="*80)
    print("PRODUCTION TRAINING - FULL WIKITEXT-103")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {gpu_mem:.1f} GB")
        
        if gpu_mem < 8:
            print("   ⚠️  Warning: 6GB GPU will be slow")
            print("   💡 Consider using Google Colab T4 (free, 15GB)")
    
    # PRODUCTION CONFIG
    config = CONFIGS['tiny']
    config.vocab_size = 50257
    config.max_seq_len = 512  # Longer sequences!
    config.use_gradient_checkpointing = True
    
    print(f"\n📋 Configuration:")
    print(f"   Model: tiny (63M params)")
    print(f"   Max sequence: 512 tokens")
    print(f"   Dataset: FULL WikiText-103")
    print(f"   Batch size: 4")
    print(f"   Gradient accumulation: 16")
    print(f"   Effective batch: 64")
    print(f"   Epochs: 10")
    
    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare FULL dataset
    data = prepare_full_dataset(tokenizer, max_length=512)
    
    train_dataset = PreTokenizedDataset(data['train'])
    val_dataset = PreTokenizedDataset(data['val'])
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
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
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = SpectralLanguageModel(config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Model ready: {num_params:.1f}M parameters")
    
    # Optimizer with cosine lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    scaler = TorchGradScaler('cuda')
    
    # LR scheduler
    num_epochs = 10
    num_training_steps = len(train_loader) * num_epochs // 16  # Account for grad accumulation
    warmup_steps = num_training_steps // 10
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\n{'='*80}")
    print("STARTING PRODUCTION TRAINING")
    print(f"{'='*80}")
    print(f"Total steps: {num_training_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")
    
    gradient_accumulation_steps = 16
    log_interval = 100
    
    output_dir = Path('checkpoints_production')
    output_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"📊 Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        epoch_start = time.time()
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            # Forward with mixed precision
            with torch_autocast('cuda'):
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
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                perplexity = math.exp(avg_loss)
                current_lr = scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{current_lr:.2e}'
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
        
        # Save checkpoint every epoch
        checkpoint_path = output_dir / f'spectral_epoch_{epoch+1}.pth'
        model_to_save = model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
            'global_step': global_step,
            'val_loss': avg_val_loss,
            'val_perplexity': val_perplexity
        }, checkpoint_path)
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            best_path = output_dir / 'spectral_best.pth'
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, best_path)
            
            print(f"   🌟 New best model! Saved: {best_path}")
    
    print(f"\n{'='*80}")
    print("🎉 PRODUCTION TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Best model: {output_dir / 'spectral_best.pth'}")
    print(f"\nTest with:")
    print(f"  python inference.py --checkpoint {output_dir / 'spectral_best.pth'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
