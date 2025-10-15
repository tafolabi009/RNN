"""
PRODUCTION TRAINING SCRIPT - Spectral Neural Networks
=====================================================

Train Spectral models on REAL datasets with proper tokenization.

Features:
- ✅ Proper BPE tokenization (HuggingFace)
- ✅ Real datasets: WikiText-103, OpenWebText, C4
- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient accumulation
- ✅ Learning rate scheduling
- ✅ Checkpointing
- ✅ Wandb logging
- ✅ Multi-GPU support

Usage:
    # Train on WikiText-103
    python train_production.py --dataset wikitext --model_size base --epochs 20
    
    # Train on larger dataset
    python train_production.py --dataset openwebtext --model_size large --batch_size 8
    
    # Resume from checkpoint
    python train_production.py --resume checkpoints/spectral_base_latest.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict

# HuggingFace libraries
try:
    from transformers import GPT2TokenizerFast, AutoTokenizer
    from datasets import load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers and datasets not installed. Install with:")
    print("   pip install transformers datasets")

# Wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Logging disabled.")

# Our model
sys.path.insert(0, str(Path(__file__).parent))
from resonance_nn.spectral_optimized import SpectralLanguageModel, SpectralConfig, CONFIGS


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_size: str = 'base'
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # Data
    dataset: str = 'wikitext'  # wikitext, openwebtext, c4
    train_split: str = 'train'
    val_split: str = 'validation'
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 20
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    
    # Optimization
    use_amp: bool = True  # Mixed precision
    amp_dtype: str = 'fp16'  # fp16 or bf16
    use_gradient_checkpointing: bool = False
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    use_wandb: bool = False
    wandb_project: str = 'spectral-lm'
    
    # Checkpoints
    output_dir: str = 'checkpoints'
    resume: Optional[str] = None
    
    # Distributed
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    def __post_init__(self):
        """Create output directory"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    """Dataset for language modeling with proper tokenization"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        return_tensors: bool = True
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt' if self.return_tensors else None
        )
        
        if self.return_tensors:
            input_ids = encoded['input_ids'].squeeze(0)
            
            # Create labels (shifted input_ids)
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = self.tokenizer.pad_token_id or 0
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        else:
            return encoded


def load_text_dataset(
    dataset_name: str,
    split: str = 'train',
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Load dataset from HuggingFace datasets.
    
    Supported: wikitext, openwebtext, c4
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and datasets required. Install with: pip install transformers datasets")
    
    print(f"\n📦 Loading {dataset_name} ({split})...")
    
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        texts = [ex['text'] for ex in dataset if len(ex['text'].strip()) > 0]
    
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext', split=split)
        texts = [ex['text'] for ex in dataset]
    
    elif dataset_name == 'c4':
        dataset = load_dataset('c4', 'en', split=split, streaming=True)
        # For streaming datasets, take first N samples
        texts = []
        for i, ex in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            texts.append(ex['text'])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if max_samples:
        texts = texts[:max_samples]
    
    print(f"✅ Loaded {len(texts):,} texts")
    
    return texts


def create_dataloaders(
    config: TrainingConfig,
    tokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Load datasets
    train_texts = load_text_dataset(
        config.dataset,
        config.train_split,
        config.max_train_samples
    )
    
    val_texts = load_text_dataset(
        config.dataset,
        config.val_split,
        config.max_val_samples
    )
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, config.max_seq_len)
    
    # Create dataloaders
    train_sampler = DistributedSampler(train_dataset) if config.world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Production-grade trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and config.amp_dtype == 'fp16' else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Wandb
        if config.use_wandb and WANDB_AVAILABLE and config.rank == 0:
            wandb.init(
                project=config.wandb_project,
                config=asdict(config),
                name=f"spectral_{config.model_size}"
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate, betas=(0.9, 0.95))
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            else:
                progress = (step - self.config.warmup_steps) / max(1, len(self.train_loader) * self.config.num_epochs - self.config.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler
    
    def train_step(self, batch: Dict) -> Tuple[float, float]:
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward
        with autocast(enabled=self.config.use_amp, dtype=torch.float16 if self.config.amp_dtype == 'fp16' else torch.bfloat16):
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0  # Pad token
            )
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Calculate perplexity
        perplexity = torch.exp(loss * self.config.gradient_accumulation_steps)
        
        return loss.item() * self.config.gradient_accumulation_steps, perplexity.item()
    
    def optimizer_step(self):
        """Update optimizer with gradient clipping"""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", disable=self.config.rank != 0):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save training checkpoint"""
        if self.config.rank != 0:
            return
        
        checkpoint_path = Path(self.config.output_dir) / f"spectral_{self.config.model_size}_{name}.pth"
        
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.module.config if hasattr(self.model, 'module') else self.model.config,
            'training_config': asdict(self.config),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        print(f"   Epoch: {self.epoch}, Step: {self.global_step}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"TRAINING SPECTRAL LANGUAGE MODEL")
        print(f"{'='*80}")
        print(f"Model size: {self.config.model_size}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.use_amp} ({self.config.amp_dtype})")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        # Load checkpoint if resuming
        if self.config.resume:
            self.load_checkpoint(self.config.resume)
        
        self.model.train()
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            if self.config.rank == 0:
                print(f"\n📊 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            epoch_steps = 0
            
            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}",
                disable=self.config.rank != 0
            )
            
            for step, batch in progress_bar:
                # Train step
                loss, perplexity = self.train_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0 and self.config.rank == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'ppl': f'{perplexity:.2f}',
                            'lr': f'{lr:.2e}'
                        })
                        
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/perplexity': perplexity,
                                'train/learning_rate': lr,
                                'train/epoch': epoch,
                                'train/step': self.global_step
                            })
                    
                    # Evaluation
                    if self.global_step % self.config.eval_interval == 0:
                        if self.config.rank == 0:
                            print(f"\n🔍 Evaluating at step {self.global_step}...")
                        
                        metrics = self.evaluate()
                        
                        if self.config.rank == 0:
                            print(f"   Val Loss: {metrics['val_loss']:.4f}")
                            print(f"   Val Perplexity: {metrics['val_perplexity']:.2f}")
                            
                            if self.config.use_wandb and WANDB_AVAILABLE:
                                wandb.log({
                                    'val/loss': metrics['val_loss'],
                                    'val/perplexity': metrics['val_perplexity'],
                                    'val/step': self.global_step
                                })
                            
                            # Save best model
                            if metrics['val_loss'] < self.best_val_loss:
                                self.best_val_loss = metrics['val_loss']
                                self.save_checkpoint('best')
                                print(f"   💾 New best model saved!")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint('latest')
            
            # End of epoch
            if self.config.rank == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"\n✅ Epoch {epoch + 1} complete!")
                print(f"   Avg Loss: {avg_loss:.4f}")
                print(f"   Avg Perplexity: {math.exp(avg_loss):.2f}")
                
                self.save_checkpoint(f'epoch_{epoch + 1}')
        
        if self.config.rank == 0:
            print(f"\n{'='*80}")
            print("🎉 TRAINING COMPLETE!")
            print(f"{'='*80}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best validation perplexity: {math.exp(self.best_val_loss):.2f}")
            print(f"Total steps: {self.global_step}")
            print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Spectral Language Model')
    
    # Model
    parser.add_argument('--model_size', type=str, default='base', choices=list(CONFIGS.keys()))
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    
    # Data
    parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'openwebtext', 'c4'])
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='spectral-lm')
    
    # Checkpoints
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TRANSFORMERS_AVAILABLE:
        print("❌ transformers and datasets required!")
        print("   Install with: pip install transformers datasets")
        return
    
    # Create config
    config = TrainingConfig(**vars(args))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    # Update vocab size
    config.vocab_size = len(tokenizer)
    
    # Create model
    print(f"\n🏗️  Creating model: {config.model_size}")
    model_config = CONFIGS[config.model_size]
    model_config.vocab_size = config.vocab_size
    model_config.max_seq_len = config.max_seq_len
    model_config.use_gradient_checkpointing = config.use_gradient_checkpointing
    
    model = SpectralLanguageModel(model_config)
    model = model.to(device)
    
    print(f"✅ Model created: {model.get_num_params()/1e6:.1f}M parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, tokenizer)
    
    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader, device)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
