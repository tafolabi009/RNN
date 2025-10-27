"""
Benchmark: Spectral Neural Network vs Transformer
==================================================

Compares forward time and memory usage between:
- SpectralLanguageModel (FFT-based, O(n log n))
- Standard Transformer (attention-based, O(n²))

Sequence lengths tested: [512, 8192, 65536]
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resonance_nn import create_spectral_lm, SpectralConfig


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class StandardTransformer(nn.Module):
    """Standard transformer for comparison"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embed_dim) if embed_dim != hidden_dim else nn.Identity()
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        x = self.input_proj(x)
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        
        # Process through blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)
        logits = self.lm_head(x)
        
        return logits


def measure_time_and_memory(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: str,
    num_warmup: int = 3,
    num_runs: int = 10
) -> Tuple[float, float, float]:
    """
    Measure forward pass time and memory usage
    
    Returns:
        avg_time: Average time in milliseconds
        peak_memory: Peak memory in MB
        throughput: Tokens per second
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(input_ids)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    
    # Memory
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        peak_memory = 0  # Not available for CPU
    
    # Throughput
    batch_size, seq_len = input_ids.shape
    throughput = (batch_size * seq_len) / (avg_time / 1000)  # tokens/sec
    
    return avg_time, peak_memory, throughput


def run_benchmark(
    seq_lengths: List[int] = [512, 8192, 65536],
    vocab_size: int = 50257,
    embed_dim: int = 768,
    hidden_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    batch_size: int = 2,
    device: str = None
) -> Dict:
    """
    Run comprehensive benchmark comparing Spectral vs Transformer
    
    Returns:
        results: Dictionary with benchmark results
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("BENCHMARK: Spectral Neural Network vs Standard Transformer")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Sequence lengths: {seq_lengths}")
    print("="*80)
    
    results = {
        'spectral': {},
        'transformer': {},
        'speedup': {},
        'memory_ratio': {}
    }
    
    for seq_len in seq_lengths:
        print(f"\n{'='*80}")
        print(f"Testing sequence length: {seq_len:,}")
        print(f"{'='*80}")
        
        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Skip transformer for very long sequences (OOM risk)
        if seq_len > 16384 and device == 'cuda':
            mem_available = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB
            if mem_available < 40:  # Less than 40GB VRAM
                print(f"\n⚠️  Skipping Transformer @ {seq_len} (likely OOM with {mem_available:.1f}GB VRAM)")
                results['transformer'][seq_len] = None
                
                # Only test Spectral
                print(f"\n📊 Spectral Model @ {seq_len:,} tokens:")
                try:
                    config = SpectralConfig(
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        max_seq_len=max(seq_lengths),
                        sparsity=0.15
                    )
                    spectral = create_spectral_lm('base', vocab_size=vocab_size, max_seq_len=max(seq_lengths))
                    spectral = spectral.to(device)
                    
                    spec_time, spec_mem, spec_throughput = measure_time_and_memory(spectral, input_ids, device)
                    
                    print(f"  ✓ Time: {spec_time:.2f} ms")
                    print(f"  ✓ Memory: {spec_mem:.2f} MB")
                    print(f"  ✓ Throughput: {spec_throughput:,.0f} tokens/sec")
                    
                    results['spectral'][seq_len] = {
                        'time': spec_time,
                        'memory': spec_mem,
                        'throughput': spec_throughput
                    }
                    
                    del spectral
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    results['spectral'][seq_len] = None
                
                continue
        
        # Test Spectral Model
        print(f"\n📊 Spectral Model @ {seq_len:,} tokens:")
        try:
            config = SpectralConfig(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max(seq_lengths),
                sparsity=0.15
            )
            spectral = create_spectral_lm('base', vocab_size=vocab_size, max_seq_len=max(seq_lengths))
            spectral = spectral.to(device)
            
            spec_time, spec_mem, spec_throughput = measure_time_and_memory(spectral, input_ids, device)
            
            print(f"  ✓ Time: {spec_time:.2f} ms")
            print(f"  ✓ Memory: {spec_mem:.2f} MB")
            print(f"  ✓ Throughput: {spec_throughput:,.0f} tokens/sec")
            
            results['spectral'][seq_len] = {
                'time': spec_time,
                'memory': spec_mem,
                'throughput': spec_throughput
            }
            
            del spectral
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['spectral'][seq_len] = None
        
        # Test Transformer
        print(f"\n📊 Standard Transformer @ {seq_len:,} tokens:")
        try:
            transformer = StandardTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=max(seq_lengths),
                dropout=0.1
            ).to(device)
            
            trans_time, trans_mem, trans_throughput = measure_time_and_memory(transformer, input_ids, device)
            
            print(f"  ✓ Time: {trans_time:.2f} ms")
            print(f"  ✓ Memory: {trans_mem:.2f} MB")
            print(f"  ✓ Throughput: {trans_throughput:,.0f} tokens/sec")
            
            results['transformer'][seq_len] = {
                'time': trans_time,
                'memory': trans_mem,
                'throughput': trans_throughput
            }
            
            del transformer
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  ✗ Out of Memory!")
                results['transformer'][seq_len] = None
            else:
                print(f"  ✗ Error: {e}")
                results['transformer'][seq_len] = None
        
        # Compute speedup
        if results['spectral'][seq_len] and results['transformer'][seq_len]:
            speedup = results['transformer'][seq_len]['time'] / results['spectral'][seq_len]['time']
            mem_ratio = results['spectral'][seq_len]['memory'] / results['transformer'][seq_len]['memory']
            
            results['speedup'][seq_len] = speedup
            results['memory_ratio'][seq_len] = mem_ratio
            
            print(f"\n🚀 Comparison @ {seq_len:,} tokens:")
            print(f"  Speedup: {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")
            print(f"  Memory: {mem_ratio:.2f}x {'MORE' if mem_ratio > 1 else 'LESS'}")
    
    return results


def print_summary(results: Dict):
    """Print summary table"""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n📊 Forward Pass Time (ms):")
    print(f"{'Seq Length':<15} {'Spectral':<15} {'Transformer':<15} {'Speedup':<15}")
    print("-" * 60)
    
    for seq_len in sorted(results['spectral'].keys()):
        spec = results['spectral'][seq_len]
        trans = results['transformer'][seq_len]
        
        spec_str = f"{spec['time']:.2f}" if spec else "N/A"
        trans_str = f"{trans['time']:.2f}" if trans else "OOM"
        speedup_str = f"{results['speedup'].get(seq_len, 0):.2f}x" if seq_len in results['speedup'] else "N/A"
        
        print(f"{seq_len:<15,} {spec_str:<15} {trans_str:<15} {speedup_str:<15}")
    
    print("\n💾 Peak Memory (MB):")
    print(f"{'Seq Length':<15} {'Spectral':<15} {'Transformer':<15} {'Ratio':<15}")
    print("-" * 60)
    
    for seq_len in sorted(results['spectral'].keys()):
        spec = results['spectral'][seq_len]
        trans = results['transformer'][seq_len]
        
        spec_str = f"{spec['memory']:.2f}" if spec else "N/A"
        trans_str = f"{trans['memory']:.2f}" if trans else "OOM"
        ratio_str = f"{results['memory_ratio'].get(seq_len, 0):.2f}x" if seq_len in results['memory_ratio'] else "N/A"
        
        print(f"{seq_len:<15,} {spec_str:<15} {trans_str:<15} {ratio_str:<15}")
    
    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80)


if __name__ == '__main__':
    # Run benchmark
    results = run_benchmark(
        seq_lengths=[512, 8192, 65536],
        batch_size=2,
        num_layers=12,
        num_heads=12
    )
    
    # Print summary
    print_summary(results)
