"""
Quick Benchmark: Spectral vs Transformer (Small Version)
========================================================

Faster benchmark with smaller models for quick testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run the benchmark with smaller settings
if __name__ == '__main__':
    from benchmarks.compare_transformer import run_benchmark, print_summary
    
    print("\n🚀 Running QUICK benchmark (smaller models)...")
    
    results = run_benchmark(
        seq_lengths=[512, 2048, 8192],  # Smaller lengths
        vocab_size=5000,  # Smaller vocab
        embed_dim=256,    # Smaller dimensions
        hidden_dim=256,
        num_layers=4,     # Fewer layers
        num_heads=4,
        batch_size=2,
        device='cpu'
    )
    
    print_summary(results)
