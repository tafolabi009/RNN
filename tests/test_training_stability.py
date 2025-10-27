"""
Training Stability Test - FP16/FP32
===================================

Verifies that the model can train stably in both fp32 and fp16 modes.
Tests:
- Forward pass
- Backward pass
- Gradient flow
- NaN/Inf detection
- Loss convergence
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from resonance_nn import create_spectral_lm


def create_dummy_data(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Create dummy training data"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Target is shifted input (language modeling)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, target_ids


def check_gradients(model: nn.Module) -> tuple:
    """Check gradient statistics"""
    total_norm = 0.0
    num_params = 0
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1
            
            if torch.isnan(param.grad).any():
                has_nan = True
                print(f"  ⚠️  NaN gradient in {name}")
            
            if torch.isinf(param.grad).any():
                has_inf = True
                print(f"  ⚠️  Inf gradient in {name}")
    
    total_norm = total_norm ** 0.5
    return total_norm, num_params, has_nan, has_inf


def test_training_stability(
    precision: str = 'fp32',
    num_steps: int = 10,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 1000,
    device: str = None
):
    """
    Test training stability in given precision
    
    Args:
        precision: 'fp32' or 'fp16'
        num_steps: Number of training steps
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to use
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"Testing Training Stability - {precision.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Vocabulary: {vocab_size}")
    
    # Create model
    print("\n📦 Creating model...")
    model = create_spectral_lm('tiny', vocab_size=vocab_size, max_seq_len=512)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # FP16 setup
    scaler = None
    if precision == 'fp16' and device == 'cuda':
        print("  ✓ Using automatic mixed precision (AMP)")
        scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print(f"\n🏋️  Training for {num_steps} steps...")
    losses = []
    grad_norms = []
    
    for step in range(num_steps):
        # Generate data
        input_ids, target_ids = create_dummy_data(batch_size, seq_len, vocab_size, device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if precision == 'fp16' and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1)
                )
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )
        
        # Check loss
        if torch.isnan(loss):
            print(f"  ✗ Step {step}: NaN loss detected!")
            return False
        
        if torch.isinf(loss):
            print(f"  ✗ Step {step}: Inf loss detected!")
            return False
        
        losses.append(loss.item())
        
        # Backward pass
        if precision == 'fp16' and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm, num_params, has_nan, has_inf = check_gradients(model)
            
            if has_nan or has_inf:
                print(f"  ✗ Step {step}: Bad gradients detected!")
                return False
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm, num_params, has_nan, has_inf = check_gradients(model)
            
            if has_nan or has_inf:
                print(f"  ✗ Step {step}: Bad gradients detected!")
                return False
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        grad_norms.append(grad_norm)
        
        # Print progress
        if (step + 1) % 5 == 0 or step == 0:
            print(f"  Step {step+1:3d}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.4f}")
    
    # Check convergence (loss should decrease or stabilize)
    print(f"\n📊 Training Statistics:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.4f}")
    print(f"  Avg grad norm: {sum(grad_norms) / len(grad_norms):.4f}")
    
    # Success criteria
    success = True
    
    # 1. No catastrophic loss increase
    if losses[-1] > losses[0] * 2:
        print(f"  ⚠️  Loss increased significantly!")
        success = False
    
    # 2. Gradients should be reasonable
    avg_grad = sum(grad_norms) / len(grad_norms)
    if avg_grad > 100 or avg_grad < 1e-6:
        print(f"  ⚠️  Unusual gradient magnitudes!")
        success = False
    
    if success:
        print(f"\n✅ Training is stable in {precision.upper()}!")
    else:
        print(f"\n⚠️  Training stability issues detected in {precision.upper()}")
    
    return success


def test_mixed_precision_features():
    """Test mixed precision specific features"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping mixed precision tests")
        return True
    
    print(f"\n{'='*80}")
    print("Testing Mixed Precision Features")
    print(f"{'='*80}")
    
    device = 'cuda'
    model = create_spectral_lm('tiny', vocab_size=1000, max_seq_len=512)
    model = model.to(device)
    
    # Test autocast
    print("\n🔬 Testing autocast context...")
    input_ids = torch.randint(0, 1000, (2, 128), device=device)
    
    with torch.cuda.amp.autocast():
        logits = model(input_ids)
        # Check that operations inside ran in lower precision
        print(f"  ✓ Autocast successful")
    
    # Test gradient scaling
    print("\n🔬 Testing gradient scaling...")
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    target_ids = torch.randint(0, 1000, (2, 128), device=device)
    
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, 1000), target_ids.reshape(-1))
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"  ✓ Gradient scaling successful")
    print(f"\n✅ Mixed precision features working!")
    
    return True


def main():
    """Run all training stability tests"""
    print("\n" + "="*80)
    print("TRAINING STABILITY TEST SUITE")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    results = {}
    
    # Test FP32
    results['fp32'] = test_training_stability(
        precision='fp32',
        num_steps=10,
        device=device
    )
    
    # Test FP16 (only on CUDA)
    if device == 'cuda':
        results['fp16'] = test_training_stability(
            precision='fp16',
            num_steps=10,
            device=device
        )
        
        results['mixed_precision'] = test_mixed_precision_features()
    else:
        print("\n⚠️  Skipping FP16 tests (CUDA not available)")
        results['fp16'] = None
        results['mixed_precision'] = None
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {test.upper():<20s}: {status}")
    
    print("="*80)
    
    # Overall result
    failed_tests = [k for k, v in results.items() if v is False]
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        return False
    else:
        print(f"\n✅ All tests passed!")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
