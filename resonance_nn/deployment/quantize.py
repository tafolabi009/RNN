"""
Edge deployment optimization for spectral networks
Features: INT8 quantization, pruning, knowledge distillation
Target: Mobile/edge devices with limited compute
"""
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Any, Optional
import copy


class QuantizedSpectralLayer(nn.Module):
    """Quantization-friendly spectral layer"""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Layers
        self.norm = nn.LayerNorm(d_model)
        self.spectral_weights = nn.Parameter(torch.randn(257, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Note: FFT operations are kept in FP32
        residual = x
        
        # Quantize input
        x_quant = self.quant(x)
        x = self.dequant(x_quant)
        
        x = self.norm(x)
        
        # Spectral processing (FP32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weights = torch.sigmoid(self.spectral_weights)
        x_fft = x_fft * weights
        x = torch.fft.irfft(x_fft, n=residual.size(1), dim=1, norm='ortho')
        
        # Quantize again
        x = self.quant(x)
        x = self.dequant(x)
        
        x = residual + self.dropout(x)
        x = x + self.ffn(x)
        
        return x


class QuantizableSpectralClassifier(nn.Module):
    """Spectral classifier ready for quantization"""
    
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_layers=4,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Embedding (not quantized)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Spectral layers
        self.layers = nn.ModuleList([
            QuantizedSpectralLayer(d_model, dropout)
            for _ in range(num_layers)
        ])
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = self.embed_dropout(x)
        
        # Quantize
        x = self.quant(x)
        
        # Spectral layers
        for layer in self.layers:
            x = layer(x)
        
        # Dequantize
        x = self.dequant(x)
        
        # Classification
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        x = self.classifier(x)
        
        return x


def quantize_model(model: nn.Module, calibration_loader, device) -> nn.Module:
    """
    Quantize model to INT8 for faster inference
    
    Args:
        model: Model to quantize
        calibration_loader: Data loader for calibration
        device: Device to run on
    
    Returns:
        Quantized model
    """
    # Prepare model for quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Fuse modules
    model_fused = torch.quantization.fuse_modules(model, [['norm', 'dropout']], inplace=False)
    
    # Prepare
    model_prepared = torch.quantization.prepare(model_fused)
    
    # Calibrate with sample data
    print("Calibrating quantized model...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            if batch_idx >= 100:  # Use 100 batches for calibration
                break
            data = data.to(device)
            model_prepared(data)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    print("✓ Model quantized to INT8")
    return model_quantized


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Prune model weights
    
    Args:
        model: Model to prune
        amount: Fraction of weights to prune (0.0 - 1.0)
    
    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune
    
    print(f"Pruning {amount*100:.0f}% of model weights...")
    
    # Prune linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    
    # Count remaining parameters
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    
    print(f"✓ Model pruned: {nonzero_params:,} / {total_params:,} params remaining")
    print(f"  Sparsity: {(1 - nonzero_params/total_params)*100:.1f}%")
    
    return model


class KnowledgeDistillation:
    """Knowledge distillation for model compression"""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        
        self.teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Combined distillation + hard label loss"""
        
        # Soft targets from teacher
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        distill_loss = nn.functional.kl_div(
            soft_prob,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard label loss
        hard_loss = nn.functional.cross_entropy(student_logits, labels)
        
        # Combined
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return loss, distill_loss.item(), hard_loss.item()
    
    def train_student(
        self,
        train_loader,
        optimizer,
        num_epochs: int = 5
    ):
        """Train student model with distillation"""
        
        self.student.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_distill = 0
            total_hard = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                
                # Student forward
                optimizer.zero_grad()
                student_logits = self.student(data)
                
                # Loss
                loss, distill_loss, hard_loss = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )
                
                loss.backward()
                optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                total_distill += distill_loss
                total_hard += hard_loss
                
                pred = student_logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f} '
                          f'(Distill: {distill_loss:.4f}, Hard: {hard_loss:.4f})')
            
            acc = 100. * correct / total
            print(f'Epoch {epoch+1} Complete: '
                  f'Loss: {total_loss/len(train_loader):.4f} '
                  f'Acc: {acc:.2f}%')
        
        print("✓ Student training complete")


def compare_model_sizes(original_model, compressed_models: Dict[str, nn.Module]):
    """Compare sizes of original vs compressed models"""
    
    print("\n" + "=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)
    
    # Save models to get actual file sizes
    import tempfile
    import os
    
    # Original model
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(original_model.state_dict(), f.name)
        original_size_mb = os.path.getsize(f.name) / 1024 / 1024
        os.unlink(f.name)
    
    print(f"{'Model':<20} {'Size (MB)':<15} {'vs Original':<15} {'Speedup Est.'}")
    print("-" * 80)
    print(f"{'Original':<20} {original_size_mb:<14.2f} {'baseline':<15} {'1.0x'}")
    
    for name, model in compressed_models.items():
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size_mb = os.path.getsize(f.name) / 1024 / 1024
            os.unlink(f.name)
        
        ratio = original_size_mb / size_mb
        
        # Estimate speedup (rough approximation)
        if 'quantized' in name.lower():
            speedup = 2.0  # INT8 is ~2x faster
        elif 'pruned' in name.lower():
            speedup = 1.5  # 30% pruning ~1.5x faster
        elif 'distilled' in name.lower():
            speedup = 1.8  # Smaller model ~1.8x faster
        else:
            speedup = 1.0
        
        print(f"{name:<20} {size_mb:<14.2f} {ratio:<14.2f}x {speedup:.1f}x")
    
    print("=" * 80)


def create_edge_optimized_model(
    original_model: nn.Module,
    calibration_loader,
    train_loader,
    device,
    methods: list = ['quantize', 'prune', 'distill']
) -> Dict[str, nn.Module]:
    """
    Create edge-optimized versions of model
    
    Args:
        original_model: Original model
        calibration_loader: Data for quantization calibration
        train_loader: Data for distillation training
        device: Device
        methods: List of optimization methods to apply
    
    Returns:
        Dictionary of optimized models
    """
    optimized_models = {}
    
    # 1. Quantization
    if 'quantize' in methods:
        print("\n" + "=" * 80)
        print("QUANTIZATION")
        print("=" * 80)
        
        # Create quantizable version
        quantizable = copy.deepcopy(original_model)
        quantized = quantize_model(quantizable, calibration_loader, device)
        optimized_models['Quantized (INT8)'] = quantized
    
    # 2. Pruning
    if 'prune' in methods:
        print("\n" + "=" * 80)
        print("PRUNING")
        print("=" * 80)
        
        pruned = copy.deepcopy(original_model)
        pruned = prune_model(pruned, amount=0.3)
        optimized_models['Pruned (30%)'] = pruned
    
    # 3. Knowledge Distillation
    if 'distill' in methods:
        print("\n" + "=" * 80)
        print("KNOWLEDGE DISTILLATION")
        print("=" * 80)
        
        # Create smaller student model
        if hasattr(original_model, 'd_model'):
            student = type(original_model)(
                vocab_size=original_model.embedding.num_embeddings,
                d_model=original_model.d_model // 2,  # Half the size
                num_layers=max(2, len(original_model.layers) // 2),
                num_classes=original_model.classifier.out_features
            ).to(device)
            
            # Train with distillation
            distiller = KnowledgeDistillation(original_model, student, device)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
            distiller.train_student(train_loader, optimizer, num_epochs=3)
            
            optimized_models['Distilled (0.5x)'] = student
    
    return optimized_models


if __name__ == '__main__':
    from resonance_nn.fast_spectral import FastSpectralClassifier
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Original model
    model = FastSpectralClassifier(
        vocab_size=10000,
        d_model=256,
        num_layers=4,
        num_classes=2
    ).to(device)
    
    print(f"Original model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dummy data
    train_data = TensorDataset(
        torch.randint(0, 10000, (1000, 512)),
        torch.randint(0, 2, (1000,))
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Create optimized versions
    optimized = create_edge_optimized_model(
        model,
        calibration_loader=train_loader,
        train_loader=train_loader,
        device=device,
        methods=['prune', 'distill']  # Skip quantize for now (complex setup)
    )
    
    # Compare sizes
    compare_model_sizes(model, optimized)
    
    print("\n✓ Edge optimization complete!")
