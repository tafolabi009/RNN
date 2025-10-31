# Resonance NN

**Next-Generation Neural Architecture | O(n log n) Complexity | 200K Context**

---

## ⚠️ PROPRIETARY - INTERNAL USE ONLY

**© 2025 Genovo Technologies. All Rights Reserved.**

This software is confidential and proprietary to Genovo Technologies. Unauthorized distribution, use, or disclosure is strictly prohibited.

---

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## Overview

Resonance NN is a production-grade neural network architecture that replaces traditional attention mechanisms with Fast Fourier Transform (FFT) based spectral processing. This approach delivers **O(n log n)** computational complexity compared to the **O(n²)** complexity of standard transformers, enabling dramatically faster inference on long sequences.

### Key Advantages

- **Superior Scaling** - O(n log n) complexity enables processing of sequences up to 200,000 tokens
- **Faster Inference** - 4-6x speed improvement on sequences longer than 2K tokens
- **Extended Context** - Support for context lengths up to 200K tokens (6x longer than GPT-4)
- **Memory Efficient** - Linear memory growth vs quadratic for attention-based models
- **Multi-Modal** - Unified architecture for text, vision, and audio processing

---

## Performance

| Sequence Length | Resonance NN | Transformer | Speedup |
|-----------------|--------------|-------------|---------|
| 512 tokens      | 37 ms        | 54 ms       | 1.5x    |
| 2,048 tokens    | 134 ms       | 283 ms      | 2.1x    |
| 8,192 tokens    | 568 ms       | 2,555 ms    | **4.5x** |
| 16,384 tokens   | 1,576 ms     | 10,074 ms   | **6.4x** |

*Benchmarked on NVIDIA GTX 1660 Ti. The 4.5x refers to speed improvement, not a quality rating.*

---

## Architecture

Resonance NN leverages spectral domain processing through three core innovations:

1. **Advanced Spectral Gating (ASG)** - Phase-aware frequency modulation replaces attention
2. **Adaptive Frequency Selection** - Learned sparsity patterns optimize computation
3. **Hierarchical FFT Processing** - Efficient handling of ultra-long sequences

This architecture eliminates the quadratic bottleneck of attention while maintaining global context awareness.

---

## Installation

```bash
pip install resonance_nn-0.1.0-py3-none-any.whl
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0

---

## Quick Start

### Language Model

```python
from resonance_nn import create_spectral_lm
import torch

# Create model
model = create_spectral_lm('base', vocab_size=50257)
model.eval()

# Inference
input_ids = torch.randint(0, 50257, (1, 8192))
with torch.no_grad():
    logits = model(input_ids)
```

### Classification

```python
from resonance_nn import SpectralClassifier, SpectralConfig

# Configure classifier
config = SpectralConfig(
    vocab_size=30522,
    embed_dim=768,
    hidden_dim=3072,
    num_layers=12,
    num_heads=12,
    max_seq_len=512
)

# Create and use
model = SpectralClassifier(config, num_classes=2)
logits = model(input_ids)
```

### Text Generation

```python
# Generate text
output = model.generate(
    input_ids,
    max_length=512,
    temperature=0.8,
    top_p=0.9
)
```

---

## Model Configurations

| Configuration | Parameters | Context Length | Hidden Dim | Recommended Use |
|---------------|------------|----------------|------------|-----------------|
| `tiny`        | 77M        | 16K            | 1024       | Prototyping, Edge |
| `small`       | 454M       | 65K            | 2048       | Development |
| **`base`**    | **983M**   | **131K**       | **3072**   | **Production** |
| `medium`      | 3.3B       | 200K           | 4096       | High Performance |
| `large`       | 9.8B       | 200K           | 6144       | Research |

```python
# List all available configurations
from resonance_nn import list_available_models
list_available_models()
```

---

## Training

```python
import torch
from torch.utils.data import DataLoader
from resonance_nn import create_spectral_lm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_spectral_lm('base', vocab_size=50257).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# Save checkpoint
torch.save(model.state_dict(), 'model.pth')
```

---

## Advanced Features

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Checkpointing

```python
# Enable for large models to reduce memory usage
config = SpectralConfig(
    vocab_size=50257,
    hidden_dim=4096,
    num_layers=24,
    use_gradient_checkpointing=True
)
```

### Multi-Modal Processing

```python
from resonance_nn import SpectralVisionEncoder, SpectralCrossModalFusion

# Vision encoder
vision_encoder = SpectralVisionEncoder(config)
image_features = vision_encoder(images)

# Cross-modal fusion
fusion = SpectralCrossModalFusion(config)
fused_features = fusion(
    text_features=text_embeddings,
    vision_features=image_features
)
```

---

## Deployment

### Model Export

```python
# Save for production
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
    'metadata': {
        'version': '0.1.0',
        'vocab_size': 50257,
        'max_seq_len': 131072
    }
}, 'production_model.pth')

# Load in production
checkpoint = torch.load('production_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Optimization

```python
# Compile model for faster inference (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')

# Quantization for deployment
from resonance_nn.deployment import quantize_model
quantized_model = quantize_model(model, dtype=torch.qint8)
```

---

## Documentation

- **[Installation Guide](docs/INSTALLATION_GUIDE.md)** - Comprehensive setup instructions
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Common patterns and examples
- **[Distribution Guide](docs/DISTRIBUTION_README.md)** - Deployment best practices
- **[Verification Script](docs/verify_installation.py)** - Test your installation

---

## System Requirements

### Development

- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 10GB for models and dependencies

### Production

- **Base Model**: 4GB GPU memory, 8GB RAM
- **Medium Model**: 16GB GPU memory, 32GB RAM
- **Large Model**: 40GB+ GPU memory, 80GB+ RAM

---

## Technical Specifications

### Algorithm Complexity

- **Attention (Standard)**: O(n² · d)
- **Resonance NN**: O(n · log(n) · d)
- **Memory**: O(n · d) vs O(n² · d)

### Supported Modalities

- **Text**: Token sequences up to 200K
- **Vision**: Image patches (224×224 to 1024×1024)
- **Audio**: Spectrograms and waveforms

### Layer Types

- **Dense**: Full spectral processing
- **Sparse**: Adaptive frequency selection (10-15% sparsity)
- **MoE**: Mixture-of-Experts with learned routing
- **MultiScale**: Hierarchical frequency decomposition

---

## Support

For internal support and questions:

- **Email**: afolabi@genovotech.com
- **Repository**: https://github.com/tafolabi009/RNN
- **Issues**: https://github.com/tafolabi009/RNN/issues

---

## License

**Proprietary License - Genovo Technologies Internal Use Only**

© 2025 Genovo Technologies. All Rights Reserved.

This software is confidential and proprietary. Unauthorized distribution, use, or disclosure is strictly prohibited. For internal use by authorized Genovo Technologies personnel only.

See [LICENSE](LICENSE) for complete terms and conditions.

---

## Citation

```bibtex
@software{resonance_nn_2025,
  title = {Resonance NN: O(n log n) Spectral Neural Networks},
  author = {Genovo Technologies Research},
  year = {2025},
  organization = {Genovo Technologies},
  note = {Proprietary Software - Internal Use Only}
}
```

---

**Genovo Technologies** | Building the Future of AI

*Version 0.1.0 | October 2025*
