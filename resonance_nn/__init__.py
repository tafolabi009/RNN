"""
Spectral Neural Networks - Production Optimized (v2.0)
=======================================================

COMPLETE REWRITE with world-class improvements:
- ✅ RoPE position encoding (like LLaMA)
- ✅ Multi-head frequency decomposition
- ✅ 32K context length
- ✅ Optimized FFT operations
- ✅ BPE tokenization support
- ✅ No more gibberish generation!

O(n log n) complexity, 100M-100B+ parameters
"""

from resonance_nn.spectral_optimized import (
    # Core
    SpectralConfig,
    LayerType,
    CONFIGS,
    # Layers
    RotaryPositionEmbedding,
    MultiHeadFrequencyLayer,
    SpectralLayer,
    # Models
    SpectralLanguageModel,
    # Factory functions
    create_spectral_lm,
)

__version__ = '2.0.0'
__all__ = [
    'SpectralConfig',
    'LayerType',
    'CONFIGS',
    'RotaryPositionEmbedding',
    'MultiHeadFrequencyLayer',
    'SpectralLayer',
    'SpectralLanguageModel',
    'create_spectral_lm',
]
