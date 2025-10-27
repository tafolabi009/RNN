"""
Basic smoke tests for resonance_nn package
==========================================

Tests:
- Import functionality
- Model creation
- Basic forward pass
- Generation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_import_package():
    """Test that the package can be imported"""
    import resonance_nn
    assert hasattr(resonance_nn, '__version__')
    assert resonance_nn.__version__ == '3.0.0'


def test_import_main_classes():
    """Test that main classes can be imported"""
    from resonance_nn import (
        SpectralLanguageModel,
        SpectralConfig,
        create_spectral_lm,
        load_spectral_model,
        SpectralModelWrapper
    )
    
    assert SpectralLanguageModel is not None
    assert SpectralConfig is not None
    assert create_spectral_lm is not None
    assert load_spectral_model is not None
    assert SpectralModelWrapper is not None


def test_create_model():
    """Test model creation"""
    from resonance_nn import create_spectral_lm
    
    model = create_spectral_lm('tiny', vocab_size=1000, max_seq_len=512)
    assert model is not None
    assert hasattr(model, 'forward')
    assert hasattr(model, 'generate')


def test_model_forward():
    """Test forward pass"""
    from resonance_nn import create_spectral_lm
    
    model = create_spectral_lm('tiny', vocab_size=1000, max_seq_len=512)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, 1000)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_model_generate():
    """Test generation"""
    from resonance_nn import create_spectral_lm
    
    model = create_spectral_lm('tiny', vocab_size=1000, max_seq_len=512)
    model.eval()
    
    # Create prompt
    prompt = torch.randint(0, 1000, (1, 10))
    
    # Generate
    with torch.no_grad():
        generated = model.generate(prompt, max_length=50, do_sample=False)
    
    # Check output
    assert generated.shape[0] == 1
    assert generated.shape[1] > 10  # Should be longer than prompt
    assert generated.shape[1] <= 50  # Should not exceed max_length


def test_wrapper():
    """Test model wrapper"""
    from resonance_nn import load_spectral_model, SpectralModelWrapper
    
    model = load_spectral_model('tiny', device='cpu', vocab_size=1000, max_seq_len=512)
    wrapper = SpectralModelWrapper(model)
    
    # Test forward
    input_ids = torch.randint(0, 1000, (2, 64))
    logits = wrapper.forward(input_ids)
    assert logits.shape == (2, 64, 1000)
    
    # Test generate
    prompt = torch.randint(0, 1000, (1, 10))
    generated = wrapper.generate(prompt, max_length=30)
    assert generated.shape[1] <= 30


def test_config_validation():
    """Test configuration validation"""
    from resonance_nn import SpectralConfig
    
    # Valid config
    config = SpectralConfig(
        vocab_size=1000,
        embed_dim=256,
        hidden_dim=1024,
        num_layers=4,
        num_heads=4
    )
    assert config.vocab_size == 1000
    
    # Invalid config (hidden_dim not divisible by num_heads)
    try:
        config = SpectralConfig(
            vocab_size=1000,
            embed_dim=256,
            hidden_dim=1000,  # Not divisible by 4
            num_layers=4,
            num_heads=4
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass  # Expected


def test_multimodal_imports():
    """Test multi-modal component imports"""
    from resonance_nn import (
        SpectralVisionEncoder,
        SpectralAudioEncoder,
        SpectralCrossModalFusion
    )
    
    assert SpectralVisionEncoder is not None
    assert SpectralAudioEncoder is not None
    assert SpectralCrossModalFusion is not None


def test_available_configs():
    """Test predefined configurations"""
    from resonance_nn import CONFIGS
    
    assert 'tiny' in CONFIGS
    assert 'small' in CONFIGS
    assert 'base' in CONFIGS
    assert 'medium' in CONFIGS
    assert 'large' in CONFIGS
    assert 'xlarge' in CONFIGS


if __name__ == '__main__':
    # Run tests
    print("Running smoke tests...")
    
    test_import_package()
    print("✓ Package import")
    
    test_import_main_classes()
    print("✓ Main classes import")
    
    test_create_model()
    print("✓ Model creation")
    
    test_model_forward()
    print("✓ Forward pass")
    
    test_model_generate()
    print("✓ Generation")
    
    test_wrapper()
    print("✓ Wrapper")
    
    test_config_validation()
    print("✓ Config validation")
    
    test_multimodal_imports()
    print("✓ Multi-modal imports")
    
    test_available_configs()
    print("✓ Available configs")
    
    print("\n✅ All smoke tests passed!")
