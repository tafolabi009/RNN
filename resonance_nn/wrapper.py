"""Convenience wrapper utilities for loading and using Spectral models.

This small helper makes it easy to import the architecture from other
projects. Keep it intentionally lightweight — it doesn't add tokenizers
or heavy IO logic.

Usage:
    from resonance_nn import load_spectral_model
    model = load_spectral_model('base', device='cpu')

    wrapper = SpectralModelWrapper(model)
    outputs = wrapper.forward(input_ids)
"""
from typing import Optional

import torch

from .spectral_optimized import create_spectral_lm


def load_spectral_model(
    size: str = 'base',
    device: Optional[str] = None,
    vocab_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    pretrained_path: Optional[str] = None,
    **config_overrides,
):
    """Create and optionally load weights into a SpectralLanguageModel.

    Args:
        size: model size key from CONFIGS (tiny/small/base/...)
        device: device string like 'cpu' or 'cuda'
        vocab_size: optional override for vocab size
        max_seq_len: optional override for sequence length
        pretrained_path: optional path to a saved state_dict or checkpoint
        **config_overrides: additional SpectralConfig overrides

    Returns:
        model: an initialized (and optionally loaded) nn.Module
    """
    model = create_spectral_lm(size, vocab_size=vocab_size, max_seq_len=max_seq_len, **config_overrides)

    if pretrained_path is not None:
        # Load checkpoint: accept either raw state_dict or dict with 'state_dict'
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            # Best-effort load (some checkpoints may have prefixes)
            new_state = {}
            for k, v in state.items():
                new_key = k.replace('module.', '')
                new_state[new_key] = v
            model.load_state_dict(new_state, strict=False)

    if device is not None:
        model.to(device)

    model.eval()
    return model


class SpectralModelWrapper:
    """Tiny wrapper around a Spectral model providing convenience methods.

    This keeps the repository agnostic to tokenizers or other heavy deps.
    Use this in downstream projects as a starting point.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def to(self, device: str):
        self.model.to(device)
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Run a forward pass and return logits."""
        return self.model(input_ids, attention_mask=attention_mask)

    def generate(self, input_ids: torch.Tensor, **generate_kwargs):
        """Proxy to model.generate for autoregressive decoding."""
        return self.model.generate(input_ids, **generate_kwargs)
