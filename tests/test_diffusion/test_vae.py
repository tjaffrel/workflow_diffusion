"""Tests for VAE placeholder — encode/decode are identity."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not available")
VAE = None

try:
    from diffuse_materials.vae import VAE
except ImportError:
    pytest.skip("diffuse_materials.vae not importable", allow_module_level=True)


def test_encode_identity():
    vae = VAE()
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(vae.encode(x), x)


def test_decode_identity():
    vae = VAE()
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(vae.decode(x), x)


def test_latent_channels():
    vae = VAE()
    assert vae.config.latent_channels == 4
