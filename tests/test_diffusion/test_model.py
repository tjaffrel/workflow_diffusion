"""Tests for DiT model architecture — verifies shapes and forward pass."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not available")
DiT = None
AttentionType = None
RotaryType = None

try:
    from diffuse_materials.model import DiT, AttentionType, RotaryType
except ImportError:
    pytest.skip("diffuse_materials.model not importable", allow_module_level=True)


@pytest.fixture
def small_dit():
    """A small DiT for fast testing (2 layers, 64 dim)."""
    return DiT(
        in_channels=4,
        patch_size=2,
        dim=64,
        num_layers=2,
        num_heads=4,
        action_dim=10,
        max_frames=4,
    )


class TestDiTShapes:
    def test_forward_output_shape(self, small_dit):
        B, T, H, W, C = 2, 4, 8, 8, 4
        x = torch.randn(B, T, H, W, C)
        t = torch.randint(0, 1000, (B, T))
        actions = torch.randn(B, T, 10)
        out = small_dit(x, t, actions)
        assert out.shape == (B, T, H, W, C)

    def test_patchify_shape(self, small_dit):
        B, T, H, W, C = 2, 4, 8, 8, 4
        x = torch.randn(B, T, H, W, C)
        patched = small_dit.patchify(x)
        assert patched.shape == (B, T, H // 2, W // 2, 64)

    def test_timestep_embedding_shape(self, small_dit):
        t = torch.tensor([0, 500, 999])
        emb = small_dit.timestep_embedding(t, dim=256)
        assert emb.shape == (3, 256)

    def test_different_batch_sizes(self, small_dit):
        for B in [1, 3]:
            T, H, W, C = 4, 8, 8, 4
            x = torch.randn(B, T, H, W, C)
            t = torch.randint(0, 1000, (B, T))
            actions = torch.randn(B, T, 10)
            out = small_dit(x, t, actions)
            assert out.shape == (B, T, H, W, C)


class TestAttentionTypes:
    def test_attention_type_enum(self):
        assert AttentionType.SPATIAL == "spatial"
        assert AttentionType.TEMPORAL == "temporal"

    def test_rotary_type_enum(self):
        assert RotaryType.STANDARD == "standard"
        assert RotaryType.PIXEL == "pixel"
