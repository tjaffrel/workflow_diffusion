"""Tests for DDIM diffusion — schedule, q_sample, loss shapes."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not available")
Diffusion = None

try:
    from diffuse_materials.diffusion import Diffusion
except ImportError:
    pytest.skip("diffuse_materials.diffusion not importable", allow_module_level=True)


@pytest.fixture
def diffusion():
    return Diffusion(timesteps=100, sampling_timesteps=5)


class TestDiffusionSchedule:
    def test_alphas_cumprod_shape(self, diffusion):
        assert diffusion.alphas_cumprod.shape == (100,)

    def test_alphas_cumprod_decreasing(self, diffusion):
        ac = diffusion.alphas_cumprod
        assert (ac[:-1] >= ac[1:]).all()

    def test_alphas_cumprod_range(self, diffusion):
        ac = diffusion.alphas_cumprod
        assert ac.min() > 0
        assert ac.max() <= 1


class TestQSample:
    def test_q_sample_shape(self, diffusion):
        B, T, H, W, C = 2, 4, 8, 8, 4
        x = torch.randn(B, T, H, W, C)
        t = torch.randint(0, 100, (B, T))
        noise = torch.randn_like(x)
        noisy = diffusion.q_sample(x, t, noise)
        assert noisy.shape == x.shape

    def test_q_sample_at_t0_close_to_x(self, diffusion):
        B, T, H, W, C = 1, 1, 4, 4, 4
        x = torch.randn(B, T, H, W, C)
        t = torch.zeros(B, T, dtype=torch.long)
        noise = torch.randn_like(x)
        noisy = diffusion.q_sample(x, t, noise)
        assert torch.allclose(noisy, x, atol=0.5)


class TestLossFn:
    def test_loss_scalar(self, diffusion):
        B, T, H, W, C = 2, 4, 8, 8, 4
        from diffuse_materials.model import DiT

        model = DiT(
            in_channels=4, patch_size=2, dim=64,
            num_layers=1, num_heads=4, action_dim=10, max_frames=4,
        )
        x = torch.randn(B, T, H, W, C)
        actions = torch.randn(B, T, 10)
        loss = diffusion.loss_fn(model, x, actions)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestSchedulingMatrix:
    def test_pyramid_shape(self, diffusion):
        horizon = 3
        mat = diffusion.generate_pyramid_scheduling_matrix(horizon)
        expected_height = diffusion.sampling_timesteps + horizon
        assert mat.shape == (expected_height, horizon)

    def test_pyramid_values_clipped(self, diffusion):
        mat = diffusion.generate_pyramid_scheduling_matrix(4)
        assert mat.min() >= 0
        assert mat.max() <= diffusion.sampling_timesteps
