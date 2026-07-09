from __future__ import annotations

import torch

from latentbrain.models.neural_sde import NeuralSDE, NeuralSDEConfig


def _model(diffusion_scale: float = 0.0) -> NeuralSDE:
    return NeuralSDE(
        NeuralSDEConfig(
            input_dim=5,
            output_dim=7,
            encoder_hidden_dim=8,
            drift_hidden_dim=9,
            diffusion_hidden_dim=6,
            latent_dim=4,
            factor_dim=3,
            dropout=0.0,
            min_rate_hz=1.0e-4,
            max_rate_hz=500.0,
            dt_seconds=0.02,
            diffusion_scale=diffusion_scale,
        )
    )


def test_forward_returns_required_shapes() -> None:
    model = _model()
    output = model(torch.ones(2, 11, 5))

    assert set(output) >= {
        "rates_hz",
        "factors",
        "latents",
        "z0_mean",
        "z0_logvar",
        "drift",
        "diffusion",
    }
    assert output["rates_hz"].shape == (2, 11, 7)
    assert output["factors"].shape == (2, 11, 3)
    assert output["latents"].shape == (2, 11, 4)
    assert output["drift"].shape == (2, 11, 4)
    assert output["diffusion"].shape == (2, 11, 4)


def test_deterministic_limit_and_eval_mode_are_deterministic() -> None:
    x = torch.ones(2, 9, 5)
    model = _model(diffusion_scale=0.0)
    model.train()
    torch.manual_seed(1)
    first = model(x)["rates_hz"]
    torch.manual_seed(2)
    second = model(x)["rates_hz"]
    assert torch.allclose(first, second)

    stochastic = _model(diffusion_scale=0.03)
    stochastic.eval()
    first_eval = stochastic(x)["rates_hz"]
    second_eval = stochastic(x)["rates_hz"]
    assert torch.allclose(first_eval, second_eval)


def test_training_outputs_finite_positive_rates_and_reproducible_noise() -> None:
    x = torch.ones(2, 9, 5)
    model = _model(diffusion_scale=0.03)
    model.train()
    generator = torch.Generator().manual_seed(123)
    first = model(x, generator=generator)
    generator = torch.Generator().manual_seed(123)
    second = model(x, generator=generator)

    assert torch.isfinite(first["rates_hz"]).all()
    assert torch.isfinite(first["drift"]).all()
    assert torch.isfinite(first["diffusion"]).all()
    assert torch.all(first["rates_hz"] > 0.0)
    assert torch.allclose(first["latents"], second["latents"])
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
