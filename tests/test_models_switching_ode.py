from __future__ import annotations

import torch

from latentbrain.models.switching_ode import SwitchingODE, SwitchingODEConfig, mix_regime_drifts


def _model() -> SwitchingODE:
    return SwitchingODE(
        SwitchingODEConfig(
            input_dim=5,
            output_dim=7,
            encoder_hidden_dim=8,
            drift_hidden_dim=9,
            latent_dim=4,
            factor_dim=3,
            n_regimes=3,
            regime_hidden_dim=6,
            regime_temperature=0.75,
            dropout=0.0,
            min_rate_hz=1.0e-4,
            max_rate_hz=500.0,
            dt_seconds=0.02,
        )
    )


def test_forward_returns_required_shapes_and_probabilities_sum_to_one() -> None:
    model = _model()
    output = model(torch.ones(2, 11, 5))

    assert set(output) >= {
        "rates_hz",
        "factors",
        "latents",
        "z0_mean",
        "z0_logvar",
        "regime_probs",
        "regime_logits",
        "regime_drifts",
        "mixed_drift",
    }
    assert output["rates_hz"].shape == (2, 11, 7)
    assert output["factors"].shape == (2, 11, 3)
    assert output["latents"].shape == (2, 11, 4)
    assert output["regime_probs"].shape == (2, 11, 3)
    assert output["regime_drifts"].shape == (2, 11, 3, 4)
    assert output["mixed_drift"].shape == (2, 11, 4)
    assert torch.allclose(output["regime_probs"].sum(dim=-1), torch.ones(2, 11), atol=1e-6)


def test_eval_mode_is_deterministic_and_training_outputs_are_finite_positive() -> None:
    x = torch.ones(2, 9, 5)
    model = _model()
    model.eval()
    first = model(x)
    second = model(x)
    assert torch.allclose(first["rates_hz"], second["rates_hz"])

    model.train()
    output = model(x, generator=torch.Generator().manual_seed(123))
    assert torch.isfinite(output["rates_hz"]).all()
    assert torch.isfinite(output["mixed_drift"]).all()
    assert torch.isfinite(output["regime_probs"]).all()
    assert torch.all(output["rates_hz"] > 0.0)
    occupancy = output["regime_probs"].mean(dim=(0, 1))
    assert torch.isfinite(occupancy).all()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_mix_regime_drifts_matches_probability_weighted_sum() -> None:
    probs = torch.tensor([[[0.25, 0.75]]])
    drifts = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])

    mixed = mix_regime_drifts(probs, drifts)

    assert torch.allclose(mixed, torch.tensor([[[5.0, 7.0]]]))
