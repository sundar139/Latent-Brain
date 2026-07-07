from __future__ import annotations

import torch


def poisson_nll_torch(
    counts: torch.Tensor,
    rates_hz: torch.Tensor,
    bin_size_ms: int,
    include_constant: bool = True,
) -> torch.Tensor:
    """Return summed Poisson negative log likelihood for rates in Hz."""
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    expected_counts = torch.clamp(rates_hz, min=torch.finfo(rates_hz.dtype).tiny) * (
        bin_size_ms / 1000.0
    )
    log_likelihood = counts * torch.log(expected_counts) - expected_counts
    if include_constant:
        log_likelihood = log_likelihood - torch.lgamma(counts + 1.0)
    return -torch.sum(log_likelihood)


def gaussian_kl_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Average KL(q(z|x) || N(0, I)) over batch, summing latent dimensions."""
    kl_by_sample = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - torch.exp(logvar), dim=-1)
    return torch.mean(kl_by_sample)


def lfads_elbo_loss(
    heldin_counts: torch.Tensor,
    heldin_rates_hz: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    bin_size_ms: int,
    kl_beta: float,
) -> dict[str, torch.Tensor]:
    """ELBO loss with reconstruction NLL summed over time/neurons and averaged by batch."""
    batch_size = max(int(heldin_counts.shape[0]), 1)
    reconstruction_loss = (
        poisson_nll_torch(heldin_counts, heldin_rates_hz, bin_size_ms) / batch_size
    )
    kl_loss = gaussian_kl_standard_normal(posterior_mean, posterior_logvar)
    beta = heldin_counts.new_tensor(float(kl_beta))
    total = reconstruction_loss + beta * kl_loss
    return {
        "loss": total,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "kl_beta": beta,
    }
