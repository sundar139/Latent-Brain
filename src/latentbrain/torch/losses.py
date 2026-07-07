from __future__ import annotations

import torch


def _validate_positive_finite_rates(rates_hz: torch.Tensor) -> None:
    if not torch.isfinite(rates_hz).all().item():
        msg = "rates_hz must be finite"
        raise ValueError(msg)
    if torch.any(rates_hz <= 0.0).item():
        msg = "rates_hz must be positive"
        raise ValueError(msg)


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
    _validate_positive_finite_rates(rates_hz)
    expected_counts = torch.clamp(rates_hz, min=torch.finfo(rates_hz.dtype).tiny) * (
        bin_size_ms / 1000.0
    )
    log_likelihood = counts * torch.log(expected_counts) - expected_counts
    if include_constant:
        log_likelihood = log_likelihood - torch.lgamma(counts + 1.0)
    return -torch.sum(log_likelihood)


def masked_poisson_loss(
    counts: torch.Tensor,
    rates_hz: torch.Tensor,
    bin_size_ms: int,
    include_constant: bool = True,
    normalization: str = "mean",
) -> torch.Tensor:
    """Poisson NLL with explicit normalization for masked neural targets."""
    if counts.shape != rates_hz.shape:
        msg = (
            "counts and rates_hz must have matching shapes; "
            f"got {counts.shape} and {rates_hz.shape}"
        )
        raise ValueError(msg)
    nll = poisson_nll_torch(counts, rates_hz, bin_size_ms, include_constant)
    if normalization == "sum":
        return nll
    if normalization in {"mean", "per_observed_spike_bin"}:
        return nll / max(int(counts.numel()), 1)
    if normalization == "batch_mean":
        return nll / max(int(counts.shape[0]), 1)
    msg = "normalization must be sum, mean, batch_mean, or per_observed_spike_bin"
    raise ValueError(msg)


def gaussian_kl_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Average KL(q(z|x) || N(0, I)) over batch, summing latent dimensions."""
    kl_by_sample = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - torch.exp(logvar), dim=-1)
    return torch.mean(kl_by_sample)


def _unique_1d_indices(indices: torch.Tensor, name: str, output_dim: int) -> torch.Tensor:
    flat = indices.detach().to(dtype=torch.long).reshape(-1)
    if flat.numel() == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if torch.any(flat < 0).item() or torch.any(flat >= output_dim).item():
        msg = f"{name} contains indices outside output dimension {output_dim}"
        raise ValueError(msg)
    if torch.unique(flat).numel() != flat.numel():
        msg = f"{name} must contain unique indices"
        raise ValueError(msg)
    return flat


def _validate_disjoint_indices(heldin_indices: torch.Tensor, heldout_indices: torch.Tensor) -> None:
    overlap = torch.isin(heldin_indices.detach().cpu(), heldout_indices.detach().cpu())
    if torch.any(overlap).item():
        msg = "heldin_indices and heldout_indices must be disjoint"
        raise ValueError(msg)


def lfads_elbo_loss(
    heldin_counts: torch.Tensor,
    heldin_rates_hz: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    bin_size_ms: int,
    kl_beta: float,
) -> dict[str, torch.Tensor]:
    """ELBO loss with reconstruction NLL summed over time/neurons and averaged by batch."""
    reconstruction_loss = masked_poisson_loss(
        heldin_counts,
        heldin_rates_hz,
        bin_size_ms,
        include_constant=True,
        normalization="batch_mean",
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


def lfads_cosmoothing_loss(
    heldin_counts: torch.Tensor,
    heldout_counts: torch.Tensor,
    all_rates_hz: torch.Tensor,
    heldin_indices: torch.Tensor,
    heldout_indices: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    bin_size_ms: int,
    kl_beta: float,
    heldin_loss_weight: float,
    heldout_loss_weight: float,
    normalization: str,
) -> dict[str, torch.Tensor]:
    """LFADS-style masked co-smoothing objective from held-in inputs to all-neuron rates."""
    if heldin_loss_weight < 0.0 or heldout_loss_weight < 0.0:
        msg = "loss weights must be non-negative"
        raise ValueError(msg)
    if heldin_loss_weight == 0.0 and heldout_loss_weight == 0.0:
        msg = "at least one loss weight must be positive"
        raise ValueError(msg)
    heldin_flat = _unique_1d_indices(heldin_indices, "heldin_indices", all_rates_hz.shape[-1])
    heldout_flat = _unique_1d_indices(heldout_indices, "heldout_indices", all_rates_hz.shape[-1])
    _validate_disjoint_indices(heldin_flat, heldout_flat)
    heldin_rates = all_rates_hz.index_select(dim=2, index=heldin_flat.to(all_rates_hz.device))
    heldout_rates = all_rates_hz.index_select(dim=2, index=heldout_flat.to(all_rates_hz.device))
    heldin_loss = masked_poisson_loss(
        heldin_counts,
        heldin_rates,
        bin_size_ms,
        include_constant=True,
        normalization=normalization,
    )
    heldout_loss = masked_poisson_loss(
        heldout_counts,
        heldout_rates,
        bin_size_ms,
        include_constant=True,
        normalization=normalization,
    )
    kl_loss = gaussian_kl_standard_normal(posterior_mean, posterior_logvar)
    beta = heldin_counts.new_tensor(float(kl_beta))
    total = (
        float(heldin_loss_weight) * heldin_loss
        + float(heldout_loss_weight) * heldout_loss
        + beta * kl_loss
    )
    return {
        "loss": total,
        "heldin_reconstruction_loss": heldin_loss,
        "heldout_prediction_loss": heldout_loss,
        "kl_loss": kl_loss,
        "kl_beta": beta,
    }
