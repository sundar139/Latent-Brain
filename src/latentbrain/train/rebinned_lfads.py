from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def _run_id(bin_size_ms: int) -> str:
    return f"bin_{bin_size_ms}ms"


def build_rebinned_lfads_train_config(
    base_config: dict[str, Any],
    bin_size_ms: int,
    window_bins: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one masked co-smoothing LFADS-style training config for a rebinned dataset."""
    config = copy.deepcopy(base_config)
    settings = dict(config["lfads_settings"])
    config["dataset"]["bin_size_ms"] = int(bin_size_ms)
    config["data"] = {
        "input_neuron_group": "heldin",
        "target_neuron_group": "heldout",
        "max_time_bins": int(window_bins),
        "batch_size": int(settings["batch_size"]),
        "num_workers": 0,
        "drop_last": False,
    }
    config["model"] = {
        "name": "lfads_gru",
        "input_dim": None,
        "output_dim": "all",
        "encoder_hidden_dim": int(settings["encoder_hidden_dim"]),
        "generator_hidden_dim": int(settings["generator_hidden_dim"]),
        "latent_dim": int(settings["latent_dim"]),
        "factor_dim": int(settings["factor_dim"]),
        "dropout": float(settings["dropout"]),
        "min_rate_hz": float(settings["min_rate_hz"]),
        "max_rate_hz": float(settings["max_rate_hz"]),
    }
    config["training"] = {
        "seed": int(config["splits"]["seed"]),
        "epochs": int(settings["epochs"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "gradient_clip_norm": float(settings["gradient_clip_norm"]),
        "heldin_loss_weight": float(settings["heldin_loss_weight"]),
        "heldout_loss_weight": float(settings["heldout_loss_weight"]),
        "kl_warmup_epochs": int(settings["kl_warmup_epochs"]),
        "loss_normalization": str(settings["loss_normalization"]),
        "checkpoint_metric": "validation_total_loss",
        "checkpoint_mode": "min",
        "device": str(config["runtime"]["device"]),
    }
    config["evaluation"] = {
        "evaluate_splits": list(base_config["evaluation"]["evaluate_splits"]),
        "primary_split": str(base_config["evaluation"].get("primary_split", "validation")),
        "direct_model_primary": True,
        "also_evaluate_factor_decoder": True,
        "behavior_decoder_enabled": True,
        "baseline_references": dict(
            base_config.get("evaluation", {}).get("baseline_references", {})
        ),
    }
    config["reporting"] = {"output_dir": str(output_dir)}
    config["run_id"] = _run_id(bin_size_ms)
    return config


def build_rebinned_lfads_eval_config(
    base_config: dict[str, Any],
    bin_size_ms: int,
    window_bins: int,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one held-out evaluation config for a rebinned LFADS-style checkpoint."""
    config = build_rebinned_lfads_train_config(base_config, bin_size_ms, window_bins, output_dir)
    settings = dict(base_config["lfads_settings"])
    config["model"]["checkpoint_path"] = str(checkpoint_path)
    config["evaluation_mode"] = {
        "use_direct_model_rates_for_heldout": True,
        "also_evaluate_factor_decoder": True,
    }
    config["heldout_decoder"] = {
        "name": "ridge",
        "alpha": 1000.0,
        "fit_intercept": True,
        "min_rate_hz": float(settings["min_rate_hz"]),
        "max_rate_hz": float(settings["max_rate_hz"]),
        "standardize_factors": True,
        "train_trials_only": True,
    }
    config["behavior_decoder"] = {
        "enabled": True,
        "alpha": 100.0,
        "fit_intercept": True,
        "standardize_factors": True,
        "standardize_targets": True,
        "target_prefixes": ["hand_pos", "cursor_pos"],
        "derive_velocity": True,
        "velocity_method": "central_difference",
        "train_trials_only": True,
    }
    config["reference"] = {"name": "train_mean_rate", "fit_train_trials_only": True}
    return config


def build_rate_initialized_lfads_train_config(
    base_config: dict[str, Any],
    bin_size_ms: int,
    window_bins: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the readout-bias-initialized masked co-smoothing config."""
    config = copy.deepcopy(base_config)
    settings = dict(config["initialized_lfads"])
    refs = dict(config["references"])
    config["dataset"]["bin_size_ms"] = int(bin_size_ms)
    config["data"] = {
        "input_neuron_group": "heldin",
        "target_neuron_group": "heldout",
        "max_time_bins": int(window_bins),
        "batch_size": int(settings["batch_size"]),
        "num_workers": 0,
        "drop_last": False,
    }
    config["model"] = {
        "name": "lfads_gru",
        "input_dim": None,
        "output_dim": "all",
        "encoder_hidden_dim": int(settings["encoder_hidden_dim"]),
        "generator_hidden_dim": int(settings["generator_hidden_dim"]),
        "latent_dim": int(settings["latent_dim"]),
        "factor_dim": int(settings["factor_dim"]),
        "dropout": float(settings["dropout"]),
        "min_rate_hz": float(settings["min_rate_hz"]),
        "max_rate_hz": float(settings["max_rate_hz"]),
    }
    config["training"] = {
        "seed": int(config["splits"]["seed"]),
        "epochs": int(settings["epochs"]),
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "gradient_clip_norm": float(settings["gradient_clip_norm"]),
        "heldin_loss_weight": float(settings["heldin_loss_weight"]),
        "heldout_loss_weight": float(settings["heldout_loss_weight"]),
        "kl_warmup_epochs": int(settings["kl_warmup_epochs"]),
        "loss_normalization": str(settings["loss_normalization"]),
        "checkpoint_metric": "validation_total_loss",
        "checkpoint_mode": "min",
        "device": str(config["runtime"]["device"]),
        "initialize_readout_bias_from_train_rates": bool(
            settings["initialize_readout_bias_from_train_rates"]
        ),
    }
    config["evaluation"] = {
        "evaluate_splits": list(base_config["evaluation"]["evaluate_splits"]),
        "primary_split": str(base_config["evaluation"].get("primary_split", "validation")),
        "direct_model_primary": True,
        "also_evaluate_factor_decoder": True,
        "behavior_decoder_enabled": True,
        "baseline_references": {
            "window_matched_mean_rate_validation_bits_per_spike": float(
                refs["same_bin_mean_rate_validation_bits_per_spike"]
            ),
            "window_matched_factor_latent_validation_bits_per_spike": float(
                refs["same_bin_factor_latent_validation_bits_per_spike"]
            ),
            "previous_lfads_masked_direct_validation_bits_per_spike": float(
                refs["previous_20ms_lfads_validation_bits_per_spike"]
            ),
        },
    }
    config["reporting"] = {"output_dir": str(output_dir)}
    config["run_id"] = str(settings["output_dir_name"])
    return config
