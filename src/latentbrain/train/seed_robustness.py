from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from latentbrain.data.schemas import NeuralDataset, NeuronMask, TrialSplit
from latentbrain.data.splits import create_neuron_mask, create_trial_split
from latentbrain.data.validation import validate_neuron_mask, validate_trial_split
from latentbrain.eval.latent_baseline import run_factor_latent_baseline
from latentbrain.eval.seed_robustness import (
    RESULT_COLUMNS,
    paired_seed_differences,
    summarize_method_scores,
    summarize_seed_robustness,
)
from latentbrain.paths import get_repo_root, resolve_configured_path
from latentbrain.randomness import seed_everything
from latentbrain.train.neural_ode_objectives import (
    _train_and_evaluate_run,
    build_neural_ode_objective_train_config,
)
from latentbrain.train.neural_sde_tuning import (
    _json_default,
    _load_dataset,
    _resolve_processed_path,
    _validate_cuda,
    _verify_reference_zero,
)

NEURAL_METHOD_TYPES = frozenset({"neural_ode", "neural_ode_objective"})
METHOD_TYPES = frozenset({"factor_latent", *NEURAL_METHOD_TYPES})

# Canonical neural parameters. `best_config.yaml` carries a stale `model` block whose
# objective weights are the pre-variant defaults; `training` is the authoritative record.
_NEURAL_MODEL_KEYS = (
    "encoder_hidden_dim",
    "drift_hidden_dim",
    "diffusion_hidden_dim",
    "latent_dim",
    "factor_dim",
    "min_rate_hz",
    "max_rate_hz",
    "dt_seconds",
    "batch_size",
    "loss_normalization",
    "model_dropout",
)
_NEURAL_TRAINING_KEYS = (
    "epochs",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "heldin_loss_weight",
    "heldout_loss_weight",
    "kl_warmup_epochs",
    "kl_scale",
    "scheduler",
)
_NEURAL_DEFAULTS: dict[str, Any] = {
    "diffusion_hidden_dim": 32,
    "loss_normalization": "per_observed_spike_bin",
    "model_dropout": 0.0,
    "min_rate_hz": 1.0e-4,
    "max_rate_hz": 500.0,
    "dt_seconds": 0.02,
    "diffusion_scale": 0.0,
    "zero_count_weight": 1.0,
    "positive_count_weight": 1.0,
    "rate_calibration_loss_weight": 0.0,
}


def make_seed_run_id(method_name: str, seed: int) -> str:
    return f"{method_name}/seed_{seed}"


def config_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_seed_plan(config: dict[str, Any]) -> pd.DataFrame:
    """Deterministic method x seed plan. Every method receives the identical seed list."""
    seeds = [int(seed) for seed in config["seeds"]]
    if len(seeds) < 3:
        msg = "at least three seeds are required"
        raise ValueError(msg)
    if len(set(seeds)) != len(seeds):
        msg = "seeds must be unique"
        raise ValueError(msg)
    methods = list(config["methods"])
    names = [str(method["name"]) for method in methods]
    if len(set(names)) != len(names):
        msg = "method names must be unique"
        raise ValueError(msg)
    for method in methods:
        if str(method["type"]) not in METHOD_TYPES:
            msg = f"unknown method type: {method['type']}"
            raise ValueError(msg)
    split_seed_mode = str(config["splits"]["split_seed_mode"])
    if split_seed_mode not in {"fixed", "varied"}:
        msg = "splits.split_seed_mode must be fixed or varied"
        raise ValueError(msg)
    if str(config["splits"]["initialization_seed_mode"]) != "varied":
        msg = "splits.initialization_seed_mode must be varied"
        raise ValueError(msg)
    fixed_split_seed = int(config["splits"]["split_seed"])
    rows = [
        {
            "method_name": str(method["name"]),
            "method_type": str(method["type"]),
            "valid_model": bool(method.get("valid_model", True)),
            "seed": seed,
            # The split seed never absorbs the initialization seed: with split_seed_mode
            # `fixed` every job shares one split and neuron mask, so score spread across
            # seeds is initialization/training variance only.
            "split_seed": fixed_split_seed if split_seed_mode == "fixed" else seed,
            "initialization_seed": seed,
            "run_id": make_seed_run_id(str(method["name"]), seed),
            "notes": str(method.get("notes", "")),
        }
        for method in methods
        for seed in seeds
    ]
    return pd.DataFrame(rows)


def _neural_params_from_source(source: dict[str, Any]) -> dict[str, Any]:
    model = dict(source.get("model", {}))
    training = dict(source.get("training", {}))
    params: dict[str, Any] = {}
    for key in _NEURAL_MODEL_KEYS:
        if key in model:
            params[key] = model[key]
    for key in _NEURAL_TRAINING_KEYS:
        if key in training:
            params[key] = training[key]
    if "drift_regularization_scale" in training:
        params["drift_regularization"] = training["drift_regularization_scale"]
    for key in ("zero_count_weight", "positive_count_weight", "rate_calibration_loss_weight"):
        if key in training:
            params[key] = training[key]
    dropout = dict(training.get("input_dropout", {}))
    if dropout:
        params["input_dropout_rate"] = float(dropout.get("rate", 0.0))
    params["diffusion_scale"] = 0.0
    return params


def load_or_build_method_config(config: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    """Resolve a method's parameters from `config_source` if present, else `fallback_config`."""
    method_type = str(method["type"])
    fallback = copy.deepcopy(dict(method.get("fallback_config", {})))
    if method_type == "factor_latent":
        return fallback | {"source": "fallback_config"}
    params: dict[str, Any] = dict(_NEURAL_DEFAULTS) | fallback
    source_path = method.get("config_source")
    resolved_source = "fallback_config"
    if source_path:
        path = resolve_configured_path(str(source_path), get_repo_root())
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                msg = f"malformed method config source: {path}"
                raise ValueError(msg)
            loaded = _neural_params_from_source(raw)
            if loaded:
                params = params | loaded
                resolved_source = str(path)
    if float(params["diffusion_scale"]) != 0.0:
        msg = "neural methods must force diffusion_scale == 0.0"
        raise ValueError(msg)
    params["source"] = resolved_source
    return params


def _base_model_from_params(method_name: str, params: dict[str, Any]) -> dict[str, Any]:
    base = {key: value for key, value in params.items() if key != "source"}
    return base | {
        "name": method_name,
        "input_neuron_group": "heldin",
        "output_dim": "all",
        "dropout": float(base.get("model_dropout", 0.0)),
        "diffusion_scale": 0.0,
        "checkpoint_metric": "validation_total_loss",
        "checkpoint_mode": "min",
        "save_unified_checkpoints": True,
        "evaluate_checkpoints_by_unified_metric": True,
    }


def _split_and_mask(
    dataset: NeuralDataset, split_seed: int, config: dict[str, Any]
) -> tuple[TrialSplit, NeuronMask]:
    split = create_trial_split(
        dataset.trial_ids,
        float(config["splits"]["train_fraction"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
        seed=split_seed,
    )
    mask = create_neuron_mask(
        dataset.spikes.shape[2],
        float(config["splits"]["heldout_neuron_fraction"]),
        seed=split_seed,
    )
    validate_trial_split(split, dataset.trial_ids)
    validate_neuron_mask(mask, dataset.spikes.shape[2])
    return split, mask


def _factor_latent_run_config(
    params: dict[str, Any], initialization_seed: int, evaluate_splits: list[str], behavior: bool
) -> dict[str, Any]:
    return {
        "features": {
            "input_neuron_group": "heldin",
            "target_neuron_group": "heldout",
            "smoothing": {
                "method": "gaussian",
                "sigma_ms": float(params["smoothing_sigma_ms"]),
                "truncate": 4.0,
            },
            "convert_to_hz": True,
            "standardize_features": bool(params["standardize_features"]),
        },
        "latent_model": {
            "name": "factor_analysis",
            "latent_dim": int(params["latent_dim"]),
            # Factor analysis uses randomized SVD, so the initialization seed is the one
            # stochastic knob this baseline exposes.
            "random_state": int(initialization_seed),
            "max_iter": int(params["max_iter"]),
            "tol": float(params["tol"]),
            "train_trials_only": True,
        },
        "heldout_decoder": {
            "name": "ridge",
            "alpha": float(params["heldout_decoder_alpha"]),
            "fit_intercept": True,
            "min_rate_hz": float(params["min_rate_hz"]),
            "max_rate_hz": float(params["max_rate_hz"]),
            "train_trials_only": True,
        },
        "behavior_decoder": {
            "enabled": bool(behavior),
            "target_prefixes": ["hand_pos", "cursor_pos"],
            "derive_velocity": True,
            "velocity_method": "central_difference",
            "alpha": 100.0,
            "fit_intercept": True,
            "standardize_targets": True,
        },
        "reference": {"name": "train_mean_rate", "fit_train_trials_only": True},
        "evaluation": {"evaluate_splits": list(evaluate_splits), "primary_split": "validation"},
    }


def _run_factor_latent_job(
    config: dict[str, Any],
    params: dict[str, Any],
    dataset: NeuralDataset,
    split: TrialSplit,
    mask: NeuronMask,
    initialization_seed: int,
    run_dir: Path,
) -> dict[str, float]:
    seed_everything(initialization_seed)
    behavior = bool(config["evaluation"].get("behavior_decoder_enabled", True))
    run_config = _factor_latent_run_config(
        params, initialization_seed, list(config["evaluation"]["evaluate_splits"]), behavior
    )
    split_metrics, neuron_metrics, behavior_metrics, latent_summary, _metadata = (
        run_factor_latent_baseline(dataset, split, mask, run_config)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8"
    )
    scores = split_metrics.copy()
    scores.insert(0, "prediction_source", "factor_decoder")
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    split_metrics.to_csv(evaluation_dir / "split_metrics.csv", index=False)
    neuron_metrics.to_csv(evaluation_dir / "neuron_metrics.csv", index=False)
    behavior_metrics.to_csv(evaluation_dir / "behavior_metrics.csv", index=False)
    latent_summary.to_csv(evaluation_dir / "factor_summary.csv", index=False)
    by_split = split_metrics.set_index("split")
    validation_r2 = float("nan")
    if not behavior_metrics.empty:
        rows = behavior_metrics[behavior_metrics["split"] == "validation"]
        validation_r2 = float("nan") if rows.empty else float(rows["r2"].mean())
    metrics = {
        "validation_unified_bits_per_spike": float(by_split.loc["validation", "bits_per_spike"]),
        "validation_poisson_nll": float(by_split.loc["validation", "poisson_nll"]),
        "validation_behavior_mean_r2": validation_r2,
        # The factor decoder is this baseline's only prediction source.
        "validation_factor_decoder_unified_bits_per_spike": float(
            by_split.loc["validation", "bits_per_spike"]
        ),
        "train_unified_bits_per_spike": float(by_split.loc["train", "bits_per_spike"]),
        "test_unified_bits_per_spike": float(by_split.loc["test", "bits_per_spike"]),
    }
    (run_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _neural_split_metric(scores: pd.DataFrame, split: str, source: str, column: str) -> float:
    rows = scores[(scores["split"] == split) & (scores["prediction_source"] == source)]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows.iloc[0][column])


def _run_neural_job(
    config: dict[str, Any],
    method_name: str,
    params: dict[str, Any],
    dataset: NeuralDataset,
    split: TrialSplit,
    mask: NeuronMask,
    split_seed: int,
    initialization_seed: int,
    window_bins: int,
    run_dir: Path,
) -> dict[str, float]:
    base = copy.deepcopy(config)
    base["_window_bins"] = window_bins
    base["base_model"] = _base_model_from_params(method_name, params)
    # `_train_and_evaluate_run` seeds from training.seed, which comes from splits.seed here.
    # The split and mask are passed in explicitly, so this only drives initialization.
    base["splits"] = dict(base["splits"]) | {"seed": initialization_seed}
    base["reporting"] = dict(base["reporting"])
    run_config = build_neural_ode_objective_train_config(base, {"name": method_name}, run_dir)
    run_config["splits"]["split_seed"] = split_seed
    _train_and_evaluate_run(run_config, method_name, dataset, split, mask)
    scores = pd.read_csv(run_dir / "evaluation" / "split_metrics.csv")
    scores.insert(0, "reference_name", "train_heldout_mean_rate")
    scores.insert(0, "valid_model", True)
    scores.to_csv(run_dir / "unified_scores.csv", index=False)
    behavior_path = run_dir / "evaluation" / "behavior_metrics.csv"
    validation_r2 = float("nan")
    if behavior_path.exists():
        behavior = pd.read_csv(behavior_path)
        rows = behavior[behavior["split"] == "validation"] if "split" in behavior else behavior
        validation_r2 = float("nan") if rows.empty else float(rows["r2"].mean())
    return {
        "validation_unified_bits_per_spike": _neural_split_metric(
            scores, "validation", "direct_model", "bits_per_spike"
        ),
        "validation_poisson_nll": _neural_split_metric(
            scores, "validation", "direct_model", "poisson_nll"
        ),
        "validation_behavior_mean_r2": validation_r2,
        "validation_factor_decoder_unified_bits_per_spike": _neural_split_metric(
            scores, "validation", "factor_decoder", "bits_per_spike"
        ),
        "train_unified_bits_per_spike": _neural_split_metric(
            scores, "train", "direct_model", "bits_per_spike"
        ),
        "test_unified_bits_per_spike": _neural_split_metric(
            scores, "test", "direct_model", "bits_per_spike"
        ),
    }


def run_seed_robustness(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    _resolve_processed_path(config)
    plan = build_seed_plan(config)
    methods_by_name = {str(method["name"]): dict(method) for method in config["methods"]}
    needs_cuda = any(
        str(method["type"]) in NEURAL_METHOD_TYPES for method in methods_by_name.values()
    )
    gpu_name = _validate_cuda(config) if needs_cuda else "cpu"
    dataset, dataset_hash, window_bins = _load_dataset(config)
    repo_root = get_repo_root()
    output_dir = resolve_configured_path(str(config["reporting"]["output_dir"]), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = dict(config["references"])
    resolved: dict[str, dict[str, Any]] = {
        name: load_or_build_method_config(config, method)
        for name, method in methods_by_name.items()
    }
    hashes = {name: config_hash(params) for name, params in resolved.items()}
    split_cache: dict[int, tuple[TrialSplit, NeuronMask]] = {}
    reference_zero: float | None = None
    rows: list[dict[str, Any]] = []
    for _, job in plan.iterrows():
        method_name = str(job["method_name"])
        method_type = str(job["method_type"])
        split_seed = int(job["split_seed"])
        initialization_seed = int(job["initialization_seed"])
        if split_seed not in split_cache:
            split_cache[split_seed] = _split_and_mask(dataset, split_seed, config)
        split, mask = split_cache[split_seed]
        if reference_zero is None:
            reference_zero = _verify_reference_zero(dataset, split, mask, config)
        run_dir = output_dir / "runs" / method_name / f"seed_{initialization_seed}"
        params = resolved[method_name]
        if method_type == "factor_latent":
            metrics = _run_factor_latent_job(
                config, params, dataset, split, mask, initialization_seed, run_dir
            )
        else:
            metrics = _run_neural_job(
                config,
                method_name,
                params,
                dataset,
                split,
                mask,
                split_seed,
                initialization_seed,
                window_bins,
                run_dir,
            )
        bits = float(metrics["validation_unified_bits_per_spike"])
        row = dict.fromkeys(RESULT_COLUMNS)
        row.update(
            {
                "method_name": method_name,
                "method_type": method_type,
                "seed": initialization_seed,
                "split_seed": split_seed,
                "initialization_seed": initialization_seed,
                "config_hash": hashes[method_name],
                "valid_model": bool(job["valid_model"]),
                "status": "completed",
                **metrics,
                "beats_train_mean_reference": bits
                > float(refs["train_mean_validation_bits_per_spike"]),
                "beats_factor_latent_single_seed_reference": bits
                > float(refs["factor_latent_single_seed_reference"]),
                "beats_neural_ode_refinement_single_seed_reference": bits
                > float(refs["neural_ode_refinement_single_seed_reference"]),
                "output_dir": str(run_dir),
                "notes": str(job["notes"]),
            }
        )
        rows.append(row)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    statistics = dict(config["statistics"])
    method_summary = summarize_method_scores(
        results,
        int(statistics["bootstrap_repeats"]),
        float(statistics["confidence_interval"]),
        int(statistics["bootstrap_seed"]),
    )
    refs["train_mean_validation_bits_per_spike"] = float(reference_zero or 0.0)
    summary = summarize_seed_robustness(results, method_summary, refs)
    summary.update(
        {
            "dataset_name": config["dataset"]["name"],
            "dataset_hash": dataset_hash,
            "cuda_device": gpu_name,
            "bin_size_ms": int(config["binning"]["target_bin_size_ms"]),
            "window_seconds": float(config["window"]["duration_seconds"]),
            "window_bins": window_bins,
            "split_seed_mode": str(config["splits"]["split_seed_mode"]),
            "split_seed": int(config["splits"]["split_seed"]),
            "initialization_seed_mode": str(config["splits"]["initialization_seed_mode"]),
            "seed_list_shared_across_methods": True,
            "method_config_hashes": hashes,
            "method_config_sources": {
                name: params.get("source", "fallback_config") for name, params in resolved.items()
            },
            "confidence_interval": float(statistics["confidence_interval"]),
            "bootstrap_repeats": int(statistics["bootstrap_repeats"]),
            "bootstrap_seed": int(statistics["bootstrap_seed"]),
            "output_dir": str(output_dir),
        }
    )
    return results, summary


def build_seed_effects(results: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    frames = [
        paired_seed_differences(results, method, reference_method)
        for method in sorted(results["method_name"].unique())
        if method != reference_method
    ]
    if not frames:
        return paired_seed_differences(results, reference_method, reference_method)
    return pd.concat(frames, ignore_index=True)


def build_carried_forward_config(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    carried = summary.get("carried_forward_method")
    if not carried:
        return {}
    method = next(
        (dict(item) for item in config["methods"] if str(item["name"]) == str(carried)), None
    )
    if method is None:
        return {}
    return {
        "carried_forward_method": carried,
        "carried_forward_reason": summary.get("carried_forward_reason"),
        "method": method,
        "resolved_config": load_or_build_method_config(config, method),
        "config_hash": summary.get("method_config_hashes", {}).get(str(carried)),
        "seeds": [int(seed) for seed in config["seeds"]],
        "split_seed": int(config["splits"]["split_seed"]),
        "official_benchmark_claim": False,
    }


def method_summary_from_results(results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    statistics = dict(config["statistics"])
    return summarize_method_scores(
        results,
        int(statistics["bootstrap_repeats"]),
        float(statistics["confidence_interval"]),
        int(statistics["bootstrap_seed"]),
    )


__all__ = [
    "build_carried_forward_config",
    "build_seed_effects",
    "build_seed_plan",
    "config_hash",
    "load_or_build_method_config",
    "make_seed_run_id",
    "method_summary_from_results",
    "run_seed_robustness",
]
