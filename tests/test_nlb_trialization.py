from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latentbrain.data.nlb import (
    dataframe_to_behavior_tensor,
    dataframe_to_trial_sequences,
    dataframe_to_trial_tensor,
    select_test_nwb_files,
    select_train_nwb_file,
)
from latentbrain.data.validation import validate_trial_sequences


def _trial_dataframe(
    *,
    include_spikes: bool = True,
    include_heldout: bool = False,
    spike_value: float = 1.0,
    nan_spike: bool = False,
) -> pd.DataFrame:
    trial_ids = [10, 10, 10, 10, 20, 20, 20]
    trial_times = [pd.Timedelta(milliseconds=value) for value in [0, 5, 10, 15, 0, 5, 10]]
    data: dict[tuple[str, str | int], list[object]] = {
        ("trial_id", ""): trial_ids,
        ("trial_time", ""): trial_times,
        ("cursor_pos", "x"): [0.0] * 7,
        ("hand_pos", "y"): [1.0] * 7,
    }
    if include_spikes:
        first_spike = np.nan if nan_spike else spike_value
        data[("spikes", 1)] = [first_spike, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0]
        data[("spikes", 2)] = [0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    if include_heldout:
        data[("heldout_spikes", 3)] = [3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0]
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys()))


def test_multiindex_spikes_only_convert_to_trial_tensor() -> None:
    spikes, trial_ids, time_ms, metadata = dataframe_to_trial_tensor(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=True,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )

    assert spikes.shape == (2, 3, 2)
    assert spikes.dtype == np.int64
    np.testing.assert_array_equal(trial_ids, np.array([10, 20]))
    np.testing.assert_array_equal(time_ms, np.array([0.0, 5.0, 10.0]))
    assert metadata["heldout_spikes_present"] is False
    assert metadata["trialization"]["original_trial_lengths"] == [4, 3]
    assert metadata["trialization"]["trial_time_range_ms"] == [0.0, 15.0]
    assert metadata["behavior"]["present"] is True
    assert metadata["behavior"]["column_counts"] == {"cursor_pos": 1, "hand_pos": 1}


def test_heldout_spikes_are_concatenated_after_heldin_spikes() -> None:
    spikes, _, _, metadata = dataframe_to_trial_tensor(
        _trial_dataframe(include_heldout=True),
        signal_types=["spikes", "heldout_spikes"],
        combine_heldout_spikes=True,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )

    assert spikes.shape == (2, 3, 3)
    assert metadata["spike_column_counts"] == {"spikes": 2, "heldout_spikes": 1}
    assert metadata["heldout_spikes_present"] is True


def test_variable_length_trials_crop_to_min_and_record_lengths() -> None:
    spikes, _, _, metadata = dataframe_to_trial_tensor(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )

    assert spikes.shape[1] == 3
    assert metadata["trialization"]["original_trial_lengths"] == [4, 3]
    assert metadata["trialization"]["min_length"] == 3
    assert metadata["trialization"]["max_length"] == 4
    assert metadata["trialization"]["cropping_occurred"] is True


def test_missing_spike_signal_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="spikes.*available signal types"):
        dataframe_to_trial_tensor(
            _trial_dataframe(include_spikes=False),
            signal_types=["spikes"],
            combine_heldout_spikes=True,
            variable_length_policy="crop_to_min",
            bin_size_ms=5,
        )


def test_non_integer_spike_values_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        dataframe_to_trial_tensor(
            _trial_dataframe(spike_value=0.5),
            signal_types=["spikes"],
            combine_heldout_spikes=True,
            variable_length_policy="crop_to_min",
            bin_size_ms=5,
        )


def test_nan_spike_values_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        dataframe_to_trial_tensor(
            _trial_dataframe(nan_spike=True),
            signal_types=["spikes"],
            combine_heldout_spikes=True,
            variable_length_policy="crop_to_min",
            bin_size_ms=5,
        )


def test_preferred_train_file_selection_chooses_behavior_ecephys_train() -> None:
    files = [
        Path("sub-Jenkins_ses-small_desc-test_ecephys.nwb"),
        Path("sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"),
    ]

    selected = select_train_nwb_file(files, "*desc-train_behavior+ecephys.nwb")

    assert selected.name == "sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"


def test_test_file_is_detected_but_not_selected_for_target_extraction() -> None:
    files = [
        Path("sub-Jenkins_ses-small_desc-test_ecephys.nwb"),
        Path("sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"),
    ]

    train_file = select_train_nwb_file(files, "*desc-train_behavior+ecephys.nwb")
    test_files = select_test_nwb_files(files, "*desc-test_ecephys.nwb")

    assert train_file not in test_files
    assert [path.name for path in test_files] == ["sub-Jenkins_ses-small_desc-test_ecephys.nwb"]


def test_behavior_groups_concatenate_with_stable_names_and_spike_window() -> None:
    spikes, trial_ids, _, _ = dataframe_to_trial_tensor(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=True,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )

    behavior, names, metadata = dataframe_to_behavior_tensor(
        _trial_dataframe(),
        behavior_signal_types=["hand_pos", "cursor_pos"],
        trial_ids=trial_ids,
        n_time_bins=spikes.shape[1],
        require_behavior=True,
        allow_behavior_nans=False,
        behavior_variable_length_policy="crop_to_spike_window",
    )

    assert behavior is not None
    assert behavior.shape[:2] == spikes.shape[:2]
    assert behavior.shape == (2, 3, 2)
    assert names == ["hand_pos_y", "cursor_pos_x"]
    assert metadata["groups_found"] == ["hand_pos", "cursor_pos"]
    assert metadata["cropped_to_spike_window"] is True


def test_missing_required_behavior_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="required behavior signals"):
        dataframe_to_behavior_tensor(
            _trial_dataframe(),
            behavior_signal_types=["eye_pos"],
            trial_ids=np.array([10, 20], dtype=np.int64),
            n_time_bins=3,
            require_behavior=True,
            allow_behavior_nans=False,
            behavior_variable_length_policy="crop_to_spike_window",
        )


def test_missing_optional_behavior_returns_none() -> None:
    behavior, names, metadata = dataframe_to_behavior_tensor(
        _trial_dataframe(),
        behavior_signal_types=["eye_pos"],
        trial_ids=np.array([10, 20], dtype=np.int64),
        n_time_bins=3,
        require_behavior=False,
        allow_behavior_nans=False,
        behavior_variable_length_policy="crop_to_spike_window",
    )

    assert behavior is None
    assert names is None
    assert metadata["present"] is False


def _equal_length_dataframe() -> pd.DataFrame:
    trial_ids = [10, 10, 10, 20, 20, 20]
    trial_times = [pd.Timedelta(milliseconds=value) for value in [0, 5, 10, 0, 5, 10]]
    data: dict[tuple[str, str | int], list[object]] = {
        ("trial_id", ""): trial_ids,
        ("trial_time", ""): trial_times,
        ("spikes", 1): [1.0, 0.0, 2.0, 0.0, 1.0, 3.0],
        ("spikes", 2): [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
    }
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data.keys()))


def test_spike_conservation_holds_for_equal_length_trials() -> None:
    spikes, _, _, metadata = dataframe_to_trial_tensor(
        _equal_length_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )
    conservation = metadata["ingestion_summary"]["spike_conservation"]

    assert conservation["conserved"] is True
    assert conservation["raw_spike_count"] == int(spikes.sum()) == 11
    assert conservation["excluded_spike_count"] == 0
    assert conservation["excluded_bins"] == 0


def test_spike_conservation_quantifies_cropped_bins() -> None:
    _, _, _, metadata = dataframe_to_trial_tensor(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )
    conservation = metadata["ingestion_summary"]["spike_conservation"]

    assert conservation["conserved"] is False
    assert conservation["excluded_bins"] == 1
    assert conservation["raw_spike_count"] == 9
    assert conservation["trialized_spike_count"] == 8
    assert conservation["excluded_spike_count"] == 1
    assert conservation["exclusion_reason"] == "crop_to_min"


def test_trialization_is_deterministic_across_repeated_calls() -> None:
    first = dataframe_to_trial_tensor(
        _equal_length_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )
    second = dataframe_to_trial_tensor(
        _equal_length_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        variable_length_policy="crop_to_min",
        bin_size_ms=5,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_trial_sequences_preserve_variable_lengths_without_global_crop() -> None:
    sequences = dataframe_to_trial_sequences(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        behavior_signal_types=["hand_pos", "cursor_pos"],
        bin_size_ms=5,
    )
    validate_trial_sequences(sequences)

    assert [trial.shape[0] for trial in sequences.spikes] == [4, 3]
    assert sequences.metadata["global_crop_applied"] is False
    np.testing.assert_array_equal(sequences.trial_lengths, np.array([4, 3]))
    np.testing.assert_array_equal(sequences.trial_ids, np.array([10, 20]))


def test_trial_sequences_conserve_every_raw_spike() -> None:
    sequences = dataframe_to_trial_sequences(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        behavior_signal_types=[],
        bin_size_ms=5,
    )

    # dataframe_to_trial_tensor drops 1 spike to crop_to_min; the ragged view drops none.
    assert sequences.metadata["raw_spike_count"] == 9
    assert sequences.metadata["sequence_spike_count"] == 9
    assert sequences.metadata["excluded_spike_count"] == 0
    assert sequences.metadata["spikes_conserved"] is True


def test_trial_sequences_keep_behavior_aligned_with_spikes() -> None:
    sequences = dataframe_to_trial_sequences(
        _trial_dataframe(),
        signal_types=["spikes"],
        combine_heldout_spikes=False,
        behavior_signal_types=["cursor_pos", "hand_pos"],
        bin_size_ms=5,
    )

    assert sequences.behavior is not None
    assert sequences.behavior_names == ["cursor_pos_x", "hand_pos_y"]
    for spikes, behavior in zip(sequences.spikes, sequences.behavior, strict=True):
        assert spikes.shape[0] == behavior.shape[0]


def test_trial_sequences_are_deterministic() -> None:
    kwargs = {
        "signal_types": ["spikes"],
        "combine_heldout_spikes": False,
        "behavior_signal_types": ["hand_pos"],
        "bin_size_ms": 5,
    }
    first = dataframe_to_trial_sequences(_trial_dataframe(), **kwargs)
    second = dataframe_to_trial_sequences(_trial_dataframe(), **kwargs)

    np.testing.assert_array_equal(first.trial_ids, second.trial_ids)
    for left, right in zip(first.spikes, second.spikes, strict=True):
        np.testing.assert_array_equal(left, right)
