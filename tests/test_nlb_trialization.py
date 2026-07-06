from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from latentbrain.data.nlb import (
    dataframe_to_behavior_tensor,
    dataframe_to_trial_tensor,
    select_test_nwb_files,
    select_train_nwb_file,
)


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
