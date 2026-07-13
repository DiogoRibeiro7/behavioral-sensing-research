"""Tests for missing-data utilities."""
import pandas as pd

from sensor_modeling.utils import (
    forward_fill,
    handle_missing_data,
    interpolate_linear,
)


def test_forward_fill():
    """Forward-fill replaces missing value with previous."""
    df = pd.DataFrame({"a": [1, None, 3]})
    filled = forward_fill(df)
    assert filled.loc[1, "a"] == 1


def test_interpolate_linear():
    """Linear interpolation fills interior missing values."""
    df = pd.DataFrame({"a": [1, None, 3]})
    interp = interpolate_linear(df)
    assert interp.loc[1, "a"] == 2


def test_gap_aware_preserves_long_outages_and_metadata():
    """Long gaps should remain missing and be reported in the metadata."""
    df = pd.DataFrame({"a": [1.0, None, None, None, 5.0]})
    result = handle_missing_data(df, strategy="gap_aware", max_gap=2)
    assert pd.isna(result.data.loc[1, "a"])
    assert pd.isna(result.data.loc[2, "a"])
    assert pd.isna(result.data.loc[3, "a"])
    assert result.long_gap_mask["a"].sum() == 3
    assert result.summary["a"]["longest_gap"] == 3
    assert result.summary["a"]["imputed"] == 0


def test_gap_aware_fills_short_gaps_and_tracks_imputed_cells():
    """Short gaps should be filled and marked as imputed."""
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0]})
    result = handle_missing_data(df, strategy="gap_aware", max_gap=2)
    assert result.data.loc[1, "a"] == 2.0
    assert bool(result.imputed_mask.loc[1, "a"])
    assert result.summary["a"]["remaining_missing"] == 0


def test_forward_fill_leaves_leading_gap_unresolved():
    """Plain forward-fill should not invent values before the first observation."""
    df = pd.DataFrame({"a": [None, 2.0, None]})
    result = handle_missing_data(df, strategy="forward_fill")
    assert pd.isna(result.data.loc[0, "a"])
    assert result.data.loc[2, "a"] == 2.0
    assert result.summary["a"]["remaining_missing"] == 1


def test_flag_strategy_supports_mixed_dtypes_and_indicator_columns():
    """Flag mode should preserve values and add missingness indicators."""
    df = pd.DataFrame(
        {
            "numeric": [1.0, None, 3.0],
            "label": ["ok", None, "alert"],
        }
    )
    result = handle_missing_data(df, strategy="flag", add_indicators=True)
    assert pd.isna(result.data.loc[1, "numeric"])
    assert pd.isna(result.data.loc[1, "label"])
    assert "numeric_was_missing" in result.data.columns
    assert "label_was_missing" in result.data.columns
    assert bool(result.data.loc[1, "numeric_was_missing"])
    assert bool(result.data.loc[1, "label_was_missing"])


def test_drop_strategy_removes_rows_with_missing_values():
    """Drop mode should remove incomplete rows and preserve clean rows."""
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, None]})
    result = handle_missing_data(df, strategy="drop")
    assert list(result.data.index) == [0]
    assert result.summary["a"]["remaining_missing"] == 0
