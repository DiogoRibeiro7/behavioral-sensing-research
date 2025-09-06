"""Tests for missing-data utilities."""
import pandas as pd

from sensor_modeling.utils import forward_fill, interpolate_linear


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
