"""Utilities for handling missing sensor data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class MissingDataResult:
    """Container for imputed data and uncertainty metadata.

    Attributes
    ----------
    data
        Data after applying the selected missing-data strategy.
    original_missing_mask
        Boolean mask identifying cells that were missing in the input.
    imputed_mask
        Boolean mask identifying cells that were missing originally and were
        filled by the chosen strategy.
    remaining_missing_mask
        Boolean mask identifying cells still missing after processing.
    long_gap_mask
        Boolean mask marking gaps longer than ``max_gap`` when provided.
    summary
        Per-column metadata about missingness, imputation, and longest gaps.
    """

    data: pd.DataFrame
    original_missing_mask: pd.DataFrame
    imputed_mask: pd.DataFrame
    remaining_missing_mask: pd.DataFrame
    long_gap_mask: pd.DataFrame
    summary: dict[str, dict[str, Any]]


def _gap_lengths(mask: pd.Series) -> pd.Series:
    """Return the length of each missing run for every position."""
    lengths = pd.Series(0, index=mask.index, dtype=int)
    run_length = 0
    run_index: list[Any] = []
    for idx, is_missing in mask.items():
        if bool(is_missing):
            run_length += 1
            run_index.append(idx)
            continue
        if run_index:
            lengths.loc[run_index] = run_length
            run_length = 0
            run_index = []
    if run_index:
        lengths.loc[run_index] = run_length
    return lengths


def _long_gap_mask(df: pd.DataFrame, max_gap: int | None) -> pd.DataFrame:
    """Return a boolean mask for missing runs exceeding ``max_gap``."""
    if max_gap is None:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    mask = df.isna()
    long_gap = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        lengths = _gap_lengths(mask[col])
        long_gap[col] = lengths > max_gap
    return long_gap


def _apply_numeric_interpolation(df: pd.DataFrame, max_gap: int | None) -> pd.DataFrame:
    """Interpolate numeric columns while leaving unsupported dtypes untouched."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        result.loc[:, numeric_cols] = result.loc[:, numeric_cols].interpolate(
            method="linear",
            limit=max_gap,
            limit_area="inside",
        )
    return result


def _build_summary(
    original_missing: pd.DataFrame,
    imputed_mask: pd.DataFrame,
    remaining_missing: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Summarize missingness and imputation per column."""
    summary: dict[str, dict[str, Any]] = {}
    for col in original_missing.columns:
        gap_lengths = _gap_lengths(original_missing[col])
        summary[col] = {
            "original_missing": int(original_missing[col].sum()),
            "imputed": int(imputed_mask[col].sum()),
            "remaining_missing": int(remaining_missing[col].sum()),
            "longest_gap": int(gap_lengths.max()) if len(gap_lengths) else 0,
            "missing_ratio": float(original_missing[col].mean()),
        }
    return summary


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    *,
    max_gap: int | None = None,
    add_indicators: bool = False,
) -> MissingDataResult:
    """Apply a missing-data strategy and return imputation metadata.

    Parameters
    ----------
    df
        Sensor data indexed by time.
    strategy
        One of ``"forward_fill"``, ``"interpolate"``, ``"gap_aware"``,
        ``"drop"``, or ``"flag"``.
    max_gap
        Maximum consecutive missing samples to fill. Longer gaps remain missing
        for ``"forward_fill"``, ``"interpolate"``, and ``"gap_aware"``.
    add_indicators
        When ``True``, append ``"<column>_was_missing"`` boolean columns.
    """

    if max_gap is not None and max_gap < 1:
        raise ValueError("max_gap must be at least 1 when provided")

    original = df.copy()
    original_missing = original.isna()
    long_gap = _long_gap_mask(original, max_gap)
    data = original.copy()

    if strategy in {"forward_fill", "ffill"}:
        data = data.ffill(limit=max_gap)
    elif strategy == "interpolate":
        data = _apply_numeric_interpolation(data, max_gap)
        non_numeric = data.columns.difference(data.select_dtypes(include="number").columns)
        if len(non_numeric) > 0:
            data.loc[:, non_numeric] = data.loc[:, non_numeric].ffill(limit=max_gap)
    elif strategy == "gap_aware":
        data = _apply_numeric_interpolation(data, max_gap)
        data = data.ffill(limit=max_gap).bfill(limit=max_gap)
        data = data.mask(long_gap)
    elif strategy == "drop":
        data = data.dropna()
    elif strategy == "flag":
        pass
    else:
        raise ValueError(f"Unsupported missing-data strategy: {strategy}")

    if add_indicators:
        for col in original.columns:
            data[f"{col}_was_missing"] = original_missing[col]

    if strategy == "drop":
        kept_missing = original_missing.loc[data.index]
        imputed_mask = pd.DataFrame(False, index=data.index, columns=original.columns)
        remaining_missing = data[original.columns].isna()
        long_gap = long_gap.loc[data.index]
        summary = _build_summary(kept_missing, imputed_mask, remaining_missing)
    else:
        imputed_mask = original_missing & ~data[original.columns].isna()
        remaining_missing = data[original.columns].isna()
        summary = _build_summary(original_missing, imputed_mask, remaining_missing)

    return MissingDataResult(
        data=data,
        original_missing_mask=original_missing if strategy != "drop" else original_missing.loc[data.index],
        imputed_mask=imputed_mask,
        remaining_missing_mask=remaining_missing,
        long_gap_mask=long_gap,
        summary=summary,
    )


def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values in a DataFrame."""
    return handle_missing_data(df, strategy="forward_fill").data


def interpolate_linear(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate interior missing values in numeric columns."""
    return handle_missing_data(df, strategy="interpolate").data
