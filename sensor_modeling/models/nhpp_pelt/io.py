from __future__ import annotations

import csv
import glob
import os
from typing import List, Sequence

import numpy as np

Array1D = np.ndarray


def load_days_from_npy_folder(folder: str) -> List[Array1D]:
    """
    Load days from a folder of .npy files.
    Each .npy must contain a 1-D float array of event times in [0, delta].
    Files are sorted by filename to define day order.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.npy")))
    days: List[Array1D] = []
    for p in paths:
        arr = np.load(p)
        if arr.ndim != 1:
            raise ValueError(f"{p}: expected 1-D array.")
        days.append(arr.astype(float, copy=False))
    return days


def load_days_from_csv_folder(folder: str, column: str = "time") -> List[Array1D]:
    """
    Load days from a folder of .csv files (one file per day).
    The CSV must have a header row including `column` with event times (floats).
    Files are sorted by filename to define day order.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    days: List[Array1D] = []
    for p in paths:
        vals: List[float] = []
        with open(p, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(f"{p}: missing '{column}' column.")
            for row in reader:
                try:
                    vals.append(float(row[column]))
                except (TypeError, ValueError):
                    continue  # skip malformed rows
        days.append(np.asarray(vals, float))
    return days


def save_days_npz(days: Sequence[Array1D], path: str) -> None:
    """
    Save list of days to an .npz that works with nhpp.cli (--input).
    """
    obj = np.array([np.asarray(d, float) for d in days], dtype=object)
    np.savez(path, days=obj)
