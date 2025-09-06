"""Integration tests for the command line interface."""

import subprocess
import sys
from pathlib import Path

from sensor_modeling.utils.data_io import simulate_sensor_data


def test_cli_runs(tmp_path: Path) -> None:
    """CLI runs end-to-end with synthetic data."""
    dataset = simulate_sensor_data(n_days=1, n_sensors=2)
    csv_path = tmp_path / "data.csv"
    dataset.data.to_csv(csv_path)
    cmd = [
        sys.executable,
        "-m",
        "sensor_modeling.cli",
        "bernoulli-ar",
        str(csv_path),
        "sensor_0",
    ]
    subprocess.check_call(cmd)
