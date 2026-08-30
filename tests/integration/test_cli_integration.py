"""Integration tests for the command line interface."""

import json
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


def test_analysis_pipeline_cli_runs(tmp_path: Path) -> None:
    """Analysis pipeline CLI writes reports to the requested output directory."""
    dataset = simulate_sensor_data(n_days=1, n_sensors=2)
    csv_path = tmp_path / "data.csv"
    out_dir = tmp_path / "reports"
    dataset.data.to_csv(csv_path)
    cmd = [
        sys.executable,
        "-m",
        "sensor_modeling.analysis.pipeline",
        str(csv_path),
        str(out_dir),
    ]
    subprocess.check_call(cmd)
    assert (out_dir / "analysis.tex").exists()
    assert (out_dir / "dashboard.html").exists()
    assert (out_dir / "analysis_fhir.json").exists()


def _run_cli(*args: str) -> str:
    """Run the CLI and return its stdout."""
    return subprocess.check_output(
        [sys.executable, "-m", "sensor_modeling.cli", *args],
        text=True,
        stderr=subprocess.STDOUT,
    )


def test_demo_runs_end_to_end_and_writes_structured_output(tmp_path: Path) -> None:
    """The worked example runs from the command line with a fixed seed."""
    output = tmp_path / "demo.json"
    stdout = _run_cli(
        "demo",
        "--days",
        "24",
        "--seed",
        "7",
        "--step-minutes",
        "30",
        "--output",
        str(output),
    )
    assert "END-TO-END DEMONSTRATION" in stdout
    assert "STATE INFERENCE" in stdout
    assert "no part of this system is a medical device" in stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario"]["seed"] == 7
    assert payload["scenario"]["faults"]
    assert 0.0 <= payload["state_inference"]["balanced_accuracy"] <= 1.0
    assert "sensor_health" in payload


def test_demo_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    """Two runs of the same command must produce identical numbers."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    for path in (first, second):
        _run_cli(
            "demo",
            "--days",
            "20",
            "--seed",
            "3",
            "--step-minutes",
            "60",
            "--output",
            str(path),
        )
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_ablation_command_reports_paired_differences(tmp_path: Path) -> None:
    """The ablation experiment runs and reports paired comparisons."""
    output = tmp_path / "ablation.json"
    stdout = _run_cli(
        "ablate",
        "--days",
        "4",
        "--seeds",
        "1",
        "2",
        "--step-minutes",
        "60",
        "--output",
        str(output),
    )
    assert "Paired sensor ablation" in stdout
    assert "95% CI" in stdout

    # The artefact wraps the findings in provenance: results sit under
    # "results", with configuration, seeds, environment and metric
    # definitions alongside them.
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "sensor_ablation"
    assert payload["seeds"] == [1, 2]
    assert payload["environment"]["python"]
    assert payload["metric_definitions"]["balanced_accuracy"]

    results = payload["results"]
    assert "all_modalities" in results["summary"]
    assert results["summary"]["minimal_door_bed"]["n_sensors"] == 2


def test_attribution_command_writes_a_provenance_record(tmp_path: Path) -> None:
    """Regression: wrapping results in provenance changed the artefact shape."""
    output = tmp_path / "attribution.json"
    _run_cli(
        "attribution",
        "--days",
        "3",
        "--seed",
        "5",
        "--step-minutes",
        "60",
        "--output",
        str(output),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "attribution_comparison"
    assert payload["seeds"] == [5]
    assert "scenarios" in payload["results"]
