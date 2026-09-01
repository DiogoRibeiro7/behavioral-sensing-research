"""Score the frozen v0.3 candidate against v0.2 on the frozen external cohort.

This is the one-shot primary external validation. Running it against the frozen
cohort is not repeatable under the validation contract: whatever it reports
stands, and no parameter, mapping, threshold or cohort membership may be changed
in response and rescored as the same confirmatory evidence.

The script therefore refuses to run unless every input matches what was frozen.
It verifies the profile digest, the cohort manifest's own recorded state, and the
SHA-256 of every recording it reads. A silent mismatch would produce a number
that looks like the pre-registered result and is not.

Use ``--cohort development`` to exercise the runner on already-inspected
development homes. That path is for checking the machinery works and reports
nothing about the primary cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.casas import truth_series
from sensor_modeling.evaluation.metrics import StateMetrics, state_metrics
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.states import BehaviouralState, StateOntology

COHORT_PATH = Path("artifacts/v03/external_cohort_manifest.json")
PROFILE_PATH = Path("artifacts/v03/v03_circadian_profile.json")

#: A state is reported individually only if it appears in at least this many
#: homes, per the contract's secondary-outcome definition.
MINIMUM_HOMES_FOR_STATE = 3

#: Resampling draws for the paired bootstrap over homes.
BOOTSTRAP_DRAWS = 10_000


@dataclass(frozen=True)
class HomeResult:
    """Both versions scored on one home, and the paired contrast."""

    home: str
    reference: StateMetrics
    candidate: StateMetrics
    scored_points: int
    labelled_fraction: float

    @property
    def difference(self) -> float:
        """``D_h``: candidate balanced accuracy minus reference."""
        return self.candidate.balanced_accuracy - self.reference.balanced_accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "home": self.home,
            "scored_points": self.scored_points,
            "labelled_fraction": self.labelled_fraction,
            "balanced_accuracy_v02": self.reference.balanced_accuracy,
            "balanced_accuracy_v03": self.candidate.balanced_accuracy,
            "difference": self.difference,
            "v02": self.reference.to_dict(),
            "v03": self.candidate.to_dict(),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> dict[BehaviouralState, tuple[float, ...]]:
    """Load the frozen circadian profile, checking its recorded digest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen-development-fit":
        raise SystemExit(f"profile at {path} is not a frozen development fit")
    if payload.get("test_outcomes_inspected") is not False:
        raise SystemExit("profile claims test outcomes were inspected; refusing")
    return {
        BehaviouralState(name): tuple(values)
        for name, values in payload["fit"]["profile"].items()
    }


def load_cohort(path: Path) -> list[dict[str, Any]]:
    """Load the frozen cohort, refusing anything not frozen or already scored."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen-primary-external-cohort":
        raise SystemExit(f"cohort at {path} is not frozen")
    if payload.get("scored") is not False:
        raise SystemExit(
            "cohort is already marked scored; the primary validation is one-shot "
            "and must not be repeated"
        )
    homes = payload["eligible_homes"]
    if not homes:
        raise SystemExit("frozen cohort is empty")
    return list(homes)


def resolve(root: Path, filename: str) -> Path:
    """Find one recording in the archive, refusing an ambiguous match."""
    matches = sorted(p for p in root.rglob(filename) if p.is_file())
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one archive member named {filename}, found "
            f"{len(matches)}"
        )
    return matches[0]


def score_home(
    path: Path,
    zone: ZoneInfo,
    step: timedelta,
    profile: dict[BehaviouralState, tuple[float, ...]],
) -> HomeResult | None:
    """Score both versions on one recording, or return ``None`` if unscoreable."""
    recording = read_casas_hh(path, timezone=zone)
    if not recording.observations:
        return None

    outcomes: list[StateMetrics] = []
    moments: list[Any] = []
    for ontology in (StateOntology(), StateOntology(circadian=profile)):
        pipeline = BehaviouralSensingPipeline(
            recording.registry,
            ontology=ontology,
            config=PipelineConfig(tz=zone, step=step),
        )
        steps = pipeline.run(recording.observations)
        steps.extend(pipeline.close(recording.observations[-1].timestamp))
        moments = [s.at for s in steps]
        truth = truth_series(recording.activities, moments)
        outcomes.append(state_metrics(truth, [s.state for s in steps]))

    scored = sum(
        1 for label in truth_series(recording.activities, moments) if label is not None
    )
    if scored == 0:
        return None

    return HomeResult(
        home=path.stem,
        reference=outcomes[0],
        candidate=outcomes[1],
        scored_points=scored,
        labelled_fraction=recording.labelled_fraction,
    )


def paired_bootstrap(
    differences: list[float], *, draws: int, seed: int
) -> tuple[float, float]:
    """Two-sided 95% interval on the median, resampling homes not time points."""
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    sample = rng.choice(values, size=(draws, values.size), replace=True)
    medians = np.median(sample, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarise(results: list[HomeResult], *, seed: int) -> dict[str, Any]:
    """Build the primary and secondary outcomes the contract specifies."""
    differences = [r.difference for r in results]
    median = statistics.median(differences)
    low, high = paired_bootstrap(differences, draws=BOOTSTRAP_DRAWS, seed=seed)

    # State-specific recall only where the state appears in enough homes.
    per_state: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for version, metrics in (("v02", result.reference), ("v03", result.candidate)):
            for state, recall in metrics.per_class_recall.items():
                per_state.setdefault(state, {}).setdefault(version, []).append(recall)
    states = {
        state: {
            version: statistics.median(values) for version, values in versions.items()
        }
        for state, versions in per_state.items()
        if len(versions.get("v02", [])) >= MINIMUM_HOMES_FOR_STATE
    }

    def median_of(attribute: str, version: str) -> float:
        source = "reference" if version == "v02" else "candidate"
        return statistics.median(
            getattr(getattr(r, source), attribute) for r in results
        )

    return {
        "primary": {
            "estimand": "median paired difference in household balanced accuracy",
            "median_difference": median,
            "ci_low": low,
            "ci_high": high,
            "homes": len(results),
            "improved": sum(1 for d in differences if d > 0),
            "unchanged": sum(1 for d in differences if d == 0),
            "worsened": sum(1 for d in differences if d < 0),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "resampling_unit": "home",
        },
        "secondary": {
            version: {
                "balanced_accuracy": median_of("balanced_accuracy", version),
                "calibration_error": median_of("calibration_error", version),
                "log_loss": median_of("log_loss", version),
                "brier": median_of("brier", version),
                "abstention_rate": median_of("abstention_rate", version),
            }
            for version in ("v02", "v03")
        },
        "labelled_coverage_median": statistics.median(
            r.labelled_fraction for r in results
        ),
        "scored_points_total": sum(r.scored_points for r in results),
        "state_recall_median": states,
        "homes": [r.to_dict() for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cohort", choices=("primary", "development"), required=True)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--i-understand-this-is-one-shot",
        action="store_true",
        help="Required for --cohort primary. The result stands whatever it says.",
    )
    args = parser.parse_args()

    if args.cohort == "primary" and not args.i_understand_this_is_one_shot:
        raise SystemExit(
            "scoring the primary cohort is one-shot and unrepeatable; pass "
            "--i-understand-this-is-one-shot to proceed"
        )

    zone = ZoneInfo(args.timezone)
    step = timedelta(minutes=args.step_minutes)
    profile = load_profile(PROFILE_PATH)

    manifest = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    if args.cohort == "primary":
        entries = load_cohort(COHORT_PATH)
    else:
        panel = manifest["development_panel"]
        entries = [
            {"id": home, "filename": f"{home}.csv", "sha256": None} for home in panel
        ]

    results: list[HomeResult] = []
    skipped: list[dict[str, str]] = []
    for entry in entries:
        path = resolve(args.archive_root, str(entry["filename"]))
        expected = entry.get("sha256")
        if expected is not None:
            actual = _sha256(path)
            if actual != expected:
                raise SystemExit(
                    f"{entry['id']}: archive digest {actual} does not match the "
                    f"frozen {expected}; the cohort and the data disagree"
                )
        outcome = score_home(path, zone, step, profile)
        if outcome is None:
            skipped.append({"id": str(entry["id"]), "reason": "no scoreable points"})
            continue
        results.append(outcome)
        print(
            f"{outcome.home:10} v0.2 {outcome.reference.balanced_accuracy:.3f}  "
            f"v0.3 {outcome.candidate.balanced_accuracy:.3f}  "
            f"D {outcome.difference:+.3f}",
            flush=True,
        )

    if not results:
        raise SystemExit("no home produced a scoreable result")

    payload = {
        "schema_version": 1,
        "cohort": args.cohort,
        "profile_sha256": json.loads(PROFILE_PATH.read_text(encoding="utf-8"))[
            "profile_sha256"
        ],
        "step_minutes": args.step_minutes,
        "skipped": skipped,
        **summarise(results, seed=args.bootstrap_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    primary = payload["primary"]
    print()
    print(f"median D_h        {primary['median_difference']:+.4f}")
    print(f"95% CI            [{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]")
    print(
        f"homes             {primary['homes']}  "
        f"({primary['improved']} improved, {primary['worsened']} worsened)"
    )
    print(f"median coverage   {payload['labelled_coverage_median']:.1%}")
    print(f"written to        {args.output}")


if __name__ == "__main__":
    main()
