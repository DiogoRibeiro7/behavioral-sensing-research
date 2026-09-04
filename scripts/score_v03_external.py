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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from sensor_modeling.datasets import read_casas_hh
from sensor_modeling.datasets.casas import truth_series
from sensor_modeling.evaluation.metrics import StateMetrics, state_metrics
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.states import BehaviouralState, StateOntology
from sensor_modeling.utils import text_file_sha256

COHORT_PATH = Path("artifacts/v03/external_cohort_manifest.json")
PROFILE_PATH = Path("artifacts/v03/v03_circadian_profile.json")
DECLARATION_PATH = Path("artifacts/v03/external_cohort_freeze_declaration.json")

#: Marker written after a successful primary run. Its existence is what makes
#: the validation one-shot in practice rather than only in prose.
CONSUMED_PATH = Path("artifacts/v03/external_primary_scored.json")

#: Evaluation grid for the primary run.
#:
#: Neither the candidate specification nor the validation contract fixes a step
#: size or timezone, which is a gap in the pre-registration. They are pinned
#: here to the values every development result used, and overriding them is
#: refused on the primary path: a different grid changes the circadian
#: alignment and the number of scored points, so it would not be the
#: pre-registered comparison.
FROZEN_TIMEZONE = "America/Los_Angeles"
FROZEN_STEP_MINUTES = 5

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


def _canonical_fit_sha256(fit: dict[str, Any]) -> str:
    """Digest of the fit block, matching the freeze validator byte for byte."""
    payload = json.dumps(
        fit, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_profile(path: Path) -> dict[BehaviouralState, tuple[float, ...]]:
    """Load the frozen circadian profile, recomputing its digest.

    Reading ``profile_sha256`` and trusting it would accept an edited profile
    carrying the old hash, which is precisely the substitution the digest exists
    to prevent. The digest is recomputed from the fit block and compared.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen-development-fit":
        raise SystemExit(f"profile at {path} is not a frozen development fit")
    if payload.get("test_outcomes_inspected") is not False:
        raise SystemExit("profile claims test outcomes were inspected; refusing")

    recorded = payload.get("profile_sha256")
    actual = _canonical_fit_sha256(payload["fit"])
    if recorded != actual:
        raise SystemExit(
            f"profile digest mismatch: recorded {recorded}, recomputed {actual}; "
            "the profile has changed since it was frozen"
        )
    return {
        BehaviouralState(name): tuple(values)
        for name, values in payload["fit"]["profile"].items()
    }


def load_cohort(path: Path, declaration: Path) -> list[dict[str, Any]]:
    """Load the frozen cohort, checking it against its freeze declaration.

    Trusting the manifest's own membership and hashes would let an edited
    manifest be scored as frozen. The declaration records the manifest digest
    separately, so the manifest is verified against it before any home is read.
    """
    if not declaration.exists():
        raise SystemExit(f"freeze declaration not found at {declaration}")
    freeze = json.loads(declaration.read_text(encoding="utf-8"))
    expected = freeze.get("manifest_sha256")
    actual = text_file_sha256(path)
    if expected != actual:
        raise SystemExit(
            f"cohort manifest digest mismatch: declaration records {expected}, "
            f"file is {actual}; the frozen cohort has changed"
        )
    if freeze.get("scored") is not False:
        raise SystemExit("freeze declaration already records the cohort as scored")

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
        # close() re-emits the final estimate as a reporting step carrying the
        # day summary. Extending with it would score that one point twice, so
        # it is flushed for its side effects and discarded here.
        steps = pipeline.run(recording.observations)
        pipeline.close(recording.observations[-1].timestamp)
        moments = [s.at for s in steps]
        truth = truth_series(recording.activities, moments)

        # Computed before scoring: state_metrics raises when nothing is
        # labelled, so a later guard would never be reached and an unscoreable
        # home would abort the whole run instead of being skipped.
        scored = sum(1 for label in truth if label is not None)
        if scored == 0:
            return None
        outcomes.append(state_metrics(truth, [s.state for s in steps]))

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

    if args.cohort == "primary":
        if not args.i_understand_this_is_one_shot:
            raise SystemExit(
                "scoring the primary cohort is one-shot and unrepeatable; pass "
                "--i-understand-this-is-one-shot to proceed"
            )
        if CONSUMED_PATH.exists():
            raise SystemExit(
                f"the primary cohort has already been scored; see {CONSUMED_PATH}. "
                "Rescoring it would not be the same confirmatory validation."
            )
        if args.timezone != FROZEN_TIMEZONE or args.step_minutes != FROZEN_STEP_MINUTES:
            raise SystemExit(
                "the primary run uses the frozen evaluation grid "
                f"({FROZEN_TIMEZONE}, {FROZEN_STEP_MINUTES} minutes); a different "
                "grid changes the circadian alignment and the scored points, so "
                "it would not be the pre-registered comparison"
            )

    zone = ZoneInfo(args.timezone)
    step = timedelta(minutes=args.step_minutes)
    profile = load_profile(PROFILE_PATH)

    manifest = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    if args.cohort == "primary":
        entries = load_cohort(COHORT_PATH, DECLARATION_PATH)
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

    if args.cohort == "primary":
        # Written after the result exists, so a crash mid-run does not burn the
        # one shot, and before the summary is printed, so the marker cannot be
        # skipped by someone interrupting at the last moment.
        CONSUMED_PATH.write_text(
            json.dumps(
                {
                    "status": "primary-external-cohort-scored",
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                    "result_file": str(args.output),
                    "result_sha256": _sha256(args.output),
                    "profile_sha256": payload["profile_sha256"],
                    "homes": payload["primary"]["homes"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
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
