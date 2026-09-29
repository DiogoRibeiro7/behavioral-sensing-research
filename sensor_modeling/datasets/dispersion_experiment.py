"""Phase 3.3 follow-up: the pre-specified evaluation of the hurdle negative binomial.

The posterior predictive checks found the fitted hurdle model's zero-truncated
Poisson active count systematically under-dispersed. The hurdle negative
binomial (``hurdle_nb`` in :mod:`~sensor_modeling.datasets.channel_models`)
keeps the hurdle's silence and replaces the active count with a zero-truncated
negative binomial. This experiment compares three channel observation models
on identical information and inference:

- ``declared``: the declared Poisson rates;
- ``hurdle``: the fitted hurdle-Poisson, the Phase 3.3 follow-up's model;
- ``hurdle_nb``: the fitted hurdle negative binomial.

Each is scored with current windows, ``I0``, and in the filter's recursion over
every window, ``R``. Every setting, comparison and criterion is declared in
:class:`DispersionProtocol` and frozen in a committed file before any household
is scored.

The questions
-------------
Against the hurdle-Poisson model, in each setting, does the negative binomial:

1. recover some of the ``home_active`` recall the hurdle-Poisson model lost;
2. preserve its calibration improvement over the declared rates;
3. improve log loss;
4. avoid degrading balanced accuracy?

Extra complexity is not accepted because one metric improves. The decision
rule requires a log-loss gain, the proper score, together with every guard:
calibration preserved, and neither balanced accuracy nor ``home_active`` recall
worse.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.households import household_values, recall_of
from ..evaluation.metrics import PredictionMetrics
from ..evaluation.provenance import (
    ExperimentRecord,
    InputArtifact,
    ModelRecord,
    ReportedInterval,
)
from ..fusion.regime import ONLINE
from ..states.ontology import BehaviouralState, StateOntology
from .casas import CasasRecording
from .channel_models import (
    DECLARED,
    HURDLE,
    HURDLE_NB,
    MAX_DISPERSION,
    ChannelStatistics,
    FittedChannels,
    combine_statistics,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    total_loglik,
    ztnb_log_pmf,
    ztnb_rate,
)
from .information_sets import (
    EvidenceChannel,
    EvidenceResolution,
    InformationComponent,
    InformationSet,
    build_feature_table,
)
from .matched_evaluation import (
    MATCHED_METRIC_DEFINITIONS,
    HouseholdSplit,
    _finite,
    _Household,
    _regular_moments,
    online_evidence,
)
from .recoverable_gap import (
    DEFAULT_METRICS,
    GENERATIVE_CONFIGURATION,
    FrozenSplits,
    GapProtocol,
    _cell,
    _level,
    _paired,
    _score,
)
from .restricted_filter import restricted_posteriors
from .silence_dependence import concentration
from .time_prior_experiment import MINIMAL_DIFFERENCES, verdict

PROTOCOL_SCHEMA = "dispersion-protocol/1"
RESULT_SCHEMA = "dispersion-results/1"

S = BehaviouralState

CURRENT = "I0"
RECURSION = "R"
SETTINGS = (CURRENT, RECURSION)

MODELS = (DECLARED, HURDLE, HURDLE_NB)

#: The state whose recall the hurdle-Poisson model lost, and question 1 reads.
FOCUS_STATE = S.HOME_ACTIVE

#: The quiet-run overconfidence slope's name and scale.
SLOPE = "accumulation_slope"


def cell_key(model: str, setting: str) -> str:
    """The name of one scored cell: model and setting."""
    return f"{model}@{setting}"


@dataclass(frozen=True)
class Comparison:
    """One pre-specified paired comparison within one setting."""

    key: str
    role: str
    question: str
    model: str
    reference: str
    setting: str

    def to_dict(self) -> dict[str, str]:
        """Return a serialisable form."""
        return {
            "key": self.key,
            "role": self.role,
            "question": self.question,
            "model": cell_key(self.model, self.setting),
            "reference": cell_key(self.reference, self.setting),
        }


ESTIMANDS: tuple[Comparison, ...] = (
    Comparison(
        "N0",
        "primary",
        "the negative binomial against the hurdle-Poisson, current windows",
        HURDLE_NB,
        HURDLE,
        CURRENT,
    ),
    Comparison(
        "NR",
        "primary",
        "the same, the recursion",
        HURDLE_NB,
        HURDLE,
        RECURSION,
    ),
    Comparison(
        "D0",
        "secondary",
        "the negative binomial against the declared rates, current windows: is the "
        "calibration improvement preserved",
        HURDLE_NB,
        DECLARED,
        CURRENT,
    ),
    Comparison(
        "DR",
        "secondary",
        "the same, the recursion",
        HURDLE_NB,
        DECLARED,
        RECURSION,
    ),
    Comparison(
        "H0",
        "context",
        "the hurdle-Poisson against the declared rates, current windows, as in "
        "the Phase 3.3 follow-up",
        HURDLE,
        DECLARED,
        CURRENT,
    ),
    Comparison(
        "HR",
        "context",
        "the same, the recursion",
        HURDLE,
        DECLARED,
        RECURSION,
    ),
)

#: For each setting, its primary estimand and the one against the declared rates.
PRIMARY = {CURRENT: ("N0", "D0"), RECURSION: ("NR", "DR")}

CRITERIA: dict[str, str] = {
    "verdicts": "favours model: mean >= delta and lower bound > 0; favours "
    "reference: mean <= -delta and upper bound < 0; negligible: the whole "
    "interval within (-delta, delta); uncertain: anything else",
    "recall": "question 1, from the primary estimand's home_active recall: "
    "recovers if it favours the negative binomial, no change if negligible, "
    "loses more if it favours the hurdle-Poisson, uncertain otherwise",
    "calibration": "question 2: preserved if calibration error does not favour "
    "the hurdle-Poisson in the primary estimand and favours the negative "
    "binomial against the declared rates; lost if it favours the hurdle-Poisson, "
    "or is negligible or favours the declared rates against them; uncertain "
    "otherwise",
    "log_loss": "question 3, from the primary estimand: improves, no change, "
    "worsens or uncertain",
    "balanced_accuracy": "question 4, from the primary estimand: not degraded if "
    "negligible or favouring the negative binomial, degraded if favouring the "
    "hurdle-Poisson, uncertain otherwise",
    "adopt": "log loss favours the negative binomial, calibration is preserved, "
    "balanced accuracy is not degraded, and home_active recall does not favour "
    "the hurdle-Poisson",
    "trade-off": "log loss favours the negative binomial, but calibration is "
    "lost, or balanced accuracy or home_active recall favours the hurdle-Poisson",
    "reject": "log loss is negligible or favours the hurdle-Poisson: the extra "
    "parameter is not accepted, whatever else improves",
    "inconclusive": "anything else",
    "overall": "adopt if adopted in both settings, reject if rejected in both, "
    "and otherwise not adopted, with each setting's decision reported",
    "complexity": "extra complexity is never accepted because one metric "
    "improves: adoption needs the log-loss gain and every guard",
}


def recall_answer(recall: str) -> str:
    """Question 1, from the primary estimand's home_active recall verdict."""
    return {
        "favours model": "recovers",
        "negligible": "no change",
        "favours reference": "loses more",
    }.get(recall, "uncertain")


def calibration_answer(primary: str, against_declared: str) -> str:
    """Question 2, from calibration error's verdicts in the two estimands."""
    if primary == "favours reference" or against_declared in (
        "negligible",
        "favours reference",
    ):
        return "lost"
    if against_declared == "favours model":
        return "preserved"
    return "uncertain"


def log_loss_answer(loss: str) -> str:
    """Question 3, from the primary estimand's log-loss verdict."""
    return {
        "favours model": "improves",
        "negligible": "no change",
        "favours reference": "worsens",
    }.get(loss, "uncertain")


def accuracy_answer(accuracy: str) -> str:
    """Question 4, from the primary estimand's balanced-accuracy verdict."""
    if accuracy in ("negligible", "favours model"):
        return "not degraded"
    if accuracy == "favours reference":
        return "degraded"
    return "uncertain"


def decision(verdicts: Mapping[str, str], recall: str, calibration: str) -> str:
    """Adopt, trade-off, reject or inconclusive, for one setting."""
    loss = verdicts["log_loss"]
    if loss in ("negligible", "favours reference"):
        return "reject"
    if loss != "favours model":
        return "inconclusive"
    if (
        calibration == "lost"
        or verdicts["balanced_accuracy"] == "favours reference"
        or recall == "favours reference"
    ):
        return "trade-off"
    if calibration == "preserved" and verdicts["balanced_accuracy"] in (
        "negligible",
        "favours model",
    ):
        return "adopt"
    return "inconclusive"


def overall_decision(decisions: Mapping[str, str]) -> str:
    """Adopt, reject or not adopted, across the settings."""
    values = set(decisions.values())
    if values == {"adopt"}:
        return "adopt"
    if values == {"reject"}:
        return "reject"
    return "not adopted"


NB_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "restricted generative filter with fitted hurdle negative-binomial "
    "channels",
    "emissions": "per channel and state: P(silent) = pi, and a zero-truncated "
    "negative binomial with mean mu and dispersion alpha for an active "
    "window's count",
    "fitted": "pi and the active mean as for the hurdle; alpha maximises the "
    "profile likelihood of the training households' active-count table plus "
    "the pseudo-windows, Poisson-shaped at the same mean; per fold, channel and "
    "state, on the fold's training households only",
}

HURDLE_CONFIGURATION: dict[str, object] = {
    **GENERATIVE_CONFIGURATION,
    "model": "restricted generative filter with fitted hurdle channels",
    "emissions": "per channel and state: P(silent) = pi, and a zero-truncated "
    "Poisson with rate mu for an active window's count",
    "fitted": "pi and mu per fold, per channel and state, on the fold's training "
    "households' labelled windows only, shrunk toward the declared model by the "
    "pseudo-windows",
}


@dataclass(frozen=True)
class DispersionProtocol:
    """Everything the experiment fixes before any household is scored."""

    folds: tuple[HouseholdSplit, ...]
    splits_sha256: str
    resolution: EvidenceResolution = field(default_factory=EvidenceResolution)
    pseudo_windows: float = 12.0
    max_dispersion: float = MAX_DISPERSION
    accumulation_steps: int = 36
    extreme_dispersion: float = 10.0
    min_active_windows: int = 100
    flat_gain: float = 1.0
    seed: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    resamples: int = 10_000
    confidence: float = 0.95
    minimal_differences: Mapping[str, float] = field(
        default_factory=lambda: {**MINIMAL_DIFFERENCES, SLOPE: 0.02}
    )
    rare_share: float = 0.05
    name: str = "phase3-hurdle-negative-binomial"

    def __post_init__(self) -> None:
        """Validate through the recoverable-gap protocol, then the rest."""
        base = self.as_gap_protocol()
        if not isinstance(self.splits_sha256, str) or len(self.splits_sha256) != 64:
            raise ValueError("splits_sha256 must be a SHA-256 hex digest")
        for name in ("pseudo_windows", "extreme_dispersion", "flat_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.max_dispersion != MAX_DISPERSION:
            raise ValueError("max_dispersion is the model's bound, MAX_DISPERSION")
        if not self.extreme_dispersion < self.max_dispersion:
            raise ValueError("extreme_dispersion must lie below max_dispersion")
        for name in ("accumulation_steps", "min_active_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2")
        missing = sorted(
            {*base.metrics, "recall", SLOPE} - set(self.minimal_differences)
        )
        if missing or any(
            not math.isfinite(v) or v <= 0.0 for v in self.minimal_differences.values()
        ):
            raise ValueError(
                f"every compared quantity needs a positive minimal difference; "
                f"missing {missing}"
            )
        if not 0.0 < self.rare_share < 1.0:
            raise ValueError("rare_share must lie in (0, 1)")
        object.__setattr__(self, "folds", base.folds)
        object.__setattr__(self, "metrics", base.metrics)
        object.__setattr__(self, "minimal_differences", dict(self.minimal_differences))

    def as_gap_protocol(self) -> GapProtocol:
        """The equivalent recoverable-gap protocol, for validation and helpers."""
        return GapProtocol(
            tuple(self.folds),
            seed=self.seed,
            metrics=tuple(self.metrics),
            resamples=self.resamples,
            confidence=self.confidence,
        )

    @property
    def information_set(self) -> InformationSet:
        """``I0``: each channel's current window."""
        return InformationSet(
            "current",
            frozenset({InformationComponent.CURRENT_EVIDENCE}),
            self.resolution,
        )

    @property
    def homes(self) -> tuple[str, ...]:
        """Every held-out household, sorted."""
        return tuple(sorted(home for fold in self.folds for home in fold.test))

    def to_dict(self) -> dict[str, object]:
        """Return the protocol as a stable JSON-serialisable declaration."""
        info = self.information_set
        return {
            "schema": PROTOCOL_SCHEMA,
            "name": self.name,
            "motivation": "the posterior predictive checks found the "
            "zero-truncated Poisson active count systematically under-dispersed, "
            "with a variance growing with the square of the mean; their routing "
            "named within-state temporal dependence as the next family, which a "
            "count distribution cannot address, and this experiment measures what "
            "the over-dispersed count alone does",
            "households": {
                "splits_file": "artifacts/phase1/household_splits.json",
                "splits_sha256": self.splits_sha256,
                "folds": [
                    {**fold.to_dict(), "sha256": fold.sha256()} for fold in self.folds
                ],
                "use": "each household is held out once; its fold's training "
                "households fit every channel model",
            },
            "settings": {
                CURRENT: {**info.to_dict(), "sha256": info.sha256()},
                RECURSION: "the filter's recursion fed every window of the "
                "recording from the stationary distribution; every model receives "
                "the same windows",
            },
            "models": {
                DECLARED: GENERATIVE_CONFIGURATION,
                HURDLE: HURDLE_CONFIGURATION,
                HURDLE_NB: NB_CONFIGURATION,
            },
            "identical": "prior, transition, channels, windows and inference; the "
            "models differ only in each channel's observation model",
            "fitting": {
                "pseudo_windows": self.pseudo_windows,
                "max_dispersion": self.max_dispersion,
                "hurdle": "pi = (Z + k pi0) / (W + k); m = (S + k m0) / (P + k); "
                "mu solves mu / (1 - exp(-mu)) = m",
                "hurdle_nb": "the same pi and m; alpha maximises the profile "
                "likelihood of the active-count table plus k pseudo-windows "
                "distributed as the zero-truncated Poisson with mean m, over {0} "
                "and [1e-4, max_dispersion]; mu solves the zero-truncated negative "
                "binomial's mean for m",
            },
            "metrics": list(self.metrics),
            "calibration": "expected calibration error over 10 confidence bins",
            "per_state_recall": {
                "estimands": "every estimand",
                "focus_state": FOCUS_STATE.value,
                "rare_rule": f"share of labelled time below {self.rare_share} in "
                "either fold's training households",
            },
            "quiet_runs": {
                "statistic": "Phase 3.3's accumulation slope: within the predicted "
                "state, the least-squares slope per hour of confidence minus "
                "correctness on the position in a run of consecutive fully silent "
                "windows, over labelled windows at positions 1 to "
                f"{self.accumulation_steps}",
                "setting": "the recursion only: a fully silent window has the "
                "same likelihood under both hurdle families, so with current "
                "windows their slopes coincide",
                "comparison": "paired household difference, lower is better",
            },
            "estimands": [estimand.to_dict() for estimand in ESTIMANDS],
            "questions": {
                "1": "does the negative binomial recover some of the home_active "
                "recall the hurdle-Poisson lost",
                "2": "does it preserve the calibration improvement",
                "3": "does it improve log loss",
                "4": "does it avoid degrading balanced accuracy",
                "settings": "each question is answered in each setting from its "
                "primary estimand, N0 or NR, and for calibration also D0 or DR",
            },
            "dispersion": {
                "recorded": "each fold's alpha, size, mu and active mean per "
                "channel and state, with its active training windows",
                "extreme": f"alpha of at least {self.extreme_dispersion:g}",
                "low_data": f"fewer than {self.min_active_windows} active training "
                "windows",
                "flat": "the observed table's log-likelihood at alpha exceeds that "
                f"at alpha / 2 by less than {self.flat_gain:g} nat: weakly "
                "identified",
                "classes": "poisson (alpha 0); moderate (below extreme); extreme, "
                "low data; extreme, flat; extreme, supported",
                "question": "extreme estimates indicate insufficient data if more "
                "than half of them are low data; otherwise they are not a data "
                "shortage",
            },
            "minimal_differences": dict(self.minimal_differences),
            "criteria": dict(CRITERIA),
            "bootstrap": {
                "unit": "household",
                "statistics": "mean and median paired differences",
                "resamples": self.resamples,
                "confidence": self.confidence,
                "interval": "percentile",
            },
            "seeds": {"bootstrap": self.seed},
            "inference": "online filter: I0 and R read no window after the scored one",
            "tuning": "none: every setting is declared here, before scoring",
        }

    def sha256(self) -> str:
        """SHA-256 of the canonical declaration."""
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


def declared_protocol(splits: FrozenSplits) -> DispersionProtocol:
    """The protocol as declared for the development panel's frozen folds."""
    return DispersionProtocol(splits.folds, splits.sha256)


def check_frozen_protocol(protocol: DispersionProtocol, path: Path) -> str:
    """Refuse to run unless *protocol* is exactly the one frozen at *path*."""
    raw = Path(path).read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen != {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}:
        raise ValueError(
            f"the protocol in {path} differs from the one this code declares; "
            "a frozen protocol cannot change after scoring begins"
        )
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


# ----------------------------------------------------------------------------
# Dispersion estimates
# ----------------------------------------------------------------------------
def dispersion_class(
    alpha: float, active: int, flatness: float | None, protocol: DispersionProtocol
) -> str:
    """Which declared class one channel and state's dispersion estimate is in."""
    if alpha == 0.0:
        return "poisson"
    if alpha < protocol.extreme_dispersion:
        return "moderate"
    if active < protocol.min_active_windows:
        return "extreme, low data"
    if flatness is not None and flatness < protocol.flat_gain:
        return "extreme, flat"
    return "extreme, supported"


def _table_loglik(
    statistics: ChannelStatistics, state: int, active_mean: float, alpha: float
) -> float:
    """The observed active-count table's log-likelihood at *alpha*, keeping the mean."""
    table = statistics.active_counts
    if table is None:  # pragma: no cover - home_statistics always builds it
        raise ValueError("the statistics carry no active-count table")
    weights = table.frequencies[state]
    kept = weights > 0
    rate = ztnb_rate(active_mean, alpha)
    return float(np.dot(weights[kept], ztnb_log_pmf(table.values[kept], rate, alpha)))


def dispersion_estimates(
    population: FittedChannels, protocol: DispersionProtocol
) -> list[dict[str, Any]]:
    """Every channel and state's dispersion estimate, its support and its class."""
    rows: list[dict[str, Any]] = []
    for channel in sorted(population.statistics):
        statistics = population.statistics[channel]
        parameters = population.parameters(channel)
        alphas = population.dispersion(channel)
        for i, state in enumerate(population.states):
            alpha = float(alphas[i])
            mean = float(parameters["active_mean"][i])
            active = int(statistics.active[i])
            if not (math.isfinite(alpha) and math.isfinite(mean)):
                continue
            gain = flatness = None
            if active > 0:
                at_alpha = _table_loglik(statistics, i, mean, alpha)
                gain = at_alpha - _table_loglik(statistics, i, mean, 0.0)
                if alpha > 0.0:
                    flatness = at_alpha - _table_loglik(statistics, i, mean, alpha / 2)
            rows.append(
                {
                    "channel": channel.name,
                    "room": channel.room,
                    "modality": channel.modality.value,
                    "state": state.value,
                    "active_windows": active,
                    "dispersion": alpha,
                    "size": 1.0 / alpha if alpha > 0.0 else None,
                    "at_bound": alpha >= protocol.max_dispersion,
                    "rate": ztnb_rate(mean, alpha),
                    "active_mean": mean,
                    "gain_over_poisson": gain,
                    "flatness": flatness,
                    "class": dispersion_class(alpha, active, flatness, protocol),
                }
            )
    return rows


def extreme_reading(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether extreme dispersion estimates indicate insufficient data."""
    extreme = [r for r in rows if str(r["class"]).startswith("extreme")]
    low = [r for r in extreme if r["class"] == "extreme, low data"]
    if not extreme:
        answer = "no extreme estimates"
    elif 2 * len(low) > len(extreme):
        answer = "insufficient data"
    else:
        answer = "not a data shortage"
    return {
        "extreme": len(extreme),
        "low_data": len(low),
        "flat": sum(1 for r in extreme if r["class"] == "extreme, flat"),
        "supported": sum(1 for r in extreme if r["class"] == "extreme, supported"),
        "answer": answer,
    }


# ----------------------------------------------------------------------------
# One household
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DispersionResult:
    """A completed experiment: its record and where it was written."""

    record: ExperimentRecord
    path: Path | None = None


def household_scores(
    recording: CasasRecording,
    protocol: DispersionProtocol,
    population: FittedChannels,
    *,
    household: str,
    ontology: StateOntology,
) -> dict[str, Any]:
    """Every model's scores and quiet-run slope for one held-out household."""
    space = tuple(ontology.states)
    resolution = protocol.resolution
    families = {
        model: population.models(recording.registry, resolution, model, ontology)
        for model in MODELS
    }
    moments = _regular_moments(recording, resolution.step)
    table = build_feature_table(
        recording, protocol.information_set, moments, household=household
    )
    built = _Household.build(household, "test", table, recording)
    scores: dict[str, PredictionMetrics] = {}
    for model, models in families.items():
        beliefs = restricted_posteriors(table, models, ontology)[built.labelled]
        scores[cell_key(model, CURRENT)] = _score(built.labels, beliefs, space)

    counts, rows, labels, _ = household_channel_counts(
        recording, resolution, ontology, household=household
    )
    transition = ontology.transition(resolution.step)
    silent_count = np.sum([counts[c] == 0 for c in counts], axis=0)
    slopes: dict[str, float | None] = {}
    for model, models in families.items():
        _, posterior = filter_recursion(
            total_loglik(models, counts), transition, ontology.stationary()
        )
        scores[cell_key(model, RECURSION)] = _score(
            [space[i] for i in labels], posterior[rows], space
        )
        slopes[model] = concentration(
            posterior,
            rows,
            labels,
            silent_count,
            len(counts),
            accumulation_steps=protocol.accumulation_steps,
            steps_per_hour=3600.0 / resolution.step.total_seconds(),
        )[SLOPE]
    return {
        "scores": scores,
        "labelled": int(rows.size),
        "moments_sha256": built.moments_sha256,
        "channels": [c.name for c in sorted(counts)],
        "untrained_channels": sorted(
            c.name for c in counts if c not in population.statistics
        ),
        "accumulation_slope": slopes,
    }


# ----------------------------------------------------------------------------
# Across households
# ----------------------------------------------------------------------------
def _shares(
    fold: HouseholdSplit,
    space: tuple[BehaviouralState, ...],
    statistics: Mapping[str, Mapping[EvidenceChannel, ChannelStatistics]],
) -> dict[str, float]:
    """Each state's share of labelled windows in the fold's training households."""
    totals = np.zeros(len(space))
    for home in fold.train:
        parts = statistics[home]
        if parts:
            totals += next(iter(parts.values())).windows
    return {state.value: float(v) for state, v in zip(space, totals / totals.sum())}


def run_dispersion(
    recordings: Mapping[str, CasasRecording],
    protocol: DispersionProtocol,
    *,
    data_source: str,
    inputs: Sequence[InputArtifact] = (),
    output_dir: Path | None = None,
) -> DispersionResult:
    """Run the pre-specified experiment and build its record."""
    homes = sorted({h for fold in protocol.folds for h in (*fold.train, *fold.test)})
    missing = sorted(set(homes) - set(recordings))
    if missing:
        raise ValueError(f"no recording for households {missing}")
    ontology = StateOntology()
    space = tuple(ontology.states)
    gap = protocol.as_gap_protocol()
    statistics = {
        home: home_statistics(
            recordings[home], protocol.resolution, ontology, household=home
        )
        for home in homes
    }
    fits: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        population = combine_statistics(
            {h: statistics[h] for h in fold.train},
            states=space,
            pseudo_windows=protocol.pseudo_windows,
        )
        fits[fold.name] = {
            "population": population,
            "shares": _shares(fold, space, statistics),
            "dispersion": dispersion_estimates(population, protocol),
        }
    rare = sorted(
        s.value
        for s in space
        if any(f["shares"][s.value] < protocol.rare_share for f in fits.values())
    )

    scores: dict[str, dict[str, PredictionMetrics]] = {}
    households: dict[str, dict[str, Any]] = {}
    for fold in protocol.folds:
        for home in sorted(fold.test):
            scored = household_scores(
                recordings[home],
                protocol,
                fits[fold.name]["population"],
                household=home,
                ontology=ontology,
            )
            for key, metrics in scored.pop("scores").items():
                scores.setdefault(key, {})[home] = metrics
            households[home] = {"fold": fold.name, **scored}
    households = dict(sorted(households.items()))

    def values(model: str, setting: str, metric: Any) -> dict[str, float | None]:
        return household_values(scores[cell_key(model, setting)], metric)

    def slopes(model: str) -> dict[str, float | None]:
        return {h: e["accumulation_slope"][model] for h, e in households.items()}

    cells = [
        {
            **_cell(model, setting, scores[cell_key(model, setting)], space, gap),
            "key": cell_key(model, setting),
        }
        for setting in SETTINGS
        for model in MODELS
    ]
    quiet_runs = {model: _level(slopes(model), gap) for model in MODELS}

    estimands: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        comparisons = {
            metric: _paired(
                values(estimand.model, estimand.setting, metric),
                values(estimand.reference, estimand.setting, metric),
                metric,
                gap,
            )
            for metric in protocol.metrics
        }
        verdicts = {
            metric: verdict(comparison, protocol.minimal_differences[metric])
            for metric, comparison in comparisons.items()
        }
        recall = {}
        for state in space:
            comparison = _paired(
                values(estimand.model, estimand.setting, recall_of(state)),
                values(estimand.reference, estimand.setting, recall_of(state)),
                "balanced_accuracy",
                gap,
            )
            recall[state.value] = {
                "rare": state.value in rare,
                "comparison": comparison,
                "verdict": verdict(comparison, protocol.minimal_differences["recall"]),
            }
        entry: dict[str, Any] = {
            **estimand.to_dict(),
            "setting": estimand.setting,
            "comparisons": comparisons,
            "verdicts": verdicts,
            "per_state_recall": recall,
        }
        if estimand.setting == RECURSION:
            comparison = _paired(
                slopes(estimand.model),
                slopes(estimand.reference),
                SLOPE,
                gap,
                higher_is_better=False,
            )
            entry["quiet_runs"] = {
                "comparison": comparison,
                "verdict": verdict(comparison, protocol.minimal_differences[SLOPE]),
            }
        estimands.append(entry)

    by_key = {e["key"]: e for e in estimands}
    questions: dict[str, dict[str, str]] = {}
    for setting, (primary, against_declared) in PRIMARY.items():
        verdicts = by_key[primary]["verdicts"]
        recall = by_key[primary]["per_state_recall"][FOCUS_STATE.value]["verdict"]
        calibration = calibration_answer(
            verdicts["calibration_error"],
            by_key[against_declared]["verdicts"]["calibration_error"],
        )
        questions[setting] = {
            "home_active_recall": recall_answer(recall),
            "calibration": calibration,
            "log_loss": log_loss_answer(verdicts["log_loss"]),
            "balanced_accuracy": accuracy_answer(verdicts["balanced_accuracy"]),
            "decision": decision(verdicts, recall, calibration),
        }

    estimates = {fold: fit["dispersion"] for fold, fit in fits.items()}
    every = [row for rows_ in estimates.values() for row in rows_]
    classes = sorted({str(r["class"]) for r in every})
    dispersion = {
        "estimates": estimates,
        "by_state": {
            state.value: {
                c: sum(
                    1 for r in every if r["state"] == state.value and r["class"] == c
                )
                for c in classes
            }
            for state in space
        },
        "by_channel": {
            channel: {
                c: sum(1 for r in every if r["channel"] == channel and r["class"] == c)
                for c in classes
            }
            for channel in sorted({str(r["channel"]) for r in every})
        },
        "extreme": extreme_reading(every),
    }

    results = {
        "result_schema": RESULT_SCHEMA,
        "status": "pre-specified; development panel, which earlier work has inspected",
        "states": [state.value for state in space],
        "focus_state": FOCUS_STATE.value,
        "rare_states": rare,
        "households": households,
        "cells": cells,
        "quiet_runs": quiet_runs,
        "estimands": estimands,
        "questions": questions,
        "conclusion": overall_decision(
            {setting: q["decision"] for setting, q in questions.items()}
        ),
        "dispersion": dispersion,
        "fitted": {
            fold: {
                "population": {
                    **fit["population"].to_dict(),
                    "sha256": fit["population"].sha256(),
                },
                "training_shares": fit["shares"],
            }
            for fold, fit in fits.items()
        },
    }
    record = ExperimentRecord(
        experiment=protocol.name,
        configuration={**protocol.to_dict(), "protocol_sha256": protocol.sha256()},
        inference=ONLINE,
        evidence=online_evidence(recordings, protocol.homes, protocol.resolution.step),
        seeds=[protocol.seed],
        results=_finite(results),
        data_source=data_source,
        metric_definitions={
            **MATCHED_METRIC_DEFINITIONS,
            SLOPE: "Within the predicted state, the least-squares slope per hour "
            "of confidence minus correctness on the position in a run of "
            "consecutive fully silent windows; zero in expectation for a "
            "calibrated model. Lower is better.",
            "dispersion": "The negative binomial's alpha: the active count's "
            "untruncated variance is mu + alpha mu^2.",
            "gain_over_poisson": "The observed active-count table's log-likelihood "
            "at the fitted alpha minus at alpha 0, keeping the active mean.",
            "flatness": "The same table's log-likelihood at alpha minus at alpha / "
            "2: small when alpha is weakly identified.",
        },
        notes=[
            "Every setting, estimand and criterion was declared in the protocol "
            "before any household was scored.",
            "Every model has the same prior, transition, channels, windows and "
            "inference; only each channel's observation model differs.",
            "Channel parameters, including the dispersion, are fitted per fold on "
            "training households only.",
            "Verdicts compare effect sizes and household bootstrap intervals with "
            "declared minimal differences; they are not significance tests.",
        ],
        inputs=list(inputs),
        split={
            "folds": [{**f.to_dict(), "sha256": f.sha256()} for f in protocol.folds]
        },
        information_set={
            **protocol.information_set.to_dict(),
            "sha256": protocol.information_set.sha256(),
        },
        preprocessing={
            "moments": "one per step from each recording's first observation to "
            "its last, identical for every model",
            "labels": "truth_series of each recording's annotations; unlabelled "
            "moments are not scored",
            "step_seconds": protocol.resolution.step.total_seconds(),
        },
        models=[
            ModelRecord(
                DECLARED,
                GENERATIVE_CONFIGURATION,
                "restricted_posteriors and filter_recursion",
            ),
            ModelRecord(
                HURDLE,
                HURDLE_CONFIGURATION,
                "restricted_posteriors and filter_recursion",
            ),
            ModelRecord(
                HURDLE_NB,
                NB_CONFIGURATION,
                "restricted_posteriors and filter_recursion",
            ),
        ],
        household_metrics={
            key: {h: m.to_dict() for h, m in sorted(per_home.items())}
            for key, per_home in sorted(scores.items())
        },
        intervals=_intervals(estimands),
    )
    path = (
        record.write(Path(output_dir) / f"{protocol.name}.json")
        if output_dir is not None
        else None
    )
    return DispersionResult(record=record, path=path)


def _intervals(estimands: Sequence[Mapping[str, Any]]) -> list[ReportedInterval]:
    reported: list[ReportedInterval] = []
    for entry in estimands:
        compared = dict(entry["comparisons"])
        if "quiet_runs" in entry:
            compared[SLOPE] = entry["quiet_runs"]["comparison"]
        for metric, comparison in compared.items():
            for statistic in ("mean", "median"):
                estimate = comparison[statistic]
                if estimate["interval"] is None:
                    continue
                reported.append(
                    ReportedInterval(
                        label=f"{entry['key']}: {statistic} {metric} difference",
                        estimate=estimate["estimate"],
                        low=estimate["interval"]["low"],
                        high=estimate["interval"]["high"],
                        confidence=estimate["interval"]["confidence"],
                        method=estimate["interval"]["method"],
                        unit="household",
                        n=comparison["n"],
                    )
                )
    return reported
