"""Household results that keep the inference regime they were produced under.

A metric per household does not say when the estimates behind it could have
been reported. An online filter's estimates exist at their own moments. A
fixed-lag smoother's exist only ``lag`` later, and they read evidence a live
system does not yet have. Compared as plain numbers, a smoothing gain reads
like an online one. This module keeps the regime and the evidence timestamps
with the values, and refuses to drop them:

- :class:`RegimeResult` holds one value per household, the regime and the
  evidence summary of the scored estimates. A summary whose estimates read
  past the regime's horizon is refused, so smoothed estimates cannot be
  labelled online.
- :func:`compare_results` compares two labelled results. Plain mappings are
  refused. Results of different regimes are refused unless the caller asks for
  a smoothing gain, and the comparison is then labelled as one.
- :func:`pool_results` combines results of one regime only, each household once.
- :meth:`RegimeComparison.online_gain` returns the comparison only when both
  sides are online filters.

:func:`~sensor_modeling.evaluation.compare_households` still takes plain
mappings. The experiments that call it record their regime at the level of the
whole record (:class:`~sensor_modeling.evaluation.ExperimentRecord`), which
refuses anything but one regime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..fusion.regime import (
    EvidenceLeakageError,
    EvidenceSummary,
    InferenceRegime,
    NotEnumerated,
    check_evidence,
    require_same_regime,
)
from .households import (
    HouseholdComparison,
    HouseholdSummary,
    compare_households,
    summarise_households,
)

#: What a comparison is: within one regime, or a labelled smoothing gain.
ONLINE_COMPARISON = "online"
SMOOTHED_COMPARISON = "smoothed"
SMOOTHING_GAIN = "smoothing gain"


@dataclass(frozen=True)
class RegimeResult:
    """One metric value per household, with the regime and evidence behind it.

    Attributes
    ----------
    regime
        The regime whose estimates were scored.
    values
        One value per household, ``None`` where it is missing. Stored read-only.
    evidence
        The prediction and evidence timestamps of the scored estimates, or,
        for a causal regime only, why they were not listed.
    """

    regime: InferenceRegime
    values: Mapping[str, float | None]
    evidence: EvidenceSummary | NotEnumerated

    def __post_init__(self) -> None:
        """Refuse a result without a regime, or with evidence beyond it."""
        if not isinstance(self.regime, InferenceRegime):
            raise TypeError("a result needs the InferenceRegime that produced it")
        if not isinstance(self.evidence, EvidenceSummary | NotEnumerated):
            raise TypeError(
                "a result needs an EvidenceSummary, or a NotEnumerated reason"
            )
        check_evidence(self.evidence, self.regime)
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def summary(self) -> HouseholdSummary:
        """The values across households, each counted once."""
        return summarise_households(self.values)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, with the regime and its evidence."""
        return {
            "inference": {
                **self.regime.to_dict(),
                "evidence": self.evidence.to_dict(),
            },
            "values": dict(sorted(self.values.items())),
        }


@dataclass(frozen=True)
class RegimeComparison:
    """A household comparison, labelled with both sides' regimes.

    Attributes
    ----------
    model, reference
        The regimes of the two sides.
    comparison
        The paired household comparison. A positive difference favours the
        model.
    """

    model: InferenceRegime
    reference: InferenceRegime
    comparison: HouseholdComparison

    @property
    def kind(self) -> str:
        """``"online"``, ``"smoothed"`` or ``"smoothing gain"``."""
        if self.model != self.reference:
            return SMOOTHING_GAIN
        return ONLINE_COMPARISON if self.model.is_online else SMOOTHED_COMPARISON

    @property
    def label(self) -> str:
        """What was compared, which no report may leave out."""
        if self.kind == SMOOTHING_GAIN:
            return (
                f"smoothing gain: the {self.model.label} against the "
                f"{self.reference.label}; the smoothed estimates are reported "
                f"{self.model.delay.total_seconds() / 60.0:g} min after their "
                "moments"
            )
        return f"both sides {self.model.label}"

    def online_gain(self, what: str) -> HouseholdComparison:
        """The comparison, if *what* may be reported as an online gain.

        Raises
        ------
        EvidenceLeakageError
            Unless both sides are online filters.
        """
        if self.kind != ONLINE_COMPARISON:
            raise EvidenceLeakageError(
                f"{what} is an online gain only when both sides are online "
                f"filters, but this is {self.label}"
            )
        return self.comparison

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable form, with both regimes."""
        return {
            "kind": self.kind,
            "label": self.label,
            "model_inference": self.model.to_dict(),
            "reference_inference": self.reference.to_dict(),
            "comparison": self.comparison.to_dict(),
        }


def compare_results(
    model: RegimeResult,
    reference: RegimeResult,
    *,
    smoothing_gain: bool = False,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    interval: str = "percentile",
) -> RegimeComparison:
    """Compare two labelled results across households.

    Parameters
    ----------
    model, reference
        The two sides. Both must be :class:`RegimeResult`; plain mappings carry
        no regime and are refused.
    smoothing_gain
        Must be true exactly when the regimes differ. A smoothing gain compares
        a fixed-lag smoother (the model) against the online filter (the
        reference), and is labelled as such.
    higher_is_better, confidence, resamples, seed, interval
        As for :func:`~sensor_modeling.evaluation.compare_households`.

    Raises
    ------
    TypeError
        If either side is not labelled with its regime.
    EvidenceLeakageError
        If the regimes differ without ``smoothing_gain``, if ``smoothing_gain``
        is given for one regime, or if it is not a smoother against the online
        filter.
    """
    for side, result in (("model", model), ("reference", reference)):
        if not isinstance(result, RegimeResult):
            raise TypeError(
                f"the {side} is not labelled with its inference regime; wrap its "
                "values in a RegimeResult"
            )
    if model.regime == reference.regime:
        if smoothing_gain:
            raise EvidenceLeakageError(
                f"both sides are the {model.regime.label}, so their difference is "
                "not a smoothing gain"
            )
    elif not smoothing_gain:
        raise EvidenceLeakageError(
            f"the model is the {model.regime.label} and the reference the "
            f"{reference.regime.label}; compare results of one regime, or pass "
            "smoothing_gain=True to report the difference as a smoothing gain"
        )
    elif model.regime.is_online or not reference.regime.is_online:
        raise EvidenceLeakageError(
            "a smoothing gain compares a fixed-lag smoother, the model, against "
            f"the online filter, the reference, not the {model.regime.label} "
            f"against the {reference.regime.label}"
        )
    comparison = compare_households(
        model.values,
        reference.values,
        higher_is_better=higher_is_better,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        interval=interval,
    )
    return RegimeComparison(model.regime, reference.regime, comparison)


def pool_results(
    results: Iterable[RegimeResult], what: str = "a pooled result"
) -> RegimeResult:
    """Combine results of one regime into one, each household once.

    Raises
    ------
    TypeError
        If a result is not labelled with its regime.
    EvidenceLeakageError
        If the results come from different regimes.
    ValueError
        If there are no results, or a household appears in more than one.
    """
    parts = list(results)
    for part in parts:
        if not isinstance(part, RegimeResult):
            raise TypeError(
                f"{what} would include a result that is not labelled with its "
                "inference regime"
            )
    regime = require_same_regime((part.regime for part in parts), what)
    values: dict[str, float | None] = {}
    for part in parts:
        repeated = sorted(set(values) & set(part.values))
        if repeated:
            raise ValueError(f"{what} would count households {repeated} more than once")
        values.update(part.values)
    summaries = [p.evidence for p in parts if isinstance(p.evidence, EvidenceSummary)]
    reasons = [
        p.evidence.reason for p in parts if isinstance(p.evidence, NotEnumerated)
    ]
    evidence: EvidenceSummary | NotEnumerated = (
        EvidenceSummary.combine(summaries)
        if not reasons
        else NotEnumerated("; ".join(dict.fromkeys(reasons)))
    )
    return RegimeResult(regime, values, evidence)
