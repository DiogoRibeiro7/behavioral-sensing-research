"""Tests for the observation-model mismatch diagnostic.

The diagnostic asks how surprising a window's evidence is under every state.
Its count laws must be exact and normalised, its tails valid, its standardised
surprise calibrated under the model, and it must tell typical evidence from
extreme counts and contradictory channels, while treating missing evidence and
known sensor failures as no evidence rather than as novelty.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.channel_models import (
    HURDLE,
    HURDLE_NB,
    HurdleChannel,
    HurdleNBChannel,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    poisson_channel,
    total_loglik,
)
from sensor_modeling.datasets.information_sets import (
    EvidenceChannel,
    EvidenceResolution,
)
from sensor_modeling.datasets.observation_mismatch import (
    CountLaw,
    PatternReference,
    count_law,
    failure_flags,
    household_mismatch,
    mismatch_trace,
    model_description,
    pattern_reference,
    training_support,
)
from sensor_modeling.datasets.structural_models import fit_samples
from sensor_modeling.evaluation import (
    ArtifactError,
    ExperimentRecord,
    MismatchReport,
    MismatchTrace,
    load_record,
)
from sensor_modeling.evaluation.mismatch import (
    AVAILABLE,
    FAILED,
    LEVELS,
    MISSING,
    validate_mismatch,
)
from sensor_modeling.fusion import ONLINE, NotEnumerated
from sensor_modeling.observations import Modality
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology


def law_of(silence: Any, rate: Any, dispersion: Any) -> CountLaw:
    """A count law from plain sequences."""
    return CountLaw(np.array(silence), np.array(rate), np.array(dispersion))


STATES = ("away", "home", "sleep")
HOME = STATES.index("home")

LAWS = {
    "bedroom": law_of([0.97, 0.6, 0.5], [0.5, 2.0, 1.5], [0.0, 0.0, 0.3]),
    "door": law_of([0.9, 0.95, 0.999], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
    "kitchen": law_of([0.95, 0.4, 0.97], [0.5, 4.0, 0.5], [0.0, 0.5, 0.0]),
}
CHANNEL = EvidenceChannel("kitchen", Modality.MOTION)


def sample(law: CountLaw, state: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """*n* counts from *law* under *state*: silence, or a zero-truncated draw."""
    pi, mu, alpha = law.silence[state], law.rate[state], law.dispersion[state]
    out = np.zeros(n)
    active = rng.random(n) >= pi
    need = int(active.sum())
    draws: list[float] = []
    while len(draws) < need:
        size = 4 * need + 10
        batch = (
            rng.poisson(mu, size)
            if alpha == 0.0
            else rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mu), size)
        )
        draws.extend(batch[batch > 0].tolist())
    out[active] = draws[:need]
    return out


def trace(
    counts: dict[str, Any], laws: dict[str, CountLaw] | None = None, **kwargs: Any
) -> MismatchTrace:
    """A trace of *counts* against *laws*."""
    laws = LAWS if laws is None else laws
    windows = len(next(iter(counts.values())))
    return mismatch_trace(
        laws,
        {k: np.asarray(v, dtype=float) for k, v in counts.items()},
        household="h",
        timestamps=[
            f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}" for i in range(windows)
        ],
        states=STATES,
        **kwargs,
    )


def typical(n: int, seed: int = 1, state: int = HOME) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: sample(law, state, n, rng) for name, law in LAWS.items()}


# ----------------------------------------------------------------------------
class TestCountLaw:
    """Exact, normalised, log-space count laws."""

    @pytest.mark.parametrize("name", sorted(LAWS))
    def test_normalised_with_exact_tails(self, name: str) -> None:
        law = LAWS[name]
        grid = np.arange(0.0, 4000.0)
        log_p = law.log_pmf(grid)
        np.testing.assert_allclose(logsumexp(log_p, axis=0), 0.0, atol=1e-10)
        lower = np.logaddexp.accumulate(log_p, axis=0)
        upper = np.logaddexp.accumulate(log_p[::-1], axis=0)[::-1]
        np.testing.assert_allclose(law.log_lower(grid[:400]), lower[:400], atol=1e-10)
        np.testing.assert_allclose(law.log_upper(grid[:400]), upper[:400], atol=1e-10)
        # Every count is in exactly one of the tails, and the count in both.
        both = np.exp(law.log_upper(grid[:400])) + np.exp(law.log_lower(grid[:400]))
        np.testing.assert_allclose(both - np.exp(log_p[:400]), 1.0, atol=1e-12)

    def test_moments_are_the_entropy_and_varentropy(self) -> None:
        for law in LAWS.values():
            log_p = law.log_pmf(np.arange(0.0, 4000.0))
            p = np.exp(log_p)
            entropy = -(p * log_p).sum(axis=0)
            varentropy = (p * (-log_p - entropy) ** 2).sum(axis=0)
            found = law.moments()
            np.testing.assert_allclose(found[0], entropy, atol=1e-12)
            np.testing.assert_allclose(found[1], varentropy, atol=1e-10)

    def test_moments_at_extreme_fits(self) -> None:
        # Near-zero rates, and dispersions at the fit's bound: a light tail and
        # a heavy one, where library quantile functions fail.
        law = law_of([0.5, 0.5, 0.9], [1e-9, 0.02, 3.0], [0.0, 100.0, 100.0])
        log_p = law.log_pmf(np.arange(0.0, 400_000.0))
        p = np.exp(log_p)
        entropy = -(p * log_p).sum(axis=0)
        np.testing.assert_allclose(logsumexp(log_p, axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(law.moments()[0], entropy, atol=1e-9)
        assert law._moment_top() < 100_000

    def test_the_fitted_models_laws(self) -> None:
        counts = np.arange(0.0, 60.0)
        silence, rate = np.array([0.9, 0.3, 0.05]), np.array([0.5, 3.0, 20.0])
        hurdle = HurdleChannel(CHANNEL, silence, rate)
        # The model's likelihood drops log n!, which a surprise must keep.
        np.testing.assert_allclose(
            count_law(hurdle).log_pmf(counts),
            hurdle.loglik(counts) - gammaln(counts + 1.0)[:, None],
            atol=1e-10,
        )
        nb = HurdleNBChannel(CHANNEL, silence, rate, np.array([0.5, 2.0, 100.0]))
        np.testing.assert_allclose(
            count_law(nb).log_pmf(counts), nb.log_pmf(counts), atol=1e-12
        )
        mean = np.array([0.2, 5.0, 40.0])
        law = count_law(poisson_channel(CHANNEL, mean))
        np.testing.assert_allclose(
            law.log_pmf(counts), poisson.logpmf(counts[:, None], mean), atol=1e-10
        )
        np.testing.assert_allclose(
            law.log_upper(counts[1:]),
            poisson.logsf(counts[1:, None] - 1.0, mean),
            atol=1e-10,
        )
        with pytest.raises(TypeError, match="no count law"):
            count_law(object())  # type: ignore[arg-type]

    def test_extreme_counts_stay_finite_and_ordered(self) -> None:
        # Enumerated lower tails up to 1e5, and the complement beyond the limit.
        counts = np.array([0.0, 10.0, 100.0, 1e3, 1e4, 1e5, 3e6])
        for law in LAWS.values():
            for values in (law.log_pmf(counts), law.log_upper(counts)):
                assert np.all(np.isfinite(values))
            upper = law.log_upper(counts)
            assert np.all(np.diff(upper, axis=0) <= 1e-12)
            assert np.all(upper >= law.log_pmf(counts) - 1e-12)
            assert np.all(np.isfinite(law.log_lower(counts)))

    def test_underflowing_tails_are_summed_directly(self) -> None:
        law = LAWS["kitchen"]
        counts = np.arange(150.0, 400.0)
        log_p = law.log_pmf(np.arange(150.0, 5000.0))
        upper = np.logaddexp.accumulate(log_p[::-1], axis=0)[::-1][: counts.size]
        assert upper.min() < -1000.0
        np.testing.assert_allclose(law.log_upper(counts), upper, rtol=1e-12)

    @pytest.mark.parametrize(
        ("silence", "rate", "dispersion", "message"),
        [
            ([0.0], [1.0], [0.0], "strictly in"),
            ([0.5], [0.0], [0.0], "positive"),
            ([0.5], [1.0], [-1.0], "non-negative"),
            ([0.5, 0.5], [1.0], [0.0], "one value per state"),
        ],
    )
    def test_invalid_laws(
        self,
        silence: list[float],
        rate: list[float],
        dispersion: list[float],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            law_of(silence, rate, dispersion)

    def test_invalid_counts(self) -> None:
        with pytest.raises(ValueError, match="whole"):
            LAWS["door"].log_pmf([1.5])
        with pytest.raises(ValueError, match="non-negative"):
            LAWS["door"].log_upper([-1.0])


# ----------------------------------------------------------------------------
class TestSyntheticEvidence:
    """Typical, extreme, contradictory, missing and failed evidence."""

    def test_typical_evidence_is_typical(self) -> None:
        n = 4000
        found = trace(typical(n))
        z = found.standardised[:, HOME]
        # Under the state that produced it, the surprise has its own mean and variance.
        assert abs(z.mean()) < 4.0 / math.sqrt(n)
        assert 0.85 < z.var() < 1.15
        # Each channel's two-sided tail is a valid p-value: rarely small.
        tails = found.log_two_sided[:, :, HOME]
        for level in (0.05, 0.01):
            assert np.all((tails < math.log(level)).mean(axis=0) <= level + 0.01)
        summary = found.summary()
        assert summary["evidence_windows"] == n
        assert summary["levels"]["0.0001"]["channel_tail"] <= 0.002
        assert summary["levels"]["1e-06"]["burst"] == 0.0
        assert summary["least_standardised"]["median"] < 0.5

    def test_extreme_event_counts(self) -> None:
        counts = typical(50)
        counts["kitchen"][7] = 500.0
        found = trace(counts, training={"kitchen": np.array([2.0, 25.0, 3.0])})
        assert found.best_upper[7, found.channels.index("kitchen")] < math.log(1e-9)
        assert found.beyond_training is not None
        assert np.flatnonzero(found.beyond_training.any(axis=1)).tolist() == [7]
        assert found.least_standardised[7] > 50.0
        best = int(found.best_state[7])
        assert found.channels[int(np.argmax(found.excess[7, :, best]))] == "kitchen"
        summary = found.summary()
        assert summary["levels"]["1e-09"]["burst"] == pytest.approx(1 / 50)
        assert summary["beyond_training"] == pytest.approx(1 / 50)
        assert summary["channels"]["kitchen"]["beyond_training"] == 1

    def test_contradictory_sensor_evidence(self) -> None:
        # Kitchen activity only at home, bedroom activity only asleep.
        laws = {
            "bedroom": law_of([0.99, 0.99, 0.1], [1.0, 1.0, 6.0], [0.0, 0.0, 0.0]),
            "kitchen": law_of([0.99, 0.1, 0.99], [1.0, 6.0, 1.0], [0.0, 0.0, 0.0]),
        }
        found = trace({"bedroom": [6.0, 0.0], "kitchen": [6.0, 6.0]}, laws=laws)
        # Each channel alone is typical of some state...
        assert np.all(found.best_two_sided[0] > math.log(0.05))
        # ...but no state explains both.
        assert found.least_standardised[0] > 4.0
        assert found.least_standardised[1] < 1.0
        assert math.exp(found.best_pattern_tail[0]) == pytest.approx(0.01, rel=1e-6)
        assert found.best_pattern_tail[1] > math.log(0.5)
        # At the best state, the contradicting channel carries the excess.
        best = int(found.best_state[0])
        other = "bedroom" if STATES[best] == "home" else "kitchen"
        assert found.channels[int(np.argmax(found.excess[0, :, best]))] == other

    def test_all_missing_evidence_is_no_evidence(self) -> None:
        counts = typical(20)
        for name in counts:
            counts[name][[3, 4]] = np.nan
        found = trace(counts, predicted=np.full((20, 3), 1 / 3))
        assert found.evidence.tolist() == [i not in (3, 4) for i in range(20)]
        assert np.all(found.status[[3, 4]] == MISSING)
        for values in (
            found.loglik,
            found.standardised,
            found.pattern_tail,
            found.predictive_loglik,
            found.max_loglik,
        ):
            assert values is not None and np.all(np.isnan(values[[3, 4]]))
        assert found.best_state[[3, 4]].tolist() == [-1, -1]
        records = found.windows()
        assert records[3]["evidence"] is False and "loglik" not in records[3]
        json.dumps(records, allow_nan=False)
        summary = found.summary()
        assert (summary["evidence_windows"], summary["no_evidence_windows"]) == (18, 2)
        assert summary["channel_windows"]["missing"] == 6
        # The empty windows count as neither typical nor surprising.
        same = trace(
            {k: v[~np.isin(np.arange(20), [3, 4])] for k, v in counts.items()},
            predicted=np.full((18, 3), 1 / 3),
        )
        assert summary["levels"] == same.summary()["levels"]

    def test_known_sensor_failure_is_not_novelty(self) -> None:
        counts = typical(40)
        stuck = [5, 6, 7]
        counts["kitchen"][stuck] = 500.0
        dead = [20, 21]
        counts["bedroom"][dead] = 0.0
        failed = {
            "kitchen": np.isin(np.arange(40), stuck),
            "bedroom": np.isin(np.arange(40), dead),
        }
        training = {"kitchen": np.array([2.0, 25.0, 3.0])}
        unflagged = trace(counts, training=training)
        assert unflagged.summary()["levels"]["1e-09"]["burst"] == pytest.approx(3 / 40)
        found = trace(counts, failed=failed, training=training)
        kitchen = found.channels.index("kitchen")
        assert np.all(found.status[stuck, kitchen] == FAILED)
        summary = found.summary()
        assert summary["levels"]["1e-09"]["burst"] == 0.0
        assert summary["beyond_training"] == 0.0
        assert summary["channel_windows"]["failed"] == 5
        assert summary["channels"]["kitchen"]["failed"] == 3
        # A failed channel is left out exactly: the window is scored on the rest.
        rest = trace(
            {k: v[stuck] for k, v in counts.items() if k != "kitchen"},
            laws={k: v for k, v in LAWS.items() if k != "kitchen"},
        )
        np.testing.assert_allclose(found.loglik[stuck], rest.loglik, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            found.standardised[stuck], rest.standardised, rtol=0, atol=1e-12
        )
        # A dead sensor's silence is not in the activity pattern.
        assert not found.available[dead, found.channels.index("bedroom")].any()

    def test_every_channel_failed_is_no_evidence(self) -> None:
        counts = typical(5)
        failed = {name: np.array([True, False, False, False, False]) for name in counts}
        found = trace(counts, failed=failed)
        assert found.evidence.tolist() == [False, True, True, True, True]
        assert found.summary()["no_evidence_windows"] == 1


# ----------------------------------------------------------------------------
class TestMeasures:
    """Contributions, the predictive, patterns and support."""

    def test_channel_excess_sums_to_the_window(self) -> None:
        found = trace(typical(200))
        total = np.where(found.available[:, :, None], found.excess, 0.0).sum(axis=1)
        np.testing.assert_allclose(
            total, -found.loglik - found.expected_surprise, atol=1e-10
        )

    def test_best_measures_are_over_the_states(self) -> None:
        found = trace(typical(100))
        np.testing.assert_array_equal(found.max_loglik, found.loglik.max(axis=1))
        np.testing.assert_array_equal(found.best_state, found.loglik.argmax(axis=1))
        np.testing.assert_array_equal(found.best_upper, found.log_upper.max(axis=2))
        np.testing.assert_array_equal(
            found.least_standardised, found.standardised.min(axis=1)
        )

    def test_the_predictive_under_a_certain_prediction(self) -> None:
        counts = typical(100)
        predicted = np.zeros((100, 3))
        predicted[:, HOME] = 1.0
        found = trace(counts, predicted=predicted)
        assert found.predictive_log_pmf is not None
        np.testing.assert_allclose(found.predictive_log_pmf, found.log_pmf[:, :, HOME])
        assert found.predictive_loglik is not None
        np.testing.assert_allclose(found.predictive_loglik, found.loglik[:, HOME])
        gap = found.predictive_gap
        assert gap is not None and np.all(gap <= 0.0)
        at_best = found.best_state == HOME
        np.testing.assert_allclose(gap[at_best], 0.0, atol=1e-12)

    def test_the_predictive_mixes_the_states(self) -> None:
        counts = typical(30)
        predicted = np.tile([0.2, 0.5, 0.3], (30, 1))
        found = trace(counts, predicted=predicted)
        expected = logsumexp(np.log(predicted)[:, None, :] + found.log_upper, axis=2)
        upper = np.minimum(0.0, expected)
        tail = np.minimum(
            0.0,
            math.log(2.0)
            + np.minimum(
                upper,
                logsumexp(np.log(predicted)[:, None, :] + found.log_lower, axis=2),
            ),
        )
        assert found.predictive_two_sided is not None
        np.testing.assert_allclose(found.predictive_two_sided, tail)

    def test_pattern_tail_by_enumeration(self) -> None:
        found = trace(
            {"bedroom": [0, 3, 0, 2], "door": [0, 0, 1, 1], "kitchen": [4, 0, 0, 5]}
        )
        silence = np.stack([LAWS[c].silence for c in found.channels])
        for t in range(4):
            active = found.active[t]
            own = np.where(active[:, None], np.log1p(-silence), np.log(silence)).sum(0)
            np.testing.assert_allclose(found.pattern_loglik[t], own)
            every = []
            for bits in range(8):
                on = np.array([(bits >> j) & 1 for j in range(3)], dtype=bool)
                every.append(
                    np.where(on[:, None], np.log1p(-silence), np.log(silence)).sum(0)
                )
            table = np.array(every)
            for s in range(3):
                mass = logsumexp(table[table[:, s] <= own[s] + 1e-9, s])
                assert found.pattern_tail[t, s] == pytest.approx(mass, abs=1e-12)

    def test_training_support_and_unseen_patterns(self) -> None:
        reference = PatternReference(
            channels=("bedroom", "door", "kitchen", "hall"),
            states=STATES,
            households=("a", "b"),
            # bedroom=1, door=2, kitchen=4, hall=8
            instrumented=np.array([7, 7, 7, 15, 3]),
            active=np.array([4, 4, 1, 12, 1]),
            labels=np.array([1, 1, 2, 0, 2]),
        )
        found = trace(
            {"bedroom": [0, 1, 1], "door": [0, 0, 1], "kitchen": [3, 0, 2]},
            patterns=reference,
        )
        assert found.pattern_support is not None
        # Kitchen alone: two home windows show it, and the hall window, whose
        # pattern on these three channels is also kitchen alone.
        assert found.pattern_support[0].tolist() == [1, 2, 0]
        assert found.pattern_comparable is not None
        assert found.pattern_comparable.tolist() == [4, 4, 4]
        assert found.pattern_support[1].tolist() == [0, 0, 1]
        assert found.unseen_pattern is not None
        assert found.unseen_pattern.tolist() == [False, False, True]
        # An unknown channel leaves nothing comparable: unknown, not unseen.
        unknown = mismatch_trace(
            {**LAWS, "attic": LAWS["door"]},
            {
                "bedroom": np.zeros(1),
                "door": np.zeros(1),
                "kitchen": np.ones(1),
                "attic": np.ones(1),
            },
            household="h",
            timestamps=["t"],
            states=STATES,
            patterns=reference,
        )
        assert unknown.pattern_comparable is not None
        assert unknown.pattern_comparable.tolist() == [0]
        assert unknown.unseen_pattern is not None and not unknown.unseen_pattern.any()
        assert unknown.summary()["unseen_pattern"] is None

    def test_training_maxima_that_are_unknown_never_exceed(self) -> None:
        counts = typical(20)
        counts["door"][0] = 99.0
        found = trace(counts, training={"door": np.full(3, np.nan)})
        assert found.beyond_training is not None and not found.beyond_training.any()


# ----------------------------------------------------------------------------
class TestTrace:
    """Layout, serialisation and determinism."""

    def test_round_trip_and_digest(self, tmp_path: Path) -> None:
        counts = typical(30)
        counts["door"][2] = np.nan
        found = trace(
            counts,
            failed={"kitchen": np.arange(30) == 5},
            predicted=np.tile([0.2, 0.5, 0.3], (30, 1)),
            training={"kitchen": np.array([2.0, 25.0, np.nan])},
            truth=np.full(30, HOME),
        )
        first = found.write(tmp_path / "a.json.gz")
        assert (
            trace(
                counts,
                failed={"kitchen": np.arange(30) == 5},
                predicted=np.tile([0.2, 0.5, 0.3], (30, 1)),
                training={"kitchen": np.array([2.0, 25.0, np.nan])},
                truth=np.full(30, HOME),
            ).write(tmp_path / "b.json.gz")
            == first
        )
        back = MismatchTrace.read(tmp_path / "a.json.gz", first)
        assert json.dumps(back.summary(), sort_keys=True) == json.dumps(
            found.summary(), sort_keys=True
        )
        np.testing.assert_array_equal(back.log_upper, found.log_upper)
        with pytest.raises(ValueError, match="digest"):
            MismatchTrace.read(tmp_path / "a.json.gz", "0" * 64)
        json.dumps(found.windows(), allow_nan=False)
        json.dumps(found.summary(), allow_nan=False)

    def test_invalid_layouts_are_refused(self) -> None:
        found = trace(typical(3))
        fields = dict(found.__dict__)
        for key in list(fields):
            if key not in MismatchTrace.__dataclass_fields__:
                del fields[key]
        status = found.status.copy()
        status[0, 0] = 7
        with pytest.raises(ValueError, match="status"):
            MismatchTrace(**{**fields, "status": status})
        counts = found.counts.copy()
        counts[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN exactly"):
            MismatchTrace(**{**fields, "counts": counts})
        pmf = found.log_pmf.copy()
        pmf[0, 0, 0] = 0.5
        with pytest.raises(ValueError, match="log-probabilities"):
            MismatchTrace(**{**fields, "log_pmf": pmf})
        with pytest.raises(ValueError, match="distribution"):
            MismatchTrace(**{**fields, "predicted": np.ones((3, 3))})
        with pytest.raises(ValueError, match="cover exactly"):
            trace({"door": [1.0]})

    def test_the_status_codes(self) -> None:
        assert (AVAILABLE, MISSING, FAILED) == (0, 1, 2)


# ----------------------------------------------------------------------------
def report() -> MismatchReport:
    return MismatchReport.from_traces(
        [trace(typical(20))], {"family": "hurdle_nb", "fitted_on": ["t1", "t2"]}
    )


class TestRecord:
    """The report in an experiment record, at schema 1.5."""

    def record(self, **changes: Any) -> ExperimentRecord:
        settings: dict[str, Any] = {
            "experiment": "mismatch",
            "configuration": {},
            "inference": ONLINE,
            "evidence": NotEnumerated("a unit-test record"),
            "data_source": "synthetic-test",
            "observation_mismatch": report(),
        }
        settings.update(changes)
        return ExperimentRecord(**settings)

    def test_the_section_round_trips(self, tmp_path: Path) -> None:
        path = self.record().write(tmp_path / "record.json")
        payload = load_record(path)
        assert payload["schema_version"] == "1.5"
        section = payload["observation_mismatch"]
        assert section["levels"] == list(LEVELS)
        assert section["households"]["h"]["summary"]["windows"] == 20
        assert ExperimentRecord.from_dict(payload).observation_mismatch == report()
        assert (
            load_record(
                self.record(observation_mismatch=None).write(tmp_path / "n.json")
            )["observation_mismatch"]
            is None
        )

    def test_a_1_4_record_migrates_without_the_section(self, tmp_path: Path) -> None:
        payload = self.record(observation_mismatch=None).to_dict()
        del payload["observation_mismatch"]
        payload["schema_version"] = "1.4"
        path = tmp_path / "old.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        migrated = load_record(path)
        assert migrated["migrated_from"] == "1.4"
        assert migrated["observation_mismatch"] is None

    def test_invalid_sections_are_refused(self, tmp_path: Path) -> None:
        good = report().to_dict()
        assert validate_mismatch(good) == []
        scored_on_training = {**good, "model": {"family": "x", "fitted_on": ["h"]}}
        assert any("used to fit" in p for p in validate_mismatch(scored_on_training))
        bad_share = json.loads(json.dumps(good))
        bad_share["households"]["h"]["summary"]["levels"]["0.01"]["burst"] = 1.5
        assert any("share" in p for p in validate_mismatch(bad_share))
        bad_trace = json.loads(json.dumps(good))
        bad_trace["households"]["h"]["trace"] = {"file": "x", "sha256": "short"}
        assert any("SHA-256" in p for p in validate_mismatch(bad_trace))
        assert validate_mismatch({"levels": []})
        with pytest.raises(ValueError, match="fitted_on"):
            MismatchReport.from_traces([trace(typical(3))], {"family": "x"})
        payload = self.record().to_dict()
        payload["observation_mismatch"] = bad_share
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="share"):
            load_record(path)
        with pytest.raises(TypeError, match="MismatchReport"):
            self.record(observation_mismatch={"not": "a report"})


# ----------------------------------------------------------------------------
# On simulated households
# ----------------------------------------------------------------------------
RESOLUTION = EvidenceResolution()
ONTOLOGY = StateOntology()


def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=2, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 5)}


@pytest.fixture(scope="module")
def population(homes: dict[str, CasasRecording]) -> Any:
    training = ["sim1", "sim2", "sim3"]
    statistics = {
        h: home_statistics(homes[h], RESOLUTION, ONTOLOGY, household=h)
        for h in training
    }
    return fit_samples(
        statistics, training, states=tuple(ONTOLOGY.states), pseudo_windows=12.0
    )["all"]


class TestOnHouseholds:
    """The household helper over simulated recordings."""

    def test_the_prediction_is_the_recursion(
        self, homes: dict[str, CasasRecording], population: Any
    ) -> None:
        recording = homes["sim4"]
        models = population.models(recording.registry, RESOLUTION, HURDLE_NB, ONTOLOGY)
        found = household_mismatch(
            recording,
            models,
            household="sim4",
            resolution=RESOLUTION,
            ontology=ONTOLOGY,
            fitted=population,
        )
        counts, rows, labels, _ = household_channel_counts(
            recording, RESOLUTION, ONTOLOGY, household="sim4"
        )
        predicted, _ = filter_recursion(
            total_loglik(models, counts),
            ONTOLOGY.transition(RESOLUTION.step),
            ONTOLOGY.stationary(),
        )
        assert found.predicted is not None
        np.testing.assert_allclose(found.predicted, predicted[rows], atol=1e-12)
        np.testing.assert_array_equal(found.truth, labels)
        assert found.channels == tuple(sorted(c.name for c in models))
        assert found.evidence.all()
        support = training_support(population, models)
        assert found.training_max is not None
        np.testing.assert_array_equal(
            found.training_max, np.stack([support[c] for c in found.channels])
        )
        summary = found.summary()
        json.dumps(summary, allow_nan=False)
        assert summary["windows"] == len(rows)

    def test_training_households_are_refused(
        self, homes: dict[str, CasasRecording], population: Any
    ) -> None:
        recording = homes["sim1"]
        models = population.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        with pytest.raises(ValueError, match="fit the population"):
            household_mismatch(
                recording,
                models,
                household="sim1",
                resolution=RESOLUTION,
                fitted=population,
            )

    def test_failures_and_patterns(
        self, homes: dict[str, CasasRecording], population: Any
    ) -> None:
        recording = homes["sim4"]
        models = population.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        counts, rows, _, moments = household_channel_counts(
            recording, RESOLUTION, ONTOLOGY, household="sim4"
        )
        channel = sorted(models)[0]
        start = moments[int(rows[10])] - timedelta(minutes=2)
        end = start + timedelta(minutes=30)
        reference = pattern_reference(
            homes, ["sim1", "sim2", "sim3"], resolution=RESOLUTION, ontology=ONTOLOGY
        )
        found = household_mismatch(
            recording,
            models,
            household="sim4",
            resolution=RESOLUTION,
            ontology=ONTOLOGY,
            fitted=population,
            patterns=reference,
            failures={channel: [(start, end)]},
        )
        flags = failure_flags(moments, RESOLUTION.step, [(start, end)])[rows]
        column = found.channels.index(channel.name)
        assert flags.sum() >= 6
        np.testing.assert_array_equal(found.status[:, column] == FAILED, flags)
        assert found.pattern_comparable is not None
        assert np.all(found.pattern_comparable == reference.labels.size)
        assert found.summary()["unseen_pattern"] is not None
        with pytest.raises(ValueError, match="pattern reference"):
            household_mismatch(
                homes["sim2"],
                population.models(homes["sim2"].registry, RESOLUTION, HURDLE, ONTOLOGY),
                household="sim2",
                resolution=RESOLUTION,
                patterns=reference,
            )
        with pytest.raises(ValueError, match="end after"):
            failure_flags(moments, RESOLUTION.step, [(end, start)])

    def test_a_report_of_households(
        self, homes: dict[str, CasasRecording], population: Any, tmp_path: Path
    ) -> None:
        recording = homes["sim4"]
        models = population.models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        found = household_mismatch(
            recording,
            models,
            household="sim4",
            resolution=RESOLUTION,
            ontology=ONTOLOGY,
            fitted=population,
        )
        digest = found.write(tmp_path / "sim4.json.gz")
        section = MismatchReport.from_traces(
            [found],
            model_description(HURDLE, population),
            {"sim4": {"file": "sim4.json.gz", "sha256": digest}},
        )
        assert section.model == {
            "family": HURDLE,
            "fitted_on": ["sim1", "sim2", "sim3"],
        }
        record = ExperimentRecord(
            experiment="mismatch",
            configuration={},
            inference=ONLINE,
            evidence=NotEnumerated("a unit-test record"),
            data_source="synthetic-test",
            observation_mismatch=section,
        )
        payload = load_record(record.write(tmp_path / "record.json"))
        assert payload["observation_mismatch"]["households"]["sim4"]["trace"] == {
            "file": "sim4.json.gz",
            "sha256": digest,
        }
