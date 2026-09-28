"""Strict leakage tests for the inference-regime contract (ROADMAP 3.5).

A synthetic sequence is streamed one window at a time. Its later evidence
genuinely changes the posterior of an earlier window, and the tests show:

- an online estimate, once reported, never changes as later evidence arrives,
  and cannot be edited in place;
- a fixed-lag smoother's estimate for a window changes only while the evidence
  it may read is still arriving, and is fixed from its horizon on;
- lag zero reports exactly the online estimates, at the same moments.

The same is checked on the online pipeline. The remaining tests pin what each
reported estimate records (its prediction timestamp, the latest evidence it
read and when it could be reported) and that the evaluation refuses to
compare, pool or relabel results across regimes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sensor_modeling.datasets.channel_models import filter_recursion
from sensor_modeling.evaluation import (
    RegimeResult,
    compare_households,
    compare_results,
    pool_results,
)
from sensor_modeling.fusion import (
    ONLINE,
    EvidenceLeakageError,
    EvidenceSummary,
    InferenceRegime,
    NotEnumerated,
    ReportedEstimate,
    StateEstimate,
    evidence_from_dict,
    regime_beliefs,
    regime_estimates,
    smooth_estimates,
)
from sensor_modeling.online import BehaviouralSensingPipeline, PipelineConfig
from sensor_modeling.simulation import HouseholdConfig, SimulationResult, simulate

STEP = timedelta(minutes=5)
#: A sticky two-state chain, A and B.
TRANSITION = np.array([[0.95, 0.05], [0.05, 0.95]])
PRIOR = np.array([0.5, 0.5])
#: The window whose estimate later evidence corrects.
TURN = 5
START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def evidence() -> np.ndarray:
    """``(windows, 2)`` log-likelihoods: B, then an ambiguous turn, then A.

    Window 5 itself says nothing and window 6 says A decisively, so the online
    estimate of window 5 is B and any smoother that reads window 6 corrects it.
    """
    rows = [np.log([0.3, 0.7])] * TURN
    rows += [np.zeros(2), np.log([0.99, 0.01])]
    rows += [np.log([0.7, 0.3])] * 5
    return np.array(rows)


WINDOWS = len(evidence())
MOMENTS = [START + STEP * k for k in range(WINDOWS)]


def filtered(windows: int = WINDOWS) -> np.ndarray:
    """The online filter's beliefs after the first *windows* have arrived."""
    return filter_recursion(evidence()[:windows], TRANSITION, PRIOR)[1]


def as_streamed(regime: InferenceRegime) -> dict[int, list[np.ndarray]]:
    """Each window's estimate as the stream grows.

    ``as_streamed(regime)[t][k]`` is the estimate for window ``t`` computed
    when window ``t + k`` has just arrived, from the evidence so far only.
    """
    versions: dict[int, list[np.ndarray]] = {}
    for arrived in range(WINDOWS):
        beliefs = regime_beliefs(filtered(arrived + 1), TRANSITION, regime)
        for row in range(arrived + 1):
            versions.setdefault(row, []).append(beliefs[row])
    return versions


def published(regime: InferenceRegime) -> dict[int, ReportedEstimate]:
    """What a live system publishes: each estimate once, when it is available.

    At each arrival the window whose horizon has just arrived is published.
    When the stream ends, the windows still waiting are published with the
    evidence there is, as for a recording that ends.
    """
    out: dict[int, ReportedEstimate] = {}
    for arrived in range(WINDOWS):
        row = arrived - regime.lag_steps
        if row < 0:
            continue
        beliefs = regime_beliefs(filtered(arrived + 1), TRANSITION, regime)
        out[row] = ReportedEstimate(
            prediction_at=MOMENTS[row],
            evidence_until=MOMENTS[arrived],
            available_at=MOMENTS[row] if regime.is_online else MOMENTS[arrived],
            belief=beliefs[row],
            regime=regime,
        )
    final = regime_beliefs(filtered(), TRANSITION, regime)
    for row in range(WINDOWS):
        if row not in out:
            out[row] = ReportedEstimate(
                MOMENTS[row], MOMENTS[-1], MOMENTS[-1], final[row], regime
            )
    return out


# ----------------------------------------------------------------------------
# Streaming the synthetic sequence
# ----------------------------------------------------------------------------
class TestStreaming:
    def test_later_evidence_genuinely_changes_the_turn(self) -> None:
        online_turn = as_streamed(ONLINE)[TURN][0]
        smoothed_turn = as_streamed(InferenceRegime.smoother(1, STEP))[TURN][1]
        assert online_turn.argmax() == 1  # B, reading up to the turn
        assert smoothed_turn.argmax() == 0  # A, the truth, after window 6

    def test_an_online_estimate_never_changes_after_it_is_reported(self) -> None:
        for row, versions in as_streamed(ONLINE).items():
            for later in versions[1:]:
                np.testing.assert_array_equal(later, versions[0])
        # The decisive window arrives after the turn is reported, and the
        # reported estimate for the turn still says B.
        assert published(ONLINE)[TURN].belief.argmax() == 1

    @pytest.mark.parametrize("lag", [1, 2, 3])
    def test_a_smoothed_estimate_changes_only_while_its_evidence_arrives(
        self, lag: int
    ) -> None:
        streamed = as_streamed(InferenceRegime.smoother(lag, STEP))
        for row, versions in streamed.items():
            horizon = min(lag, len(versions) - 1)
            for version in versions[horizon:]:
                np.testing.assert_array_equal(version, versions[horizon])
        turn = streamed[TURN]
        # Before the decisive window arrives, nothing later has been read.
        np.testing.assert_array_equal(turn[0], as_streamed(ONLINE)[TURN][0])
        # It arrives one window later, within every positive lag, and the
        # estimate moves to A; from the horizon on it is fixed.
        assert turn[0].argmax() == 1 and turn[1].argmax() == 0
        assert not np.array_equal(turn[1], turn[0])

    def test_the_improvement_waits_for_the_permitted_window(self) -> None:
        # From the window before the turn, the decisive window is two ahead.
        # With lag one it is never read; with lag two it is read on arrival.
        before = TURN - 1
        one = as_streamed(InferenceRegime.smoother(1, STEP))[before]
        two = as_streamed(InferenceRegime.smoother(2, STEP))[before]
        online_before = as_streamed(ONLINE)[before][0]
        for version in one:
            np.testing.assert_allclose(version, online_before, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(two[1], online_before, rtol=0.0, atol=1e-12)
        assert two[2][0] > online_before[0] + 0.05

    @pytest.mark.parametrize(
        "regime",
        [ONLINE, *(InferenceRegime.smoother(lag, STEP) for lag in range(4))],
        ids=lambda regime: regime.label,
    )
    def test_what_is_published_is_what_the_evaluation_scores(
        self, regime: InferenceRegime
    ) -> None:
        # The offline estimates of a completed recording are exactly what a
        # live system would have published, and when it would have.
        live = published(regime)
        offline = regime_estimates(MOMENTS, filtered(), TRANSITION, regime)
        for row, estimate in enumerate(offline):
            assert estimate.prediction_at == live[row].prediction_at
            assert estimate.evidence_until == live[row].evidence_until
            assert estimate.available_at == live[row].available_at
            np.testing.assert_array_equal(estimate.belief, live[row].belief)

    def test_zero_lag_smoothing_reproduces_online_filtering(self) -> None:
        zero = regime_estimates(
            MOMENTS, filtered(), TRANSITION, InferenceRegime.smoother(0, STEP)
        )
        online = regime_estimates(MOMENTS, filtered(), TRANSITION, ONLINE)
        for smoothed, live in zip(zero, online, strict=True):
            np.testing.assert_array_equal(smoothed.belief, live.belief)
            assert smoothed.evidence_until == live.evidence_until == live.prediction_at
            assert smoothed.available_at == live.available_at == live.prediction_at
            assert not smoothed.uses_future_evidence
        assert EvidenceSummary.of_estimates(zero) == EvidenceSummary.of_estimates(
            online
        )
        # Equal numbers, but still labelled as what was run.
        assert zero[0].regime.label == "fixed-lag smoother, lag 0 windows (0 min)"
        assert zero[0].regime.is_causal and zero[0].regime.delay == timedelta(0)


# ----------------------------------------------------------------------------
# The online pipeline
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def household() -> SimulationResult:
    return simulate(HouseholdConfig(days=1, seed=3))


def stream_pipeline(
    result: SimulationResult, until: datetime | None = None
) -> list[tuple[StateEstimate, np.ndarray]]:
    """Stream observations in time order, as a live system receives them.

    Returns each estimate as it is released, with a copy of its belief taken
    at that moment. With *until*, the stream stops there.
    """
    config = PipelineConfig(tz=result.config.tz, step=STEP)
    pipeline = BehaviouralSensingPipeline(result.registry, config=config)
    released: list[tuple[StateEstimate, np.ndarray]] = []

    def collect(now: datetime) -> None:
        for step in pipeline.advance(now):
            released.append((step.state, step.state.belief.copy()))

    for observation in sorted(result.observations, key=lambda o: o.timestamp):
        if until is not None and observation.timestamp > until:
            break
        pipeline.push(observation)
        collect(observation.timestamp)
    if until is not None:
        collect(until + config.lateness_tolerance)
    return released


class TestOnlinePipeline:
    def test_a_reported_estimate_is_never_revised(
        self, household: SimulationResult
    ) -> None:
        released = stream_pipeline(household)
        assert len(released) > 100
        # Everything that arrived after an estimate was released left it as it
        # was, and it cannot be edited in place.
        for estimate, at_release in released:
            np.testing.assert_array_equal(estimate.belief, at_release)
        with pytest.raises(ValueError, match="read-only"):
            released[0][0].belief[0] = 1.0

    def test_estimates_do_not_depend_on_later_observations(
        self, household: SimulationResult
    ) -> None:
        full = stream_pipeline(household)
        cutoff = full[len(full) // 2][0].at
        early = stream_pipeline(household, until=cutoff)
        assert [e.at for e, _ in early] == [e.at for e, _ in full if e.at <= cutoff]
        for (truncated, _), (complete, _) in zip(early, full):
            np.testing.assert_array_equal(truncated.belief, complete.belief)

    def test_a_smoother_revises_what_the_filter_reported(
        self, household: SimulationResult
    ) -> None:
        # The later observations carry information about earlier moments, so
        # the online estimates stay fixed only because the filter declines to
        # read them. A smoother returns new estimates and leaves them intact.
        estimates = [estimate for estimate, _ in stream_pipeline(household)]
        before = [estimate.belief.copy() for estimate in estimates]
        smoothed = smooth_estimates(estimates, lag=2, step=STEP)
        assert any(
            not np.allclose(s.belief, e.belief) for s, e in zip(smoothed, estimates)
        )
        for estimate, belief in zip(estimates, before):
            np.testing.assert_array_equal(estimate.belief, belief)


# ----------------------------------------------------------------------------
# What a reported estimate records
# ----------------------------------------------------------------------------
class TestReportedEstimate:
    MOMENT = START
    BELIEF = np.array([0.4, 0.6])

    def test_an_online_estimate_reads_and_is_available_at_its_moment(self) -> None:
        estimate = ReportedEstimate(
            self.MOMENT, self.MOMENT, self.MOMENT, self.BELIEF, ONLINE
        )
        assert estimate.lead == timedelta(0) and not estimate.uses_future_evidence
        with pytest.raises(EvidenceLeakageError, match="may read evidence up to"):
            ReportedEstimate(
                self.MOMENT,
                self.MOMENT + timedelta(seconds=1),
                self.MOMENT,
                self.BELIEF,
                ONLINE,
            )

    def test_a_smoothed_estimate_waits_for_its_last_evidence(self) -> None:
        regime = InferenceRegime.smoother(2, STEP)
        later = self.MOMENT + 2 * STEP
        estimate = ReportedEstimate(self.MOMENT, later, later, self.BELIEF, regime)
        assert estimate.lead == regime.delay == timedelta(minutes=10)
        assert estimate.uses_future_evidence
        with pytest.raises(EvidenceLeakageError, match="is available at"):
            ReportedEstimate(self.MOMENT, later, self.MOMENT, self.BELIEF, regime)
        with pytest.raises(EvidenceLeakageError, match="may read evidence up to"):
            ReportedEstimate(
                self.MOMENT, later + STEP, later + STEP, self.BELIEF, regime
            )

    def test_evidence_cannot_precede_the_moment(self) -> None:
        with pytest.raises(ValueError, match="at least the evidence of its moment"):
            ReportedEstimate(
                self.MOMENT, self.MOMENT - STEP, self.MOMENT, self.BELIEF, ONLINE
            )

    def test_a_reported_belief_is_read_only_and_detached(self) -> None:
        source = self.BELIEF.copy()
        estimate = ReportedEstimate(
            self.MOMENT, self.MOMENT, self.MOMENT, source, ONLINE
        )
        with pytest.raises(ValueError, match="read-only"):
            estimate.belief[0] = 1.0
        source[0] = 1.0
        assert estimate.belief[0] == 0.4

    def test_regime_estimates_record_their_timestamps(self) -> None:
        regime = InferenceRegime.smoother(2, STEP)
        estimates = regime_estimates(MOMENTS, filtered(), TRANSITION, regime)
        for row, estimate in enumerate(estimates):
            horizon = MOMENTS[min(row + 2, WINDOWS - 1)]
            assert estimate.prediction_at == MOMENTS[row]
            assert estimate.evidence_until == estimate.available_at == horizon
        summary = EvidenceSummary.of_estimates(estimates)
        assert summary.max_lead == regime.delay
        assert summary.latest_evidence == MOMENTS[-1]
        summary.check(regime)
        with pytest.raises(EvidenceLeakageError, match="600 s past their moments"):
            summary.check(ONLINE)

    @pytest.mark.parametrize(
        ("moments", "message"),
        [
            (MOMENTS[:-1], "one moment is needed per window"),
            (MOMENTS[:1] * WINDOWS, "strictly increasing"),
            (
                [START + timedelta(minutes=10) * k for k in range(WINDOWS)],
                "spaced by its step",
            ),
        ],
    )
    def test_regime_estimates_refuse_moments_that_do_not_fit(
        self, moments: list[datetime], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            regime_estimates(
                moments, filtered(), TRANSITION, InferenceRegime.smoother(1, STEP)
            )


class TestEvidenceSummary:
    def test_it_round_trips(self) -> None:
        summary = EvidenceSummary.of(
            [(MOMENTS[0], MOMENTS[1]), (MOMENTS[3], MOMENTS[3])]
        )
        assert summary.predictions == 2
        assert summary.max_lead == STEP
        assert evidence_from_dict(summary.to_dict()) == summary
        reason = NotEnumerated("an aggregate study")
        assert evidence_from_dict(reason.to_dict()) == reason

    def test_combining_keeps_the_extremes(self) -> None:
        early = EvidenceSummary.online(MOMENTS[:3])
        late = EvidenceSummary.of([(MOMENTS[5], MOMENTS[7])])
        combined = EvidenceSummary.combine([early, late])
        assert combined.predictions == 4
        assert combined.first_prediction == MOMENTS[0]
        assert combined.latest_evidence == MOMENTS[7]
        assert combined.max_lead == 2 * STEP

    @pytest.mark.parametrize(
        ("pairs", "message"),
        [
            ([], "at least one prediction"),
            ([(MOMENTS[1], MOMENTS[0])], "at least the evidence of its moment"),
        ],
    )
    def test_impossible_summaries_are_refused(
        self, pairs: list[tuple[datetime, datetime]], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            EvidenceSummary.of(pairs)

    def test_an_inconsistent_summary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="beyond the largest lead"):
            EvidenceSummary(1, MOMENTS[0], MOMENTS[0], MOMENTS[2], timedelta(0))

    @pytest.mark.parametrize(
        "payload",
        [
            {"enumerated": False},
            {"enumerated": False, "reason": "x", "extra": 1},
            {"reason": "x"},
            {"enumerated": "yes"},
        ],
    )
    def test_malformed_evidence_is_refused(self, payload: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="enumerated summary or a reason"):
            evidence_from_dict(payload)

    def test_a_blank_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="say why"):
            NotEnumerated("  ")


# ----------------------------------------------------------------------------
# The evaluation
# ----------------------------------------------------------------------------
SMOOTHER = InferenceRegime.smoother(2, STEP)


def result(regime: InferenceRegime, values: dict[str, float | None]) -> RegimeResult:
    estimates = regime_estimates(MOMENTS, filtered(), TRANSITION, regime)
    return RegimeResult(regime, values, EvidenceSummary.of_estimates(estimates))


ONLINE_VALUES = {"h1": 0.50, "h2": 0.60, "h3": 0.55}
SMOOTHED_VALUES = {"h1": 0.58, "h2": 0.66, "h3": 0.61}


class TestRegimeResults:
    def test_smoothed_estimates_cannot_be_labelled_online(self) -> None:
        estimates = regime_estimates(MOMENTS, filtered(), TRANSITION, SMOOTHER)
        with pytest.raises(EvidenceLeakageError, match="the online filter may read"):
            RegimeResult(
                ONLINE, SMOOTHED_VALUES, EvidenceSummary.of_estimates(estimates)
            )
        with pytest.raises(EvidenceLeakageError, match="must be recorded"):
            RegimeResult(SMOOTHER, SMOOTHED_VALUES, NotEnumerated("not listed"))

    def test_a_result_needs_its_regime(self) -> None:
        with pytest.raises(TypeError, match="InferenceRegime"):
            RegimeResult("online", ONLINE_VALUES, NotEnumerated("x"))  # type: ignore[arg-type]

    def test_unlabelled_results_are_not_compared(self) -> None:
        with pytest.raises(TypeError, match="not labelled with its inference regime"):
            compare_results(SMOOTHED_VALUES, result(ONLINE, ONLINE_VALUES))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="compare_results"):
            compare_households(
                result(SMOOTHER, SMOOTHED_VALUES),  # type: ignore[arg-type]
                result(ONLINE, ONLINE_VALUES),  # type: ignore[arg-type]
            )

    def test_online_and_smoothed_results_are_not_compared_silently(self) -> None:
        smoothed, online = result(SMOOTHER, SMOOTHED_VALUES), result(
            ONLINE, ONLINE_VALUES
        )
        with pytest.raises(EvidenceLeakageError, match="smoothing_gain=True"):
            compare_results(smoothed, online, resamples=200)

    def test_a_smoothing_gain_is_labelled_and_never_online(self) -> None:
        gain = compare_results(
            result(SMOOTHER, SMOOTHED_VALUES),
            result(ONLINE, ONLINE_VALUES),
            smoothing_gain=True,
            resamples=200,
        )
        assert gain.kind == "smoothing gain"
        assert gain.label == (
            "smoothing gain: the fixed-lag smoother, lag 2 windows (10 min) against "
            "the online filter; the smoothed estimates are reported 10 min after "
            "their moments"
        )
        assert gain.comparison.mean.value == pytest.approx(0.0666666, abs=1e-6)
        payload = gain.to_dict()
        assert payload["model_inference"]["causal"] is False
        assert payload["model_inference"]["delay_seconds"] == 600.0
        assert payload["reference_inference"]["mode"] == "online_filter"
        with pytest.raises(EvidenceLeakageError, match="online gain only"):
            gain.online_gain("the headline")

    @pytest.mark.parametrize(
        ("model", "reference"),
        [(ONLINE, SMOOTHER), (SMOOTHER, InferenceRegime.smoother(1, STEP))],
    )
    def test_a_smoothing_gain_is_a_smoother_against_the_filter(
        self, model: InferenceRegime, reference: InferenceRegime
    ) -> None:
        with pytest.raises(EvidenceLeakageError, match="a smoothing gain compares"):
            compare_results(
                result(model, SMOOTHED_VALUES),
                result(reference, ONLINE_VALUES),
                smoothing_gain=True,
            )

    def test_one_regime_is_not_a_smoothing_gain(self) -> None:
        with pytest.raises(EvidenceLeakageError, match="not a smoothing gain"):
            compare_results(
                result(ONLINE, SMOOTHED_VALUES),
                result(ONLINE, ONLINE_VALUES),
                smoothing_gain=True,
            )

    def test_an_online_comparison_is_an_online_gain(self) -> None:
        comparison = compare_results(
            result(ONLINE, SMOOTHED_VALUES),
            result(ONLINE, ONLINE_VALUES),
            resamples=200,
        )
        assert comparison.kind == "online"
        assert comparison.label == "both sides online filter"
        assert comparison.online_gain("the headline") is comparison.comparison
        smoothed = compare_results(
            result(SMOOTHER, SMOOTHED_VALUES),
            result(SMOOTHER, ONLINE_VALUES),
            resamples=200,
        )
        assert smoothed.kind == "smoothed"
        with pytest.raises(EvidenceLeakageError, match="online gain only"):
            smoothed.online_gain("the headline")

    def test_zero_lag_gain_is_zero_where_it_is_the_same_estimate(self) -> None:
        zero = InferenceRegime.smoother(0, STEP)
        gain = compare_results(
            result(zero, ONLINE_VALUES),
            result(ONLINE, ONLINE_VALUES),
            smoothing_gain=True,
            resamples=200,
        )
        assert gain.kind == "smoothing gain"
        assert set(gain.comparison.differences) == {0.0}

    def test_pooling_keeps_one_regime_and_each_household_once(self) -> None:
        pooled = pool_results(
            [result(ONLINE, {"h1": 0.5}), result(ONLINE, {"h2": 0.6})]
        )
        assert pooled.regime == ONLINE
        assert dict(pooled.values) == {"h1": 0.5, "h2": 0.6}
        assert pooled.summary().n == 2
        assert pooled.to_dict()["inference"]["evidence"]["predictions"] == 2 * WINDOWS
        with pytest.raises(EvidenceLeakageError, match="different inference regimes"):
            pool_results([result(ONLINE, {"h1": 0.5}), result(SMOOTHER, {"h2": 0.6})])
        with pytest.raises(ValueError, match="more than once"):
            pool_results([result(ONLINE, {"h1": 0.5}), result(ONLINE, {"h1": 0.6})])
        with pytest.raises(TypeError, match="not labelled"):
            pool_results([result(ONLINE, {"h1": 0.5}), {"h2": 0.6}])  # type: ignore[list-item]

    def test_a_pool_with_unlisted_evidence_says_why(self) -> None:
        pooled = pool_results(
            [
                RegimeResult(ONLINE, {"h1": 0.5}, NotEnumerated("aggregate")),
                result(ONLINE, {"h2": 0.6}),
            ]
        )
        assert pooled.evidence == NotEnumerated("aggregate")

    def test_values_are_read_only(self) -> None:
        values = dict(ONLINE_VALUES)
        labelled = result(ONLINE, values)
        values["h1"] = 0.0
        assert labelled.values["h1"] == 0.5
        with pytest.raises(TypeError):
            labelled.values["h1"] = 0.0  # type: ignore[index]
