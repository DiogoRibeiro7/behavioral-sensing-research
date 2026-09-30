"""Tests keeping online filtering and fixed-lag smoothing apart.

They check, on a synthetic sequence where later evidence genuinely corrects an
earlier estimate:
- the online estimate reads nothing after its moment;
- a fixed-lag smoother improves the estimate with its permitted lag, and reads
  nothing beyond it;
- zero-lag smoothing is exactly online filtering.

They also check that the distinction survives into records, reports and the
matched evaluation: the regime is recorded with its lag, a smoother cannot be
presented as online, and results from different regimes are never combined.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    HouseholdSplit,
    ModelSpec,
    nested_information_sets,
)
from sensor_modeling.datasets.channel_models import filter_recursion
from sensor_modeling.datasets.gap_summary import inference_line
from sensor_modeling.datasets.matched_evaluation import (
    FeatureRows,
    LabelledRows,
    StatePredictions,
    held_out_metrics,
    online_evidence,
    run_matched_evaluation,
)
from sensor_modeling.evaluation import ArtifactError, load_record
from sensor_modeling.evaluation.provenance import (
    DECLARED_INFERENCE,
    ExperimentRecord,
    validate_record,
)
from sensor_modeling.fusion import (
    ONLINE,
    EvidenceLeakageError,
    EvidenceSummary,
    InferenceMode,
    InferenceRegime,
    NotEnumerated,
    assert_respects_horizon,
    check_evidence_access,
    regime_beliefs,
    regime_estimates,
    require_online,
    require_same_regime,
)
from sensor_modeling.simulation import HouseholdConfig, simulate

ROOT = Path(__file__).resolve().parents[1]
#: Fields a migration changes; everything else is carried over unchanged.
MIGRATED = {"schema_version", "migrated_from", "inference", "structural_disagreement"}
STEP = timedelta(minutes=5)
#: A sticky two-state chain, A and B.
TRANSITION = np.array([[0.95, 0.05], [0.05, 0.95]])
PRIOR = np.array([0.5, 0.5])
#: The window whose estimate later evidence corrects.
TURN = 5


def evidence() -> np.ndarray:
    """``(windows, 2)`` log-likelihoods: B, then an ambiguous turn, then A.

    The resident has moved to A by window 5, but window 5 itself says nothing.
    Window 6 says A decisively. Online, window 5 still looks like B; with one
    window of lag, the smoother sees window 6 and corrects it.
    """
    rows = [np.log([0.3, 0.7])] * TURN
    rows += [np.zeros(2), np.log([0.99, 0.01])]
    rows += [np.log([0.7, 0.3])] * 5
    return np.array(rows)


def online(loglik: np.ndarray) -> np.ndarray:
    return filter_recursion(loglik, TRANSITION, PRIOR)[1]


def smoothed(lag: int) -> Any:
    regime = InferenceRegime.smoother(lag, STEP)

    def estimate(loglik: np.ndarray) -> np.ndarray:
        return regime_beliefs(online(loglik), TRANSITION, regime)

    return estimate


# ----------------------------------------------------------------------------
# The synthetic sequence
# ----------------------------------------------------------------------------
class TestFutureEvidence:
    def test_later_evidence_genuinely_corrects_the_turn(self) -> None:
        filtered = online(evidence())[TURN]
        revised = smoothed(1)(evidence())[TURN]
        assert filtered.argmax() == 1  # online still says B
        assert revised.argmax() == 0  # the smoother says A, the truth
        assert revised[0] > filtered[0] + 0.3

    def test_the_online_estimate_reads_nothing_after_its_moment(self) -> None:
        assert_respects_horizon(online, evidence(), ONLINE)
        flipped = evidence()
        flipped[TURN + 1] = np.log([0.01, 0.99])
        np.testing.assert_array_equal(online(flipped)[TURN], online(evidence())[TURN])
        assert not np.allclose(
            smoothed(1)(flipped)[TURN], smoothed(1)(evidence())[TURN]
        )

    @pytest.mark.parametrize("lag", [1, 2, 3])
    def test_a_smoother_reads_at_most_its_lag(self, lag: int) -> None:
        assert_respects_horizon(
            smoothed(lag), evidence(), InferenceRegime.smoother(lag, STEP)
        )
        with pytest.raises(EvidenceLeakageError, match="changed when only windows"):
            assert_respects_horizon(
                smoothed(lag), evidence(), InferenceRegime.smoother(lag - 1, STEP)
            )

    def test_the_improvement_waits_for_its_permitted_window(self) -> None:
        # From the window before the turn, one window of lag reads only the
        # uninformative turn, so it reports exactly the online estimate. Two
        # windows of lag reach the decisive window and move toward A.
        before = TURN - 1
        filtered = online(evidence())
        one, two = smoothed(1)(evidence()), smoothed(2)(evidence())
        np.testing.assert_allclose(one[before], filtered[before], atol=1e-12)
        assert two[before][0] > filtered[before][0] + 0.05

    def test_zero_lag_smoothing_is_online_filtering(self) -> None:
        zero = InferenceRegime.smoother(0, STEP)
        filtered = online(evidence())
        np.testing.assert_array_equal(
            regime_beliefs(filtered, TRANSITION, zero), filtered
        )
        np.testing.assert_array_equal(
            regime_beliefs(filtered, TRANSITION, ONLINE), filtered
        )
        assert zero.mode is InferenceMode.FIXED_LAG_SMOOTHER
        assert zero.label == "fixed-lag smoother, lag 0 windows (0 min)"
        assert_respects_horizon(smoothed(0), evidence(), ONLINE)

    def test_a_smoother_presented_as_online_is_caught(self) -> None:
        with pytest.raises(EvidenceLeakageError):
            assert_respects_horizon(smoothed(1), evidence(), ONLINE)
        with pytest.raises(EvidenceLeakageError, match="must be online"):
            require_online(InferenceRegime.smoother(1, STEP), "an alert")


class TestEvidenceAccess:
    MOMENT = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def test_online_reads_nothing_after_its_moment(self) -> None:
        check_evidence_access(ONLINE, self.MOMENT, self.MOMENT)
        with pytest.raises(EvidenceLeakageError):
            check_evidence_access(
                ONLINE, self.MOMENT, self.MOMENT + timedelta(seconds=1)
            )

    def test_a_smoother_reads_up_to_its_lag_and_no_further(self) -> None:
        regime = InferenceRegime.smoother(2, STEP)
        check_evidence_access(regime, self.MOMENT, self.MOMENT + timedelta(minutes=10))
        with pytest.raises(EvidenceLeakageError):
            check_evidence_access(
                regime, self.MOMENT, self.MOMENT + timedelta(minutes=10, seconds=1)
            )


# ----------------------------------------------------------------------------
# The regime itself
# ----------------------------------------------------------------------------
class TestRegime:
    def test_labels_name_the_mode_and_the_lag(self) -> None:
        assert ONLINE.label == "online filter"
        assert InferenceRegime.smoother(1, STEP).label == (
            "fixed-lag smoother, lag 1 window (5 min)"
        )
        assert InferenceRegime.smoother(3, STEP).lag == timedelta(minutes=15)

    def test_an_online_regime_has_no_lag(self) -> None:
        with pytest.raises(EvidenceLeakageError, match="fixed-lag smoother"):
            InferenceRegime(InferenceMode.ONLINE_FILTER, 1)
        with pytest.raises(ValueError, match="no step"):
            InferenceRegime(InferenceMode.ONLINE_FILTER, 0, STEP)

    @pytest.mark.parametrize(
        ("lag", "step", "message"),
        [
            (-1, STEP, "non-negative"),
            (1, None, "window step"),
            (1, timedelta(0), "window step"),
        ],
    )
    def test_a_smoother_needs_a_valid_lag_and_step(
        self, lag: int, step: timedelta | None, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            InferenceRegime(InferenceMode.FIXED_LAG_SMOOTHER, lag, step)

    @pytest.mark.parametrize("regime", [ONLINE, InferenceRegime.smoother(2, STEP)])
    def test_it_round_trips_and_refuses_a_wrong_label(
        self, regime: InferenceRegime
    ) -> None:
        payload = json.loads(json.dumps(regime.to_dict()))
        assert InferenceRegime.from_dict(payload) == regime
        payload["label"] = "online filter" if not regime.is_online else "smoothed"
        with pytest.raises(EvidenceLeakageError, match="does not describe"):
            InferenceRegime.from_dict(payload)

    def test_results_from_different_regimes_are_not_combined(self) -> None:
        smoother = InferenceRegime.smoother(1, STEP)
        assert require_same_regime([ONLINE, ONLINE], "a table") == ONLINE
        with pytest.raises(EvidenceLeakageError, match="different inference regimes"):
            require_same_regime([ONLINE, smoother], "a table")
        with pytest.raises(EvidenceLeakageError):
            require_same_regime(
                [smoother, InferenceRegime.smoother(2, STEP)], "a table"
            )


# ----------------------------------------------------------------------------
# Records and reports
# ----------------------------------------------------------------------------
START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
MOMENTS = [START + STEP * k for k in range(len(evidence()))]


def reported_evidence(regime: InferenceRegime) -> EvidenceSummary:
    """The evidence summary of *regime*'s estimates on the synthetic sequence."""
    estimates = regime_estimates(MOMENTS, online(evidence()), TRANSITION, regime)
    return EvidenceSummary.of_estimates(estimates)


def record(**changes: Any) -> ExperimentRecord:
    regime = changes.get("inference", ONLINE)
    settings: dict[str, Any] = {
        "experiment": "regimes",
        "configuration": {},
        "inference": ONLINE,
        "evidence": reported_evidence(regime),
        "data_source": "synthetic-test",
    }
    settings.update(changes)
    return ExperimentRecord(**settings)


class TestRecords:
    def test_a_record_cannot_leave_the_regime_to_a_default(self) -> None:
        with pytest.raises(TypeError, match="inference"):
            ExperimentRecord(experiment="x", configuration={})  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="evidence"):
            ExperimentRecord(  # type: ignore[call-arg]
                experiment="x", configuration={}, inference=ONLINE
            )

    def test_a_smoothed_record_carries_its_mode_and_lag(self, tmp_path: Path) -> None:
        regime = InferenceRegime.smoother(2, STEP)
        path = record(inference=regime).write(tmp_path / "smoothed.json")
        payload = load_record(path)
        assert payload["inference"] == {
            "mode": "fixed_lag_smoother",
            "lag_steps": 2,
            "step_seconds": 300.0,
            "label": "fixed-lag smoother, lag 2 windows (10 min)",
            "causal": False,
            "delay_seconds": 600.0,
            "provenance": DECLARED_INFERENCE,
            "evidence": {
                "enumerated": True,
                "predictions": len(MOMENTS),
                "first_prediction": MOMENTS[0].isoformat(),
                "last_prediction": MOMENTS[-1].isoformat(),
                "latest_evidence": MOMENTS[-1].isoformat(),
                "max_lead_seconds": 600.0,
            },
        }
        rebuilt = ExperimentRecord.from_dict(payload)
        assert rebuilt.inference == regime
        assert rebuilt.evidence == reported_evidence(regime)

    def test_an_online_record_is_causal_with_no_delay(self) -> None:
        inference = record().to_dict()["inference"]
        assert inference["causal"] is True
        assert inference["delay_seconds"] == 0.0
        assert inference["evidence"]["max_lead_seconds"] == 0.0
        assert inference["evidence"]["latest_evidence"] == MOMENTS[-1].isoformat()

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"label": "online filter"}, "does not describe"),
            ({"mode": "online_filter"}, "lag is zero"),
            ({"provenance": ""}, "provenance"),
            ({"causal": True}, "is not causal"),
            ({"delay_seconds": 0.0}, "has a delay of 600"),
            ({"evidence": NotEnumerated("not listed").to_dict()}, "must be recorded"),
        ],
    )
    def test_a_mislabelled_regime_is_refused(
        self, change: dict[str, Any], message: str
    ) -> None:
        payload = record(inference=InferenceRegime.smoother(2, STEP)).to_dict()
        payload["inference"].update(change)
        with pytest.raises(ArtifactError, match=message):
            validate_record(payload)

    def test_smoothed_estimates_cannot_be_recorded_as_online(self) -> None:
        smoothed_evidence = reported_evidence(InferenceRegime.smoother(2, STEP))
        with pytest.raises(EvidenceLeakageError, match="600 s past their moments"):
            record(inference=ONLINE, evidence=smoothed_evidence)
        payload = record().to_dict()
        payload["inference"]["evidence"] = smoothed_evidence.to_dict()
        with pytest.raises(ArtifactError, match="the online filter may read 0 s"):
            validate_record(payload)

    def test_a_smoother_must_list_the_future_information_it_used(self) -> None:
        with pytest.raises(EvidenceLeakageError, match="must be recorded"):
            record(
                inference=InferenceRegime.smoother(1, STEP),
                evidence=NotEnumerated("not listed"),
            )
        # A causal regime may say why it did not list them.
        record(evidence=NotEnumerated("an aggregate study"))
        record(
            inference=InferenceRegime.smoother(0, STEP),
            evidence=NotEnumerated("an aggregate study"),
        )

    def test_an_older_record_is_attested_online_on_migration(
        self, tmp_path: Path
    ) -> None:
        payload = record().to_dict()
        payload["schema_version"] = "1.1"
        del payload["inference"]
        path = tmp_path / "older.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        migrated = load_record(path)
        assert migrated["migrated_from"] == "1.1"
        assert migrated["inference"]["mode"] == "online_filter"
        assert migrated["inference"]["provenance"].startswith(
            "attested on migration from schema 1.1"
        )
        assert migrated["inference"]["causal"] is True
        assert migrated["inference"]["evidence"] == {
            "enumerated": False,
            "reason": "written at schema 1.1, before prediction and evidence "
            "timestamps were recorded; its estimates read no evidence after their "
            "prediction moments",
        }
        assert inference_line(migrated) == (
            "- Inference regime: online filter (attested on migration from schema 1.1)."
        )

    def test_a_published_1_2_record_is_migrated_to_the_current_schema(self) -> None:
        path = ROOT / "artifacts" / "phase3" / "phase3-partial-pooling.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == "1.2"
        migrated = load_record(path)
        assert migrated["migrated_from"] == "1.2"
        assert migrated["inference"] == {
            **raw["inference"],
            "causal": True,
            "delay_seconds": 0.0,
            "evidence": {
                "enumerated": False,
                "reason": "written at schema 1.2, before prediction and evidence "
                "timestamps were recorded; its estimates read no evidence after "
                "their prediction moments",
            },
        }
        assert {k: v for k, v in migrated.items() if k not in MIGRATED} == {
            k: v for k, v in raw.items() if k not in MIGRATED
        }
        assert ExperimentRecord.from_dict(migrated).inference == ONLINE

    def test_a_smoothed_1_2_record_is_refused_not_migrated(
        self, tmp_path: Path
    ) -> None:
        payload = record().to_dict()
        payload["schema_version"] = "1.2"
        regime = InferenceRegime.smoother(1, STEP).to_dict()
        payload["inference"] = {
            "mode": regime["mode"],
            "lag_steps": regime["lag_steps"],
            "step_seconds": regime["step_seconds"],
            "label": regime["label"],
            "provenance": DECLARED_INFERENCE,
        }
        path = tmp_path / "smoothed-1.2.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError, match="a smoother must; re-run it"):
            load_record(path)

    def test_reports_label_the_regime(self) -> None:
        assert inference_line(record().to_dict()) == (
            "- Inference regime: online filter (declared by the experiment that "
            "wrote the record)."
        )
        smoothed_line = inference_line(
            record(inference=InferenceRegime.smoother(1, STEP)).to_dict()
        )
        assert "fixed-lag smoother, lag 1 window (5 min)" in smoothed_line

    @pytest.mark.parametrize(
        ("artifact", "module"),
        [
            ("phase1/phase1-recoverable-information-gap.json", "gap_summary"),
            ("phase3/phase3-hierarchical-time-prior.json", "time_prior_summary"),
            ("phase3/phase3-explicit-history.json", "history_summary"),
            ("phase3/phase3-correlated-silence.json", "silence_summary"),
            ("phase3/phase3-fitted-rates.json", "rates_summary"),
        ],
    )
    def test_every_published_report_labels_its_regime(
        self, artifact: str, module: str
    ) -> None:
        import importlib

        render = importlib.import_module(
            f"sensor_modeling.datasets.{module}"
        ).render_summary
        summary = render(load_record(ROOT / "artifacts" / artifact))
        assert "- Inference regime: online filter" in summary


# ----------------------------------------------------------------------------
# The matched evaluation
# ----------------------------------------------------------------------------
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


class Constant:
    """Predicts the training state frequencies for every row."""

    def fit(self, training: LabelledRows, development: LabelledRows | None) -> None:
        counts = np.array([training.labels.count(s) for s in training.states], float)
        self.probabilities = (counts + 1.0) / (counts + 1.0).sum()
        self.states = training.states

    def predict(self, rows: FeatureRows) -> StatePredictions:
        probabilities = np.tile(self.probabilities, (len(rows), 1))
        return StatePredictions(
            tuple(self.states[int(i)] for i in probabilities.argmax(axis=1)),
            probabilities,
        )


class DeclaredSmoother(Constant):
    inference_regime = InferenceRegime.smoother(1, STEP)


class UndeclaredSmoother(Constant):
    """Blends each row's prediction with the next row's: it reads later rows."""

    def predict(self, rows: FeatureRows) -> StatePredictions:
        values = rows.values[:, 0]
        weight = np.nan_to_num(np.concatenate([values[1:], [0.0]])) > 0
        probabilities = np.tile(self.probabilities, (len(rows), 1))
        probabilities[weight] = probabilities[weight][:, ::-1]
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return StatePredictions(
            tuple(self.states[int(i)] for i in probabilities.argmax(axis=1)),
            probabilities,
        )


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 4)}


def evaluate(homes: dict[str, CasasRecording], model: type, **options: Any) -> Any:
    return run_matched_evaluation(
        homes,
        split=HouseholdSplit("fold", train=("sim1", "sim2"), test=("sim3",)),
        information_set=nested_information_sets()[0],
        models=(ModelSpec("model", lambda seed: model()),),
        seed=0,
        data_source="synthetic-test",
        resamples=100,
        **options,
    )


class TestMatchedEvaluation:
    def test_an_online_run_records_and_labels_its_regime(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        run = evaluate(homes, Constant)
        assert run.regime == ONLINE
        payload = run.record.to_dict()
        assert payload["inference"]["mode"] == "online_filter"
        assert payload["results"]["inference"] == "online filter"
        # The scored moments are the held-out home's labelled moments, each
        # read up to itself.
        step = nested_information_sets()[0].resolution.step
        assert run.evidence == online_evidence(homes, ("sim3",), step)
        assert payload["inference"]["evidence"] == run.evidence.to_dict()
        assert payload["inference"]["evidence"]["max_lead_seconds"] == 0.0

    def test_a_smoothing_evaluation_is_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        with pytest.raises(EvidenceLeakageError, match="matched evaluation"):
            evaluate(homes, Constant, regime=InferenceRegime.smoother(1, STEP))

    def test_a_model_declaring_a_smoother_is_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        with pytest.raises(EvidenceLeakageError, match="model 'model'"):
            evaluate(homes, DeclaredSmoother)

    def test_a_model_reading_later_rows_is_refused(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        with pytest.raises(ValueError, match="depend only on the row"):
            evaluate(homes, UndeclaredSmoother)

    def test_runs_from_different_regimes_are_not_pooled(
        self, homes: dict[str, CasasRecording]
    ) -> None:
        run = evaluate(homes, Constant)
        smoothed_run = replace(run, regime=InferenceRegime.smoother(1, STEP))
        with pytest.raises(EvidenceLeakageError, match="different inference regimes"):
            held_out_metrics([run, smoothed_run], "model")
