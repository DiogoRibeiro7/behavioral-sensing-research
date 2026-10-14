"""Tests for the ContextAct transfer test, on simulated homes.

They guard what the result rests on:
- everything is fitted on the development homes only, exactly as the held-out
  confirmation fitted it, and the run refuses a fit or a prior that is not the
  confirmed one;
- the recursions are the combined-prior experiment's, scored day by day, and the
  online pipeline's opt-in gives the same beliefs;
- the baselines are scored on the same windows of the same days;
- the protocol is frozen, pins its records and its code, and refuses any change;
- the pages are generated from the frozen file and the record.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording, HouseholdSplit
from sensor_modeling.datasets import transfer_experiment as transfer
from sensor_modeling.datasets.combined_prior_experiment import MODELS, household_scores
from sensor_modeling.datasets.heldout_experiment import training_fit
from sensor_modeling.datasets.phase3_formulation import (
    Phase3Formulation,
    parameters_payload,
    write_parameters,
)
from sensor_modeling.datasets.rates_experiment import FittedRatesProtocol
from sensor_modeling.datasets.recoverable_gap import load_frozen_splits
from sensor_modeling.datasets.transfer_experiment import (
    INPUTS_FILE,
    LEVEL_READING,
    MIN_DAY_WINDOWS,
    PINNED_RECORDS,
    PROTOCOL_FILE,
    RECORD_FILE,
    TransferProtocol,
    TransferResult,
    check_frozen_protocol,
    check_pinned_records,
    code_changes,
    day_key,
    day_scores,
    declared_protocol,
    inputs_report,
    labelled_days,
    pipeline_difference,
    read_periods,
    recursion_posteriors,
    reproduced_fit,
    run_transfer,
    split_days,
    write_protocol,
)
from sensor_modeling.datasets.transfer_summary import (
    PROTOCOL_PAGE,
    RESULTS_PAGE,
    render_page,
    render_protocol,
)
from sensor_modeling.evaluation import load_record
from sensor_modeling.external.contextact import DESCRIPTION, HOUSEHOLDS, LOGS
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import BehaviouralState as S
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "phase1" / "household_splits.json"
ONTOLOGY = StateOntology()
FOLDS = (
    HouseholdSplit("a", train=("sim2", "sim4"), test=("sim1", "sim3")),
    HouseholdSplit("b", train=("sim1", "sim3"), test=("sim2", "sim4")),
)
JULY, NOVEMBER = HOUSEHOLDS
ABSENT = "A4H-2017-01"
PERIODS = (JULY, NOVEMBER, ABSENT)


def simulated(seed: int) -> CasasRecording:
    sim = simulate(HouseholdConfig(days=3, seed=seed))
    return CasasRecording(
        sim.registry,
        sim.observations,
        tuple(
            ActivityInterval(e.state.value, e.start, e.end, e.state)
            for e in sim.truth.episodes
        ),
    )


def base_protocol() -> FittedRatesProtocol:
    return FittedRatesProtocol(folds=FOLDS, splits_sha256="0" * 64, resamples=200)


def eligible(*periods: str) -> dict[str, Any]:
    return {p: {"eligible": True, "blocking": [], "issues": {}} for p in periods}


@pytest.fixture(scope="module")
def training() -> dict[str, CasasRecording]:
    return {f"sim{seed}": simulated(seed) for seed in range(1, 5)}


def without_bed_awake(recording: CasasRecording) -> CasasRecording:
    """A period in which `bed_awake` is never labelled, as in ContextAct."""
    return CasasRecording(
        recording.registry,
        recording.observations,
        tuple(a for a in recording.activities if a.state is not S.BED_AWAKE),
    )


@pytest.fixture(scope="module")
def periods() -> dict[str, CasasRecording]:
    return {
        JULY: without_bed_awake(simulated(5)),
        NOVEMBER: without_bed_awake(simulated(6)),
    }


@pytest.fixture(scope="module")
def formulation(
    training: dict[str, CasasRecording], tmp_path_factory: pytest.TempPathFactory
) -> Phase3Formulation:
    """Parameters fitted on the simulated development homes, as the package's."""
    fit = training_fit(training, tuple(sorted(training)), base_protocol(), ONTOLOGY)
    path = write_parameters(
        parameters_payload(fit.channels, fit.periodic, {"fitted": "simulated"}),
        tmp_path_factory.mktemp("parameters") / "phase3_parameters.json",
    )
    return Phase3Formulation.load(
        path,
        channels_sha256=fit.channels.sha256(),
        periodic_prior_sha256=fit.periodic.sha256(),
    )


@pytest.fixture(scope="module", autouse=True)
def shipped(formulation: Phase3Formulation) -> Iterator[None]:
    """The package's parameters are the simulated ones throughout."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            Phase3Formulation, "load", classmethod(lambda cls, *a, **k: formulation)
        )
        yield


def small_protocol(
    formulation: Phase3Formulation, inputs: dict[str, Any] | None = None
) -> TransferProtocol:
    return TransferProtocol(
        base_protocol(),
        periods=PERIODS,
        inputs=inputs,
        channels_sha256=formulation_channels(formulation),
    )


def frozen_inputs(formulation: Phase3Formulation) -> dict[str, Any]:
    """An inputs report stub, written under the declaration without it."""
    return {
        "periods": dict.fromkeys(PERIODS, {}),
        "summary": {},
        "sha256": "0" * 64,
        "declared_protocol_sha256": small_protocol(formulation).sha256(),
        "code_sha256": transfer.code_digest(),
    }


def formulation_channels(formulation: Phase3Formulation) -> str:
    from sensor_modeling.datasets.phase3_formulation import canonical_sha256

    return canonical_sha256(formulation.channels)


@pytest.fixture(scope="module")
def result(
    training: dict[str, CasasRecording],
    periods: dict[str, CasasRecording],
    formulation: Phase3Formulation,
    tmp_path_factory: pytest.TempPathFactory,
) -> TransferResult:
    validation = {
        **eligible(JULY, NOVEMBER),
        ABSENT: {"eligible": False, "blocking": ["mixed_awareness"], "issues": {}},
    }
    return run_transfer(
        training,
        periods,
        validation,
        small_protocol(formulation),
        data_source="simulator",
        output_dir=tmp_path_factory.mktemp("transfer"),
    )


# ----------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------
class TestFit:
    def test_it_reproduces_the_fit_and_uses_the_shipped_prior(
        self, training: dict[str, CasasRecording], formulation: Phase3Formulation
    ) -> None:
        fit, difference = reproduced_fit(
            training, small_protocol(formulation), formulation
        )
        assert fit.periodic is formulation.prior
        assert difference <= transfer.PRIOR_TOLERANCE

    def test_it_refuses_channels_that_are_not_the_confirmed_ones(
        self, training: dict[str, CasasRecording], formulation: Phase3Formulation
    ) -> None:
        protocol = replace(small_protocol(formulation), channels_sha256="0" * 64)
        with pytest.raises(ValueError, match="not the held-out confirmation's"):
            reproduced_fit(training, protocol, formulation)

    def test_it_refuses_a_prior_that_is_not_the_shipped_one(
        self, training: dict[str, CasasRecording], formulation: Phase3Formulation
    ) -> None:
        moved = replace(
            formulation.prior, population=formulation.prior.population + 1e-6
        )
        other = Phase3Formulation(formulation.channels, moved)
        with pytest.raises(ValueError, match="from the shipped one"):
            reproduced_fit(training, small_protocol(formulation), other)


# ----------------------------------------------------------------------------
# Days, the recursions and the pipeline
# ----------------------------------------------------------------------------
class TestDays:
    def test_a_day_is_a_local_calendar_date(self) -> None:
        paris = ZoneInfo("Europe/Paris")
        assert day_key(JULY, datetime(2016, 7, 25, 23, 55, tzinfo=paris)) == (
            f"{JULY}/2016-07-25"
        )
        assert day_key(JULY, datetime(2016, 7, 26, 0, 0, tzinfo=paris)) == (
            f"{JULY}/2016-07-26"
        )

    def test_a_period_across_a_change_of_utc_offset_is_refused(
        self, periods: dict[str, CasasRecording]
    ) -> None:
        paris = ZoneInfo("Europe/Paris")
        moments = [
            datetime(2016, 10, 30, 1, 55, tzinfo=paris),
            datetime(2016, 10, 30, 3, 0, tzinfo=paris),
        ]
        with pytest.raises(ValueError, match="change of UTC offset"):
            transfer.check_clock(periods[JULY], JULY, moments)

    def test_a_day_with_too_few_labelled_windows_is_unscored(self) -> None:
        kept, below = split_days({"a": MIN_DAY_WINDOWS, "b": MIN_DAY_WINDOWS - 1})
        assert kept == ["a"]
        assert below == {"b": MIN_DAY_WINDOWS - 1}

    def test_the_days_partition_the_periods_scores(
        self,
        training: dict[str, CasasRecording],
        periods: dict[str, CasasRecording],
        formulation: Phase3Formulation,
    ) -> None:
        protocol = small_protocol(formulation)
        fit, _ = reproduced_fit(training, protocol, formulation)
        recording = periods[JULY]
        posteriors, rows, truth, moments = recursion_posteriors(
            recording, fit, period=JULY, resolution=protocol.resolution
        )
        days = list(labelled_days(JULY, rows, moments))
        assert len(days) >= 3
        per_day = day_scores(JULY, posteriors, rows, truth, moments, days)
        whole = household_scores(
            recording, fit, household=JULY, ontology=ONTOLOGY, protocol=protocol.scoring
        )
        for model in MODELS:
            summed = sum(
                np.array(per_day[model][d].confusion.to_dict()["counts"]) for d in days
            )
            np.testing.assert_array_equal(
                summed, np.array(whole[model].confusion.to_dict()["counts"])
            )

    def test_the_online_pipeline_is_the_recursion(
        self,
        training: dict[str, CasasRecording],
        periods: dict[str, CasasRecording],
        formulation: Phase3Formulation,
    ) -> None:
        protocol = small_protocol(formulation)
        fit, _ = reproduced_fit(training, protocol, formulation)
        recording = periods[NOVEMBER]
        posteriors, _, _, moments = recursion_posteriors(
            recording, fit, period=NOVEMBER, resolution=protocol.resolution
        )
        difference = pipeline_difference(
            recording, posteriors["filter_periodic_hurdle"], moments
        )
        assert difference <= transfer.PIPELINE_TOLERANCE


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
class TestRun:
    def test_it_writes_a_valid_record(self, result: TransferResult) -> None:
        assert result.path is not None
        payload = load_record(result.path)
        results = payload["results"]
        assert [e["key"] for e in results["estimands"]] == [
            "K1",
            "K2",
            "K3",
            "G1",
            "G2",
            "L1",
            "C1",
        ]
        assert results["check"]["windows"] == "identical"
        assert results["check"]["excluded_periods"] == [ABSENT]
        assert set(results["check"]["periods"]) == {JULY, NOVEMBER}
        assert results["conclusion"] in {"success", "failure", "inconclusive"}

    def test_every_unit_is_a_day(self, result: TransferResult) -> None:
        metrics = result.record.household_metrics
        days = set(metrics["filter_periodic_hurdle@R"])
        assert days == set(metrics["diagnostic@I3"])
        assert all(day.split("/")[0] in {JULY, NOVEMBER} for day in days)
        assert len(days) == result.record.results["check"]["days"]

    def test_the_level_and_chance_estimands(self, result: TransferResult) -> None:
        entries = {e["key"]: e for e in result.record.results["estimands"]}
        assert entries["L1"]["conclusion"] in set(LEVEL_READING.values())
        chance = entries["C1"]
        assert chance["reference"] == "chance@day"
        assert chance["comparisons"]["balanced_accuracy"]["n"] + chance[
            "days_left_out"
        ] == (result.record.results["check"]["days"])

    def test_a_state_never_labelled_is_listed_not_compared(
        self, result: TransferResult
    ) -> None:
        k1 = next(e for e in result.record.results["estimands"] if e["key"] == "K1")
        assert k1["per_state_recall"]["bed_awake"] == {
            "comparison": None,
            "verdict": "never labelled",
        }
        assert k1["per_state_recall"]["sleeping"]["comparison"] is not None

    def test_too_few_days_leave_k1_inconclusive(self, result: TransferResult) -> None:
        results = result.record.results
        k1 = next(e for e in results["estimands"] if e["key"] == "K1")
        assert results["check"]["days"] < transfer.MIN_SCORED_DAYS
        assert k1["conclusion"] == results["conclusion"] == "inconclusive"
        assert "scored days" in k1["note"]

    def test_the_pages_render_without_nan(
        self, result: TransferResult, formulation: Phase3Formulation
    ) -> None:
        assert result.path is not None
        page = render_page(load_record(result.path))
        assert "nan" not in page.lower()
        protocol = small_protocol(formulation, frozen_inputs(formulation))
        frozen = {**protocol.to_dict(), "protocol_sha256": protocol.sha256()}
        assert "nan" not in render_protocol(frozen).lower()

    def test_it_refuses_when_no_day_is_labelled(
        self,
        training: dict[str, CasasRecording],
        periods: dict[str, CasasRecording],
        formulation: Phase3Formulation,
    ) -> None:
        bare = {
            p: CasasRecording(r.registry, r.observations, ())
            for p, r in periods.items()
        }
        with pytest.raises(ValueError, match="no day has enough labelled windows"):
            run_transfer(
                training,
                bare,
                eligible(JULY, NOVEMBER),
                small_protocol(formulation),
                data_source="simulator",
            )

    def test_it_refuses_when_the_pipeline_differs(
        self,
        training: dict[str, CasasRecording],
        periods: dict[str, CasasRecording],
        formulation: Phase3Formulation,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(transfer, "pipeline_difference", lambda *a, **k: 1.0)
        with pytest.raises(ValueError, match="online pipeline"):
            run_transfer(
                training,
                periods,
                eligible(JULY, NOVEMBER),
                small_protocol(formulation),
                data_source="simulator",
            )

    def test_it_refuses_periods_and_homes_it_does_not_expect(
        self,
        training: dict[str, CasasRecording],
        periods: dict[str, CasasRecording],
        formulation: Phase3Formulation,
    ) -> None:
        protocol = small_protocol(formulation)
        with pytest.raises(ValueError, match="does not name"):
            run_transfer(
                training, {"other": periods[JULY]}, {}, protocol, data_source="x"
            )
        with pytest.raises(ValueError, match="no recording for training homes"):
            run_transfer({}, periods, {}, protocol, data_source="x")


# ----------------------------------------------------------------------------
# The inputs report, reading ContextAct, and the protocol
# ----------------------------------------------------------------------------
class TestInputs:
    def test_it_reports_without_scoring(
        self,
        training: dict[str, CasasRecording],
        periods: dict[str, CasasRecording],
        formulation: Phase3Formulation,
    ) -> None:
        validation = {
            **eligible(JULY, NOVEMBER),
            ABSENT: {"eligible": False, "blocking": ["x"], "issues": {}},
        }
        report = inputs_report(
            training, periods, validation, small_protocol(formulation)
        )
        assert report["scored"].startswith("nothing")
        assert (
            report["declared_protocol_sha256"] == small_protocol(formulation).sha256()
        )
        assert report["dry_run"]["completed"]
        assert report["summary"]["eligible_periods"] == [JULY, NOVEMBER]
        assert report["periods"][ABSENT] == {"validation": validation[ABSENT]}
        july = report["periods"][JULY]
        assert july["recursions_finite"]
        assert july["pipeline_difference"] <= transfer.PIPELINE_TOLERANCE
        assert set(july["scored_days"]) <= set(july["labelled_windows_by_day"])
        assert "hall_motion" not in july["instrumented"]

    def test_it_reads_contextact_and_keeps_no_examples(self, tmp_path: Path) -> None:
        from test_external_contextact import HEADER, LOG, VARIABLES, write_xlsx

        write_xlsx(tmp_path / DESCRIPTION, [HEADER, *VARIABLES])
        for name in LOGS.values():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(LOG) + "\n", encoding="latin-1")
        recordings, validation = read_periods(tmp_path)
        assert set(recordings) == set(HOUSEHOLDS)
        for entry in validation.values():
            assert entry["eligible"]
            assert all(
                set(i) == {"severity", "count"} for i in entry["issues"].values()
            )
        rooms = {spec.sensor_id: spec.room for spec in recordings[JULY].registry}
        assert rooms["Presence_Table"] == "living"


class TestProtocol:
    def test_it_is_stable_and_declares_its_settings(
        self, formulation: Phase3Formulation
    ) -> None:
        protocol = small_protocol(formulation)
        assert protocol.sha256() == small_protocol(formulation).sha256()
        payload: dict[str, Any] = protocol.to_dict()
        json.dumps(payload, allow_nan=False)
        assert payload["bootstrap"]["unit"] == "day"
        assert payload["dataset"]["periods"] == list(PERIODS)
        assert payload["minimal_differences"]["chance"] == transfer.CHANCE_DIFFERENCE

    def test_it_refuses_inconsistent_declarations(
        self, formulation: Phase3Formulation
    ) -> None:
        with pytest.raises(ValueError, match="distinct periods"):
            TransferProtocol(base_protocol(), periods=(JULY, JULY))
        with pytest.raises(ValueError, match="training home"):
            TransferProtocol(base_protocol(), periods=("sim1",))
        with pytest.raises(ValueError, match="does not cover"):
            small_protocol(formulation, {"periods": {JULY: {}}})

    def test_a_frozen_protocol_refuses_any_change(
        self, formulation: Phase3Formulation, tmp_path: Path
    ) -> None:
        protocol = small_protocol(formulation, frozen_inputs(formulation))
        path = tmp_path / "protocol.json"
        with pytest.raises(ValueError, match="without its inputs report"):
            write_protocol(small_protocol(formulation), path)
        stale = {**frozen_inputs(formulation), "declared_protocol_sha256": "0" * 64}
        with pytest.raises(ValueError, match="another declaration"):
            write_protocol(small_protocol(formulation, stale), path)
        moved = {**frozen_inputs(formulation), "code_sha256": "0" * 64}
        with pytest.raises(ValueError, match="sources changed"):
            write_protocol(small_protocol(formulation, moved), path)
        write_protocol(protocol, path)
        assert check_frozen_protocol(protocol, path)
        with pytest.raises(ValueError, match="differs"):
            check_frozen_protocol(replace(protocol, name="other"), path)
        assert code_changes(path) == []

    def test_it_refuses_a_changed_pinned_record_or_source(
        self, formulation: Phase3Formulation, tmp_path: Path
    ) -> None:
        for name in PINNED_RECORDS:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        check_pinned_records(tmp_path)
        first = tmp_path / next(iter(PINNED_RECORDS))
        first.write_text(first.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(ValueError, match="not the file the protocol pins"):
            check_pinned_records(tmp_path)

        protocol = small_protocol(formulation, frozen_inputs(formulation))
        path = tmp_path / "protocol.json"
        write_protocol(protocol, path)
        copy = tmp_path / "repository"
        shutil.copytree(ROOT / "sensor_modeling", copy / "sensor_modeling")
        for name in transfer.PINNED_FILES:
            if not name.startswith("sensor_modeling/"):
                (copy / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, copy / name)
        assert code_changes(path, copy) == []
        changed = copy / "sensor_modeling" / "datasets" / "transfer_experiment.py"
        changed.write_text(changed.read_text(encoding="utf-8") + "\n# x\n")
        assert code_changes(path, copy) == [
            "sensor_modeling/datasets/transfer_experiment.py"
        ]


# ----------------------------------------------------------------------------
# The frozen files, once they exist
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / PROTOCOL_FILE).exists(), reason="not yet frozen")
class TestFrozenProtocol:
    def test_the_frozen_protocol_is_the_declared_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(ROOT)
        protocol = declared_protocol(load_frozen_splits(SPLITS))
        check_frozen_protocol(protocol, ROOT / PROTOCOL_FILE)
        assert protocol.inputs is not None
        assert protocol.inputs["schema"] == transfer.INPUTS_SCHEMA
        assert (ROOT / INPUTS_FILE).exists()

    def test_its_page_is_generated_from_it(self) -> None:
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)


@pytest.mark.skipif(not (ROOT / RECORD_FILE).exists(), reason="not yet run")
class TestRecord:
    def test_it_ran_the_frozen_protocol_on_a_clean_tree(self) -> None:
        payload = load_record(ROOT / RECORD_FILE)
        frozen = json.loads((ROOT / PROTOCOL_FILE).read_text(encoding="utf-8"))
        assert payload["configuration"]["protocol_sha256"] == frozen["protocol_sha256"]
        assert payload["environment"]["git_dirty"] == "false"

    def test_its_page_is_generated_from_it(self) -> None:
        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(load_record(ROOT / RECORD_FILE))
