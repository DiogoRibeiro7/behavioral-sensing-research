"""The sleep-mat declaration: what it pins, and that it is what was frozen."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from sensor_modeling.datasets.sleep_mat_protocol import (
    PLANNED_CHOICE,
    PLANNED_COVERAGE,
    PLANNED_READINGS,
    PLANNING_RECORD,
    PLANNING_RECORD_SHA256,
    PUBLISHED_RECORD,
    PUBLISHED_RECORD_SHA256,
    SILENT_HOME_RECORD,
    SILENT_HOME_RECORD_SHA256,
    SleepMatProtocol,
    check_frozen_protocol,
    code_changes,
    declared_protocol,
    file_sha256,
    pinned_sources,
)
from sensor_modeling.datasets.sleep_mat_summary import PROTOCOL_PAGE, render_protocol

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/sleep_mat/sleep_mat_protocol.json"


def planning() -> dict:
    return json.loads((ROOT / PLANNING_RECORD).read_text(encoding="utf-8"))["results"]


class TestWhatItPins:
    @pytest.mark.parametrize(
        ("path", "digest"),
        [
            (PLANNING_RECORD, PLANNING_RECORD_SHA256),
            (PUBLISHED_RECORD, PUBLISHED_RECORD_SHA256),
            (SILENT_HOME_RECORD, SILENT_HOME_RECORD_SHA256),
        ],
    )
    def test_each_pinned_record_is_the_file_in_the_repository(
        self, path: str, digest: str
    ) -> None:
        assert file_sha256(ROOT / path) == digest

    def test_the_planned_coverage_is_the_planning_records(self) -> None:
        results = planning()
        cells = results["tracking"] + results["level"]
        for flag, name in (
            (False, "without_the_outlying_homes"),
            (True, "with_the_outlying_homes"),
        ):
            chosen = [cell for cell in cells if cell["outliers"] is flag]
            for method, bounds in PLANNED_COVERAGE[name].items():
                values = [cell[method]["coverage"] for cell in chosen]
                assert (round(min(values), 3), round(max(values), 3)) == bounds
        assert results["chosen"] == PLANNED_CHOICE

    def test_the_quoted_readings_are_the_planning_records(self) -> None:
        results = planning()
        tracking = {
            (c["centre"], c["spread"], c["contamination"], c["outliers"]): c
            for c in results["tracking"]
        }
        level = {(c["bias"], c["spread"], c["outliers"]): c for c in results["level"]}
        found = {
            "c1_truth_every_home_at_0_5": tracking[(0.5, 0.0, 0.0, False)]["truth"],
            "c1_truth_homes_spread_0_6_about_0_5": tracking[(0.5, 0.6, 0.0, False)][
                "truth"
            ],
            "c1_truth_outlying_homes_rest_at_0_6": tracking[(0.6, 0.0, 0.0, True)][
                "truth"
            ],
            "c2_agrees_no_bias_spread_1": level[(0.0, 1.0, False)]["student_t"][
                "verdicts"
            ].get("agrees", 0.0),
            "c2_agrees_no_bias_spread_2": level[(0.0, 2.0, False)]["student_t"][
                "verdicts"
            ].get("agrees", 0.0),
        }
        assert {k: round(v, 3) for k, v in found.items()} == PLANNED_READINGS

    def test_the_planning_was_made_from_a_clean_commit(self) -> None:
        record = json.loads((ROOT / PLANNING_RECORD).read_text(encoding="utf-8"))
        assert record["environment"]["git_dirty"] == "false"

    def test_the_scripts_that_freeze_and_run_it_are_pinned(self) -> None:
        sources = pinned_sources(ROOT)
        assert "scripts/freeze_sleep_mat.py" in sources
        assert "scripts/run_sleep_mat.py" in sources
        assert "sensor_modeling/datasets/sleep_mat_experiment.py" in sources


class TestTheDeclaration:
    def test_it_serialises_without_a_missing_number(self) -> None:
        protocol = declared_protocol()
        json.dumps(protocol.to_dict(), allow_nan=False)
        assert protocol.sha256() == declared_protocol().sha256()

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"min_matched_days": 2}, "three days"),
            ({"tracking_margin": 1.0}, "tracking_margin"),
            ({"level_margin_hours": 0.0}, "level_margin_hours"),
            ({"shifts_hours": (0.0,)}, "shift of zero"),
            ({"lags_hours": (1, 2)}, "include zero"),
        ],
    )
    def test_a_declaration_that_cannot_be_run_is_refused(
        self, change: dict, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(SleepMatProtocol(), **change)

    def test_the_simulated_seeds_are_distinct_and_sorted(self) -> None:
        seeds = declared_protocol().sim_seeds()
        assert len(seeds) == len(set(seeds)) == 100
        assert list(seeds) == sorted(seeds)


@pytest.mark.skipif(not FROZEN.exists(), reason="the protocol is not frozen yet")
class TestTheFrozenProtocol:
    def test_the_frozen_file_is_what_the_code_declares(self) -> None:
        check_frozen_protocol(declared_protocol(), FROZEN)

    def test_a_changed_declaration_is_refused(self) -> None:
        changed = dataclasses.replace(declared_protocol(), tracking_margin=0.4)
        with pytest.raises(ValueError, match="cannot change"):
            check_frozen_protocol(changed, FROZEN)

    def test_the_page_is_the_frozen_files(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        page = (ROOT / "docs" / PROTOCOL_PAGE).read_text(encoding="utf-8")
        assert page == render_protocol(frozen)

    def test_the_freeze_recorded_the_code(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        assert set(frozen["at_freeze"]) == {"sources", "distributions", "defaults"}
        assert set(code_changes(FROZEN, ROOT)) == {
            "sources",
            "distributions",
            "defaults",
        }


RECORD = ROOT / "artifacts/sleep_mat/sleep-mat.json"
SIMULATED = ROOT / "artifacts/sleep_mat/sleep-mat-simulated.json"
RUN_COMMIT = "c42d69a"


@pytest.mark.skipif(not RECORD.exists(), reason="the protocol has not been run yet")
class TestThePublishedRun:
    def records(self) -> tuple[dict, dict]:
        return (
            json.loads(RECORD.read_text(encoding="utf-8")),
            json.loads(SIMULATED.read_text(encoding="utf-8")),
        )

    def test_both_records_come_from_the_frozen_commit_unchanged(self) -> None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        for record in self.records():
            assert record["environment"]["git_commit"].startswith(RUN_COMMIT)
            assert record["environment"]["git_dirty"] == "false"
            assert (
                record["configuration"]["protocol_sha256"] == frozen["protocol_sha256"]
            )
        tihm, _ = self.records()
        assert not any(tihm["configuration"]["code_changed_since_the_freeze"].values())

    def test_the_criteria_are_recomputed_from_the_homes(self) -> None:
        from scipy import stats

        from sensor_modeling.datasets.sleep_mat_experiment import (
            level_reading,
            tracking_reading,
        )

        tihm, _ = self.records()
        rows = [
            row for row in tihm["household_metrics"]["off"].values() if row["included"]
        ]
        protocol = declared_protocol()
        for key, criterion, reading, margin in (
            (
                "sleep_spearman",
                "C1_the_pipeline_follows_the_mat",
                tracking_reading,
                protocol.tracking_margin,
            ),
            (
                "sleep_mean_difference",
                "C2_the_pipeline_agrees_in_level",
                level_reading,
                protocol.level_margin_hours,
            ),
        ):
            values = [row[key] for row in rows]
            n = len(values)
            mean = sum(values) / n
            sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
            half = stats.t.ppf(0.975, n - 1) * sd / n**0.5
            found = tihm["results"]["criteria"][criterion]
            assert found["estimate"]["estimate"] == pytest.approx(mean)
            assert found["estimate"]["interval"]["low"] == pytest.approx(mean - half)
            assert found["estimate"]["interval"]["high"] == pytest.approx(mean + half)
            assert found["reading"] == reading(found["estimate"], margin)

    def test_the_page_is_the_records(self) -> None:
        from sensor_modeling.datasets.sleep_mat_summary import RESULTS_PAGE, render_page

        tihm, simulated = self.records()
        page = (ROOT / "docs" / RESULTS_PAGE).read_text(encoding="utf-8")
        assert page == render_page(tihm, simulated)
