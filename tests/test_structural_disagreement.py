"""Tests for the model-structure disagreement diagnostics.

They check the measures themselves:
- zero disagreement for identical models;
- disagreement growing as models are made to contradict each other;
- symmetry;
- bounds;
- deterministic results.

They also check that the diagnostics enter the experiment record as
validated, schema-1.4 data, that only supported specifications take part, and
that the deployed decision is unchanged.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sensor_modeling.datasets import ActivityInterval, CasasRecording
from sensor_modeling.datasets.channel_models import (
    HURDLE,
    HURDLE_NB,
    filter_recursion,
    home_statistics,
    household_channel_counts,
    total_loglik,
)
from sensor_modeling.datasets.information_sets import EvidenceResolution
from sensor_modeling.datasets.structural_models import (
    CATALOGUE,
    DEFAULT_ENSEMBLE,
    MAX_SPECIFICATIONS,
    REFERENCE,
    REFUSED,
    Specification,
    check_ensemble,
    ensemble_report,
    fit_samples,
    halves,
    household_trace,
)
from sensor_modeling.evaluation import (
    ArtifactError,
    DisagreementReport,
    DisagreementTrace,
    ExperimentRecord,
    compare_posteriors,
    jensen_shannon,
    load_record,
    validate_record,
)
from sensor_modeling.evaluation.disagreement import (
    generalised_jensen_shannon,
    validate_disagreement,
)
from sensor_modeling.fusion import ONLINE, NotEnumerated
from sensor_modeling.simulation import HouseholdConfig, simulate
from sensor_modeling.states import StateOntology

ROOT = Path(__file__).resolve().parents[1]
STATES = ("a", "b", "c", "d")
ONTOLOGY = StateOntology()
RESOLUTION = EvidenceResolution()


def trace(
    posteriors: dict[str, np.ndarray], reference: str | None = None
) -> DisagreementTrace:
    windows = next(iter(posteriors.values())).shape[0]
    return compare_posteriors(
        posteriors,
        household="h",
        timestamps=[f"2026-01-01T00:{i:02d}:00+00:00" for i in range(windows)],
        states=STATES,
        reference=reference or sorted(posteriors)[0],
    )


def dirichlet(seed: int, windows: int = 50) -> np.ndarray:
    values: np.ndarray = np.random.default_rng(seed).dirichlet(
        np.full(len(STATES), 0.5), size=windows
    )
    return values


def toward(state: int, strength: float, windows: int = 1) -> np.ndarray:
    """The uniform posterior moved toward *state* by *strength*, in [0, 1]."""
    uniform = np.full(len(STATES), 1.0 / len(STATES))
    one_hot = np.eye(len(STATES))[state]
    return np.tile((1.0 - strength) * uniform + strength * one_hot, (windows, 1))


# ----------------------------------------------------------------------------
# The measures
# ----------------------------------------------------------------------------
class TestIdenticalModels:
    def test_identical_models_do_not_disagree(self) -> None:
        p = dirichlet(0)
        found = trace({"x": p, "y": p.copy(), "z": p.copy()})
        np.testing.assert_array_equal(found.pairwise, 0.0)
        np.testing.assert_array_equal(found.discordance, 0.0)
        np.testing.assert_array_equal(found.vote_disagreement, 0.0)
        np.testing.assert_array_equal(found.spread, 0.0)
        summary = found.summary()
        assert summary["unanimous"] == 1.0
        assert summary["decision_is_plurality"] == 1.0
        assert set(summary["agrees_with_decision"].values()) == {1.0}


class TestContradiction:
    def test_disagreement_grows_as_models_contradict(self) -> None:
        strengths = np.linspace(0.0, 1.0, 11)
        pairwise, discordance, spread = [], [], []
        for s in strengths:
            found = trace({"x": toward(0, s), "y": toward(1, s)})
            pairwise.append(float(found.pairwise[0, 0]))
            discordance.append(float(found.discordance[0]))
            spread.append(float(found.spread[0, 0]))
        assert pairwise[0] == 0.0 and pairwise[-1] == pytest.approx(1.0)
        assert all(b > a for a, b in zip(pairwise, pairwise[1:]))
        assert all(b > a for a, b in zip(discordance, discordance[1:]))
        assert all(b > a for a, b in zip(spread, spread[1:]))

    def test_more_contradicting_models_disagree_more(self) -> None:
        two = trace({"x": toward(0, 1.0), "y": toward(1, 1.0)})
        three = trace({"x": toward(0, 1.0), "y": toward(1, 1.0), "z": toward(2, 1.0)})
        assert three.discordance[0] > two.discordance[0]
        assert three.vote_disagreement[0] > two.vote_disagreement[0]
        # Disjoint one-hot posteriors reach every bound.
        assert three.normalised_discordance[0] == pytest.approx(1.0)
        assert three.vote_disagreement[0] == pytest.approx(1.0 - 1.0 / 3.0)

    def test_votes_count_disagreement_even_when_posteriors_are_close(self) -> None:
        close = trace({"x": toward(0, 0.02), "y": toward(1, 0.02)})
        assert close.vote_disagreement[0] == pytest.approx(0.5)
        assert close.pairwise[0, 0] < 1e-3


class TestSymmetry:
    def test_the_divergence_is_symmetric(self) -> None:
        p, q = dirichlet(1), dirichlet(2)
        np.testing.assert_array_equal(jensen_shannon(p, q), jensen_shannon(q, p))

    def test_the_order_of_the_models_does_not_matter(self) -> None:
        p, q, r = dirichlet(3), dirichlet(4), dirichlet(5)
        forward = trace({"x": p, "y": q, "z": r}, reference="x")
        backward = trace({"z": r, "y": q, "x": p}, reference="x")
        assert forward.models == backward.models
        np.testing.assert_array_equal(forward.pairwise, backward.pairwise)
        assert forward.summary() == backward.summary()

    def test_renaming_the_models_permutes_nothing_but_the_labels(self) -> None:
        p, q = dirichlet(6), dirichlet(7)
        first = trace({"x": p, "y": q})
        second = trace({"x": q, "y": p})
        np.testing.assert_array_equal(first.pairwise, second.pairwise)
        np.testing.assert_array_equal(first.discordance, second.discordance)
        np.testing.assert_array_equal(first.spread, second.spread)


class TestBounds:
    @pytest.mark.parametrize("seed", range(5))
    def test_every_measure_stays_within_its_bounds(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        models = int(rng.integers(2, 7))
        posteriors = {
            f"m{i}": rng.dirichlet(np.full(len(STATES), 0.2), size=40)
            for i in range(models)
        }
        posteriors["m0"][:5] = np.eye(len(STATES))[rng.integers(0, len(STATES), 5)]
        found = trace(posteriors)
        assert found.pairwise.min() >= 0.0 and found.pairwise.max() <= 1.0
        assert found.discordance.min() >= 0.0
        assert found.discordance.max() <= math.log2(models) + 1e-12
        assert 0.0 <= found.normalised_discordance.min()
        assert found.normalised_discordance.max() <= 1.0 + 1e-12
        assert found.vote_disagreement.min() >= 0.0
        assert found.vote_disagreement.max() <= 1.0 - 1.0 / models + 1e-12
        assert found.spread.min() >= 0.0 and found.spread.max() <= 1.0

    def test_the_generalised_divergence_of_two_is_the_pairwise_one(self) -> None:
        p, q = dirichlet(8), dirichlet(9)
        np.testing.assert_allclose(
            generalised_jensen_shannon(np.stack([p, q], axis=1)),
            jensen_shannon(p, q),
            atol=1e-12,
        )


class TestDeterminism:
    def test_the_same_inputs_give_the_same_trace_and_bytes(
        self, tmp_path: Path
    ) -> None:
        posteriors = {"x": dirichlet(10), "y": dirichlet(11), "z": dirichlet(12)}
        first, second = trace(posteriors), trace(dict(posteriors))
        assert first.summary() == second.summary()
        assert first.windows() == second.windows()
        a = first.write(tmp_path / "a.json.gz")
        b = second.write(tmp_path / "b.json.gz")
        assert a == b
        assert (tmp_path / "a.json.gz").read_bytes() == (
            tmp_path / "b.json.gz"
        ).read_bytes()

    def test_a_written_trace_reads_back_and_checks_its_digest(
        self, tmp_path: Path
    ) -> None:
        original = trace({"x": dirichlet(13), "y": dirichlet(14)})
        digest = original.write(tmp_path / "t.json.gz")
        again = DisagreementTrace.read(tmp_path / "t.json.gz", digest)
        np.testing.assert_array_equal(again.posteriors, original.posteriors)
        assert again.summary() == original.summary()
        with pytest.raises(ValueError, match="digest"):
            DisagreementTrace.read(tmp_path / "t.json.gz", "0" * 64)


class TestWindows:
    def test_each_window_records_what_the_diagnostic_promises(self) -> None:
        found = trace({"x": toward(0, 0.8, 3), "y": toward(1, 0.6, 3)})
        window = found.windows()[0]
        assert set(window) == {
            "timestamp",
            "posteriors",
            "argmax",
            "decision",
            "pairwise",
            "votes",
            "plurality",
            "vote_disagreement",
            "spread",
            "consensus",
            "consensus_state",
            "discordance",
            "unanimous",
        }
        assert window["argmax"] == {"x": "a", "y": "b"}
        assert window["decision"] == "a"
        assert set(window["pairwise"]) == {"x|y"}
        assert sum(window["consensus"].values()) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("posteriors", "message"),
        [
            ({"x": np.full((2, 4), 0.3), "y": np.full((2, 4), 0.25)}, "sum to one"),
            ({"x": toward(0, 1.0, 2), "y": toward(0, 1.0, 3)}, "same windows"),
            ({"x": -toward(0, 1.0, 2), "y": toward(0, 1.0, 2)}, "non-negative"),
            ({"x": toward(0, 1.0, 2)}, "two distinct"),
        ],
    )
    def test_invalid_posteriors_are_refused(
        self, posteriors: dict[str, np.ndarray], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            compare_posteriors(
                posteriors,
                household="h",
                timestamps=["t0", "t1"],
                states=STATES,
                reference="x",
            )

    def test_the_reference_and_truth_are_checked(self) -> None:
        with pytest.raises(ValueError, match="reference"):
            trace({"x": dirichlet(0, 2), "y": dirichlet(1, 2)}, reference="z")
        with pytest.raises(ValueError, match="truth"):
            compare_posteriors(
                {"x": dirichlet(0, 2), "y": dirichlet(1, 2)},
                household="h",
                timestamps=["t0", "t1"],
                states=STATES,
                reference="x",
                truth=np.array([0, 9]),
            )


# ----------------------------------------------------------------------------
# In the experiment record
# ----------------------------------------------------------------------------
def specification_payload(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "assumptions": [
            {
                "axis": "observation",
                "variant": name,
                "description": "test",
                "support": [
                    {"record": "r.json", "estimand": "E", "finding": "success"}
                ],
            }
        ],
    }


def report(files: dict[str, dict[str, str]] | None = None) -> DisagreementReport:
    traces = [trace({"x": dirichlet(20), "y": dirichlet(21), "z": dirichlet(22)})]
    return DisagreementReport.from_traces(
        traces, [specification_payload(n) for n in ("x", "y", "z")], files
    )


def record(**changes: Any) -> ExperimentRecord:
    settings: dict[str, Any] = {
        "experiment": "disagreement",
        "configuration": {},
        "inference": ONLINE,
        "evidence": NotEnumerated("a unit-test record"),
        "data_source": "synthetic-test",
        "structural_disagreement": report(),
    }
    settings.update(changes)
    return ExperimentRecord(**settings)


class TestRecord:
    def test_a_record_carries_and_rebuilds_the_report(self, tmp_path: Path) -> None:
        path = record().write(tmp_path / "r.json")
        payload = load_record(path)
        assert payload["schema_version"] == "1.4"
        section = payload["structural_disagreement"]
        assert section["reference"] == "x"
        assert set(section["households"]["h"]["summary"]["pairs"]) == {
            "x|y",
            "x|z",
            "y|z",
        }
        rebuilt = ExperimentRecord.from_dict(payload)
        assert rebuilt.structural_disagreement == DisagreementReport.from_dict(section)

    def test_the_section_is_optional(self) -> None:
        payload = record(structural_disagreement=None).to_dict()
        assert payload["structural_disagreement"] is None
        validate_record(payload)

    def test_trace_files_are_referenced_by_digest(self, tmp_path: Path) -> None:
        found = trace({"x": dirichlet(20), "y": dirichlet(21), "z": dirichlet(22)})
        digest = found.write(tmp_path / "h.json.gz")
        with_file = report({"h": {"file": "h.json.gz", "sha256": digest}})
        entry = with_file.to_dict()["households"]["h"]["trace"]
        assert entry == {"file": "h.json.gz", "sha256": digest}
        assert DisagreementTrace.read(
            tmp_path / entry["file"], entry["sha256"]
        ).summary() == (found.summary())

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda s: s.update(reference="w"), "reference"),
            (lambda s: s.update(distance="euclidean"), "distance"),
            (
                lambda s: s["specifications"][0]["assumptions"][0].update(support=[]),
                "without published support",
            ),
            (
                lambda s: s["households"]["h"]["summary"]["mean_pairwise"].update(
                    max=1.5
                ),
                "outside",
            ),
            (
                lambda s: s["households"]["h"]["summary"]["pairs"].pop("x|y"),
                "every pair",
            ),
            (
                lambda s: s["households"]["h"].update(
                    trace={"file": "t", "sha256": "0"}
                ),
                "SHA-256",
            ),
            (lambda s: s.update(extra=1), "exactly"),
        ],
    )
    def test_an_invalid_section_is_refused(self, change: Any, message: str) -> None:
        payload = record().to_dict()
        change(payload["structural_disagreement"])
        assert any(
            message in p
            for p in validate_disagreement(payload["structural_disagreement"])
        )
        with pytest.raises(
            ArtifactError, match="structural_disagreement|reference|distance"
        ):
            validate_record(payload)

    def test_a_1_3_record_migrates_without_the_section(self, tmp_path: Path) -> None:
        payload = record(structural_disagreement=None).to_dict()
        payload["schema_version"] = "1.3"
        del payload["structural_disagreement"]
        path = tmp_path / "old.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        migrated = load_record(path)
        assert migrated["migrated_from"] == "1.3"
        assert migrated["structural_disagreement"] is None


# ----------------------------------------------------------------------------
# Supported specifications
# ----------------------------------------------------------------------------
class TestCatalogue:
    def test_every_assumption_cites_a_published_finding(self) -> None:
        records: dict[str, Any] = {}
        for assumption in CATALOGUE.values():
            assert assumption.support
            for support in assumption.support:
                if support.record not in records:
                    records[support.record] = load_record(ROOT / support.record)[
                        "results"
                    ]
                results = records[support.record]
                if "questions" in results:  # the negative-binomial evaluation
                    assert results["questions"]["R"]["decision"] == support.finding
                else:
                    found = {e["key"]: e["conclusion"] for e in results["estimands"]}
                    assert found[support.estimand] == support.finding

    @pytest.mark.parametrize(("axis", "variant"), sorted(REFUSED))
    def test_refused_variants_say_why(self, axis: str, variant: str) -> None:
        if axis not in ("observation", "parameters", "parameter_sample"):
            assert REFUSED[(axis, variant)]
            return
        with pytest.raises(ValueError, match="not a supported variant"):
            Specification(**{axis: variant})

    def test_unknown_variants_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not in the catalogue"):
            Specification(parameter_sample="half_c")

    def test_the_default_ensemble_varies_one_assumption_at_a_time(self) -> None:
        check_ensemble(DEFAULT_ENSEMBLE)
        for specification in DEFAULT_ENSEMBLE:
            if specification != REFERENCE:
                assert len(specification.differs(REFERENCE)) == 1

    @pytest.mark.parametrize(
        ("ensemble", "message"),
        [
            ((Specification(observation=HURDLE_NB),), "reference"),
            ((REFERENCE, REFERENCE), "must differ"),
            ((REFERENCE,), "not dozens"),
        ],
    )
    def test_invalid_ensembles_are_refused(
        self, ensemble: tuple[Specification, ...], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            check_ensemble(ensemble)

    def test_an_ensemble_is_capped(self) -> None:
        many = [
            Specification(o, p, s)
            for o in (HURDLE, HURDLE_NB)
            for p in ("population", "pooled")
            for s in ("all", "half_a", "half_b")
        ]
        assert len(many) > MAX_SPECIFICATIONS
        with pytest.raises(ValueError, match="not dozens"):
            check_ensemble(many)

    def test_the_halves_split_the_training_households(self) -> None:
        split = halves(["h3", "h1", "h4", "h2", "h5"])
        assert split == {"half_a": ("h1", "h3", "h5"), "half_b": ("h2", "h4")}
        assert set(split["half_a"]).isdisjoint(split["half_b"])


# ----------------------------------------------------------------------------
# On simulated households
# ----------------------------------------------------------------------------
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


@pytest.fixture(scope="module")
def homes() -> dict[str, CasasRecording]:
    return {f"sim{s}": simulated(s) for s in range(1, 6)}


@pytest.fixture(scope="module")
def samples(homes: dict[str, CasasRecording]) -> Any:
    training = ["sim1", "sim2", "sim3", "sim4"]
    statistics = {
        h: home_statistics(homes[h], RESOLUTION, ONTOLOGY, household=h)
        for h in training
    }
    return fit_samples(
        statistics, training, states=tuple(ONTOLOGY.states), pseudo_windows=12.0
    )


def cutoff(recording: CasasRecording) -> Any:
    return min(o.timestamp for o in recording.observations) + timedelta(days=1)


class TestOnHouseholds:
    def test_the_decision_is_the_deployed_recursion(
        self, homes: dict[str, CasasRecording], samples: Any
    ) -> None:
        recording = homes["sim5"]
        found = household_trace(
            recording,
            DEFAULT_ENSEMBLE,
            samples,
            household="sim5",
            resolution=RESOLUTION,
            ontology=ONTOLOGY,
            until=cutoff(recording),
        )
        counts, rows, _, moments = household_channel_counts(
            recording, RESOLUTION, ONTOLOGY, household="sim5"
        )
        models = samples["all"].models(recording.registry, RESOLUTION, HURDLE, ONTOLOGY)
        _, deployed = filter_recursion(
            total_loglik(models, counts),
            ONTOLOGY.transition(RESOLUTION.step),
            ONTOLOGY.stationary(),
        )
        after = [r for r in rows if moments[int(r)] > cutoff(recording)]
        reference = found.models.index(REFERENCE.name)
        np.testing.assert_array_equal(found.posteriors[:, reference], deployed[after])
        np.testing.assert_array_equal(found.decision, deployed[after].argmax(axis=1))

    def test_supported_variants_disagree_and_the_trace_is_deterministic(
        self, homes: dict[str, CasasRecording], samples: Any, tmp_path: Path
    ) -> None:
        recording = homes["sim5"]

        def run() -> DisagreementTrace:
            return household_trace(
                recording,
                DEFAULT_ENSEMBLE,
                samples,
                household="sim5",
                resolution=RESOLUTION,
                ontology=ONTOLOGY,
                until=cutoff(recording),
            )

        first, second = run(), run()
        assert first.write(tmp_path / "a.gz") == second.write(tmp_path / "b.gz")
        pairs = first.summary()["pairs"]
        assert (
            pairs[f"{REFERENCE.name}|hurdle_nb/population/all"]["mean_divergence"] > 0.0
        )
        for window in first.timestamps:
            assert window > cutoff(recording).isoformat()[:10]

    def test_pooled_specifications_need_a_cut_off(
        self, homes: dict[str, CasasRecording], samples: Any
    ) -> None:
        with pytest.raises(ValueError, match="cut-off"):
            household_trace(
                homes["sim5"],
                DEFAULT_ENSEMBLE,
                samples,
                household="sim5",
                resolution=RESOLUTION,
                ontology=ONTOLOGY,
            )

    def test_a_training_household_is_refused(
        self, homes: dict[str, CasasRecording], samples: Any
    ) -> None:
        with pytest.raises(ValueError, match="used to fit the population"):
            household_trace(
                homes["sim1"],
                (REFERENCE, Specification(observation=HURDLE_NB)),
                samples,
                household="sim1",
                resolution=RESOLUTION,
                ontology=ONTOLOGY,
            )

    def test_the_report_enters_a_record(
        self, homes: dict[str, CasasRecording], samples: Any, tmp_path: Path
    ) -> None:
        recording = homes["sim5"]
        found = household_trace(
            recording,
            DEFAULT_ENSEMBLE,
            samples,
            household="sim5",
            resolution=RESOLUTION,
            ontology=ONTOLOGY,
            until=cutoff(recording),
        )
        digest = found.write(tmp_path / "sim5.json.gz")
        section = ensemble_report(
            [found],
            DEFAULT_ENSEMBLE,
            {"sim5": {"file": "sim5.json.gz", "sha256": digest}},
        )
        path = record(structural_disagreement=section).write(tmp_path / "r.json")
        payload = load_record(path)
        names = [
            s["name"] for s in payload["structural_disagreement"]["specifications"]
        ]
        assert names == sorted(s.name for s in DEFAULT_ENSEMBLE)
        assert payload["structural_disagreement"]["reference"] == REFERENCE.name
