"""Tests for the fixed protocol of the Phase 1 matched-baseline run.

The run itself needs the CASAS archive and is not part of the test suite. Its
choice of homes and folds needs no data and is checked here.
"""

from __future__ import annotations

import pytest

from scripts.run_phase1_matched_baselines import development_homes, folds


def test_the_twenty_single_resident_development_homes_are_used() -> None:
    homes = development_homes()
    assert len(homes) == 20
    assert not {"hh107", "hh121"} & set(homes)
    assert all(len(entry["sha256"]) == 64 for entry in homes.values())


def test_every_home_is_held_out_exactly_once() -> None:
    homes = sorted(development_homes())
    first, second = folds(homes)
    assert set(first.test) | set(second.test) == set(homes)
    assert not set(first.test) & set(second.test)
    assert first.train == second.test and second.train == first.test
    assert len(first.test) == len(second.test) == 10


def test_the_folds_do_not_depend_on_input_order() -> None:
    homes = list(development_homes())
    assert folds(homes) == folds(list(reversed(homes)))


@pytest.mark.parametrize("home", ["hh101", "hh130"])
def test_folds_alternate_sorted_homes(home: str) -> None:
    first, second = folds(sorted(development_homes()))
    assert (home in first.test) == (sorted(development_homes()).index(home) % 2 == 0)
