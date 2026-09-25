# Matched evaluation

A difference between two models is evidence about modelling only when both
received the same information. `run_matched_evaluation` fits and scores several
models under one [information set](INFORMATION_SETS.md). It controls what each
model sees, so a matched comparison is the default and a mismatched one is
hard to produce by accident.

## What the runner guarantees

| Guarantee | How it is enforced |
| --- | --- |
| Every model gets the same information | The runner builds the feature rows from one `InformationSet` and gives every model identical arrays. Models never see a recording, a timestamp or, at prediction time, a household identifier. |
| Held-out households never reach fitting | `fit` receives only training rows and, optionally, development rows. Held-out rows are passed to `predict` without labels. |
| Row order carries no history | Rows are presented in seeded random order. |
| Each prediction uses only its own row | Held-out rows are predicted a second time as a random half in a new order. A model whose prediction for any row changes is refused. |
| Statistics are household-level | Each held-out household is scored on its own. Models are compared by paired differences across households, never by pooling observations. |

## Plugging in a model

A model needs two methods:

- `fit(training, development)` receives `LabelledRows`: feature values, labels,
  the household of each row, and the label space `states`. `development` is
  `None` unless the split declares development households.
- `predict(rows)` receives `FeatureRows` and returns `StatePredictions`: one
  state per row, and optionally a probability matrix whose columns follow
  `rows.states`. `UNKNOWN` is an abstention and is never correct.

`predict` must be a deterministic function of each row. Log loss, Brier score
and calibration are reported only for models that return probabilities.

Wrap each model in a `ModelSpec`. The spec gives the model a name, a function
that builds a fresh model from the run's seed, and a JSON-serialisable
`configuration` that is recorded as provenance.

## Example

This example compares a class-prior baseline with a shallow decision tree on
six simulated homes. It needs no downloaded data. The models are placeholders
that show the plug-in interface; they are not proposals.

```python
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from sensor_modeling.datasets import (
    ActivityInterval,
    CasasRecording,
    HouseholdSplit,
    ModelSpec,
    StatePredictions,
    nested_information_sets,
    run_matched_evaluation,
)
from sensor_modeling.simulation import HouseholdConfig, simulate


def household(seed):
    """A simulated home, packaged like an annotated recording."""
    sim = simulate(HouseholdConfig(days=3, seed=seed))
    labels = tuple(
        ActivityInterval(e.state.value, e.start, e.end, e.state)
        for e in sim.truth.episodes
    )
    return CasasRecording(sim.registry, sim.observations, labels)


class Sklearn:
    """Adapts a scikit-learn classifier to the runner's model interface."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, training, development):
        self.estimator.fit(training.values, [s.value for s in training.labels])

    def predict(self, rows):
        known = list(self.estimator.classes_)
        scores = self.estimator.predict_proba(rows.values)
        probabilities = np.zeros((len(rows), len(rows.states)))
        for column, state in enumerate(rows.states):
            if state.value in known:
                probabilities[:, column] = scores[:, known.index(state.value)]
        labels = tuple(rows.states[i] for i in probabilities.argmax(axis=1))
        return StatePredictions(labels, probabilities)


result = run_matched_evaluation(
    {f"sim{seed}": household(seed) for seed in range(1, 7)},
    split=HouseholdSplit(
        "example", train=("sim1", "sim2", "sim3"), test=("sim4", "sim5", "sim6")
    ),
    information_set=nested_information_sets()[-1],
    models=[
        ModelSpec("prior", lambda seed: Sklearn(DummyClassifier(strategy="prior"))),
        ModelSpec(
            "tree",  # scikit-learn trees accept NaN from version 1.3
            lambda seed: Sklearn(DecisionTreeClassifier(max_depth=6, random_state=seed)),
            {"estimator": "DecisionTreeClassifier", "max_depth": 6},
        ),
    ],
    seed=0,
    data_source="simulator",
    output_dir=Path("results"),
)
for comparison in result.comparisons:
    d = comparison.difference
    print(comparison.metric, f"{d.mean_difference:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]")
```

The record is written to `results/matched_evaluation.json`.

## The record

The output is an `ExperimentRecord`, the same format as every other experiment
in this package. It contains:

| Field | Contents |
| --- | --- |
| `environment` | Git commit and dirty flag, package and library versions |
| `recorded_at`, `seeds`, `data_source` | When the run happened, its seed, and where the recordings came from |
| `configuration.information_set` | The full declaration and its `sha256` |
| `configuration.split` | Every household's role and the split's `sha256` |
| `configuration.models` | Each model's name, builder and configuration |
| `metric_definitions` | A written definition of every reported quantity |
| `results.households` | Per household: role, number of moments, number labelled, a digest of the moments, and uninstrumented channels |
| `results.models.<name>.households` | Per held-out household: balanced accuracy, per-state recall, confusion matrix, and log loss, Brier score and calibration where probabilities exist |
| `results.models.<name>.summary` | Median, mean, spread and range across households, each household counted once |
| `results.comparisons` | Paired household-level differences for every pair of models on every requested metric |

Each comparison sets `model` to the later-declared model and `reference` to the
earlier one, so declare the reference model first. The difference is oriented
so that a positive value favours `model`. The interval comes from a bootstrap
over households. A comparison that cannot be made records why in `skipped`,
for example when a model reported no probabilities or fewer than two held-out
households were scored. `confusion_summed` is descriptive only.

## Held-out households and tuning

Nothing about a held-out household reaches `fit`. Choose hyperparameters with
the development households or with household-grouped validation over the
training rows. For a development-phase experiment, use a split whose held-out
households are development homes; the frozen external cohort is reserved for
final claims.

## What the runner does not do

- **It does not run the generative filter.** The filter is recursive and reads
  raw observations, so it cannot be restricted to a declared information set.
- **It does not compare across information sets.** Each run uses exactly one
  set. Estimating an information gap means comparing runs over nested sets.
  The recorded information-set, split and moment digests make it possible to
  check that two runs used the same split and scored the same positions.
