# Experiment artifacts

Every experiment in this package writes one JSON file, an `ExperimentRecord`.
The file has to answer, months later and without the code, what was run, on
which data, with which settings and code, and what came out. A paper should be
able to quote from it directly.

This is a small artifact layer for this repository, not an
experiment-tracking platform. Records are plain files. By default they go to
`results/`, which is not under version control, because results are
regenerated from their seeds.

## Saving, loading and validating

```python
from pathlib import Path

from sensor_modeling.evaluation import ExperimentRecord, InputArtifact, load_record

record = ExperimentRecord(
    experiment="phase1-current-fold-a",
    configuration={"metrics": ["balanced_accuracy"]},
    seeds=[0],
    data_source="casas-hh",
    inputs=[InputArtifact("hh101.csv", "3f4d...", "recording", "zenodo:15708568")],
)
path = record.write(Path("results/phase1-current-fold-a.json"))

payload = load_record(path)          # a dict at the current version
again = ExperimentRecord.load(path)  # the typed record
```

- **`write()`** validates the record and writes it. An invalid record is never
  written.
- **`load_record()`** reads a file, migrates an older schema version, validates
  it and returns the payload.
- **`validate_record()`** checks a payload and raises `ArtifactError` listing
  every problem it finds, not only the first.

## Fields

| Field | Contents | Required |
| --- | --- | --- |
| `schema_version` | `MAJOR.MINOR`; currently `1.1` | yes |
| `experiment` | experiment identifier | yes |
| `recorded_at` | execution timestamp, ISO 8601 with time zone, fixed when the record is created | yes |
| `environment` | `git_commit`, `git_dirty`, `sensor_modeling` (the package version), `python`, and library versions including NumPy, SciPy, pandas and scikit-learn | yes |
| `data_source` | dataset identifier, such as `casas-hh` or `simulator` | yes |
| `inputs` | input files, each with `name`, `sha256`, `role` and `source` | 1.1 |
| `split` | household split, with its SHA-256 | 1.1 |
| `information_set` | information-set declaration, with its SHA-256 | 1.1 |
| `preprocessing` | how raw data became the scored rows | 1.1 |
| `models` | each model's `name`, `build` and `configuration` (hyperparameters) | 1.1 |
| `configuration` | every other setting that affects the outcome | yes |
| `resolved_defaults` | the library defaults in force when the record was made | yes |
| `seeds` | random seeds | yes |
| `sensor_subset` | sensors used, when the experiment varies them | yes (may be null) |
| `metric_definitions` | a written definition of every reported quantity | yes |
| `results` | aggregate results. Their layout belongs to the experiment, which names it in `configuration.result_schema` | yes |
| `household_metrics` | metrics per model, then per household | 1.1 |
| `intervals` | estimates ready to quote: `label`, `estimate`, `low`, `high`, `confidence`, `method`, `unit`, `n` | 1.1 |
| `mcse` | Monte Carlo standard errors of simulation summaries, by name | 1.1 |
| `notes` | anything a reader needs in order not to over-read the result | yes |
| `migrated_from` | the version the record was read from, when it was migrated | only after migration |

Fields marked 1.1 appear in every record written at 1.1. A 1.0 file is given
their empty defaults when it is loaded.

An interval's `unit` says what was resampled: `household` for real homes, or
`seed` for simulated trajectories. It is never timestamps; see
[Evaluation design](EVALUATION_DESIGN.md#households-not-timestamps-are-the-unit).

## Strict and deterministic

- **Canonical text.** Sorted keys, two-space indentation, UTF-8, LF line
  endings on every platform, and one trailing newline. A digest of the file
  does not depend on the machine that wrote it.
- **Fixed at creation.** The timestamp and environment are captured when the
  record is created, not when it is written. Writing the same record twice
  gives identical bytes, and so does loading and rewriting it.
- **No NaN or infinity.** JSON has neither, so every non-finite number is
  written as `null`, meaning no finite value exists. A current-version file
  that contains `NaN` or `Infinity` is refused on load.
- **Explicit conversions.** NumPy values become plain numbers or lists, enums
  their value, dates ISO 8601 text, and time deltas, time zones and paths the
  text earlier records used. Any other object is refused with its location,
  instead of being written as an opaque string.

## Versions

The version is `MAJOR.MINOR`.

- **Minor.** A minor version only adds optional fields. A reader loading an
  older minor fills them with their defaults.
- **Major.** A major version would change what an existing field means, and
  would need a migration written for it.
- **Newer files.** A reader refuses a newer minor, and any other major, rather
  than guessing at fields it does not know.

| File version | This reader (1.1) |
| --- | --- |
| `1.0` | migrated on load: new fields given defaults, non-finite numbers made `null`, `migrated_from: "1.0"` added, content otherwise unchanged |
| `1.1` | read as written |
| `1.2` or later | refused: written by a newer version of the package |
| `0.x`, `2.x` | refused: another major version |

Files written at 1.0 before `data_source` existed are migrated to `simulator`
if they carry the simulator note, which every such simulator record did, and
to `unknown` otherwise.

The writer always writes the current version. The migration was checked on
real 1.0 files: the n=100 ablation and attribution studies, and all nine
records of the Phase 1 run. Each loaded with its results unchanged, and each
rewrote byte-identically after migration.

## An example: the matched evaluation runner

`run_matched_evaluation` fills the typed fields:

- `split`, `information_set` and `models`, each as declared;
- `preprocessing`, recording how moments were chosen and labelled;
- `household_metrics`, with every held-out household's metrics per model;
- `intervals`, with every paired household comparison flattened to its mean and
  median estimates, `unit: "household"`;
- `inputs`, when the caller passes the digests of the files it read.

Its `results` hold the household summaries and the full comparisons, in the
layout named `matched-evaluation/3`.

## What this does not cover

The frozen v0.3 validation files in `artifacts/v03/` predate this schema and
keep their own frozen format. They are not experiment records and are not
migrated.
