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
from sensor_modeling.fusion import ONLINE, EvidenceSummary

record = ExperimentRecord(
    experiment="phase1-current-fold-a",
    configuration={"metrics": ["balanced_accuracy"]},
    inference=ONLINE,
    evidence=EvidenceSummary.online(scored_moments),
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
| `schema_version` | `MAJOR.MINOR`; currently `1.4` | yes |
| `experiment` | experiment identifier | yes |
| `recorded_at` | execution timestamp, ISO 8601 with time zone, fixed when the record is created | yes |
| `environment` | `git_commit`, `git_dirty`, `sensor_modeling` (the package version), `python`, and library versions including NumPy, SciPy, pandas and scikit-learn | yes |
| `data_source` | dataset identifier, such as `casas-hh` or `simulator` | yes |
| `inference` | the inference regime every estimate comes from: `mode` (`online_filter` or `fixed_lag_smoother`), `lag_steps`, `step_seconds`, `label` and `provenance` (1.2); `causal`, `delay_seconds` and `evidence`, the prediction and latest-evidence timestamps (1.3); see below | 1.2 |
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
| `structural_disagreement` | how much the experiment's model specifications disagree, window by window: the specifications with their assumptions and published support, the reference, and each household's summary and trace file; `null` if the experiment has none. See [structural disagreement](STRUCTURAL_DISAGREEMENT.md) | 1.4 |
| `notes` | anything a reader needs in order not to over-read the result | yes |
| `migrated_from` | the version the record was read from, when it was migrated | only after migration |

Fields marked 1.1 appear in every record written at 1.1. A 1.0 file is given
their empty defaults when it is loaded.

### The inference regime

`inference` says which [inference regime](INFERENCE_REGIMES.md) every estimate
in the record comes from.

A record that compares regimes, such as the
[Phase 3.5 smoothing evaluation](PHASE3_SMOOTHING.md), states the one with the
longest lag instead. That regime bounds every estimate in the record: none
reads further ahead. Its results then label every cell and comparison with its
own regime and evidence summary. Every comparison between regimes is a
labelled smoothing gain.

- **Mode and lag.** The mode is the online filter or a fixed-lag smoother. A
  smoother also records its lag in windows and the window width, so its
  reporting delay is stated in time.
- **Label.** The label is derived from the mode and lag. A record whose label
  does not describe its regime is refused.
- **Required.** `ExperimentRecord` takes `inference` as a required keyword, so
  no experiment can leave it to a default.
- **Provenance.** It says how the regime is known: declared by the writer, or
  attested when an older record was migrated.
- **Causality and delay.** `causal` says whether any estimate reads evidence
  after its moment. `delay_seconds` says how long after its moment an estimate
  can first be reported. Both follow from the regime, and a record in which
  either disagrees with it is refused.
- **Evidence timestamps.** `evidence` summarises the scored estimates. It gives
  `predictions`, `first_prediction`, `last_prediction`, `latest_evidence` and
  `max_lead_seconds`, the most future information any of them used.
  - **Beyond the regime.** A record whose largest lead exceeds its regime's
    delay is refused, so smoothed estimates cannot be recorded as online.
  - **Not listed.** A causal record that cannot list its timestamps gives
    `{"enumerated": false, "reason": ...}` instead. A smoother cannot: it
    must record the future information it used.
  - **Required.** `ExperimentRecord` takes `evidence` as a required keyword.

The [inference regimes](INFERENCE_REGIMES.md#what-each-record-states) page has
an example.

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

| File version | This reader (1.4) |
| --- | --- |
| `1.0` | migrated on load: new fields given defaults, non-finite numbers made `null`, then as `1.1` with `migrated_from: "1.0"`, content otherwise unchanged |
| `1.1` | migrated on load: `inference` added as the online filter, attested on migration, with its causality, its delay and its evidence marked not listed, and `migrated_from: "1.1"`, content otherwise unchanged |
| `1.2` | migrated on load: causality and delay derived from the regime, evidence marked not listed, and `migrated_from: "1.2"`, content otherwise unchanged. A smoothed 1.2 record is refused |
| `1.3` | migrated on load: `structural_disagreement` added as `null`, and `migrated_from: "1.3"`, content otherwise unchanged |
| `1.4` | read as written |
| `1.5` or later | refused: written by a newer version of the package |
| `0.x`, `2.x` | refused: another major version |

Files written at 1.0 before `data_source` existed are migrated to `simulator`
if they carry the simulator note, which every such simulator record did, and
to `unknown` otherwise.

1.0 and 1.1 had no inference regime. Every writer of them in this repository ran
online inference: the online pipeline, or matched information sets whose
windows close at or before each prediction moment. The repository's history
shows that no experiment code ever called the fixed-lag smoother. A migrated
record therefore says `online_filter`. Its provenance states that this was
attested on migration, not declared by the writer.

1.0 to 1.2 did not record prediction or evidence timestamps. A migrated record
marks its evidence as not listed, with the reason:

> written at schema 1.1, before prediction and evidence timestamps were
> recorded; its estimates read no evidence after their prediction moments

This is allowed only for a causal regime. A 1.2 record of a smoother would be
refused rather than migrated, since a smoother must list the future
information it used. None was published.

The published Phase 1 and Phase 3.1 to 3.3 records are 1.1 files, and the Phase
3.4 record is a 1.2 file. Each is migrated this way when read, and its
generated summary is unchanged. The files themselves are not changed.

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
- `inference.evidence`, with the held-out households' labelled moments, each
  reading evidence up to itself;
- `inputs`, when the caller passes the digests of the files it read.

Its `results` hold the household summaries and the full comparisons, in the
layout named `matched-evaluation/3`.

## What this does not cover

The frozen v0.3 validation files in `artifacts/v03/` predate this schema and
keep their own frozen format. They are not experiment records and are not
migrated.
