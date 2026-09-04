# v0.3 external-validation candidate specification

## Status

**Scored.** The primary external validation was executed once on 2026-09-04 and its
outcomes have been inspected. Everything below the Status heading is the
specification as frozen *before* that run, and is retained unchanged as the
pre-registration record; nothing in it was revised in response to the outcome.

Result: median paired difference in household balanced accuracy $+0.0091$, 95%
interval $[+0.0054, +0.0117]$, 37 of 43 homes improved. Recorded in
`artifacts/v03/external_primary_result.json` with the one-shot marker in
`artifacts/v03/external_primary_scored.json`.

This document selects the model form that will be frozen for the one-shot external validation defined in `papers/failure-aware-multimodal-behavioural-sensing/EXTERNAL_VALIDATION_CONTRACT.md`. It uses only evidence already obtained from the 22-home CASAS `hh` development panel.

## Candidate

The v0.3 external-validation candidate is:

> **v0.2 inference + the optional circadian state-dynamics profile introduced in PR #102, with all other v0.2 inference choices retained.**

The candidate therefore changes only the time dependence of the behavioural CTMC prior. It does not change the seven-state ontology, sensor/activity mapping rules, emission family, default emission-rate structure, attribution logic, health-reliability logic, abstention thresholds, dwell-time defaults, baseline logic, or alert logic.

The circadian mechanism is the implementation already merged into `StateOntology`: state-specific 24-hour stickiness multipliers modify exit rates by local hour. The mechanism remains interpretable as a time-inhomogeneous prior rather than a direct discriminative clock feature.

## Why this candidate is selected

The selection criterion is the primary external-validation metric, household balanced accuracy, evaluated only on development data.

The development evidence available before this specification is:

| Candidate component | Development result | Decision |
| --- | --- | --- |
| Circadian prior | BA 0.449 -> 0.460 on the 11-home held-out development split; improved 10/11 homes; ECE 0.312 -> 0.296 | **retain** |
| Fitted emission rates | BA 0.449 -> 0.418; ECE 0.312 -> 0.202 | do not include in the primary v0.3 candidate |
| Fitted dwell times | BA 0.449 -> 0.429; improved 4/11 homes | reject |
| Lagged/raw recent-history evidence | discriminative ablation shows signal, but recursive reuse would double-count observations already represented in the posterior unless the state-space formulation is changed | unresolved; exclude from v0.3 |

The fitted-rate result is scientifically useful for calibration, but it worsens the pre-specified primary metric in isolation. A new circadian-plus-rate combination is not introduced merely to search for a better development result after seeing component outcomes. That combination may be studied later as a separate model version.

## What remains unchanged

For the primary external comparison, v0.2 and v0.3 must use the same:

- seven latent behavioural states and `UNKNOWN` abstention semantics;
- real-data adapter and annotation interpretation;
- sensor-to-room and activity-to-state mapping rules;
- evaluation grid and household-level scoring definitions;
- emission likelihood families and v0.2 emission defaults;
- declared dwell times;
- occupancy/person-attribution model;
- health-reliability weighting;
- fusion/abstention thresholds;
- external-test cohort and exclusions frozen independently of outcomes.

Only the circadian profile differs.

## Evaluation grid

The list above requires v0.2 and v0.3 to share an evaluation grid, which makes
the paired contrast internally valid whatever the grid is. It does not record
what the grid *is*, so the absolute per-home figures were not reproducible from
the specification alone. Recorded here before any primary scoring:

| | Value |
| --- | --- |
| Inference step | 5 minutes |
| Local timezone | `America/Los_Angeles` |

These are the values every development result used, including the held-out
comparison that selected the candidate. They are not a new choice; they are the
existing one written down.

The step size and timezone both matter beyond bookkeeping. The step sets how
many points each home contributes and how much evidence each carries, and the
timezone determines which local hour the circadian profile is indexed by — a
displaced zone would silently misalign the very mechanism under test.

`scripts/score_v03_external.py` refuses any other grid on the primary path, so
the recorded values are enforced rather than merely documented.

## Development fitting of the final profile

The previously analysed development panel contains 22 `hh` recordings. CASAS metadata identify `hh107` and `hh121` as two-resident homes. They remain permanently development-only because their outcomes were already inspected, but they are not eligible to estimate the final single-resident circadian profile.

Before any primary test home is scored, one final circadian profile may therefore be estimated using the **20 single-resident homes inside the already-inspected 22-home development panel**. No untouched home may be added merely to keep the fitting count at 22.

This correction is based on resident-count metadata, not model performance. The earlier all-22 development fit is retained only as a reproducibility diagnostic and is not the immutable candidate profile.

The fitting procedure must be the same family used for the development experiment documented in PR #102: a per-state, per-local-hour profile derived from labelled occupancy frequencies/lift and converted to strictly positive stickiness multipliers accepted by `StateOntology`.

The fitting step may not inspect any primary test-home labels, sensor outcomes, prediction metrics, class recalls, or calibration results.

The fitted profile must be committed in a machine-readable file before scoring, together with:

1. the exact 22 development-panel identifiers;
2. the exact 20 single-resident identifiers used for parameter fitting and the two metadata-based exclusions;
3. the fitting code revision;
4. the profile SHA-256;
5. a declaration that no primary test-home outcome was inspected;
6. the exact v0.3 candidate Git revision that will be scored.

After that commit, the profile is immutable for the primary external test.

## No model-selection loop on the primary cohort

The primary cohort is scored once with frozen v0.2 and frozen v0.3. Its result may be positive, null, or negative. No circadian multiplier, state mapping, emission parameter, dwell time, threshold, or cohort membership may be changed in response to the primary result and then rescored as the same confirmatory external validation.

Any subsequent modification defines a new exploratory or future model version.

## Primary estimand

For eligible test home `h`, the paired difference is

```text
D_h = BA_h(v0.3) - BA_h(v0.2)
```

The primary summary is the **median** paired household difference with the interval procedure frozen in the external-validation contract. Time points are observations, not independent replications.

Secondary outcomes remain calibration error, log loss, Brier score, abstention rate, labelled/scored coverage, and state-specific recall. They do not replace the primary balanced-accuracy estimand.

## Interpretation boundary

A successful external result would establish transfer to the frozen CASAS test cohort under its available motion/door instrumentation. It would not establish clinical efficacy, general smart-home performance, or equivalence to the richer multimodal simulator deployment.

A negative result remains scientifically valid and must be reported without retuning the candidate on the test cohort.
