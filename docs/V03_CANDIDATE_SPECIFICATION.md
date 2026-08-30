# v0.3 external-validation candidate specification

## Status

**Prospective candidate specification — primary external test outcomes remain uninspected.**

This document selects the model form that will be frozen for the one-shot external validation defined in `docs/EXTERNAL_VALIDATION_CONTRACT.md`. It uses only evidence already obtained from the 22-home CASAS `hh` development panel.

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

## Development fitting of the final profile

Before any primary test home is scored, one final circadian profile may be estimated using **all 22 development-only `hh` homes**. This is model fitting, not external evaluation.

The fitting procedure must be the same family used for the development experiment documented in PR #102: a per-state, per-local-hour profile derived from labelled occupancy frequencies/lift and converted to strictly positive stickiness multipliers accepted by `StateOntology`.

The fitting step may not inspect any primary test-home labels, sensor outcomes, prediction metrics, class recalls, or calibration results.

The fitted profile must be committed in a machine-readable file before scoring, together with:

1. the exact 22 development-home identifiers;
2. the fitting code revision;
3. the profile SHA-256;
4. a declaration that no primary test-home outcome was inspected;
5. the exact v0.3 candidate Git revision that will be scored.

After that commit, the profile is immutable for the primary external test.

## No model-selection loop on the primary cohort

The primary cohort is scored once with frozen v0.2 and frozen v0.3. Its result may be positive, null, or negative. No circadian multiplier, state mapping, emission parameter, dwell time, threshold, or cohort membership may be changed in response to the primary result and then rescored as the same confirmatory external validation.

Any subsequent modification defines a new exploratory or future model version.

## Primary estimand

For eligible test home `h`,

\[
D_h = BA_h^{(v0.3)} - BA_h^{(v0.2)}.
\]

The primary summary is the mean paired household difference with the interval procedure frozen in the external-validation contract. Time points are observations, not independent replications.

Secondary outcomes remain calibration error, log loss, Brier score, abstention rate, labelled/scored coverage, and state-specific recall. They do not replace the primary balanced-accuracy estimand.

## Interpretation boundary

A successful external result would establish transfer to the frozen CASAS test cohort under its available motion/door instrumentation. It would not establish clinical efficacy, general smart-home performance, or equivalence to the richer multimodal simulator deployment.

A negative result remains scientifically valid and must be reported without retuning the candidate on the test cohort.
