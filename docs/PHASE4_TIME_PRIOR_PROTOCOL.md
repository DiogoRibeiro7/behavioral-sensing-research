# Phase 4 for the time prior: uncertainty diagnostics against confidence: the protocol

Generated entirely from the frozen file `artifacts/phase4/prior_uncertainty_protocol.json` by `sensor_modeling.datasets.prior_uncertainty_summary.render_protocol`.

**Status: pre-specified, on the development panel, which earlier work has inspected; one run; not a held-out claim.**

- **Protocol digest.** `c71a08021a7ea434ee8ef9cacc90e4c87c44e9f3565d9796b648cbb13a342865`.
- **The question.** Does any structural diagnostic produce a materially better selective-risk curve than confidence for the fitted channels with the time prior, without disproportionately rejecting difficult minority states?
- **This page reports no result.**

## What had been seen

- The Phase 4 record, for the hurdle recursion without the time prior: structural_disagreement is materially better than confidence, its AURC lower by +0.059 [+0.031, +0.087]; entropy is negligibly different, +0.002 [+0.001, +0.003]; evidence_disagreement, −0.149 [−0.183, −0.110], and predictive_mismatch, −0.040 [−0.060, −0.020], are worse; the minority-state guard covers bed_awake alone, 68 windows in 5 households; structural_disagreement retains less than confidence of bathroom_activity and kitchen_activity at coverages 0.5 and 0.7; no threshold was selected.
- The combined-prior record: in the recursion, the fitted hurdle channels with the time prior score a mean household balanced accuracy of 0.517 against 0.456 without the prior, K1 +0.061 [+0.044, +0.077], with calibration error 0.213 against 0.284.
- The held-out record: the same formulation, fitted once on the 20 development homes, scored once on 43 held-out CASAS homes, with every home's scores published.
- The activity sub-state record: two activity sub-states were inconclusive against this reference, which stays the reference.
- What the reference's predictions already show: the combined-prior record publishes every household's confusion matrix, so the pooled table the difficult-minority rule reads is known. bed_awake is the only state below the minority share whose recall, 0.044 (3 of 68 windows), is below the pooled balanced accuracy, 0.473; bathroom_activity's recall, 0.487, is above it, and kitchen_activity's, 0.547. The guard will cover bed_awake alone, in 5 households.
- Confidence on this recursion: its calibration error in every household is published by the combined-prior, held-out and activity sub-state records. Entropy, structural_disagreement and evidence_disagreement had not been computed on it for any CASAS household before this protocol was frozen; predictive_mismatch reads no prediction, so its values are the Phase 4 record's. The code was exercised on simulated homes only.

## The base protocols

- **Phase 4.** `artifacts/phase4/uncertainty_protocol.json`, SHA-256 `537dd3d319e1ec3964d2d8df370cac3d05a72fd267a10eaa148b0148f59782e1`, protocol digest `78a378f47c8adb9b1d2233383cd211b433881e167dab19b94750054182985c91`. Used: its signals, directions and missing policies, its ensemble's specifications, the evidence groups' channels and floor, the coverage grid and inspection levels, the estimands and minimal differences, the bootstrap, the difficult-minority rule, the guard and the decision rule, unchanged.
- **Time prior.** `artifacts/phase3/combined_prior_protocol.json`, SHA-256 `3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5`, protocol digest `8570de52f23ef2a30c5d66876895888bad3dc422d67fc2266440ed3ddc471429`. Used: its fold fits of the channels and the time prior on training households only, which are the fitted-rates protocol's, and its recursion with the time prior.

## What changes

- **The predictions.** `filter_periodic_hurdle`: The combined-prior recursion: fitted hurdle-Poisson channels with the fold's population parameters and the time prior's transition at each window's hour, the population prior for every held-out household, fed every window of the recording from the prior's distribution at the first window's hour; the predicted state is the posterior's most probable, and every signal ranks these same predictions.
- **The ensemble.** `hurdle/population/all`, `hurdle_nb/population/all`, `hurdle/population/half_a`, `hurdle/population/half_b`: The default ensemble without its pooled specification, which adapts to the household's own labelled windows; a deployed signal has no labels, and a cut-off would change the scored windows of every signal; every specification runs the time prior's recursion, with the fold's prior fitted on all its training households, so only their channel models differ.
- **The transition.** Structural_models refuses the time prior as the recursion's transition because it had been evaluated only on declared information sets; the combined-prior protocol has since evaluated it in the recursion, a success, and it is the reference's transition here, shared by every specification rather than varied between them.
- **The evidence groups.** Motion: motion channels; contact: door channels; context: the time prior recursion's prediction before the window.

## Signals

| Signal | Direction | Missing values | Definition |
| --- | --- | --- | --- |
| `confidence` | `higher_is_safer` | `refuse` | the reference recursion's posterior probability of its predicted state |
| `entropy` | `higher_is_riskier` | `refuse` | the reference posterior's Shannon entropy divided by log of the number of states, in [0, 1] |
| `structural_disagreement` | `higher_is_riskier` | `refuse` | the normalised discordance of the ensemble's posteriors: their generalised Jensen-Shannon divergence in bits over log2 of the number of specifications, in [0, 1] |
| `evidence_disagreement` | `higher_is_riskier` | `reject_first` | the normalised discordance of the identifiable evidence groups of the reference recursion's window: motion (the motion channels), contact (the door channel) and context (the prediction before the window), each group's posterior its channels' fitted likelihoods normalised with no prior; undefined, and rejected first, when fewer than two groups are identifiable |
| `predictive_mismatch` | `higher_is_riskier` | `refuse` | the least standardised surprise over the states: min over s of (-log p(x \| s) minus its mean under s) over its standard deviation under s, with the reference's normalised hurdle count laws |

Every signal is compared with `confidence`; the structural diagnostics are `structural_disagreement`, `evidence_disagreement`, `predictive_mismatch`.

## Evaluation

- **Calibration bins.** 10.
- **Coverages.** 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.
- **Curve bootstrap.** Confidence: 0.95; resamples: 2000; seed: 0; unit: household.
- **Framework.** Sensor_modeling.evaluation.selective.
- **Inspection.** 0.5, 0.7, 0.9.
- **Loss.** 0-1: the selective risk is the error rate.
- **Selection.** Pooled over the panel for the pooled curves; within each household for the household curves; ties at the boundary retained in part.

## Estimands

- **At inspection.** At each inspection coverage, each household's selective balanced accuracy (signal minus confidence), error and calibration error (confidence minus signal).
- **Aurc.** Primary: each household's error AURC over the grid, confidence minus signal; positive favours the signal.
- **Balanced accuracy area.** Each household's selective balanced accuracy averaged over the grid by the trapezoid rule, signal minus confidence.
- **Pairing.** Households with a value under both signals; every household counts once.
- **State coverage.** At each inspection coverage, each household's retained share of each state it has, signal minus confidence: negative when the signal rejects the state more.

| Quantity | Minimal difference |
| --- | --- |
| `aurc` | 0.01 |
| `balanced_accuracy` | 0.02 |
| `balanced_accuracy_area` | 0.02 |
| `calibration_error` | 0.02 |
| `error` | 0.02 |
| `state_coverage` | 0.05 |

- **Verdicts.** Favours signal: mean at least the minimal difference and interval above 0; favours confidence: mean at most minus it and interval below 0; negligible: interval within plus or minus it; uncertain otherwise.
- **Difficult minority states.** A state is a difficult minority state when its share of the scored windows is below 0.1 and the reference's recall on it at full coverage is below the reference's balanced accuracy, both pooled over the scored windows; fixed from the reference's predictions, before any signal is compared.
- **Guard.** A signal disproportionately rejects difficult minority states when, for any such state at any inspection coverage, the paired comparison of its retained share favours confidence: it retains less of the state than confidence does, by at least the minimal difference, with the interval below 0.

## Decision

- **Better but rejects difficult minority states.** The AURC verdict favours the signal and the guard fails.
- **Key question.** Yes when any structural signal is materially better; no otherwise.
- **Materially better.** The AURC verdict favours the signal and the guard holds.
- **Negligible difference.** The AURC verdict is negligible.
- **Uncertain.** Otherwise.
- **Worse.** The AURC verdict favours confidence.

## Bootstrap

- **Confidence.** 0.95.
- **Interval.** Percentile.
- **Resamples.** 10000.
- **Seed.** 0.
- **Statistics.** Mean and median paired differences.
- **Unit.** Household.

## The check

- Every fold's fitted channels and time prior equal the fitted-rates record's, parameter by parameter.
- The ensemble's population on every training household is the fold's fitted channels, exactly.
- Every household's scores of filter_periodic_hurdle equal the combined-prior record's.
- **Tolerance.** 1e-06.
- **Otherwise.** Nothing is reported, not even per household.

## The abstention rule's signal

- **The rule.** The structural signal whose decision is materially better, the one with the largest mean AURC difference against confidence when several are; confidence when none is; a tie in that mean goes to the signal first in alphabetical order. Entropy is not a candidate: it ranks windows by the same posterior as confidence, and Phase 4 found it negligibly different.
- **Use.** The abstention rule is selected afterwards, under its own protocol, for the signal this rule gives.
- **Thresholds here.** None selected: no coverage level or signal value is chosen for deployment.

## One run

- **Amendments.** A change to a pinned source after the freeze is declared in artifacts/phase4/prior_uncertainty_amendments.json before the run, with its reason and the digest of every source it changes; the run accepts a changed source only at the digest an amendment gives, and the record lists the amendments as an input.
- **Branch.** The run writes run-artifacts/phase4-uncertainty-time-prior, and the workflow refuses to run when that branch exists.
- **Crash.** A run that stops before writing its record reports nothing; it is fixed without looking at any outcome, the fix is declared as an amendment, and it is run again. A run that stops after writing its record is the run: the workflow still commits what it wrote, and a missing page or figure is generated afterwards from the record by the same functions.
- **Libraries.** The run refuses when numpy, scipy or pandas is at another version than the freeze recorded.
- **Runs.** One scoring run, dispatched on develop.
- **Sources.** The freeze records the digest of every file of the sensor_modeling package and of pyproject.toml, scripts/freeze_phase4_time_prior.py, scripts/run_phase4_time_prior.py, .github/workflows/run-phase4-time-prior.yml, and the versions of numpy, scipy, pandas; the run refuses when a source differs.
- **Undefined.** A signal whose missing policy is refuse and which is undefined in a scored window stops the run before it writes a record: a crash, whose amendment may make the signal's computation finite but may not change its policy.
- **Where.** The run is made only by its workflow, dispatched on develop, and the workflow refuses when the frozen protocol file has changed since the commit that froze it.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/heldout/casas-heldout-confirmation.json` | `785f6757ed9e9890dc7dfbf471660403779cc7dcc6a246586a87fd2323e2c97a` |
| `artifacts/phase3/combined_prior_protocol.json` | `3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5` |
| `artifacts/phase3/fitted_rates_protocol.json` | `f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d` |
| `artifacts/phase3/phase3-activity-substates.json` | `73e19a933de8d3bd1cf97aed8981a04bf446673f92d29b56a47f679cb525b1df` |
| `artifacts/phase3/phase3-combined-prior.json` | `57db137f57f5b298054d7a4cfe8c837d854ca9a8793329857d7803401477db59` |
| `artifacts/phase3/phase3-fitted-rates.json` | `807d60fee5adff828c5a727eedbc8858493e816bdf60a555a004ea8a035d590c` |
| `artifacts/phase4/phase4-uncertainty-diagnostics.json` | `faeb9d679ee251b493586292b9d2dd55913cab6cb986f9ad14cc4e8f3bc5c2cb` |
| `artifacts/phase4/uncertainty_protocol.json` | `537dd3d319e1ec3964d2d8df370cac3d05a72fd267a10eaa148b0148f59782e1` |

## What this cannot show

- A held-out claim: the 20 homes have been inspected in earlier work, by Phase 4 among others.
- A confirmation on the 43 held-out CASAS homes, which are used up for this formulation: its scores in every one of them are published.
- A threshold: none is selected here.
- Uncertainty in the fits: each fold's training homes share one fit, and the household bootstrap resamples the scored homes only.
- Another formulation: every signal ranks the predictions of filter_periodic_hurdle alone.
- An unbiased signal choice: the rule picks the abstention rule's signal on these 20 homes, and the largest mean among several winners is biased upward. No CASAS homes remain untouched to confirm the signal or the rule chosen with it.
- Independence from Phase 4: the homes, windows and folds are the same, and predictive_mismatch is the same window by window; only the predictions and the ensemble's recursion differ.
- A guard with reach: it covers bed_awake, 68 windows in 5 households; windows cluster within homes, and the homes of a fold share one fit.

**Tuning.** None: every setting is declared here, before scoring.
