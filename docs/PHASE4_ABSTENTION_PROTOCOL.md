# Phase 4: an abstention rule for the fitted channels with the time prior: the protocol

Generated from `artifacts/phase4/abstention_protocol.json` by `sensor_modeling.datasets.abstention_summary.render_protocol`. This page reports no result.

**Status: pre-specified selection on the development panel, which earlier work has inspected; one run; not a confirmation.** Digest `0ba0228a7f31e8dcf197cf8e9553cb64bb8d04d8744f0f53dffdfab34ee88887`.

**The question.** Which threshold on structural disagreement keeps the coverage near the target while the guarded states keep their share, and does it lower the error on homes it was not chosen on.

## What had been seen

- The step 1 record, for the fitted channels with the time prior: structural_disagreement is materially better than confidence, household error AURC 0.048 [0.033, 0.062] lower, and the declared rule picks it; full-coverage error is 0.364 pooled, 0.360 as a mean over homes. Pooled over the panel, it keeps of the proportional share of bathroom_activity and kitchen_activity 0.55 and 0.50 at coverage 0.5 and 0.70 and 0.60 at 0.7, and of away 1.17 and 1.11.
- What that record implies for this selection, pooled over all 20 homes: at coverages 0.8, 0.85, 0.9 and 0.95 the kept share of bathroom_activity is 0.76, 0.78, 0.82 and 0.86, of kitchen_activity 0.67, 0.72, 0.79 and 0.87, and of away 1.07, 1.05, 1.03 and 1.02, so the guard first holds at 0.95, and away cannot bind; there the error is 0.347 against 0.364, and the mean gain over homes is 0.017, below the minimal difference of 0.02. A1 is therefore unlikely to favour the rule, and the decision likely to be not supported or inconclusive.
- The target coverage and the guard's fraction were declared after those curves were seen, and were not changed to move that outcome. The guarded states are those the roadmap named before step 1 was run, the activity states and away; step 1's guard covered bed_awake, 68 windows in 5 homes, which this guard leaves out.

## Step 1

- **File.** Artifacts/phase4/prior_uncertainty_protocol.json.
- **Protocol sha256.** C71a08021a7ea434ee8ef9cacc90e4c87c44e9f3565d9796b648cbb13a342865.
- **Sha256.** 0cd450ef7ba3b5058da025d76a637983e48b2722b20d1b9b62403f2349dc316a.
- **Used.** Its folds, fits, predictions and ensemble; the signal is its record's rule_signal, which the run checks.

## Selection

- **Coverages.** 0.8, 0.85, 0.9, 0.95, 1.
- **Guard.** Each of away, bathroom_activity, kitchen_activity keeps at least 0.8 of its proportional share: its kept share over the coverage kept, pooled over the selection windows; a state with no selection window is not judged.
- **Rule.** The first coverage at which the guard holds; at 1, the rule never abstains.
- **Threshold.** At each coverage from the target upward, the smallest signal value at or below which that share of the selection windows lies, pooled; a window is kept when its signal is at or below the threshold, so ties are kept.
- **Undefined.** A signal undefined in any selection or held-out window stops the run before it writes a record.

## Fitting and scoring

- **Cross fitting.** For each fold, the threshold is selected on its training homes, each scored by the ensemble and time prior fitted on the fold's other training homes; it is applied to the fold's held-out homes, scored by the fold's fits as step 1 scored them.
- **Final rule.** The threshold selected the same way on all 20 homes, each scored by the ensemble and time prior fitted on the other 19.

## Estimands

- **A1.** Primary: per held-out home, the error at full coverage minus the error of the predictions the rule keeps; positive favours the rule.
- **A2.** Per held-out home, the balanced accuracy of the kept predictions minus that at full coverage.
- **Coverage.** The coverage kept, pooled and per home, described.
- **Guard held out.** The guard, pooled over the held-out homes' windows.

## Reading

- **Verdicts.** Favours the rule: mean at least the minimal difference and lower bound above 0; favours no abstention: the mirror; negligible: the interval within plus or minus it; uncertain: anything else.
- **Decision.** Supported on development data when A1 favours the rule, A2 does not favour no abstention and the guard holds on the held-out homes; not supported when A1 is negligible or favours no abstention, or the guard fails on the held-out homes; inconclusive otherwise.

## Edge cases and adoption

- **Edge cases.** A fold whose threshold never abstains gives its homes A1 and A2 of exactly 0; a home with no kept window has no A1 or A2 and is excluded from the comparison and listed as excluded; kept balanced accuracy averages the states with a kept window.
- **Adoption.** The frozen rule is adopted as the formulation's abstention rule only when the decision is supported; otherwise it is recorded and not adopted, and the online pipeline is not changed either way.

## Parameters

- **Fraction.** 0.8.
- **Minimal.** 0.02.
- **Target.** 0.8.

## Bootstrap

- **Confidence.** 0.95.
- **Resamples.** 10000.
- **Seed.** 0.
- **Unit.** Household.

## The check

- Every fold's fitted channels and time prior equal the fitted-rates record's.
- Every held-out home's reference scores equal the combined-prior record's.
- Step 1's record picked this signal.
- **Tolerance.** 1e-06; otherwise nothing is reported.

## One run

- **Amendments.** A change to a pinned source after the freeze is declared in artifacts/phase4/abstention_amendments.json with its reason and new digest; the run accepts it only at that digest and lists the file as an input.
- **Branch.** The run writes run-artifacts/phase4-abstention-rule, and the workflow refuses when it exists, or when the frozen protocol file has more than its freeze commit.
- **Crash.** A run that stops before writing its record reports nothing; it is fixed without looking at any outcome, the fix is declared as an amendment, and it is run again; a run that wrote its record is the run.
- **Runs.** One run, by .github/workflows/run-abstention-rule.yml dispatched on develop; the script refuses anywhere else.
- **Sources.** The freeze records every file of the package and pyproject.toml, scripts/freeze_abstention_rule.py, scripts/run_abstention_rule.py, .github/workflows/run-abstention-rule.yml, and the versions of numpy, scipy, pandas; the run refuses when one differs.

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
| `artifacts/phase4/phase4-uncertainty-time-prior.json` | `99addb681e893df0a988156ddb23056c7aa7fc145471501f3def895591f48022` |
| `artifacts/phase4/prior_uncertainty_protocol.json` | `0cd450ef7ba3b5058da025d76a637983e48b2722b20d1b9b62403f2349dc316a` |
| `artifacts/phase4/uncertainty_protocol.json` | `537dd3d319e1ec3964d2d8df370cac3d05a72fd267a10eaa148b0148f59782e1` |

## What this cannot show

- A confirmation: the 20 development homes have been inspected, the 43 held-out CASAS homes are used up for this formulation, and ContextAct's per-day scores of it are published too. A confirmation needs an independent dataset this formulation has not been scored on, with the rule frozen here applied once, unchanged.
- Independence of the selection: step 1 saw these homes' curves before the target and fraction were declared.
- The signal's choice: it was picked on these homes, and is not confirmed.
- Uncertainty in the fits and the selection: the homes of a fold share one fit and one threshold, and the household bootstrap resamples the scored homes without selecting the threshold again.
- The frozen threshold's own performance: A1 judges the procedure, with thresholds chosen on fits to nine homes and applied to fits to ten; the frozen threshold is chosen on fits to 19 and has no held-out estimate.

**Tuning.** None: every setting is declared here, before scoring.
