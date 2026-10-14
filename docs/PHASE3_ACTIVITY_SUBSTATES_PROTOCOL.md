# Phase 3.3 follow-up: activity sub-states in the recursion: the protocol

Generated entirely from the frozen file `artifacts/phase3/substates_protocol.json` by `sensor_modeling.datasets.substate_summary.render_protocol`.

**Status: pre-specified, on the development panel, which earlier work has inspected; not a held-out claim.**

- **Protocol digest.** `0c02781a58f8d96c4d69948bf0313c71da52dc369b2c520adb88c99b60028e01`.
- **The question.** Whether two activity sub-states in the common states, entering the filter's recursion with the time prior, recover home_active recall without costing balanced accuracy, log loss or calibration.
- **This page reports no result.**

## What had been seen

- The combined-prior record: in the recursion, the fitted hurdle channels with the time prior score a mean household balanced accuracy of 0.517 against 0.456 without the prior, K1 +0.061 [+0.044, +0.077], a success; K1's home_active recall difference is +0.002 [−0.002, +0.007], negligible, with home_active recall 0.268 with the prior and 0.266 without it.
- The fitted-rates record: in the recursion the fitted hurdle channels lose 0.231 [0.172, 0.290] of home_active recall to the declared channels (FR).
- The predictive-check record: against each cell's own hurdle, windows in quiet runs of at least an hour are 1.336 times the predicted in away, 5.187 in home_active, in the 15 households with enough runs, and 2.568 in home_inactive, each positive, and 1.116 in sleeping, negligible; bursts are positive in home_inactive, 1.596, and sleeping, 2.896, and negative in bathroom_activity and kitchen_activity; against the fold's population hurdle, quiet runs are 6.173 times the predicted in home_active, in 15 households, and 2.584 in home_inactive, positive, and negligible in away and sleeping; the own-reference quiet-run excess is positive on every channel with enough households, from 1.410 on bathroom motion to 2.561 on living-room motion; the active count's variance is 4.342 to 7.418 times the model's in the common states, above one in every household with a value; and its page describes the active counts as having too many single activations, too few mid-range counts and far too many large ones.
- The negative-binomial record: against the hurdle, the zero-truncated negative binomial's home_active recall difference is −0.055 [−0.087, −0.024] with current windows, favouring the reference, and +0.029 [−0.004, +0.063] in the recursion, uncertain; it was not adopted.
- The held-out record: on 43 held-out CASAS homes, K1's home_active recall difference is −0.000 [−0.003, +0.002], negligible.
- The ContextAct transfer record: for one independent resident, the time prior's gain held, K1 +0.049 [+0.013, +0.087], a success; with the prior, the CASAS-fitted hurdle channels lost 0.149 [0.076, 0.221] of balanced accuracy to the declared channels, with better log loss, K3 inconclusive.
- No sub-state model had been fitted to, or run on, any CASAS household before this protocol was frozen; the code was exercised on simulated homes only.

## The base protocols

- **Combined prior.** `artifacts/phase3/combined_prior_protocol.json`, SHA-256 `3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5`, protocol digest `8570de52f23ef2a30c5d66876895888bad3dc422d67fc2266440ed3ddc471429`. Used: its folds; its fold fits of the hurdle channels and the time prior on training households only, which are the fitted-rates protocol's; its recursion with the time prior; its metrics, minimal differences and household bootstrap.
- **Predictive checks.** `artifacts/phase3/predictive_protocol.json`, SHA-256 `5bbef7ef1dcb417fced2fdafe925ee4548e8e2893bb874da753072d077e72dbd`, protocol digest `7fe466e19e6a263dea2aec25abb2a0232b2b6bbd374b7a8af93e355686c03860`. Used: its cells, stretches, quiet-run statistic, replicates, seeds, sufficiency rule, minimal ratio and household reading.

## The model family

- **Chosen.** Activity sub-states: each split state has two hidden sub-states, quiet and busy, each with its own hurdle parameters on every channel, and a Markov chain between them within the state.
- The time prior's hour-dependent transition stays on the behavioural layer, so a sub-state's exit from its state is the prior's whichever sub-state it leaves. A Markov-modulated emission whose modulating chain is shared by the channels is then the same model as activity sub-states, and the two named families differ only in whether the modulation is shared or per channel.
- Shared, not per channel: the predictive checks found the quiet-run excess on every channel with enough households, and the same mixture of a low and a high intensity in every household's active counts, which one level of activity shared by the channels produces; they did not examine dependence between channels, so nothing in them asks for a chain per channel. A chain per channel would be a factorial model, 64 sub-states per state with six channels.
- **Split states.** The states whose quiet runs or bursts the predictive checks found positive against each cell's own hurdle.
- **Not split.** The short, busy states bathroom_activity and kitchen_activity, whose bursts are fewer than independence predicts, and bed_awake, with too few homes.
- **Output.** The belief in a behavioural state is the sum over its sub-states.
- **Exactness.** With each split state's two sub-states sharing the hurdle's parameters the model is exactly the reference, whatever its chain; tested to 1e-10 on simulated homes.
- **Not tested.** Exits that depend on the sub-state, which would replace the time prior's transition rather than add to it; a modulating chain per channel; more than two sub-states.

## Sub-states

- **Iterations.** 100.
- **Odds factor.** 4.
- **Persistence.** 0.9.
- **Pseudo count.** 1.
- **Split.** Away, home_active, home_inactive, sleeping.
- **Substates.** Quiet, busy.

## Fitting

- **Collapse.** Nothing stops a sub-state from holding few windows; it then keeps its own pseudo-windows' pull toward the declared model and still enters inference; each sub-state's share of the training windows is reported.
- **Convergence.** The change in the training log-likelihood over the last iteration is reported, not used; with the pseudo-windows the M-step does not maximise that likelihood, so it need not rise at every iteration.
- **E step.** Forward-backward within each stretch, each stretch entered from the entry distribution.
- **Entry.** Estimated at the first window of every stretch, which is an entry from another state or a restart after an unlabelled gap.
- **Iterations.** All of them, with no stopping rule; the training log-likelihood at each E-step is recorded.
- **M step.** Each sub-state's hurdle is the base protocol's hurdle fit with every window weighted by its sub-state probability, with the same 12 pseudo-windows toward the declared model; the chain and the entry are the expected transition and entry counts, each with the pseudo-count.
- **Names.** After fitting, the sub-state whose silence probability, averaged over the channels, is higher is called quiet; the names change no belief.
- **No window.** A split state with no training window keeps two sub-states with the hurdle's parameters.
- **Settings.** The odds factor, the persistence, a sub-state lasting 50 minutes on average, and the iterations were set before any CASAS household was fitted, and were not selected on data.
- **Start.** Both sub-states start from the fold's fitted hurdle rates; the quiet one with each channel's odds of silence multiplied by the odds factor, the busy one divided by it; the chain with the persistence on its diagonal; the entry even.
- **Unit.** One fit per fold and split state, on the labelled windows of the fold's training households, in stretches of consecutive labelled windows of the state.
- **Unsplit states.** The fold's fitted hurdle, unchanged.

## The recursion

- **Joint states.** 11: two for each of the four split states and one for each of the three others.
- **Regime.** Online: no estimate reads a window after its own.
- **Scored.** The belief in each behavioural state, the sum over its sub-states, scored by its most probable state.
- **Start.** One step before the first window, the prior's distribution at that window's hour, each split state in its entry distribution.
- **Transition.** Into each window, the time prior's at the local hour at the window's end, the population prior for every held-out household, expanded to the joint states: within a split state, the prior's probability of staying times the sub-state chain; into another state, the prior's probability times that state's entry distribution.
- **Windows.** Every window of the recording at the base step of 300 seconds; both models receive the same windows, and only labelled windows are scored.

## Runtime

Measured on simulated homes before the freeze: EM costs about 1.4 microseconds per training window and iteration, under 20 seconds per fold for the roughly 110,000 training windows a fold has in the split states; the recursion over 11 joint states costs about 14 microseconds per window.

## Models

| Model | Channels | Transition |
| --- | --- | --- |
| `filter_periodic_hurdle` | fitted hurdle, one per channel and state | the time prior's at each window's hour |
| `filter_periodic_substates` | fitted hurdle, one per channel and joint state: two sub-states in each split state, one in every other | the time prior's between behavioural states; within a split state, its sub-state chain; a state is entered in its fitted entry distribution |

## Estimands

|  | Role | Question | Model | Reference | Rule |
| --- | --- | --- | --- | --- | --- |
| W1 | primary | What two activity sub-states in the common states add to the reference formulation, the fitted hurdle channels with the time prior, in the recursion. | `filter_periodic_substates@R` | `filter_periodic_hurdle@R` | activity sub-states |

## Criteria

- **Activity sub-states.** Success when home_active recall favours the model and none of balanced accuracy, log loss, calibration error and the recall of away, home_inactive, sleeping favours the reference; trade-off when home_active recall favours the model and at least one of them favours the reference; failure when home_active recall is negligible or favours the reference; inconclusive otherwise.
- **Multiplicity.** One primary; the mechanism and per-state recall are read by their own rules and change no conclusion of W1.
- **Primary.** W1 alone decides whether the sub-states replace the hurdle channels as the reference formulation on the development panel: only a success does; a trade-off does not; a success needs an independent dataset not yet scored with this family to be confirmed.
- **Verdicts.** Favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

## Per-state recall

- **Estimands.** W1.
- **States.** Every state.
- **Verdict.** The same verdicts with the recall minimal difference; insufficient with fewer than 5 households in which the state is labelled.

## The mechanism

- **Cells.** The cells whose quiet-run statistic the predictive checks judged against the population hurdle.
- **Conclusion.** Falls when it falls in every mechanism state; falls in part when it falls in some; does not fall when it is unchanged or rises in every one; inconclusive otherwise.
- **Described.** Each split state's remaining log ratio under the model and the reference, signed and read the same way, is described beside it.
- **Household value.** For each mechanism state, the mean over a household's cells of the absolute value of the reference's log ratio minus that of the model's; positive means the model's prediction is nearer the observed long quiet runs, whichever side of them it lies.
- **Key.** M1.
- **Model.** The fold's sub-state model: each replicate draws the sub-state path of every stretch from the entry distribution and the chain, then each window's silence from its sub-state.
- **Question.** Whether the excess of long quiet runs falls.
- **Reading.** The predictive checks' household reading, with the minimal ratio 1.25: positive is falls, negative rises, negligible unchanged.
- **Reference.** The fold's population hurdle, drawn exactly as the predictive checks drew it, with their seeds.
- **States.** The states whose quiet runs the predictive checks found positive against the population hurdle: home_active and home_inactive.
- **Statistic.** The predictive checks' quiet_runs: windows in runs of at least 12 silent windows within stretches, observed over the mean of 200 replicates of the cell's windows.

## One run

- **Amendments.** A change to a pinned source after the freeze, a fix or a change merged on develop, is declared in artifacts/phase3/substates_amendments.json before the run, with its reason and the digest of every source it changes; the run accepts a changed source only at the digest an amendment gives, and the record lists the amendments as an input.
- **Branch.** The run writes run-artifacts/phase3-activity-substates, and the workflow refuses to run when that branch exists.
- **Crash.** A run that stops before writing its record reports nothing; it is fixed without looking at any outcome, the fix is declared as an amendment, and it is run again.
- **Runs.** One scoring run, dispatched on develop.
- **Sources.** The freeze records the digest of every file of the sensor_modeling package and of pyproject.toml, scripts/freeze_phase3_substates.py, scripts/run_phase3_substates.py, .github/workflows/run-phase3-substates.yml, and the versions of numpy, scipy, pandas; the run refuses when a source differs.

## Metrics and minimal differences

| Metric | Minimal difference |
| --- | --- |
| Balanced accuracy | 0.02 |
| Log loss | 0.05 |
| Brier score | 0.01 |
| Calibration error | 0.02 |
| Recall of one state | 0.05 |

## Bootstrap

- **Confidence.** 0.95.
- **Interval.** Percentile.
- **Resamples.** 10000.
- **Seed.** 0.
- **Statistics.** Mean and median paired differences.
- **Unit.** Household.

## The check

- Every fold's fitted channels and time prior equal the fitted-rates record's, parameter by parameter.
- Every household's scores of filter_periodic_hurdle in the recursion equal the combined-prior record's.
- Every cell's quiet-run statistic against the population hurdle equals the predictive-check record's.
- **Tolerance.** 1e-06.
- **Otherwise.** Nothing is reported, not even per household.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/contextact/contextact-transfer.json` | `1614872031cc35b1607fef79b55bafbe767d91f84d9e8f303e9a8ec5ed34d9b8` |
| `artifacts/heldout/casas-heldout-confirmation.json` | `785f6757ed9e9890dc7dfbf471660403779cc7dcc6a246586a87fd2323e2c97a` |
| `artifacts/phase3/combined_prior_protocol.json` | `3a0c5cad58187609601181ae841099b9144707ba2327af4173c9442812dc52a5` |
| `artifacts/phase3/fitted_rates_protocol.json` | `f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d` |
| `artifacts/phase3/phase3-combined-prior.json` | `57db137f57f5b298054d7a4cfe8c837d854ca9a8793329857d7803401477db59` |
| `artifacts/phase3/phase3-fitted-rates.json` | `807d60fee5adff828c5a727eedbc8858493e816bdf60a555a004ea8a035d590c` |
| `artifacts/phase3/phase3-hurdle-negative-binomial.json` | `e07e62e188f17bdb69c0ff1eb1657153bb864e6cb1ed32d01245c5a0cccdfc23` |
| `artifacts/phase3/phase3-hurdle-predictive-checks.json` | `024fb929a8c71bc901b05e7758bcc79287860a03142ce601dd1dd7546bf76b34` |
| `artifacts/phase3/predictive_protocol.json` | `5bbef7ef1dcb417fced2fdafe925ee4548e8e2893bb874da753072d077e72dbd` |

## What this cannot show

- A held-out claim: the 20 homes have been inspected in earlier work, and the 43 held-out CASAS homes are used up for confirmation. A success here needs an independent dataset not yet scored with this family to be confirmed.
- Which within-state dependence the data have: a success shows that two shared sub-states help, not that the dependence is shared by the channels or has two levels.
- Uncertainty in the fits: each fold's ten training homes share one fit, and the household bootstrap resamples the scored homes only, so the intervals leave out how the fits would vary.
- Selection: the family, the split states and the mechanism states were chosen from the predictive checks on these same 20 homes, so M1 is not an independent test of the mechanism.
- Transfer: the sub-states add parameters to channels that did not transfer to the one independent resident of ContextAct, so a gain here may not survive a change of homes, sensors or annotation.
- Dependence between channels: given the joint state, the channels are still independent.
- Anything about TIHM or the online pipeline, whose defaults this does not change.

**Tuning.** None: every setting is the base protocols' or declared here.
