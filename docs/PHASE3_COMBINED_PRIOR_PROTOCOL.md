# Phase 3: the time prior with the fitted channels, in the recursion: the protocol

Generated entirely from the frozen file `artifacts/phase3/combined_prior_protocol.json` by `sensor_modeling.datasets.combined_prior_summary.render_protocol`.

**Status: pre-specified, on the development panel, which earlier work has inspected; not a held-out claim.**

- **Protocol digest.** `8570de52f23ef2a30c5d66876895888bad3dc422d67fc2266440ed3ddc471429`.
- **The question.** Whether the Phase 3.1 time prior, entering the filter's recursion as an hour-dependent transition, adds to the reference formulation with fitted hurdle channels.
- **This page reports no result.**

## What had been seen

- The fitted-rates record: in the recursion the fitted hurdle channels score a mean household balanced accuracy of 0.456 against 0.417 for the declared channels, FR +0.039 [+0.015, +0.062], a success; with the time prior on the declared information sets I1 and I3 the fitted channels are a success as well (F1, F3); and the fitted Poisson channels score 0.477 in the recursion.
- The time-prior record: on I1 the prior is worth +0.131 [+0.120, +0.142] balanced accuracy to the generative model, almost all of it in the recall of away; recent history adds a negligible +0.002 to it (S6, a failure).
- The predictive checks: active counts over-dispersed and silence in long runs within away, home_active and home_inactive; and the fitted channels' loss of 0.23 of home_active recall.
- An unregistered result with the production filter and the earlier versions of both parts, fitted sensor rates and the v0.3 circadian term: together they scored 0.434 balanced accuracy, against 0.460 for the circadian term alone and 0.449 for neither, in docs/real_data.md.
- No recursion with the time prior's transition had been run on any CASAS household before this protocol was frozen; the code was exercised on simulated homes only.

## The base protocol

- **File.** `artifacts/phase3/fitted_rates_protocol.json`, SHA-256 `f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d`, protocol digest `4f5db165af7eb9178b6154b3f01db806094955571cec1b07a92623f1e8cfc8d6`.
- **Used.** Its folds, its fold fits of the channels and the time prior on training households only, its metrics, minimal differences and household bootstrap.

## The recursion

- **Channels.** Declared Poisson or fitted hurdle, as the base protocol fits them.
- **Regime.** Online: no estimate reads a window after its own.
- **Windows.** Every window of the recording at the base protocol's step of 300 seconds; both sides of every comparison receive the same windows, and only labelled windows are scored.
- **With the prior.** The transition into each window is the time prior's at the local hour at the window's end, the population prior for every held-out household; the belief one step before the first window is the prior's distribution at that window's hour.
- **Without the prior.** The ontology's transition over one step for every window, from its stationary distribution one step before the first window, as the fitted-rates record ran it.

## Models

| Model | Channels | Time prior |
| --- | --- | --- |
| `filter_declared` | declared | no |
| `filter_hurdle` | hurdle | no |
| `filter_periodic` | declared | yes |
| `filter_periodic_hurdle` | hurdle | yes |

## Estimands

|  | Role | Question | Model | Reference | Rule |
| --- | --- | --- | --- | --- | --- |
| K1 | primary | What the time prior adds to the reference formulation, the fitted hurdle channels, in the recursion. | `filter_periodic_hurdle@R` | `filter_hurdle@R` | time prior |
| K2 | secondary | What the time prior adds to the declared channels in the recursion. | `filter_periodic@R` | `filter_declared@R` | time prior |
| K3 | secondary | What the fitted channels add with the time prior in the recursion. | `filter_periodic_hurdle@R` | `filter_periodic@R` | fitted channels |

## Criteria

- **Fitted channels.** The fitted-rates rule: success when calibration error favours the model and neither log loss nor balanced accuracy favours the reference; trade-off when calibration error favours the model and balanced accuracy the reference; failure when calibration error is negligible or favours the reference.
- **Multiplicity.** K2 and K3 are secondary and read by their rules; nothing is adjusted.
- **Primary.** K1 alone decides whether the combination is adopted as the formulation the held-out confirmation will freeze.
- **Time prior.** Phase 3.1's rule: success when balanced accuracy favours the model and neither log loss nor calibration error favours the reference; failure when balanced accuracy is negligible or favours the reference; inconclusive otherwise.
- **Verdicts.** Favours model: mean >= delta and lower bound > 0; favours reference: mean <= -delta and upper bound < 0; negligible: the whole interval within (-delta, delta); uncertain: anything else.

## Per-state recall

- **Estimands.** K1, K2.
- **Verdict.** The same verdicts with the recall minimal difference.

## Interaction

- **What.** Per household, the prior's gain with the fitted channels minus its gain with the declared channels, K1 minus K2.
- **Metrics.** Balanced accuracy, log loss and calibration error.
- **Reading.** Described with its interval, not judged: a negative value means the two parts combine less than additively.

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
- Every household's scores of filter_declared and filter_hurdle in the recursion equal the fitted-rates record's.
- **Tolerance.** 1e-06.
- **Otherwise.** Nothing is reported, not even per household.

## Pinned records

| File | SHA-256 |
| --- | --- |
| `artifacts/phase3/fitted_rates_protocol.json` | `f695f972fcfcd11266a8a1a720932442e8b7d455bdec7dd6304ecab41843075d` |
| `artifacts/phase3/phase3-fitted-rates.json` | `807d60fee5adff828c5a727eedbc8858493e816bdf60a555a004ea8a035d590c` |

## What this cannot show

- A held-out result: the 20 homes have been inspected in earlier work. The confirmation on CASAS homes outside the panel needs its own protocol, frozen after this result, with the Phase 2 baselines under matched information.
- Anything about TIHM or the sleep feature: the CASAS labels are activities, not the pipeline's daily features.
