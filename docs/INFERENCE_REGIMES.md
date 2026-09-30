# Inference regimes: online filtering and fixed-lag smoothing

The filter's estimate for a moment uses only the evidence at or before it. That
is what a live system can report at that moment. A fixed-lag smoother revises
the estimate with up to `lag` later windows. It reads evidence a live system
does not yet have, and it can only be reported `lag` windows late.

Roadmap item 3.5 fixes the rule: a gain obtained with future evidence must
never be reported as an online gain. `sensor_modeling.fusion.regime` makes the
two regimes explicit, and `sensor_modeling.evaluation` refuses to compare,
pool or relabel results across them. Neither inference algorithm is changed.

## The regimes

| Mode | Reads evidence up to | Available | Causal | Label |
| --- | --- | --- | --- | --- |
| `ONLINE_FILTER` | the moment itself | at the moment | yes | `online filter` |
| `FIXED_LAG_SMOOTHER` | `lag` windows after the moment | `lag` windows after the moment | only with lag 0 | `fixed-lag smoother, lag 1 window (5 min)` |

- **`InferenceRegime` holds the mode.** An online regime has no lag. A smoother
  needs its lag in windows and the window width, so its lag is stated in time.
- **`is_causal`.** True when no estimate reads evidence after its moment: the
  online filter, and a smoother with lag zero.
- **`delay`.** How long after its moment an estimate can first be reported.
  It is zero online, and the lag for a smoother.
- **`regime_beliefs` turns filtered beliefs into what a regime reports.**
  Online, it returns them unchanged. A smoother applies the existing
  `smooth_beliefs` with the declared lag.
- **`regime_estimates` also records when each estimate exists.** It returns
  one `ReportedEstimate` per window, described below.
- **Zero lag.** A smoother with lag zero reports exactly the filtered
  estimates, at the same moments. It is still labelled a smoother, so a record
  says what was run.

## Operational differences

The two regimes answer different operational questions. They are not two
settings of one estimate.

- **When an estimate exists.**
  - An online estimate for moment `t` exists at `t`.
  - A smoothed estimate for `t` exists only at `t + lag`, once the last window
    it reads has arrived. The regime's `delay` is that wait.
- **Whether it can change.**
  - An online estimate is final when it is reported. Later evidence produces
    the next moment's estimate, never an edit of this one. Its belief array is
    read-only, in `StateEstimate` and in `ReportedEstimate`.
  - A smoothed estimate for `t` is provisional while windows `t + 1` to
    `t + lag` arrive, and final at `t + lag`. Only the final one is reported.
- **What it can be used for.**
  - Anything decided at the moment needs the online estimate. That includes an
    alert, a check-in and a live display. A smoothed estimate is not what that
    decision could have seen.
  - A smoothed estimate suits retrospective uses that can wait `lag`, such as
    a daily summary or a review of the past hours.
- **The end of a recording.** A recording can end before an estimate's full
  lag has arrived. The last estimates then read to the end of the recording,
  and are available there. `regime_estimates` records their shorter lead.
- **What a smoothing gain costs.** A smoothing gain is bought with delay. It is
  reported as a smoothing gain, with that delay, and never as an improvement
  to the online estimate.

## What each estimate records

A `ReportedEstimate` is one estimate as its regime reports it.

| Field | Meaning |
| --- | --- |
| `prediction_at` | the moment the estimate is about |
| `evidence_until` | the latest evidence it read: its moment online, up to `lag` later for a smoother |
| `available_at` | when it can first be reported: its moment online, `evidence_until` for a smoother |
| `belief` | the posterior, copied and read-only |
| `regime` | the regime that produced it |

- **What it refuses.** Construction refuses evidence beyond the regime's
  horizon, evidence before the moment, and an availability the regime cannot
  have. A smoothed estimate claiming to be available at its own moment is
  refused.
- **`lead`.** `evidence_until − prediction_at` is the future information the
  estimate used. `uses_future_evidence` says whether it is positive.

## What each record states

Every `ExperimentRecord` has an `inference` block, from schema 1.3 on. For the
tests' synthetic sequence of twelve 5-minute windows, smoothed with a lag of two
windows, it is:

```json
"inference": {
  "mode": "fixed_lag_smoother",
  "lag_steps": 2,
  "step_seconds": 300.0,
  "label": "fixed-lag smoother, lag 2 windows (10 min)",
  "causal": false,
  "delay_seconds": 600.0,
  "provenance": "declared by the experiment that wrote the record",
  "evidence": {
    "enumerated": true,
    "predictions": 12,
    "first_prediction": "2026-01-01T12:00:00+00:00",
    "last_prediction": "2026-01-01T12:55:00+00:00",
    "latest_evidence": "2026-01-01T12:55:00+00:00",
    "max_lead_seconds": 600.0
  }
}
```

An online record has `"causal": true`, `"delay_seconds": 0.0` and
`"max_lead_seconds": 0.0`.

- **Regime, lag and causality.** `mode`, `lag_steps`, `step_seconds`, `label`,
  `causal` and `delay_seconds` all follow from the regime. A record in which
  any of them disagrees with the regime is refused.
- **Timestamps.** `evidence` summarises the scored estimates. It gives how many
  there were, the first and last prediction timestamps, the latest evidence any
  of them read, and the largest lead of evidence over prediction.
- **A lead beyond the regime is refused.** A record whose largest lead exceeds
  its regime's delay is refused. So a record of smoothed estimates cannot be
  labelled online.
- **Not listed.** A causal record that cannot list its timestamps may give a
  reason instead, `{"enumerated": false, "reason": ...}`. The CLI's simulation
  studies and records migrated from earlier schemas do this. A smoother may
  not: it must record the future information it used.

Every experiment in `sensor_modeling.datasets` now lists its scored moments.
Each is a held-out household's labelled regular moment, after the household's
cut-off where an experiment has one.

A record that compares regimes states the one with the longest lag, which
bounds every estimate in it. Its results label every cell and comparison with
its own regime and evidence summary. The
[Phase 3.5 smoothing evaluation](PHASE3_SMOOTHING.md) is the first such record.
Each of its comparisons is made with `compare_results(..., smoothing_gain=True)`,
so each is a labelled smoothing gain.

## What fails loudly

- **In the regime module.**
  - `require_online` refuses a smoother wherever an online result is required.
  - `require_same_regime` refuses to combine results from different regimes,
    or from smoothers with different lags.
  - `check_evidence_access` refuses evidence observed after a regime's
    horizon.
  - `assert_respects_horizon` checks an estimator empirically. The evidence
    after each estimate's horizon is changed, and the estimate must not move.
- **In the evaluation.** `RegimeResult` holds one value per household with
  its regime and evidence summary.
  - Evidence that reads past the regime is refused, so smoothed results cannot
    be labelled online.
  - `compare_results` refuses plain mappings, which carry no regime.
  - It also refuses results of different regimes, unless the caller passes
    `smoothing_gain=True`. A smoothing gain must compare a smoother, the model,
    against the online filter, the reference. The comparison is then labelled,
    for example `smoothing gain: the fixed-lag smoother, lag 2 windows (10 min)
    against the online filter; the smoothed estimates are reported 10 min after
    their moments`.
  - `RegimeComparison.online_gain` returns the comparison only when both sides
    are online.
  - `pool_results` combines results of one regime only, each household once.
  - `compare_households` refuses a `RegimeResult`, and points to
    `compare_results`.
- **In records.** `ExperimentRecord` requires `inference` and `evidence`, with
  no defaults. A record is refused if its label, causality or delay does not
  describe its regime, or if its evidence reads past the regime.
- **In the matched evaluation.** `run_matched_evaluation` is online by
  construction, since every feature closes at or before its prediction moment.
  - It refuses a smoothing regime.
  - It refuses a model that declares an `inference_regime` other than online.
  - Its existing row-independence check refuses a model whose prediction for
    a row reads other rows.
  - Pooling folds, or comparing information sets, refuses runs from different
    regimes.
- **In generated reports.** Every generated report states its regime, in a
  line such as `Inference regime: online filter (declared by the experiment
  that wrote the record)`.

## What the tests show

`tests/test_inference_contract.py` streams a synthetic sequence one window at a
time. At the turn, later evidence genuinely corrects the online estimate.

- **Online.**
  - Once reported, no online estimate changes as later windows arrive, to the
    last bit.
  - The reported estimate for the turn stays wrong after the correcting window
    arrives.
- **Smoothed.**
  - A smoothed estimate changes only while windows within its lag arrive, and
    is fixed from its horizon on.
  - At the turn, it equals the online estimate until the correcting window
    arrives, then moves to the truth.
  - One window earlier, a one-window lag never reaches the correcting window,
    so it reports the online estimate. A two-window lag reaches it and moves.
- **Published is what is scored.** For each regime, the estimates a live
  system would publish, and when, equal the offline estimates of the completed
  recording.
- **Zero lag.** It reproduces online filtering exactly: the same beliefs, the
  same evidence and availability timestamps, and the same evidence summary.
  The smoothing gain it measures is zero in every household.
- **The online pipeline.** On a simulated day streamed through the pipeline:
  - no released estimate changes as later observations arrive, and none can be
    edited in place;
  - the estimates up to a cut-off are identical whether or not the stream
    continues past it;
  - a smoother over the same stream does revise them, returning new
    estimates.

`tests/test_inference_regimes.py` checks that a smoother of each lag reads
exactly its lag, and records, migrations and reports. The contract tests also
cover every refusal listed above.

## Existing results

- **Published records.** Every published record is online. The Phase 1 and
  Phase 3.1 to 3.3 records are 1.1 files, and the Phase 3.4 record is a 1.2
  file. None listed its prediction timestamps. When read, each is migrated:
  it gets causality and delay from its regime, and an evidence entry marked
  not listed, with the reason. The files themselves are not changed, and their
  generated summaries are identical. See
  [experiment artifacts](EXPERIMENT_ARTIFACTS.md#the-inference-regime).
- **The smoothing numbers in [real data](real_data.md).** They were measured
  outside the record layer. The table that set them beside online results
  labels each row's regime.
