# Inference regimes: online filtering and fixed-lag smoothing

The filter's estimate for a moment uses only the evidence at or before it. That
is what a live system can report at that moment. A fixed-lag smoother revises
the estimate with up to `lag` later windows. It reads evidence a live system
does not yet have, and it can only be reported `lag` windows late.

Roadmap item 3.5 fixes the rule: a gain obtained with future evidence must
never be reported as an online gain. `sensor_modeling.fusion.regime` makes the
two regimes explicit, so that they cannot be blurred by accident.

## The regimes

| Mode | Reads evidence up to | Reported | Label |
| --- | --- | --- | --- |
| `ONLINE_FILTER` | the moment itself | at the moment | `online filter` |
| `FIXED_LAG_SMOOTHER` | `lag` windows after the moment | `lag` windows late | `fixed-lag smoother, lag 1 window (5 min)` |

- **An `InferenceRegime` holds the mode.** An online regime has no lag. A
  smoother needs its lag in windows and the window width, so its delay is
  stated in time.
- **`regime_beliefs` is the one way to turn filtered beliefs into what a
  regime reports.** Online returns them unchanged. A smoother applies the
  existing `smooth_beliefs` with the declared lag. Neither algorithm is
  changed.
- **Zero lag.** A smoother with lag zero reports exactly the filtered
  estimate. It is still labelled a smoother, so a record says what was run.

## What fails loudly

- **`require_online`.** It refuses a smoother wherever an online result is
  required.
- **`require_same_regime`.** It refuses to combine results from different
  regimes, or from smoothers with different lags.
- **`check_evidence_access`.** It refuses evidence observed after a regime's
  horizon: the moment itself online, and `lag` windows later for a smoother.
- **`assert_respects_horizon`.** It checks an estimator empirically. The
  evidence after each estimate's horizon is changed, and the estimate must not
  move.
- **The record.** `ExperimentRecord` requires `inference`. It has no default,
  and a record whose label does not describe its regime is refused.
- **The matched evaluation.** `run_matched_evaluation` is online by
  construction, since every feature closes at or before its prediction
  moment.
  - It refuses a smoothing regime.
  - It refuses a model that declares an `inference_regime` other than online.
  - Its existing row-independence check refuses a model whose prediction for
    a row reads other rows.
  - Pooling folds, or comparing information sets, refuses runs from different
    regimes.
- **Generated reports.** Every generated report states its regime, in a line
  such as `Inference regime: online filter (declared by the experiment that
  wrote the record)`.

## What the tests show

On a synthetic sequence in which later evidence genuinely corrects an earlier
estimate:

- **Online.** The online estimate is wrong at the turn, and changing any later
  window leaves it unchanged.
- **One window of lag.** The smoother corrects the turn once the decisive
  window is within its lag.
- **Before the permitted window.** One window earlier, it reports exactly the
  online estimate, because the only window it may read says nothing.
- **Exactly its lag.** A smoother of each lag passes the horizon check for its
  own lag and fails it for one window less.
- **Zero lag.** Zero-lag smoothing equals online filtering exactly.
- **A smoother claiming to be online.** A smoother checked as if it were
  online is caught.

## Existing results

- **Published records.** Every record published before schema 1.2 is online.
  The fixed-lag smoother was never called by experiment code. Those records
  are read as online, attested on migration
  ([experiment artifacts](EXPERIMENT_ARTIFACTS.md#the-inference-regime)).
- **The smoothing numbers in [real data](real_data.md).** They were measured
  outside the record layer. The table that set them beside online results now
  labels each row's regime.
