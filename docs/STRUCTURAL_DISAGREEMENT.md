# Structural disagreement between model specifications

Paper 1 found that no scalar read off one posterior is a credible standalone
uncertainty signal. It tested confidence, entropy, margin, evidence strength and
information gain. ROADMAP Phase 4 asks instead for structural uncertainty: how
much a prediction depends on assumptions the evidence has not settled.

This page describes diagnostics that measure it: the disagreement, window by
window, between several fitted model specifications that the published
evidence supports.

**Status: diagnostic infrastructure.**

- **What it changes.** It sets no abstention threshold and leaves the deployed
  decision rule unchanged. It runs no evaluation.
- **The open question.** Whether disagreement predicts errors is for a later,
  pre-specified Phase 4 experiment, reported as a risk–coverage curve.

## What is measured

For each window, from the posteriors of `M` specifications over the same
windows (`sensor_modeling.evaluation.disagreement`):

| Quantity | Definition | Range |
| --- | --- | --- |
| posterior | each specification's posterior over the states | — |
| argmax | each specification's most probable state | — |
| decision | the reference specification's most probable state: the deployed decision, unchanged | — |
| pairwise divergence | the Jensen–Shannon divergence of each pair, `H((p + q)/2) − (H(p) + H(q))/2`, in bits | `[0, 1]` |
| vote disagreement | the share of specifications not voting for the plurality state | `[0, 1 − 1/M]` |
| probability spread | per state, the largest minus the smallest posterior | `[0, 1]` |
| consensus | the mean posterior, and its most probable state | — |
| discordance | the generalised Jensen–Shannon divergence, `H(mean posterior) − mean H(posterior)`, in bits | `[0, log₂ M]` |
| normalised discordance | the discordance over `log₂ M` | `[0, 1]` |
| unanimous | whether every specification votes for the same state | — |

- **Symmetry and zero.** Every divergence is symmetric and zero exactly when
  the posteriors agree.
- **Rounding floor.** Divergences below `10⁻¹²` bits are rounding and are
  reported as zero, so identical specifications disagree by exactly nothing.
- **Why Jensen–Shannon.** Unlike Kullback–Leibler divergence, it is bounded
  and symmetric, and finite even when a posterior puts zero mass on a state.
  With two specifications, the discordance equals their pairwise divergence.
- **Votes against divergence.** Vote disagreement and the divergences answer
  different questions. Two nearly uniform posteriors can vote differently while
  diverging by almost nothing. The record keeps both.
- **Ties.** The plurality goes to the earliest state in the ontology's order, so
  results are deterministic.

`DisagreementTrace.summary()` describes one household's windows:

- the mean, median, 90th percentile and maximum of the discordance, the
  normalised discordance, the mean pairwise divergence and the vote
  disagreement;
- the share of unanimous windows;
- how often the decision is the plurality;
- the mean spread of each state;
- for each pair, its mean divergence and how often its argmaxes differ;
- for each specification, how often it agrees with the decision.

## Which specifications may take part

Variations come only from assumptions the published evidence supports
(`sensor_modeling.datasets.structural_models`).

- **One variant per axis.** A specification chooses one variant on each axis.
- **Cited support.** Every variant cites the record, estimand and finding that
  support it. A test checks each citation against the published record.

| Axis | Variant | Support |
| --- | --- | --- |
| observation | `hurdle`: fitted hurdle-Poisson channels | Phase 3.3 follow-up, FR: success |
| observation | `hurdle_nb`: fitted hurdle negative binomial | negative-binomial evaluation, recursion: adopt |
| parameters | `population`: the fold's population, fitted on its training households | Phase 3.3 follow-up, FR: success |
| parameters | `pooled`: the household pooled toward the population on its labelled windows before a cut-off | Phase 3.4, P1: success; P2 (recursion): inconclusive |
| parameter sample | `all` of the fold's training households | Phase 3.3 follow-up, FR: success |
| parameter sample | `half_a`, `half_b`: the population fitted on alternate halves of the training households | the same fit's estimation uncertainty |

- **The parameter sample is causal.** It is the population's uncertainty over
  which training households it was fitted on. It reads no held-out household
  and no future window.
- **Pooling is causal too.** A pooled specification reads a household's
  labelled windows only up to its cut-off. Every specification is then scored
  only on the windows after it.

Refused, with the reason recorded:

| Variation | Why not |
| --- | --- |
| declared rates | not a fitted specification; the fitted families improve on them in every measure the Phase 3.3 follow-up scored |
| the Phase 3.1 time prior as the transition | evaluated only on declared information sets, never as the recursion's transition |
| fixed-lag smoothing | reads evidence after the window, which an online diagnostic may not |

### The reference and the default ensemble

- **The reference.** `hurdle/population/all` is the Phase 3.3 follow-up's
  recursion. Its most probable state is the decision.
- **The default ensemble.** It adds one supported alternative on each axis, so
  each specification differs from the reference in exactly one assumption:
  - `hurdle_nb/population/all`;
  - `hurdle/pooled/all`;
  - `hurdle/population/half_a` and `hurdle/population/half_b`.
- **Size.** An ensemble holds 2 to 6 distinct specifications, including the
  reference. Dozens of arbitrary perturbations are refused.
- **Shared structure.** Every specification runs the same filter recursion over
  every window, from the stationary distribution, with the same transition.
  Only its channel models differ.
- **The decision.** A test checks that the reference posterior, and with it the
  decision, is the deployed recursion's to the last bit.

## In the experiment record

Schema 1.4 adds an optional `structural_disagreement` section to every
experiment record ([experiment artifacts](EXPERIMENT_ARTIFACTS.md)). It holds:

- the distance and the trace format;
- the reference and the state order;
- every specification, with its assumptions and their published support;
- per household, the summary of its trace, and the trace's file with its
  SHA-256, or `null` if the trace is not written.

`validate_record` checks the section:

- at least two distinct specifications, each assumption with support;
- the reference among them;
- every pair covered;
- every summary within its bounds;
- every trace reference a file with a digest.

Records written before 1.4 are migrated with the section `null`.

The per-window records are too large to embed for a panel of homes, so each
household's trace is a sidecar file.

- **The format.** `DisagreementTrace.write` writes canonical gzip-compressed
  JSON with no time or name in its header, so writing the same trace twice
  gives the same bytes. It returns the digest the record keeps.
- **Reading it back.** `DisagreementTrace.read(path, sha256)` refuses a file
  that does not match its digest.
- **What it holds.** The trace stores only its inputs: the timestamps, states,
  specifications, reference, posteriors and labels. Every other quantity is
  recomputed from them.

## Using it

```python
from sensor_modeling.datasets.structural_models import (
    DEFAULT_ENSEMBLE, ensemble_report, fit_samples, household_trace,
)

samples = fit_samples(statistics, fold.train, states=states, pseudo_windows=12.0)
trace = household_trace(recording, DEFAULT_ENSEMBLE, samples, household=home,
                        resolution=resolution, until=cutoff)
digest = trace.write(output_dir / f"{home}.json.gz")
section = ensemble_report([trace], DEFAULT_ENSEMBLE,
                          {home: {"file": f"{home}.json.gz", "sha256": digest}})
record = ExperimentRecord(..., structural_disagreement=section)
```

## What this does not do

- **No thresholds.** It sets no threshold and makes no abstention.
- **No evaluation.** It does not show whether disagreement signals errors.
  That is Phase 4's pre-specified question.
- **One setting.** It covers the filter's recursion, not the declared
  information sets.
- **The production pipeline.** The online filter is not changed.
