# Exploratory mechanism plan

Status: **declared before execution. No quantity defined here has been computed on
any recording.**

This plan governs the follow-up mechanism analyses for *When Confidence Is Not
Information*. It does not amend the frozen analysis contract (commit `34f9797`,
reproduced in the manuscript's appendix) or its result
(`artifacts/paper1/uncertainty_panel.json`). H1, H2 and H3 have been evaluated and
failed, and nothing below re-tests them or can change that outcome.

The plan must not be amended in response to its results. If a defect makes an
analysis non-executable, the correction and its reason are added as a dated
addendum before any output of that analysis is inspected.

## 1. What this is, and what it is not

These analyses are **post-hoc**. They were motivated by the frozen panel result and
by a review of the manuscript, which identified explanations consistent with that
result but not tested by it:

- evidence strength averages *absolute* support, so evidence contradicting the
  prediction raises it as much as supporting evidence;
- information gain is unsigned, so a large update on an incorrect step may move the
  belief toward the true state rather than away from it, as could happen shortly
  after a real change of state;
- homes may differ in whether errors occur during quiet or active intervals.

They use the same 22 development homes that produced the result they try to
explain. Their outcomes are exploratory mechanism evidence: they can make an
explanation more or less plausible on this panel, but cannot confirm it. Nothing
here defines an abstention score, a sign-reversed diagnostic, a threshold, a
coverage target or a deployment rule.

Every quantity listed in section 5 is reported whatever it shows. No analysis may be
added, dropped, re-parameterised or re-stratified after outcomes are seen unless the
manuscript labels that change as made after inspection.

## 2. What has already been seen

The review computed the following from the frozen per-home artifact, and they are
not new evidence when reported again:

- the per-home AUCs, their ranges and the number of homes on each side of 0.5;
- a negative rank correlation across homes between confidence AUC and
  information-gain AUC (Spearman -0.51);
- per-home medians of each diagnostic on correct and incorrect steps. Information-gain
  AUC exceeds confidence AUC in exactly five homes (`hh105`, `hh106`, `hh107`,
  `hh111`, `hh124`). In those five, median confidence on incorrect steps is
  0.974–0.980, higher than in any other home (next highest: `hh102`, 0.969).

Analysis M4 tests one candidate explanation of that cross-home pattern. It is not an
independent discovery of the pattern itself.

## 3. Data, scoring and estimand

All of these are unchanged from the frozen panel.

- **Homes:** the 22-home `development_panel` in
  `artifacts/v03/external_cohort_manifest.json`, with every source file verified
  against its recorded byte size and SHA-256 digest.
- **Archive:** Zenodo record 15708568, `labeled_data.zip`, 236,037,656 bytes, MD5
  `ec37d679e85a6ae39e84994888afd514`.
- **Inference:** the default path with the circadian profile disabled, a five-minute
  step, and timestamps read in `America/Los_Angeles`.
- **Scored steps:** steps with a mapped label, after `scoring_steps` removes the
  duplicated final step. A step is correct when the reported state equals the label.
  An abstaining step reports `UNKNOWN` and is therefore incorrect.
- **Estimand:** the within-home common-language AUC `A`, with half credit for ties,
  as implemented by `_correctness_auc`, and its median over homes. For each quantity
  the favourable direction is stated below, and `A = 0.5` is the only reference
  point. Wherever a median over homes is reported, the number of contributing homes
  and the number with `A > 0.5` are reported with it.

## 4. Equivalence gate

Before computing any quantity in section 5, the runner recomputes the five frozen
diagnostics for every home. Every per-home AUC must match
`artifacts/paper1/uncertainty_panel.json` to within `1e-5`. Reruns of identical code
are known to differ by up to `2e-6`.

If any home fails the gate, the run stops and writes no mechanism output.

## 5. Analyses

### M1: signed evidence support

**Question.** Is the evidence-strength inversion produced by the absolute value?

With `s_jt` the support of sensor `j` for the reported most-probable state, and
`J_t` the sensors whose likelihood was applied (`EvidenceContribution.informative`),
compute per step:

| Quantity | Definition | Favourable direction |
| --- | --- | --- |
| Supporting magnitude `P_t` | mean over `J_t` of `max(s_jt, 0)` | higher |
| Contradicting magnitude `K_t` | mean over `J_t` of `max(-s_jt, 0)` | lower |
| Signed support `S_t` | mean over `J_t` of `s_jt`, equal to `P_t - K_t` | higher |

All three are 0 when `J_t` is empty. By construction, evidence strength is
`E_t = P_t + K_t`.

**Reported.** Median within-home `A` for `P_t`, `K_t` and `S_t`.

**Reading declared in advance.**

- If `A_K > 0.5` at the median, contradicting evidence is larger on incorrect
  steps, as the absolute-value explanation requires.
- If `A_P < 0.5` as well, supporting evidence alone is also larger on incorrect
  steps, and the absolute value does not by itself explain the inversion.

### M2: direction of large updates

**Question.** On incorrect steps with large information gain, does the update move
posterior mass toward the labelled state or away from it?

For a scored step `t` with label `y_t`, let `p_t` be its posterior. Let `p̂_{t|t-1}`
be the posterior of the immediately preceding filter update, propagated by
`StateOntology.transition` over the elapsed interval. Define

`Δ_t = p_t(y_t) - p̂_{t|t-1}(y_t)`.

`Δ_t` uses the label, so it is a mechanism diagnostic only and can never be a
candidate abstention score. The first update of each recording has no predecessor
and is excluded.

**Reconstruction gate.** For every scored step with a predecessor,
`D_KL(p_t || p̂_{t|t-1})` must match the recorded `information_gain` to within
`1e-6`. If any step in a home fails, M2 is not computed for that home, and the
number of such homes is reported.

**Reported, per home.** Let `Q75` be the 75th percentile of information gain over
that home's scored steps, computed by linear interpolation (`numpy.quantile`
default). Report:

- `U_h`: the share of incorrect steps with information gain `>= Q75` that have
  `Δ_t > 0`;
- the same share over all incorrect steps;
- the same share over correct steps with information gain `>= Q75`, for context.

For each, report the median over homes and the number of homes above 0.5. Steps with
`Δ_t = 0` exactly count as not positive, and their total is reported.

**Reading declared in advance.**

- A median `U_h` above 0.5 means large updates on incorrect steps mostly move toward
  the labelled state, which is consistent with lag.
- A median below 0.5 means they mostly move away from it.

### M3: proximity to a labelled change of state

**Question.** Is the inversion of the update-level diagnostics concentrated
just after changes in the labelled state?

A scored step is **near a change** if any of the three immediately preceding steps
(15 minutes) carries a label that differs from its own. Unlabelled preceding steps
do not create a change.

**Reported, per home and as medians over homes:**

- the share of scored steps near a change;
- the share of incorrect steps near a change;
- the within-home `A` of all five frozen diagnostics recomputed on steps **not** near
  a change, with median, contributing homes and homes above 0.5.

**Reading declared in advance.** If the median `A` for information gain and for
evidence strength moves closer to 0.5 than in the frozen result, the inversion is
partly concentrated near changes of state. The change in `|median A - 0.5|` is
reported for every diagnostic, and no minimum size is set.

### M4: quiet and active intervals

**Question.** Does whether a home's errors occur in quiet or active intervals track
the sign of its confidence and information-gain results?

A step is **quiet** when no sensor supplied any record over its interval (every
evidence contribution has `observations == 0`), and **active** otherwise.

**Reported, per home:**

- the share of scored steps that are quiet;
- accuracy on quiet steps and on active steps;
- `Q_h`, the share of incorrect steps that are quiet;
- within-stratum `A` for the five frozen diagnostics, separately for quiet and
  active steps, wherever the stratum contains both correct and incorrect steps (with
  the number of contributing homes).

**Reported, across homes:** Spearman rank correlations, with average ranks for ties,
between `Q_h` and the frozen confidence AUC, and between `Q_h` and the frozen
information-gain AUC. With 22 homes these are descriptive. No p-value or interval is
reported.

**Reading declared in advance.** A correlation below 0 with confidence AUC and above
0 with information-gain AUC fits the reading that homes whose errors fall in quiet
intervals are those where confidence is inverted and information gain is not.

## 6. Execution and outputs

- **Runner:** `scripts/analyse_uncertainty_mechanisms.py`, reusing the source
  verification of `scripts/analyse_uncertainty_panel.py`. It may be written and
  tested on synthetic inputs. It must not be run on any CASAS recording before this
  plan is merged into `develop`.
- **Workflow:** a GitHub Actions workflow that downloads and verifies the archive
  above and runs the merged runner. The artifact is the output of the first
  successful run. If an earlier run fails for a technical reason, the failure and
  the rerun are recorded in the pull request that freezes the artifact.
- **Artifact:** `artifacts/paper1/uncertainty_mechanisms.json`, frozen exactly as
  produced. It carries the per-home values for every quantity in section 5, the
  gate results, and the medians and counts.

## 7. Reporting

Results may enter the manuscript only in a section labelled exploratory that:

- cites this plan;
- reports every quantity in section 5, including gate failures and homes not
  computed;
- states which declared readings the results fit and which they do not;
- leaves the reported outcome of H1, H2 and H3 unchanged.
