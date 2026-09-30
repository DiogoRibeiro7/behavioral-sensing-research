# Disagreement between evidence groups

The filter combines every sensor as conditionally independent given the
behavioural state. Suppose motion sensors point to one state and the bed to
another. The posterior settles between them, and its confidence alone does not
show that the evidence conflicted. ROADMAP Phase 4 lists disagreement between
independent evidence channels as one of the structures an abstention rule
could use.

This page describes diagnostics that measure it. Each filter update's evidence
is split into groups of sensors of one kind. The diagnostics record what each
group supports, how far the groups disagree, and whether one of them decides
the outcome.

**Status: diagnostic infrastructure.**

- **What it changes.** It sets no abstention rule and leaves every filter
  output unchanged. It runs no evaluation.
- **The open question.** Whether evidence-group disagreement predicts errors
  is for a later, pre-specified Phase 4 experiment, reported as a
  risk–coverage curve.

## The groups

Each sensor's group comes from its modality in the sensor registry
(`sensor_modeling.evaluation.evidence_groups.MODALITY_GROUPS`):

| Group | Modalities | Evidence about |
| --- | --- | --- |
| `motion` | motion, radar | movement in the rooms |
| `contact` | door, contact, vibration | doors and objects handled |
| `bed` | bed pressure | occupancy of the bed |
| `wearable` | wearable motion, wearable physiology, proximity | the person, wherever they are |
| `other` | any other modality with an emission model | — |
| `context` | none: the filter's prediction | what the dynamics and the time of day expect |

A sensor the registry does not declare falls in `other`.

The context is the filter's belief after the transition and before the
window's evidence. It carries every earlier window's evidence, so it is only
semi-independent of the sensor groups. The sensor groups are independent
given the state, by the model's own assumption, and share nothing else within
a window.

## What each group supports

- **A sensor group's posterior** is its evidence in the window alone. The
  log-likelihoods of its available sensors are summed, tempered by reliability
  and attribution exactly as the filter tempers them, and normalised over the
  states with no prior. Without a prior, the group speaks only for its own
  evidence.
- **The context's posterior** is the prediction.
- **The full posterior** is the filter's own.

A group *favours* the states whose log posterior is within `10⁻⁹` of its
largest. Several states can tie. A bed sensor cannot tell sleep from lying
awake, so it favours both, and choosing either one would be arbitrary.

## Missing is not disagreement

A sensor's status keeps the health semantics of the filter's `StateEstimate`:

| Sensor status | Meaning | In `StateEstimate` |
| --- | --- | --- |
| `unavailable` | reliability below the filter's evidence floor (0.05 by default) | `missing` |
| `reporting` | available, and supplied records in the window | reported |
| `silent` | available, no records, and a likelihood that is not flat: a trusted event sensor's silence counts against the states that would have triggered it | `silent` |
| `no data` | available, no records, and a flat likelihood: a state or sample sensor says nothing between reports | `silent` |

A group's status follows from its sensors:

| Group status | Meaning | Compared |
| --- | --- | --- |
| `absent` | the deployment has no sensor of the group | no |
| `unavailable` | every sensor of the group is unavailable | no |
| `uninformative` | the available sensors' combined likelihood is flat | no |
| `silent` | informative, and no available sensor reported | yes |
| `reporting` | informative, and some available sensor reported | yes |
| `predicted` | the context | yes |

Only the compared groups are *identifiable*. An absent, unavailable or
uninformative group has no posterior, no divergence and no vote, so missing
evidence can never register as disagreement.

- **Partly unavailable groups.** When a group has both unavailable and
  available sensors, its posterior uses only the available ones.
- **Residual evidence.** A sensor below the floor still enters the filter,
  tempered by its low reliability. Its residual evidence is kept in the
  decomposition below, which accounts for everything the filter combined.

## What is measured

For each window, over the identifiable groups
(`GroupWindow.to_dict()`):

| Quantity | Definition | Range |
| --- | --- | --- |
| group posterior | what each identifiable group supports | — |
| favoured | each group's tied most probable states | — |
| full posterior, state, runner-up | the filter's posterior and its two most probable states | — |
| supports decision | whether a group favours the filter's state | — |
| divergence from full | Jensen–Shannon divergence of a group's posterior from the full posterior, in bits | `[0, 1]` |
| state disagreement | whether some pair of groups favours disjoint sets of states | — |
| disagreements | those pairs | — |
| vote disagreement | the share of groups not voting for the plurality state, each splitting its vote across the states it favours | `[0, 1 − 1/M]` |
| pairwise divergence | Jensen–Shannon divergence of each pair, in bits | `[0, 1]` |
| discordance | generalised Jensen–Shannon divergence of all `M` groups, `H(mean) − mean H`, in bits | `[0, log₂ M]` |
| normalised discordance | the discordance over `log₂ M` | `[0, 1]` |
| conflicts | disagreeing pairs in which each group gives its own favoured states at least twice the likelihood of every state the other favours | — |
| contributions | each group's term in the log odds of the decision (below) | — |
| dominance | the group with the largest share of those terms, its share, and whether it dominates | `[0, 1]` |

The measures reuse the Jensen–Shannon functions of
[structural disagreement](STRUCTURAL_DISAGREEMENT.md), with the same base-2
logarithms and the same `10⁻¹²`-bit rounding floor. Identical groups diverge
by exactly nothing, and every divergence is symmetric.

- **Divergence and state disagreement.** They answer different questions. Two
  groups that favour the same state with different strength diverge without
  disagreeing on the state. Two groups that each weakly favour a different
  state disagree while hardly diverging.
- **The conflict ratio.** The ratio of 2 describes the evidence and decides
  nothing. `GroupDisagreementRecorder(conflict_ratio=...)` records with
  another, and the record states the ratio used.

### Dominance

With `s*` the full posterior's most probable state and `s2` the next, the log
odds between them split exactly into one term per group:

```text
log P(s* | evidence) / P(s2 | evidence)
    = log p(s*) / p(s2)               the context
    + Σ_g [ℓ_g(s*) − ℓ_g(s2)]         each sensor group
```

Here `p` is the prediction and `ℓ_g` the group's summed log-likelihood,
unavailable sensors included. The terms sum to the full posterior's log odds.

- **Share.** A group's share is the absolute value of its term over the sum of
  the absolute values of all the terms.
- **Dominance.** The group with the largest share *dominates* when its share
  exceeds one half. It then moves the decision more than every other group
  together. Two groups that cancel exactly, with half each, dominate nothing.
- **A negative term.** A group's term is negative when it argues against the
  decision.

## In the results

`GroupDisagreementRecorder` observes one filter. It is appended to the
filter's `observers`, which the filter calls after every update with the
update's terms and estimate. It reads them and changes nothing.

- **Terms.** `MultimodalBayesFilter.evidence_terms()` computes what an update
  would combine without changing the filter: the prediction, each sensor's
  tempered log-likelihood, its reliability, attribution and record count.
  `update()` now folds exactly these terms in. On two simulated households,
  every belief, information gain, completeness and contribution is identical
  to the last bit with and without the split.
- **Only the plain recursion.** A filter that overrides `update`, such as the
  history-aware one, adds terms the recorder would not see, so it is refused.

`recorder.results(trace)` returns a JSON-safe block for an experiment record's
`results`, with layout `evidence-group-disagreement/1`. It holds:

- the definitions, the conflict ratio, the evidence floor and the states;
- each sensor's modality and group;
- the summary:
  - the windows and the compared windows;
  - the shares of compared windows with a state disagreement and with a
    conflict;
  - the mean vote disagreement and mean normalised discordance;
  - the share of windows one group dominates;
- for each group:
  - its status counts and how often it is identifiable;
  - how often it favours the decision;
  - its mean divergence from the full posterior;
  - how often it dominates;
- for each pair, its mean divergence and how often it disagrees and
  conflicts;
- the trace file and its SHA-256, or `null`.

`recorder.write(path)` writes every window's record as canonical
gzip-compressed JSON with no time in its header, so the same windows always
give the same bytes, and returns the digest. `read_trace(path, sha256)`
refuses a file that does not match it.

## Using it

```python
from sensor_modeling.evaluation import GroupDisagreementRecorder

pipeline = BehaviouralSensingPipeline(registry, config=config)
recorder = GroupDisagreementRecorder.attach(pipeline.filter)
pipeline.run(observations)

digest = recorder.write(output_dir / "evidence-groups.json.gz")
results = recorder.results({"path": "evidence-groups.json.gz", "sha256": digest})
record = ExperimentRecord(..., results={"evidence_groups": results})
```

A single window's terms can be grouped directly with
`group_window(bayes.evidence_terms(now, batch, ...), modalities, states,
evidence_floor=...)`.

## What this does not do

- **No abstention.** It sets no threshold, and no decision reads it.
- **No evaluation.** It does not show whether disagreement signals errors.
  That is Phase 4's pre-specified question.
- **Not an ablation.** Removing one sensor at a time measures how the
  posterior changes without it, not what each kind of evidence present
  supports. Here every group is read from the same update.
- **No history terms.** The history-aware filter and the declared information
  sets are not covered.
