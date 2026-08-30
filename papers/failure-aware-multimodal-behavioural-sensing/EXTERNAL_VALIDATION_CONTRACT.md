# External validation contract

Status: **pre-specified, not yet scored on the primary test cohort**.

This contract governs the real-home external validation of the paper **Failure-Aware Multimodal Behavioural Sensing**. It is frozen after the N=200 simulator result has been accepted and before any primary external-test outcomes are inspected.

## 1. Development data are not confirmatory external test data

The 22 single-resident CASAS `hh` homes already analysed in `docs/real_data.md` are designated **development-only**. Their aggregate performance, per-state recall, emission-rate fitting, supervised recoverability ceiling, and feature-ablation results have already been inspected and used to motivate model-development hypotheses. They must not be reused as the primary external-validation test set.

Any home, recording, split, or label whose outcome was inspected before this freeze is development data, even if it was called "held out" relative to a narrower fitting step.

## 2. Primary test cohort

The primary external-validation cohort consists of eligible **single-resident labelled CASAS homes outside the previously analysed `hh` development panel**.

Eligibility is determined before scoring and without reference to model performance:

1. the recording belongs to the same published CASAS release or another public CASAS recording whose raw data were not used in the 22-home development analysis;
2. the home is single-resident;
3. the recording contains activity annotations sufficient to map at least five of the seven frozen behavioural states after the prospective mapping audit;
4. motion and/or door observations can be represented through the existing canonical observation semantics without inventing sensor measurements;
5. timestamps and location/room metadata can be resolved from dataset documentation or explicit fields;
6. the recording has non-zero mapped labelled coverage.

There is **no file-size cutoff** and no performance-, coverage-, sensor-count-, or difficulty-based exclusion after scoring begins. If an otherwise eligible file cannot be processed because of a parser or provenance defect, the defect and disposition must be recorded before any model output from that file is inspected.

The exact eligible test-home identifiers and raw-file checksums must be frozen in a machine-readable cohort manifest before primary scoring.

## 3. Development / test firewall

The previously examined 22-home `hh` panel may be used to develop the v0.3 candidate, including:

- fitting trainable emission parameters;
- choosing a circadian parameterisation;
- choosing the finite history representation;
- fixing numerical stabilisation and implementation details;
- debugging dataset adapters;
- establishing deterministic activity/state mappings where the dataset vocabulary requires them.

The primary test cohort may be used only for final scoring after the v0.3 candidate is frozen. No parameter, mapping, threshold, state rule, sensor semantic, history length, circadian basis, or model structure may be changed in response to its outcomes.

A defect that makes the frozen candidate scientifically non-executable may be corrected only transparently. If the correction can affect predictions, the previous external result is invalidated and the corrected version must be treated as a new candidate with a newly documented test status; the same test outcomes may not be presented as prospectively untouched evidence.

## 4. Frozen model comparison

The external study compares, on every eligible primary test home:

- **v0.2 frozen architecture / declared defaults**: the pre-development reference corresponding to the simulator paper architecture;
- **v0.3 candidate**: the model frozen after development on the 22-home panel and before primary-test scoring.

A supervised classifier is **not** a primary comparator. It remains a development diagnostic for recoverability and must not be used to tune or select the v0.3 candidate on the primary test cohort.

## 5. Primary and secondary outcomes

### Primary outcome

The primary external-validation outcome is **household-level balanced accuracy** over mapped labelled time points.

For home \(h\), let \(B_h^{(0.2)}\) and \(B_h^{(0.3)}\) denote balanced accuracy for the reference and candidate. The primary paired contrast is

\[
D_h = B_h^{(0.3)}-B_h^{(0.2)}.
\]

The primary summary is the median \(D_h\) across eligible homes, accompanied by the full home-level distribution and a two-sided 95% paired bootstrap interval resampling homes, not time points.

This is an **improvement estimand**, not a non-inferiority test and not a field-performance threshold. A positive median difference supports improved external discrimination relative to v0.2; a non-positive result is a valid negative result.

### Secondary outcomes

Secondary outcomes are:

- calibration error;
- multiclass log loss;
- Brier score;
- abstention rate;
- mapped labelled coverage;
- state-specific recall for each frozen ontology state observed in at least three test homes;
- per-home balanced accuracy for both versions.

Coverage is always reported next to accuracy. No pooled time-point confidence intervals are permitted.

## 6. Mapping contract

The seven behavioural states remain:

- `AWAY`;
- `HOME_ACTIVE`;
- `HOME_INACTIVE`;
- `SLEEPING`;
- `BED_AWAKE`;
- `BATHROOM_ACTIVITY`;
- `KITCHEN_ACTIVITY`.

Dataset activity labels are mapped prospectively. Event annotations that mark transitions (for example, leaving or entering) must not be silently treated as persistent states. Unmapped activities remain unmapped rather than being forced into `UNKNOWN`.

Sensor semantics remain those already documented by the adapters: activation events are observations; missing events are not zeros; unsupported sensor modalities are excluded and reported rather than coerced into an invented modality.

Any new source-format adapter required for an untouched home must be tested against file-format semantics without inspecting that home's model performance.

## 7. Candidate freeze requirements

Before scoring any primary test home, the repository must contain a candidate freeze manifest recording at minimum:

- candidate Git commit;
- exact model/configuration hashes;
- exact activity-mapping table;
- exact dataset-adapter revision;
- trainable parameters and the development homes used to estimate them;
- circadian basis/knots or equivalent representation;
- history-window definition;
- state ontology;
- evaluation step size;
- metric definitions;
- primary-test cohort manifest and raw-file checksums;
- Python/package environment.

The candidate freeze commit must pass CI before primary scoring.

## 8. Statistical unit and uncertainty

The independent unit for external validation is the **home**. Repeated time points within a home are not independent replications.

Primary uncertainty is obtained by paired bootstrap over homes. If the eligible primary cohort is too small for stable interval estimation, report the individual paired home contrasts and exact cohort size rather than manufacturing time-point precision.

No post-hoc subgroup may replace the all-eligible-homes primary result. Pre-specified descriptive stratification by recording format or sensor availability is permitted only if defined before scoring.

## 9. Relationship to simulator evidence

The N=200 simulator experiment and the external-home experiment answer different questions. The external study does not re-test the simulator's numerical performance levels and does not alter the already accepted H1-H5 simulator findings.

The external question is whether the frozen architecture and its prospectively developed v0.3 candidate transfer to real homes, and whether the targeted additions motivated by development data improve that transfer.

All external results must remain bounded by the available instrumentation and annotation semantics. No result is to be described as clinical effectiveness.

## 10. Stopping rule

The primary test is run once on the complete frozen eligible cohort. There is no sequential stopping on effect direction. Missing or failed homes are resolved according to the pre-scoring eligibility and defect rules above, not according to observed performance.
