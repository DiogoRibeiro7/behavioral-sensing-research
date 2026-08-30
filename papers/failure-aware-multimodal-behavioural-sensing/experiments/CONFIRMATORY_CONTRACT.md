# Confirmatory scientific contract

**Status:** pre-specified, no confirmatory results examined.

This file is the authoritative decision contract for the confirmatory simulation study in *Failure-Aware Multimodal Behavioural Sensing*. If provisional manuscript prose differs from this file or from `config.json`, the manuscript must be corrected to match this contract before reporting confirmatory results. The contract must not be changed in response to primary-result values.

## Replication unit and seed independence

The independent replication unit is one simulated household trajectory. Time points are not replications. Production household seeds are

\[
s_i = 10000 + 4i,\qquad i=0,\ldots,N-1.
\]

The stride of four is required because H3 attribution scenarios derive degradation streams at offsets `seed + 1`, `seed + 2`, and `seed + 3`. Sharded execution is permitted only when the merged seed union is exactly the frozen seed set, with no duplicates or omissions.

The initial production size is \(N=200\). It may increase, without inspecting hypothesis direction or changing any analysis choice, only as needed to satisfy the pre-specified H2 Monte Carlo precision criterion, up to \(N=1000\).

## H1 — graceful degradation

H1 evaluates the pre-specified random-missingness rates 0%, 5%, 10%, 20%, and 40% on paired household trajectories. The 0% arm is the reference. Balanced accuracy, log loss, Brier score, calibration error, and abstention are reported at each rate, together with paired changes from the 0% reference and household-level uncertainty.

H1 is descriptive about the degradation curve; no post-result breakpoint or subset of missingness rates may be selected as the primary H1 result.

## H2 — reduced multimodal deployment

H2 has **one primary non-inferiority contrast**:

- reference: `all_modalities` (10 sensors);
- reduced deployment: `radar_door_bed_wearable` (5 sensors);
- primary metric: balanced accuracy;
- estimand:

\[
D_{H2}=\operatorname{BA}_{\mathrm{full}}-\operatorname{BA}_{\mathrm{reduced}};
\]

- absolute non-inferiority margin: \(\Delta=0.02\);
- confidence level: 95% household-paired bootstrap interval.

The reduced configuration supports the primary H2 claim iff

\[
\boxed{U_{0.95}(D_{H2}) < 0.02,}
\]

where \(U_{0.95}\) is the upper endpoint of the two-sided 95% paired bootstrap interval. The inequality is strict. Failure to satisfy it is a negative H2 result, not a reason to select another sensor subset or change the margin.

The other pre-specified sensor configurations remain secondary ablations. Log loss, Brier score, calibration error, and abstention are reported for safety/context, but they are not additional gates that can override the frozen balanced-accuracy decision rule. Failure robustness is addressed separately by H4.

The H2 Monte Carlo precision gate is the configured `MCSE <= 0.002` criterion for paired balanced-accuracy gaps. Replication may be extended prospectively for precision only; the non-inferiority margin remains fixed.

## H3 — attribution under contamination

H3 reports occupancy-aware attribution effects separately for every pre-specified scenario. The primary interpretation is directional and scenario-specific: attribution should produce negligible change in `resident_alone` and improve state/probability performance when visitor or carer contamination is present.

No pooled contaminated-scenario effect will be introduced after results are observed. Scenario-level balanced-accuracy gain, calibration gain, and visitor-detection performance are reported with household-level uncertainty.

## H4 — failure-aware versus health-naive inference

Both H4 arms receive **the same degraded observation record**: 30% random record loss plus the pre-specified `bed_pressure` dropout beginning on day 35 for 14 days.

The treatment arm uses the normal failure-aware inference chain. The control arm preserves the same missing observations and health statuses but forces sensor-health reliability weights to one. It therefore tests the value of using health reliability in inference; it does **not** replace missing observations with zeros and does **not** treat missing evidence as observed silence.

For every metric, the frozen estimand is

\[
M_{\mathrm{failure\ aware}}-M_{\mathrm{health\ naive}}.
\]

The sign must be interpreted according to the metric: positive is favourable for balanced accuracy, while negative is favourable for log loss, Brier score, and calibration error. Abstention is reported without assigning an intrinsically favourable direction.

The current confirmatory H4 regime is random missingness plus a programmed sensor outage. The exploratory activity-correlated-loss experiment remains separate exploratory evidence and must not be described as part of the confirmatory H4 estimand.

## H5 — non-additive sensor value

For each pre-specified pair \((i,j)\), H5 estimates

\[
I_{ij} = \Delta_{ij}-(\Delta_i+\Delta_j)
\]

on paired household trajectories. All four interaction pairs are reported. No pair may be promoted or removed on the basis of the confirmatory values.

## Pairing and uncertainty

All paired contrasts are household-seed contrasts. Production shards are merged and deterministically sorted by seed before summaries are computed. A merged production artifact is valid only after exact seed-union and provenance checks pass.

Uncertainty is calculated over households, never over individual time points. Bootstrap resampling is at household level with the pre-specified number of resamples.

## Result-status and provenance rules

A result may be labelled `confirmatory` only when all of the following hold:

1. the exact frozen configuration is used;
2. the exact frozen Git revision is used with a clean working tree;
3. at least 200 independent household trajectories are present;
4. the H2 Monte Carlo precision gate passes;
5. all production shards agree on config SHA-256, Git revision, and numerical environment;
6. the merged seed union exactly equals the frozen seed set.

Otherwise the result is `pilot-or-incomplete` and cannot be used as a confirmatory manuscript result.

## No post-result tuning

After the scientific freeze, confirmatory values may not be used to alter hypotheses, margins, sensor configurations, metrics, seed rules, failure regimes, bootstrap rules, or the interpretation of metric direction. Any scientifically necessary change after primary-result access starts a new explicitly versioned experiment and leaves the original frozen result intact.

All simulator outcomes remain simulator-derived and are not estimates of field performance.
