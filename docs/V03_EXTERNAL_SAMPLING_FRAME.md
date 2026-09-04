# v0.3 external sampling frame

Status: **historical — frozen prospectively, and the primary scoring has since been executed.**
This document records the sampling frame as it stood before any test home was
enumerated or scored. It is retained unchanged as the pre-registration record.
The scoring ran once on 2026-09-04; see `artifacts/v03/external_primary_result.json`.

This document narrows the external-validation sampling frame to one finite public release so the eligible cohort cannot later expand or contract opportunistically.

## Frozen source frame

Only homes contained in **CASAS Smart Home dataset v1**, Zenodo record `15708568`, file `labeled_data.zip`, are eligible for the primary v0.3 external-validation cohort.

Frozen source identity:

- DOI: `10.5281/zenodo.15708568`
- release: `v1`
- published: `2025-06-20`
- archive: `labeled_data.zip`
- archive size: `236037656` bytes
- archive checksum: `md5:ec37d679e85a6ae39e84994888afd514`

Later CASAS records, companion releases, replacements, corrections, or additional public CASAS recordings are outside the primary sampling frame. They may support later replication but cannot be added to this one-shot primary cohort.

## Resident-count gate

Resident count is taken only from the published Zenodo record metadata and is frozen in `artifacts/v03/casas_v1_resident_registry.json`.

A primary candidate must be explicitly listed as **one resident**. Two-resident, >2-resident, and unknown-resident homes are ineligible. Resident count is never inferred from filename, floor plan, activity labels, sensor behaviour, or model output.

## Development firewall

The full 22-recording historical `hh` panel remains development-only regardless of current eligibility metadata. Previous outcome inspection is sufficient for exclusion.

Therefore the primary cohort is selected as

\[
\mathcal T
=\mathcal L\cap\mathcal R_1\cap\mathcal E\setminus\mathcal D,
\]

where

- \(\mathcal L\) is exact membership in the frozen `labeled_data.zip` archive;
- \(\mathcal R_1\) is the published one-resident registry;
- \(\mathcal E\) is prospective parser/mapping eligibility under the frozen external-validation contract;
- \(\mathcal D\) is the complete 22-home development panel.

No model prediction, accuracy, calibration statistic, state recall, or v0.2/v0.3 comparison may be used to define any of these sets.

## Remaining prospective eligibility audit

For each archive member in \(\mathcal L\cap\mathcal R_1\setminus\mathcal D\), the pre-scoring audit must record:

1. raw archive path, byte size, and SHA-256;
2. home identifier and resident-count registry match;
3. source format and the exact frozen adapter used;
4. whether timestamps can be resolved without guessing;
5. whether motion and/or door events can be represented under existing observation semantics;
6. the set of prospectively mapped behavioural states represented by annotations;
7. whether at least five of the seven frozen states are represented;
8. whether mapped labelled coverage is non-zero;
9. any parser/provenance defect and its disposition.

This audit may inspect raw sensor rows and annotation semantics. It may not instantiate the behavioural inference model, produce state predictions, or compute evaluation metrics.

## Cohort freeze

The complete eligible-home manifest and raw-file checksums must be committed and pass CI before any primary scoring begins. Once that manifest is frozen, there is no performance-based home removal, replacement, or addition.

The statistical unit remains the home, and the one-shot primary estimand remains the median paired household difference

\[
D_h=B_h^{(0.3)}-B_h^{(0.2)}.
\]

Nothing in this sampling-frame freeze changes the already frozen model, profile, mapping principles, metric definitions, or analysis plan.
