# v0.3 profile freeze: verification record

This records what was checked about the development-fit circadian profile
produced by the `Freeze v0.3 circadian profile` workflow, and what was
deliberately **not** done.

It is not the freeze commit. Committing the profile as the immutable artefact,
with the provenance items required by
[the candidate specification](V03_CANDIDATE_SPECIFICATION.md), is a separate
deliberate act.

## What was produced

| | |
| --- | --- |
| Profile SHA-256 | `e5288794d591c138d29fa6a9543840122a2ea8ca23a01f068fa1b310f0206eeb` |
| Development homes | 22, each with an individual file SHA-256 |
| Size convention | `equivalent_decimal_MB_and_binary_MiB` |
| Fitting revision | `6684135` |
| `test_outcomes_inspected` | `false` |
| Source | CASAS / Zenodo record 15708568, `labeled_data.zip`, checksum verified on download |

## Reproducibility

The same profile hash was produced three times, by two independently written
implementations of the cohort reconstruction and on two different machines:

1. a local run of the guard implemented in the closed PR #116;
2. a local run of the merged guard from PR #117;
3. the workflow on a clean `ubuntu-24.04` runner, downloading the archive from
   Zenodo and verifying its checksum.

Byte-identical output across all three. For an artefact that becomes immutable
once committed, agreement between independent implementations is a stronger
check than any single run.

## Cohort determinacy

The historical prose said "under 12 MB" without recording whether MB meant
decimal or binary. Both conventions select the **same** 22 homes: the largest
included file is `hh108` at 11,901,894 bytes and the smallest excluded is
`hh104` at 13,977,124, so the cohort sits in a 2.1 MB gap and the ambiguity
never reached the data. The artefact records that equivalence rather than
silently picking a convention.

## Face validity

Each state's peak stickiness hour, from the fitted profile:

| State | Peak hour | Range |
| --- | --- | --- |
| sleeping | 04:00 | 0.25 – 2.54 |
| bed_awake | 22:00 | 0.25 – 4.00 |
| bathroom_activity | 07:00 | 0.31 – 2.45 |
| away | 11:00 | 0.39 – 1.69 |
| home_active | 12:00 | 0.25 – 1.89 |
| kitchen_activity | 18:00 | 0.25 – 2.96 |
| home_inactive | 21:00 | 0.25 – 2.48 |

Every peak falls where ordinary daily rhythm would put it. That is a weak check
in the sense that it could not have detected a subtly wrong profile, but it
would have caught a badly broken one, and it did not.

## Usability

The profile loads into `StateOntology`, passes its construction validation, and
drives the pipeline to completion on development recordings, moving balanced
accuracy in the expected direction.

**Those runs are in-sample.** The profile was fitted on all 22 development
homes, so scoring any of them measures fit rather than transfer. The figures are
not reported here as a performance estimate, and the development estimate that
justified the candidate remains the held-out 0.449 to 0.460 from the 11/11
split documented in [Real-data validation](real_data.md).

## What was not done

- The profile has **not** been committed as the immutable frozen artefact.
- No primary test home has been read, scored, or inspected in any way.
- The fifth provenance item required by the specification, the exact v0.3
  candidate revision that will be scored, has not been declared.

The one-shot external scoring remains unrun and unrepeatable by design.
