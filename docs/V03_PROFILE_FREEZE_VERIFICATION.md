# v0.3 profile freeze: verification record

This page records the pre-freeze circadian-profile checks and, importantly, why
the first reproducible profile is **not** the immutable v0.3 candidate profile.
No primary external-test outcome has been inspected.

## First reproducible run: retained as a diagnostic, not accepted

The `Freeze v0.3 circadian profile` workflow succeeded on revision `6684135`
and produced:

| | |
| --- | --- |
| Profile SHA-256 | `e5288794d591c138d29fa6a9543840122a2ea8ca23a01f068fa1b310f0206eeb` |
| Historical development panel | 22 recordings, each with an individual file SHA-256 |
| Size convention | `equivalent_decimal_MB_and_binary_MiB` |
| `test_outcomes_inspected` | `false` |
| Source | CASAS / Zenodo record 15708568, `labeled_data.zip`, checksum verified on download |

The hash was reproduced independently and the 12 MB reconstruction itself is
stable. Both decimal MB and binary MiB select the same 22-recording historical
panel.

However, resident-count metadata identify `hh107` and `hh121` as two-resident
homes. The first profile fitted all 22 recordings, so it is **rejected as the
immutable single-resident external-validation candidate profile**. Its hash is
kept here as provenance for a reproducible development diagnostic, not as the
profile to be scored externally.

This rejection is metadata-based, not performance-based. It occurred before any
primary external-test home was read or scored.

## Corrected final fitting set

All 22 previously analysed recordings remain development-only. Prior outcome
inspection is sufficient to exclude a recording from the primary test cohort,
so `hh107` and `hh121` do not return to the test pool.

The final circadian parameter fit is instead restricted to the 20
single-resident members of the same already-inspected panel:

`hh101`, `hh102`, `hh103`, `hh105`, `hh106`, `hh108`, `hh110`, `hh111`,
`hh114`, `hh118`, `hh119`, `hh120`, `hh122`, `hh123`, `hh124`, `hh125`,
`hh126`, `hh127`, `hh129`, `hh130`.

No untouched home is added merely to preserve a fitting count of 22.

The freezer and independent validator use schema version 2 to distinguish the
22-recording development panel from the 20-home parameter-fitting subset and to
record the two metadata-based exclusions explicitly. An old all-22 artifact
cannot satisfy the new acceptance contract.

## Cohort determinacy

The historical prose said "under 12 MB" without recording whether MB meant
decimal or binary. Both conventions select the same 22-recording panel: the
largest included file is `hh108` at 11,901,894 bytes and the smallest excluded
is `hh104` at 13,977,124 bytes. The ambiguity therefore does not affect panel
membership.

## What remains to be done

- Run the corrected development-only workflow after the design correction is
  merged and CI passes.
- Independently validate the schema-v2 artifact: 22 quarantined panel records,
  exactly 20 fitting homes, exclusions exactly `hh107` and `hh121`, and the
  canonical profile SHA-256.
- Commit that corrected profile as the immutable candidate artifact together
  with the exact candidate revision and remaining provenance requirements.
- Freeze the untouched single-resident primary-test cohort manifest before any
  scoring.

The one-shot external scoring remains unrun. A positive, null, or negative
primary result will be accepted without retuning the candidate on the test
cohort.
