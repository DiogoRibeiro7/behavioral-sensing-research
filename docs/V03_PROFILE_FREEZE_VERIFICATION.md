# v0.3 profile freeze: verification record

No primary external-test outcome has been inspected. This page records the
rejected first profile and the accepted corrected development-only fit.

## First reproducible run: retained as a diagnostic, not accepted

The first workflow fit all 22 historically analysed `hh` recordings and produced
profile SHA-256
`e5288794d591c138d29fa6a9543840122a2ea8ca23a01f068fa1b310f0206eeb`.
The hash was reproducible and the historical 12 MB reconstruction was stable,
but CASAS resident-count metadata identify `hh107` and `hh121` as two-resident.
The all-22 profile is therefore retained only as a reproducibility diagnostic.

This rejection was metadata-based and occurred before any primary test home was
read or scored.

## Accepted corrected profile

The corrected one-shot workflow run `33400014853` used exact fitting revision
`57dfa6bf1ad44f47e60b0aad811d360abccaa4a0`. It reconstructed the same
22-recording historical development panel, excluded `hh107` and `hh121` from
parameter estimation, and fitted the remaining 20 single-resident homes:

`hh101`, `hh102`, `hh103`, `hh105`, `hh106`, `hh108`, `hh110`, `hh111`,
`hh114`, `hh118`, `hh119`, `hh120`, `hh122`, `hh123`, `hh124`, `hh125`,
`hh126`, `hh127`, `hh129`, `hh130`.

The accepted profile SHA-256 is
`5f03753feddc90f379fada7802c18ce932ffad7bd9ee9774bb88828fde80d539`.
Independent recomputation from the canonical fit object gives the same hash.

The schema-v2 validator passed with:

- 22 unique quarantined development-panel recordings;
- exactly 20 fitting homes;
- exclusions exactly `hh107` and `hh121`;
- `fit.recordings = 20`;
- `shrinkage_hours = 2.0`;
- `minimum_multiplier = 0.25`;
- `maximum_multiplier = 4.0`;
- seven profile states with 24 hourly multipliers each;
- positive labelled duration;
- `test_outcomes_inspected = false`;
- fitting revision exactly `57dfa6bf1ad44f47e60b0aad811d360abccaa4a0`.

The workflow downloaded CASAS / Zenodo record `15708568`, verified the declared
`labeled_data.zip` checksum `md5:ec37d679e85a6ae39e84994888afd514`, and
uploaded only the fitted metadata/profile artifact. Raw CASAS data are not
committed.

## Cohort determinacy

Both decimal MB and binary MiB interpretations of the historical "under 12 MB"
rule select the same 22-recording panel. The artifact records both evaluations
and uses the stricter 12,000,000-byte effective cutoff. This ambiguity therefore
does not affect development-panel membership.

## What is now frozen

The circadian profile is immutable after its freeze PR is merged. The prospective
v0.3 identity is the exact candidate code revision plus the profile SHA-256; all
other v0.2 inference choices remain unchanged.

This is not yet the full external-validation candidate freeze. Before any primary
scoring, the untouched eligible single-resident test cohort must be frozen in a
machine-readable manifest with raw-file checksums and the remaining contract
provenance must be assembled. Until then the one-shot external test remains
blocked.
