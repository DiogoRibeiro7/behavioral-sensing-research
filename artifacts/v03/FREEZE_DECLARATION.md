# v0.3 circadian profile: freeze declaration

The profile in `v03_circadian_profile.json` is frozen as the trainable circadian
parameter artifact for v0.3. This freezes the profile; it does **not** yet
authorise primary external scoring. The full candidate freeze still requires the
untouched primary-cohort manifest and the remaining provenance items in the
external-validation contract.

## Frozen identity

| Item | Value |
| --- | --- |
| Candidate model form | v0.2 inference + circadian prior only |
| Candidate code revision | `57dfa6bf1ad44f47e60b0aad811d360abccaa4a0` |
| Profile SHA-256 | `5f03753feddc90f379fada7802c18ce932ffad7bd9ee9774bb88828fde80d539` |
| Historical development panel | 22 previously inspected `hh` recordings, each hashed in the JSON artifact |
| Parameter-fitting subset | 20 single-resident homes |
| Metadata exclusions | `hh107`, `hh121` — two-resident CASAS recordings |
| Shrinkage | 2.0 equivalent hours |
| Multiplier bounds | 0.25 to 4.0 |
| Representation | seven states × 24 local-hour multipliers |
| Primary-test outcomes inspected | **No** (`test_outcomes_inspected: false`) |

The scored v0.3 model is therefore identified prospectively by the pair

`(candidate code revision, profile SHA-256)`.

No untouched home was added to parameter fitting merely to retain a count of 22.
All 22 historical recordings remain development-only because their outcomes were
already inspected.

## Execution provenance

The accepted profile was produced by one-shot workflow run `33400014853`.
The orchestration commit was `75a7833d25109ca5714dadaa6ea6e1bc17a155ba`,
but that workflow checked out, verified and fitted the exact frozen code revision
`57dfa6bf1ad44f47e60b0aad811d360abccaa4a0`.

The uploaded artifact was `v03-circadian-profile-freeze-v2` with GitHub artifact
digest
`sha256:f9518d1e2ee3c4db210fcd5492622e4682c3c13f7c0aaa94ac5ce137236adc47`.
The workflow independently validated schema version 2 and asserted that the
20-home fitting set excludes `hh107` and `hh121`.

Source data were CASAS / Zenodo record `15708568`, file `labeled_data.zip`, size
236,037,656 bytes, with declared checksum
`md5:ec37d679e85a6ae39e84994888afd514` verified before extraction.

## Superseded diagnostic

The earlier all-22 profile
`e5288794d591c138d29fa6a9543840122a2ea8ca23a01f068fa1b310f0206eeb`
remains reproducible but is explicitly rejected as the v0.3 parameter artifact
because it fitted the two multi-resident recordings.

## Scoring remains blocked

No primary external-test home has been read, scored or inspected in producing or
validating this artifact. Before scoring, the repository must still freeze the
exact eligible untouched single-resident cohort with raw-file checksums and then
complete the candidate freeze manifest required by the external-validation
contract. A positive, null or negative primary result will then stand without
retuning or rescoring as the same confirmatory validation.
