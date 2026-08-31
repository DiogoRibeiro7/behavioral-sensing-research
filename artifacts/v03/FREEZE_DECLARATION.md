# v0.3 circadian profile: freeze declaration

The profile in `v03_circadian_profile.json` is **frozen**. Under
[the candidate specification](../../docs/V03_CANDIDATE_SPECIFICATION.md) it is
immutable for the primary external test from this commit onward.

## Required provenance

| Item | Value |
| --- | --- |
| 1. Development homes | 22, listed with individual SHA-256 in `v03_circadian_profile.json` |
| 2. Fitting code revision | `66841350704b22f59338ad74e8713ccdcd612fde` |
| 3. Profile SHA-256 | `e5288794d591c138d29fa6a9543840122a2ea8ca23a01f068fa1b310f0206eeb` |
| 4. Test outcomes inspected | **No.** Asserted in the artefact as `test_outcomes_inspected: false` |
| 5. v0.3 candidate revision to be scored | `4ba25921409daef69c8c1b3469e23ae28fe875ac` |

The candidate is v0.2 inference plus the optional circadian state-dynamics
profile, with every other v0.2 choice retained.

## Declaration

No primary test home has been read, scored, or inspected in producing this
profile. The fit used development annotations only and never ran the behavioural
inference model; the freezing script is outcome-blind by construction.

Verification of this artefact, including reproduction of the hash by two
independent implementations and on a clean runner, is recorded in
[the freeze verification](../../docs/V03_PROFILE_FREEZE_VERIFICATION.md).

## What this commit does not do

It does not score the primary cohort. That remains unrun, and under the
specification it is one-shot: its result stands whether positive, null or
negative, and no parameter, mapping, threshold or cohort membership may be
changed in response and rescored as the same confirmatory validation.
