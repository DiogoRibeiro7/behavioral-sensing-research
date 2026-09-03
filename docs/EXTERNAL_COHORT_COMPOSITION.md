# External test cohort: what it is, and how a result should be read

The eligible external-validation cohort is 43 single-resident CASAS homes
outside the 22-recording development panel, selected by the contract's criteria
and by nothing about how well they would score. This page records a property of
that cohort which changes how its result should be interpreted, written before
any scoring.

## Composition

| Family | Homes | Median labelled coverage | Median mapped states |
| --- | --- | --- | --- |
| `tm` | 25 | 71.5% | 6 |
| `hh` | 8 | 85.6% | 6 |
| `rw` | 5 | 75.7% | 6 |
| `mn` | 3 | 82.7% | 6 |
| `ihs` | 2 | 68.3% | 6 |

## The candidate has only ever seen `hh`

Every development result behind v0.3 — the emission-rate measurements, the
recoverable-signal ceiling, the feature ablation, the circadian fit, and the
held-out comparison that selected the candidate — came from `hh` recordings.

**Only 8 of the 43 eligible test homes, 19%, are from that family.** The other
81% come from instrumentation the candidate has never been exposed to, in homes
whose sensor naming this project's adapter learned to read only while screening
the cohort.

That is a deliberately harder test than the development panel implies, and it is
the right test: a method that only transfers within one research group's
deployment convention has not transferred. But it means the result carries two
questions at once, and they should not be conflated.

## How to read each outcome

**A positive result** would be strong, because it would show transfer across
instrumentation families rather than merely across homes.

**A null or negative result is ambiguous on its own.** It could mean the
circadian prior does not transfer, or that the adapter's mapping of an
unfamiliar vocabulary loses information, or that these families differ in ways
the seven-state ontology does not accommodate. The primary estimand cannot
separate those.

The pre-specified secondary outcomes help but do not resolve it. A per-family
breakdown of the paired difference is worth reporting alongside the primary
result — not as a substitute for it, and not as a basis for excluding families
after seeing their outcomes, which the contract forbids.

## What this does not license

Nothing here is a reason to narrow the cohort. Eligibility is fixed by the
contract's criteria, and dropping a family because it scored badly would be
exactly the post-hoc selection the one-shot design exists to prevent.

This page is a statement about interpretation, written before the outcome is
known, so that a disappointing result is read carefully rather than explained
away afterwards.

## A known discrepancy in the recorded registry digest

The frozen manifest records `source.registry_sha256` as
`d5f0c1086ba06cbf095206019a3cff2bebfd6cedf09ecd6b496d703443393066`. Recomputing
it from `artifacts/v03/casas_v1_resident_registry.json` as stored in the
repository gives
`2876c648a696baa8bf5f5ef5fed06cca2288113105919699e401ea7fad1a0ba1`.

The registry has not changed. The two values are the same content with different
line endings: the recorded digest was computed from a Windows working copy whose
checkout had converted LF to CRLF, and the repository stores the file with LF.
`build_external_cohort_manifest.py` now normalises line endings before hashing,
so a rebuild would record the LF value, but the frozen manifest is left exactly
as it was — editing it would change `manifest_sha256` and invalidate the freeze
declaration that records it.

This digest is recorded provenance and is not verified at scoring time, so it
gates nothing. It is documented here because an auditor recomputing it would
otherwise find a mismatch and reasonably suspect the registry had been altered.
The cohort membership was determined by the registry's content, which is
unaffected.
