# Zenodo Archive

This repository is archived on Zenodo under a **concept DOI** that always
resolves to the latest version, plus a **version DOI** for each archived
release.

- Concept record: <https://zenodo.org/records/21337272>
- Concept DOI: <https://doi.org/10.5281/zenodo.21337272>

Cite the **version DOI** when your result depends on a specific release, which
is almost always the case for a reproducible experiment. Cite the **concept
DOI** when referring to the software project across versions.

## Archived releases

| Version | Record | Version DOI | Published |
| --- | --- | --- | --- |
| 0.7.0 | <https://zenodo.org/records/22977921> | `10.5281/zenodo.22977921` | 2026-09-26 |
| 0.6.0 | <https://zenodo.org/records/22968640> | `10.5281/zenodo.22968640` | 2026-09-25 |
| 0.5.0 | <https://zenodo.org/records/22799561> | `10.5281/zenodo.22799561` | 2026-09-16 |
| 0.4.0 | <https://zenodo.org/records/22791975> | `10.5281/zenodo.22791975` | 2026-09-15 |
| 0.3.0 | <https://zenodo.org/records/22729298> | `10.5281/zenodo.22729298` | 2026-09-12 |
| 0.2.0 | <https://zenodo.org/records/22171268> | `10.5281/zenodo.22171268` | 2026-08-30 |
| 0.1.3 | <https://zenodo.org/records/21340388> | `10.5281/zenodo.21340388` | 2026-07-13 |
| 0.1.3 | <https://zenodo.org/records/21340349> | `10.5281/zenodo.21340349` | 2026-07-13 |
| 0.1.2 | <https://zenodo.org/records/21337356> | `10.5281/zenodo.21337356` | 2026-07-13 |
| 0.1.1 | <https://zenodo.org/records/21337273> | `10.5281/zenodo.21337273` | 2026-07-13 |
| 0.1.0 | <https://zenodo.org/records/17070042> | `10.5281/zenodo.17070042` | 2025-09-07 |

Zenodo holds two `0.1.3` records with different archive contents, created four
minutes apart. The later one, `21340388`, was created four seconds after the
`v0.1.3` GitHub release now on the repository was published. The earlier one
predates that release. Cite `10.5281/zenodo.21340388` for `0.1.3`.

## Two concept DOIs

The project has two concept DOIs, because the archive started a new lineage
once:

| Concept DOI | Covers | Status |
| --- | --- | --- |
| `10.5281/zenodo.21337272` | 0.1.1 onwards | **Current.** Cite this. |
| `10.5281/zenodo.17070041` | 0.1.0 only | Superseded. Does not track later releases. |

`0.1.0` was archived in September 2025 under concept `17070041`. When `v0.1.1`
was released on 2026-07-13, Zenodo archived it as the first version of a new
concept, `21337272`, rather than adding it to the existing one. Every later
release, from `0.1.2` onwards, has been added to that new concept.

This is a property of how Zenodo assigns concept records, not something the
repository chose. It cannot be undone from this side: a record's concept is
fixed when the record is created. The earlier DOI still resolves and still
correctly identifies `0.1.0`, so nothing already published is broken. What it
no longer does is stand for "all versions", which is why every citation target
in this repository uses the new concept.

The `0.2.0` record declares `isVersionOf 10.5281/zenodo.17070041`, the older
concept, so the link between the two lineages can be read from the archive
itself.

Earlier revisions of this page said that the new lineage began with `0.2.0`,
and that the `0.2.0` record declares `isNewVersionOf 10.5281/zenodo.17070042`.
The archive shows both statements were wrong.

If Zenodo support later merges the lineages, the older concept DOI becomes the
correct one again and these references should move back.

## Metadata

Archive metadata is held in [`.zenodo.json`](.zenodo.json) and must stay
consistent with [`CITATION.cff`](CITATION.cff) and `pyproject.toml`. Tests in
`tests/test_project_metadata.py` check that the versions and DOIs agree, so a
partial bump fails the build rather than reaching Zenodo.

## Citing this software

```bibtex
@software{ribeiro2026sensor,
  author    = {Ribeiro, Diogo},
  title     = {Sensor Modeling Research Toolkit},
  year      = {2026},
  version   = {0.6.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21337272},
  url       = {https://doi.org/10.5281/zenodo.21337272}
}
```

To cite a specific release, replace the concept DOI with that release's version
DOI from the table above.

## What the archive represents

Research software. It is **not** a medical device, and no claim of clinical
effectiveness is made or supported.

Releases up to `0.2.0` report results from the bundled simulator only. From
`0.3.0`, the documentation also reports results on public CASAS recordings,
in `docs/real_data.md` and, from `0.6.0`, in
`docs/PHASE1_MATCHED_BASELINES.md`. Those are development results from one
research group's instrumentation, not clinical validation. Figures from the
simulator remain simulator figures and are not field-performance estimates.
