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
| 0.2.0 | <https://zenodo.org/records/22171268> | `10.5281/zenodo.22171268` | 2026-08-30 |
| 0.1.0 | <https://zenodo.org/records/17070042> | `10.5281/zenodo.17070042` | 2025-09-07 |

## Two concept DOIs

Zenodo minted a **new concept lineage** when `0.2.0` was archived through the
GitHub release integration, rather than adding a version to the existing one.
The project therefore has two:

| Concept DOI | Covers | Status |
| --- | --- | --- |
| `10.5281/zenodo.21337272` | 0.2.0 onwards | **Current.** Cite this. |
| `10.5281/zenodo.17070041` | 0.1.0 only | Superseded. Does not track later releases. |

This is a property of how Zenodo assigns concept records, not something the
repository chose, and it cannot be undone from this side: a record's concept is
fixed when the record is created. The earlier DOI still resolves and still
correctly identifies `0.1.0`, so nothing already published is broken. What it
no longer does is stand for "all versions", which is why every citation target
in this repository was moved to the new concept.

The `0.2.0` record records `isNewVersionOf 10.5281/zenodo.17070042`, so the
link between the two lineages is machine-readable from the archive itself.

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
  version   = {0.2.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21337272},
  url       = {https://doi.org/10.5281/zenodo.21337272}
}
```

Replace the concept DOI with `10.5281/zenodo.22171268` to cite `0.2.0`
specifically.

## What the archive represents

Research software. It is **not** a medical device, and no claim of clinical
effectiveness is made or supported. Quantitative results distributed with a
release are produced by the bundled simulator and have not been validated
against real sensor data.
