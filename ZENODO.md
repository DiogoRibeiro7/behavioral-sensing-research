# Zenodo Archive

This repository is archived on Zenodo under a project-level **concept DOI**
that always resolves to the latest version, plus a **version DOI** for each
archived release.

- Concept record: <https://zenodo.org/records/17070041>
- Concept DOI: <https://doi.org/10.5281/zenodo.17070041>

Cite the **version DOI** when your result depends on a specific release, which
is almost always the case for a reproducible experiment. Cite the **concept
DOI** when referring to the software project across versions.

## Archived releases

| Version | Record | Version DOI | Published |
| --- | --- | --- | --- |
| 0.1.0 | <https://zenodo.org/records/17070042> | `10.5281/zenodo.17070042` | 2025-09-07 |
| 0.2.0 | pending | pending | pending |

`0.2.0` has not been archived. Publishing is a deliberate, manual step; see
[RELEASE.md](RELEASE.md).

## Metadata

Archive metadata is held in [`.zenodo.json`](.zenodo.json) and must stay
consistent with [`CITATION.cff`](CITATION.cff) and `pyproject.toml`. A test in
`tests/test_project_metadata.py` checks that the versions agree, so a partial
bump fails the build rather than reaching Zenodo.

## Citing this software

```bibtex
@software{ribeiro2026sensor,
  author    = {Ribeiro, Diogo},
  title     = {Sensor Modeling Research Toolkit},
  year      = {2026},
  version   = {0.2.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17070041},
  url       = {https://doi.org/10.5281/zenodo.17070041}
}
```

Replace the concept DOI with the version DOI once `0.2.0` is archived.

## What the archive represents

Research software. It is **not** a medical device, and no claim of clinical
effectiveness is made or supported. Quantitative results distributed with a
release are produced by the bundled simulator and have not been validated
against real sensor data.
