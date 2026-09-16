# Release Checklist

This project releases from `main` only. Development and integration work can
happen on `develop`, but tags and GitHub releases must point to commits on
`main`.

## Branch Policy

- `develop`: active development, refactors, documentation updates, and release
  preparation.
- `main`: stable release branch.
- Release tags: created from `main` only, using the format `vMAJOR.MINOR.PATCH`.

Do not publish a release from `develop`.

## Before Merging to Main

**First, read the status of the latest CI run on `develop`.**

The `all checks passed` job must be green. A local run is not a substitute for
required Actions checks.

Then run these on `develop` as a fast pre-check when useful:

```bash
pre-commit run --all-files
pytest -q
mkdocs build --strict
```

Confirm the release metadata is ready:

- `pyproject.toml` version is correct.
- `sensor_modeling/__init__.py` version is correct.
- `CITATION.cff` version and DOI metadata are correct.
- `.zenodo.json` is current.
- `CHANGELOG.md` has a dated entry for the release and a fresh `Unreleased`
  section.
- `README.md` badges, DOI, and citation text are current.
- `ROADMAP.md` still reflects the next planned work.

`CHANGELOG.md` is the single source of truth for release notes. Do not maintain
per-release `RELEASE_NOTES_*.md` files.

## Merge to Main

Promote the finalized release tree from `develop` to `main` through a pull
request. After merge, verify that the intended stable commit is the current
`main` head and that the required `all checks passed` status is green.

Do not create a tag before the release tree is on `main`.

## Publish the Release

The canonical release path is the GitHub Actions **Release** workflow in
`.github/workflows/release.yml`.

1. Open **Actions** in GitHub.
2. Select **Release**.
3. Choose **Run workflow**.
4. Select the `main` branch.
5. Enter the semantic version without the `v` prefix, for example `0.5.0`.
6. Run the workflow.

The workflow performs the release contract mechanically. It:

- refuses to publish from anything except `main`;
- checks out the exact stable `main` commit with full history;
- runs the release metadata consistency tests;
- verifies the requested version against `pyproject.toml`,
  `sensor_modeling/__init__.py`, `CITATION.cff`, `.zenodo.json`, and README
  citation metadata;
- requires a dated matching version section in `CHANGELOG.md`;
- uses that changelog section verbatim as the GitHub Release body;
- creates an annotated `vMAJOR.MINOR.PATCH` tag on the checked-out `main`
  commit, or verifies that an existing tag already points there;
- refuses conflicting tags or duplicate GitHub Releases;
- publishes the GitHub Release using the repository `GITHUB_TOKEN`.

A successful workflow run is the release publication record. Do not create a
second tag or duplicate release manually after it succeeds.

## Manual Fallback

Use manual git and GitHub CLI commands only if the Release workflow is
unavailable. The same invariants still apply: release from `main`, use an
annotated tag on the exact stable commit, and use the matching `CHANGELOG.md`
version section verbatim as the release body.

Example fallback:

```bash
git checkout main
git pull origin main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Then create the GitHub Release for that exact tag/commit using the matching
changelog section. Verify the tag before publishing.

## Zenodo Verification

After GitHub publishes the release:

- Confirm Zenodo created or updated the record.
- Confirm the DOI resolves.
- Confirm the Zenodo record links back to this repository.
- Confirm repository metadata links to the DOI.
- Confirm `.zenodo.json`, `CITATION.cff`, and README citation details agree.
  `pytest tests/test_project_metadata.py` checks this mechanically, so run it
  rather than reading the files.
- Replace the pending row in `ZENODO.md` with the new version DOI once the
  record exists.

## After Release

Continue development from `develop`.

A merge from `main` back into `develop` is required only when `main` contains
substantive file changes that are not already present on `develop`, for example
a hotfix made directly from the stable branch. A release promotion merge with
an identical file tree does not need to be merged back solely for ancestry.

Then:

- Keep the fresh `Unreleased` section for subsequent work.
- Open follow-up issues for deferred roadmap items when useful.

## Emergency Fix Releases

For hotfixes:

1. Branch from `main`.
2. Apply the minimal fix.
3. Run the relevant tests and pre-commit.
4. Merge the hotfix into `main`.
5. Run the **Release** workflow for the hotfix version.
6. Bring the substantive hotfix changes back into `develop`.

The rule still holds: release from `main`, never from `develop`.
