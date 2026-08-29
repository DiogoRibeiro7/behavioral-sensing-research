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

Run these checks on `develop`:

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
- `CHANGELOG.md` has an entry for the release.
- `README.md` badges, DOI, and citation text are current.
- `ROADMAP.md` still reflects the next planned work.

## Merge to Main

```bash
git checkout main
git pull origin main
git merge --no-ff develop
git push origin main
```

Verify the merge target:

```bash
git branch --show-current
git log -1 --oneline
```

The branch must be `main` before tagging.

## Tag the Release

Create an annotated tag on `main`:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Verify the tag points to `main`:

```bash
git branch --contains vX.Y.Z
git show --no-patch --decorate vX.Y.Z
```

`main` must be listed by `git branch --contains`.

## Publish the GitHub Release

Use the tag created on `main`:

```bash
gh release create vX.Y.Z \
  --target main \
  --title "vX.Y.Z" \
  --notes-file RELEASE_NOTES.md
```

If using the GitHub web UI, verify the target branch or commit is the `main`
commit for the tag.

## Zenodo Verification

After GitHub publishes the release:

- Confirm Zenodo created or updated the record.
- Confirm the DOI resolves.
- Confirm the Zenodo record links back to this repository.
- Confirm repository metadata links to the DOI.
- Confirm `.zenodo.json`, `CITATION.cff`, and README citation details agree.

## After Release

Return to `develop`:

```bash
git checkout develop
git pull origin develop
git merge --ff-only main
git push origin develop
```

Then:

- Move released changelog notes out of `Unreleased`, if needed.
- Start the next `Unreleased` section.
- Open follow-up issues for deferred roadmap items.

## Emergency Fix Releases

For hotfixes:

1. Branch from `main`.
2. Apply the minimal fix.
3. Run the relevant tests and pre-commit.
4. Merge the hotfix into `main`.
5. Tag and release from `main`.
6. Merge `main` back into `develop`.

The rule still holds: release from `main`, never from `develop`.
