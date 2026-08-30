---
name: Bug report
about: Report a reproducible problem in the toolkit
title: "bug: "
labels: bug
assignees: ""
---

## Summary

Describe the problem and the expected behavior.

## Reproduction

```python
# Minimal code or command that reproduces the issue
```

## Data Context

- Input format: CSV / JSON / HDF5 / stream / synthetic
- Timestamp layout:
- Sensor columns:
- Missing data pattern, if relevant:

## Environment

- OS:
- Python version:
- Package version or commit:

## Validation

Paste the relevant output from:

```bash
pytest -q
pre-commit run --all-files
```
