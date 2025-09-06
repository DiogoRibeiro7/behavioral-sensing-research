# Contributing Guidelines

Thank you for considering contributing to **Sensor Modeling Research Toolkit**!

## Getting Started

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Install development dependencies:
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```
3. Run formatting and tests before committing:
   ```bash
   pre-commit run --files <changed_files>
   pytest
   sphinx-build -b html docs docs/_build/html
   ```

## Pull Requests

- Ensure your branch is rebased on `main`.
- Provide clear descriptions of the problem and solution.
- Include tests and documentation for new features.
- Update the changelog in `CHANGELOG.md`.

## Reporting Issues

Please use the issue tracker to report bugs or request features. Include:

- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version)

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
- Write type hints and docstrings for public functions and classes.
- Keep functions small and focused.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).

If you have questions, feel free to contact
[Diogo Ribeiro](mailto:dfr@esmad.ipp.pt).
