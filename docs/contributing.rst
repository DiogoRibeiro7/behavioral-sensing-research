Contributing
============

Contributions are welcome! To get started:

1. Create a virtual environment and install dependencies::

      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
      pip install -r docs/requirements.txt

2. Run linters and tests before submitting a pull request::

      pre-commit run --files <changed files>
      pytest -k 'not benchmark_runtime'

3. Build the documentation locally to verify changes::

      sphinx-build -b html docs docs/_build/html

Please ensure all checks pass prior to opening a pull request.
