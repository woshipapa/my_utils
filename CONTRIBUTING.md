# Contributing to my_utils

Thanks for your interest in contributing. Small fixes are welcome as direct
PRs; for larger changes (new analyzers, new collection pipelines, architectural
rework) please open an issue first so we can discuss the direction.

## Development setup

```bash
git clone <this repo>
cd my_utils
pip install -e .
```

Run the test suite:

```bash
uv run --with pytest pytest -q tests/profiling
```

**torch is not required.** `import my_utils` is torch-free: the analysis core
(GPU spec tables, kernel taxonomy, the ncu rule engine, the triage tree) is
pure Python, and the torch-dependent utilities (`my_utils.core`, hooks, the
in-process capture backend) load lazily on first attribute access, raising a
clear `ImportError` with an install hint if torch is absent. The whole suite
runs on a machine with no GPU stack:

```bash
# Runs with no torch installed (430 tests, ~1s):
uv run --with pytest python -m pytest -q tests/profiling
```

(`python -m pytest`, not bare `pytest`, so the repo root is on `sys.path` for
an uninstalled checkout. `tests/profiling/test_no_torch_import.py` pins the
torch-free import as a regression guard.)

## Tests

All tests live in `tests/profiling/`. Two conventions to know:

- **Torch isolation.** The analysis-engine tests import their modules through
  a synthetic file-path loader (a hand-built package tree) rather than through
  `import my_utils`. Historically this was forced — the package once pulled in
  torch at import time — and it is kept deliberately: it guarantees the
  analysis core stays loadable standalone, independent of anything the parent
  package might grow. Read the header comment and the
  `_package`/`_load` helpers in `tests/profiling/_synthetic_loader.py`
  before writing tests for analysis code — new torch-free tests should use the
  same mechanism rather than `import my_utils`.
- **Behavior changes need tests.** If a PR changes what an analyzer concludes,
  what a metric means, or how a report is assembled, it should come with a test
  that pins the new behavior.

## Code style

- There is no enforced formatter; match the style of the surrounding code in
  the file you are changing, and do not reformat code unrelated to your PR.
- Target Python 3.10. PEP 701 f-string features (backslashes or reused quotes
  inside replacement fields) parse only on 3.12+ and will break imports on the
  target runtime.
- **Evidence-carrying findings — the project's core convention.** Every
  conclusion an analyzer emits must cite the metrics that produced it: rules
  read metrics through the catalog (never hard-coded metric spellings), put
  the producing numbers in the finding's `evidence`, and stay silent when the
  required metrics are absent rather than assuming zero. A missing metric is
  not a zero. PRs that add conclusions without the numbers behind them will be
  asked to add the evidence.

## Adding an analyzer or a metric

- New metrics go in `_CATALOG_LIST` in `my_utils/profiling/ncu/metric_catalog.py`
  — the single place that knows what a metric is called and what a good value
  looks like. List every architecture-specific spelling in `names`.
- New analyzer rules must map their finding categories onto the canonical
  performance axes in `my_utils/profiling/analyzers/axes.py`, so coverage
  reporting can tell an unexamined axis from a clean one.

## Pull requests

- Include tests for any behavior change.
- Keep commits atomic (one feature or fix per commit) and write commit
  messages that explain **why** the change is being made, not just what it
  does.
- Keep the PR scoped: do not touch code outside the stated purpose of the
  change, including drive-by formatting.
