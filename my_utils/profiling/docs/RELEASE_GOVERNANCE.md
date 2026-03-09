# Profiling Release Governance

## 1. Versioning

- Follow semantic versioning for the profiling subsystem.
- `MAJOR`: schema-breaking changes (`MetricEvent`/`AnalysisReport` incompatibility).
- `MINOR`: backward-compatible features (new providers/rules/CLI flags).
- `PATCH`: bug fixes and non-breaking parser updates.

## 2. Schema Compatibility

- Current schema: `MetricEvent.schema_version = "1.0"`.
- New schema versions must provide:
  - forward migration notes,
  - fallback parser for previous major version,
  - compatibility tests in `tests/profiling`.

## 3. Provider Compatibility Matrix

Each provider should declare capabilities:

- source mode (`online/offline/hybrid`)
- metric prefixes
- supported dimensions
- step/rank scope support

For external tools:

- `NsysSqliteMetricsProvider` should remain schema-adaptive.
- version metadata must be attached to tags and docs updated.

## 4. Deprecation Policy

- Deprecate in at least one `MINOR` release before removal.
- Emit runtime warning with:
  - deprecated API name,
  - replacement,
  - earliest removal version.
- Keep legacy aliases until the next major release when feasible.

## 5. Quality Gates

Before release:

1. `python -m compileall my_utils`
2. `pytest -q tests/profiling`
3. Run CLI smoke:
   - `myutils-profile list-providers`
   - `myutils-profile analyze ...`
4. Regenerate/verify docs:
   - quickstart
   - NSYS SQLite parsing guide
   - roadmap status

## 6. Plugin Contribution Contract

New providers/adapters/rules should include:

- clear `provider_id` / `rule_id`,
- capability description,
- tests (unit + at least one end-to-end fixture),
- documentation section with configuration example.

