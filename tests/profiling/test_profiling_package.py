"""Package-level health checks for my_utils.profiling: exports, docs, module reachability and subpackage import order."""

from __future__ import annotations


import re
import sys
import types
from pathlib import Path
import pytest


from _synthetic_loader import axes, metric_catalog, ncu_diagnostics, trace_quality


class TestPackageExportsResolve:
    """A documented import that does not exist is worse than no documentation."""

    @staticmethod
    def _check(init_rel, pkg_dir):
        import ast

        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        tree = ast.parse((root.parent.parent / init_rel).read_text())
        imported, exported = {}, []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                for al in node.names:
                    imported[al.asname or al.name] = node.module
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__":
                        exported = [e.value for e in node.value.elts]

        cache = {}

        def defined_in(mod):
            if mod not in cache:
                names = set()
                src = (root.parent.parent / pkg_dir / f"{mod}.py").read_text()
                for n in ast.walk(ast.parse(src)):
                    if isinstance(
                        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        names.add(n.name)
                    elif isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                names.add(t.id)
                    elif isinstance(n, ast.AnnAssign) and isinstance(
                        n.target, ast.Name
                    ):
                        names.add(n.target.id)
                cache[mod] = names
            return cache[mod]

        broken = [
            name
            for name in exported
            if name not in imported or name not in defined_in(imported[name])
        ]
        assert not broken, f"{init_rel} exports names that do not resolve: {broken}"
        assert exported, f"{init_rel} exports nothing"

    def test_analyzers_exports(self):
        self._check(
            "my_utils/profiling/analyzers/__init__.py", "my_utils/profiling/analyzers"
        )

    def test_hardware_exports(self):
        self._check(
            "my_utils/profiling/hardware/__init__.py", "my_utils/profiling/hardware"
        )


class TestHandbookExamplesAreReal:
    """Every symbol the handbook tells a reader to import must exist."""

    def test_documented_imports_resolve(self):
        import ast
        import re

        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        handbook = root / "docs" / "PERFORMANCE_ANALYSIS_HANDBOOK.md"
        if not handbook.exists():
            pytest.skip("handbook not present")
        text = handbook.read_text()

        missing = []
        for module, names in re.findall(
            r"from (my_utils\.profiling[\w.]*) import ([\w, ]+)", text
        ):
            rel = module.replace("my_utils.profiling", "").lstrip(".").replace(".", "/")
            candidates = [root / f"{rel}.py", root / rel / "__init__.py"]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                missing.append(f"{module} (module not found)")
                continue
            defined = set()
            for n in ast.walk(ast.parse(path.read_text())):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(n.name)
                elif isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            defined.add(t.id)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    defined.add(n.target.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        defined.add(a.asname or a.name.split(".")[0])
            for name in (x.strip() for x in names.split(",") if x.strip()):
                if name not in defined:
                    missing.append(f"{module}.{name}")
        assert not missing, f"handbook documents imports that do not exist: {missing}"

    def test_python_blocks_parse(self):
        import ast
        import re

        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        handbook = root / "docs" / "PERFORMANCE_ANALYSIS_HANDBOOK.md"
        if not handbook.exists():
            pytest.skip("handbook not present")
        bad = []
        for i, block in enumerate(
            re.findall(r"```python\n(.*?)```", handbook.read_text(), re.S)
        ):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                bad.append(f"block {i}: {exc.msg}")
        assert not bad, f"handbook python blocks do not parse: {bad}"


class TestDocsQuoteRealCounts:
    """Counts cited in the docs drift silently as code is added.

    Two had already drifted when this was written: the section-backed catalog
    split, and `trace_quality.py (12 checks)` when there were 13 -- the handbook
    table was also missing a row. A number in prose is a claim like any other.
    """

    def _docs(self):
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling" / "docs"
        out = {}
        for name in ("PERFORMANCE_ANALYSIS_HANDBOOK.md", "CAPABILITY_EVOLUTION.md"):
            path = root / name
            if path.exists():
                out[name] = path.read_text()
        return out

    def test_cited_counts_match_the_code(self):
        import re

        counts = {
            f"{len(metric_catalog.METRIC_CATALOG)} metrics": True,
            f"{len(axes.AXES)} axes": True,
            f"{len(metric_catalog.STALL_REASONS)} stall reasons": True,
            f"of {len(ncu_diagnostics._ANALYSIS_REQUIREMENTS)} analyses": True,
        }
        checks = len([n for n in dir(trace_quality) if n.startswith("check_")])
        counts[f"({checks} checks)"] = True

        wrong = []
        for name, text in self._docs().items():
            # Any "<n> metrics"/"<n> axes"/... that is NOT the real number is stale.
            # Anchored to the phrasings that state a fact about this codebase.
            # A bare "<n> metrics" also appears in example output and in
            # unrelated thresholds, so matching that loosely produces noise.
            for pattern, real in (
                (
                    r"catalog interprets (\d+) metrics",
                    len(metric_catalog.METRIC_CATALOG),
                ),
                (
                    r"metric_catalog\.py` [^\n]*?(\d+) metrics",
                    len(metric_catalog.METRIC_CATALOG),
                ),
                (
                    r"`METRIC_CATALOG` \((\d+) metrics\)",
                    len(metric_catalog.METRIC_CATALOG),
                ),
                (r"(\d+) axes\b", len(axes.AXES)),
                (r"(\d+) stall reasons\b", len(metric_catalog.STALL_REASONS)),
                (r"of (\d+) analyses\b", len(ncu_diagnostics._ANALYSIS_REQUIREMENTS)),
                (r"\((\d+) checks\)", checks),
            ):
                for found in re.findall(pattern, text):
                    if int(found) != real:
                        wrong.append(
                            f"{name}: '{pattern}' cites {found}, code says {real}"
                        )
        assert not wrong, "docs cite stale counts: " + "; ".join(wrong)

    def test_trace_quality_table_lists_every_check(self):
        """The 9c table must not silently omit a check."""
        text = self._docs().get("PERFORMANCE_ANALYSIS_HANDBOOK.md", "")
        if not text:
            pytest.skip("handbook not present")
        documented = set(re.findall(r"\| `(check_\w+)`", text))
        actual = {n for n in dir(trace_quality) if n.startswith("check_")}
        missing = sorted(actual - documented)
        assert not missing, (
            f"checks implemented but absent from the 9c table: {missing}"
        )


class TestNoOrphanedAnalysisModules:
    """An analysis module with no caller is dead weight that looks like coverage.

    This has happened repeatedly: throttling, nccl_bandwidth, trace_quality and
    distributed_alignment were each complete, exported, tested -- and invoked by
    nothing, so the axes they cover silently reported as gaps. measurement_context
    was added in the same session that fixed the others and immediately repeated
    the mistake. Exported and tested is not the same as reachable.
    """

    _ENTRY_POINTS = (
        "ncu/ncu_diagnostics.py",
        "ncu/ncu_report_tools.py",
        "sources/nsys_auto_analysis.py",
        "analyzers/metrics_analyzer.py",
    )

    def test_analysis_modules_are_reachable_from_an_entry_point(self):
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        entry_text = "\n".join(
            (root / rel).read_text()
            for rel in self._ENTRY_POINTS
            if (root / rel).exists()
        )
        # Modules that must be invoked, not merely importable.
        required = [
            "analyzers/axes.py",
            "analyzers/measurement_context.py",
            "analyzers/trace_quality.py",
            "hardware/throttling.py",
            "ncu/shipped_rules.py",
            "ncu/source_correlation.py",
            "ncu/sampling_validity.py",
            "ncu/section_index.py",
        ]
        orphans = []
        for rel in required:
            stem = Path(rel).stem
            if stem not in entry_text:
                orphans.append(rel)
        assert not orphans, (
            "analysis modules with no caller in any entry point "
            f"(exported and tested is not reachable): {orphans}"
        )


class TestSubpackagesImportInAnyOrder:
    """`import my_utils.profiling.sources` used to fail in a fresh interpreter.

    sources.nsys_sqlite_provider imported ..metrics, whose __init__ imported
    metrics_providers, which imported back into sources while it was still
    initialising. Normal use never hit it because profiling/__init__ happens to
    import .metrics before .sources -- which made that ordering load-bearing and
    undocumented. The re-exports are now resolved lazily.
    """

    def test_every_subpackage_imports_first(self):
        import importlib
        import importlib.util as iu

        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        failures = []
        for first in ("sources", "metrics", "analyzers", "ncu", "hardware"):
            for mod in [m for m in sys.modules if m.startswith("my_utils")]:
                del sys.modules[mod]
            pkg = types.ModuleType("my_utils")
            pkg.__path__ = [str(root.parent)]
            sys.modules["my_utils"] = pkg
            spec = iu.spec_from_file_location(
                "my_utils.profiling",
                root / "__init__.py",
                submodule_search_locations=[str(root)],
            )
            prof = iu.module_from_spec(spec)
            sys.modules["my_utils.profiling"] = prof
            pkg.profiling = prof
            try:
                importlib.import_module(f"my_utils.profiling.{first}")
                metrics = importlib.import_module("my_utils.profiling.metrics")
                assert metrics.NsysSqliteMetricsProvider is not None
            except Exception as exc:
                failures.append(f"{first} first: {type(exc).__name__}: {exc}")
            finally:
                for mod in [m for m in sys.modules if m.startswith("my_utils")]:
                    del sys.modules[mod]
        assert not failures, "subpackage import order is load-bearing: " + "; ".join(
            failures
        )
