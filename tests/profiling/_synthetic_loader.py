# SPDX-License-Identifier: Apache-2.0
"""Shared torch-free loading machinery for the profiling test suite.

The modules under test are pure analysis code with no CUDA or torch
dependency, so this helper loads them directly from their file paths under a
synthetic package root.  Going through the ``my_utils`` package would drag in
``my_utils.core.utils``, which imports torch, and that would make these tests
unrunnable exactly where they are most useful: a CI box with no GPU stack
installed.

This module also carries the synthetic nsys sqlite fixture builder
(``_init_sqlite``) used by the nsys-facing tests: those tests exercise the
file-path-based sqlite/report loaders end to end, which is the torch-free
contract proof, so the builder writes real sqlite files to disk rather than
handing objects around.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
_PKG = "_prof"


def _package(name: str) -> types.ModuleType:
    """Create (once) a synthetic package so relative imports resolve."""
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = []  # marks it as a package
    sys.modules[name] = module
    if "." in name:
        parent, _, child = name.rpartition(".")
        setattr(_package(parent), child, module)
    return module


def _load(dotted: str, relative_path: str):
    """Import one profiling module under a synthetic package root.

    The modules use relative imports (``from .metric_catalog import ...``), so
    they cannot be loaded as standalone files - they need a parent package. We
    build that package tree by hand rather than importing ``my_utils``, whose
    ``__init__`` pulls in torch.
    """
    full = f"{_PKG}.{dotted}"
    if full in sys.modules:
        return sys.modules[full]
    parent, _, leaf = full.rpartition(".")
    _package(parent)
    spec = importlib.util.spec_from_file_location(full, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    setattr(sys.modules[parent], leaf, module)
    return module


gpu_specs = _load("hardware.gpu_specs", "hardware/gpu_specs.py")
throttling = _load("hardware.throttling", "hardware/throttling.py")
kernel_taxonomy = _load("sources.kernel_taxonomy", "sources/kernel_taxonomy.py")
metric_catalog = _load("ncu.metric_catalog", "ncu/metric_catalog.py")
triage = _load("analyzers.triage", "analyzers/triage.py")
evidence = _load("analyzers.evidence", "analyzers/evidence.py")
nccl_bandwidth = _load("analyzers.nccl_bandwidth", "analyzers/nccl_bandwidth.py")
trace_quality = _load("analyzers.trace_quality", "analyzers/trace_quality.py")
ncu_diagnostics = _load("ncu.ncu_diagnostics", "ncu/ncu_diagnostics.py")

section_index = _load("ncu.section_index", "ncu/section_index.py")
axes = _load("analyzers.axes", "analyzers/axes.py")
shipped_rules = _load("ncu.shipped_rules", "ncu/shipped_rules.py")
ncu_report_tools = _load("ncu.ncu_report_tools", "ncu/ncu_report_tools.py")
nsys_auto = _load("sources.nsys_auto_analysis", "sources/nsys_auto_analysis.py")
measurement_context = _load(
    "analyzers.measurement_context", "analyzers/measurement_context.py"
)
sampling_validity = _load("ncu.sampling_validity", "ncu/sampling_validity.py")
source_correlation = _load("ncu.source_correlation", "ncu/source_correlation.py")
signal_scan = _load("ncu.signal_scan", "ncu/signal_scan.py")
source_correlation_mod = _load("ncu.source_correlation", "ncu/source_correlation.py")


# ---------------------------------------------------------------------------
# Synthetic nsys sqlite fixture builder
# ---------------------------------------------------------------------------


def _init_sqlite(path: Path, *, scale: float = 1.0) -> None:
    """Build a minimal but complete nsys-like SQLite covering all current skills."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # 鈹€鈹€ meta 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    cur.executemany(
        "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
        [
            ("NSIGHT_SYSTEMS_VERSION", "2024.7.1"),
            ("EXPORT_SCHEMA_VERSION", "3.15.1"),
        ],
    )

    # 鈹€鈹€ string table 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (1, "void gemm_kernel()"),
            (2, "ncclAllReduceRingLLKernel_sum_f16"),
            (3, "cudaLaunchKernel"),
            (4, "worker_main"),
            (5, "void attention_kernel()"),
            (101, "sm__active.avg.pct_of_peak_sustained_elapsed"),
            (102, "tensor__active.avg.pct_of_peak_sustained_elapsed"),
            (103, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ],
    )

    # 鈹€鈹€ kernel table (includes block/register columns for skill 15) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER, [end] INTEGER, streamId INTEGER, correlationId INTEGER, "
        "shortName INTEGER, demangledName INTEGER, deviceId INTEGER, "
        "blockX INTEGER, blockY INTEGER, blockZ INTEGER, "
        "registersPerThread INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, "
        "theoreticalOccupancyPct REAL)"
    )
    # rows: (start, end, stream, corr, short, demangled, dev, bx, by, bz, regs, static_smem, dyn_smem, theoretical_occupancy_pct)
    s = scale
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                0,
                int(10000 * s),
                7,
                1,
                1,
                1,
                0,
                128,
                1,
                1,
                32,
                4096,
                0,
                87.5,
            ),  # gemm stream7
            (
                int(8000 * s),
                int(20000 * s),
                8,
                2,
                2,
                2,
                0,
                256,
                1,
                1,
                40,
                0,
                0,
                62.5,
            ),  # nccl  stream8
            (
                int(25000 * s),
                int(35000 * s),
                7,
                3,
                1,
                1,
                0,
                128,
                1,
                1,
                32,
                4096,
                0,
                87.5,
            ),  # gemm stream7
            (
                int(5000 * s),
                int(12000 * s),
                9,
                4,
                5,
                5,
                0,
                64,
                1,
                1,
                48,
                8192,
                2048,
                50.0,
            ),  # attention stream9
        ],
    )

    # 鈹€鈹€ runtime table 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
        "start INTEGER, [end] INTEGER, correlationId INTEGER, nameId INTEGER, globalTid INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (0, int(2000 * s), 1, 3, 12345678),
            (int(7000 * s), int(9000 * s), 2, 3, 12345678),
            (int(24000 * s), int(24500 * s), 3, 3, 12345678),
            (int(4500 * s), int(5000 * s), 4, 3, 12345678),
        ],
    )

    # 鈹€鈹€ NVTX events 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE NVTX_EVENTS ("
        "start INTEGER, [end] INTEGER, text TEXT, textId INTEGER, eventType INTEGER, globalTid INTEGER)"
    )
    cur.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, int(22000 * s), "sample_0 step=1 rank=0", None, 59, 12345678),
            (
                int(23000 * s),
                int(36000 * s),
                "sample_0 step=2 rank=0",
                None,
                59,
                12345678,
            ),
            (0, int(10000 * s), "forward", None, 59, 12345678),
            (int(10000 * s), int(20000 * s), "backward", None, 59, 12345678),
        ],
    )

    # 鈹€鈹€ memcpy table (skill 4, 12, 17) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
        "(start INTEGER, [end] INTEGER, copyKind INTEGER, bytes INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
        [
            (0, int(1000 * s), 1, int(1024 * 1024), 0),  # H2D 1 MB
            (int(3000 * s), int(6000 * s), 2, int(2 * 1024 * 1024), 0),  # D2H 2 MB
            (int(12000 * s), int(15000 * s), 8, int(4 * 1024 * 1024), 0),  # D2D 4 MB
        ],
    )

    # 鈹€鈹€ memset table (skill 14) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET "
        "(start INTEGER, [end] INTEGER, bytes INTEGER, value INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?, ?, ?)",
        [
            (0, int(500 * s), int(8 * 1024 * 1024), 0, 0),  # zero-init 8 MB
            (int(500 * s), int(600 * s), int(1024 * 1024), 0, 0),  # zero-init 1 MB
            (int(1000 * s), int(1100 * s), int(512 * 1024), 1, 0),  # custom fill
        ],
    )

    # 鈹€鈹€ synchronization table (skill 13) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "(start INTEGER, [end] INTEGER, syncType INTEGER, streamId INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (?, ?, ?, ?, ?)",
        [
            (int(20000 * s), int(21000 * s), 1, 7, 0),  # cudaStreamSync
            (int(35000 * s), int(35500 * s), 2, 0, 0),  # cudaDeviceSync
            (int(36000 * s), int(36100 * s), 1, 8, 0),  # cudaStreamSync
        ],
    )

    # 鈹€鈹€ CPU events (skill 11) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute("CREATE TABLE COMPOSITE_EVENTS (globalTid INTEGER, cpuCycles INTEGER)")
    cur.executemany(
        "INSERT INTO COMPOSITE_EVENTS VALUES (?, ?)",
        [
            (12345678, int(1000 * s)),
            (22345678, int(500 * s)),
        ],
    )
    cur.execute("CREATE TABLE ThreadNames (globalTid INTEGER, nameId INTEGER)")
    cur.executemany(
        "INSERT INTO ThreadNames VALUES (?, ?)",
        [(12345678, 4), (22345678, 4)],
    )

    cur.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT)")
    cur.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100')")
    cur.execute("CREATE TABLE GENERIC_EVENT_SOURCES (id INTEGER, name TEXT)")
    cur.executemany(
        "INSERT INTO GENERIC_EVENT_SOURCES VALUES (?, ?)",
        [
            (1, "GpuMetrics"),
            (2, "ETW"),
        ],
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC "
        "(timestamp INTEGER, metricId INTEGER, value REAL, sourceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC VALUES (?, ?, ?, ?)",
        [
            (int(1000 * s), 101, 62.5, 1),
            (int(2000 * s), 101, 70.0, 1),
            (int(3000 * s), 102, 41.0, 1),
            (int(4000 * s), 102, 44.5, 1),
            (int(5000 * s), 103, 57.25, 1),
            (int(6000 * s), 101, 99.0, 2),
        ],
    )
    conn.commit()
    conn.close()


def _show(title: str, rows, *, limit: int = 5) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if isinstance(rows, list):
        for r in rows[:limit]:
            print(" ", r)
        if len(rows) > limit:
            print(f"  ... ({len(rows)} total)")
    elif isinstance(rows, dict):
        for k, v in rows.items():
            print(f"  {k}: {v}")
    else:
        print(" ", rows)
