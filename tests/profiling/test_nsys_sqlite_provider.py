"""Tests for my_utils.profiling.sources.nsys_sqlite_provider."""

from __future__ import annotations


from pathlib import Path


from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider


from _synthetic_loader import _init_sqlite


def test_nsys_new_skills_and_methods(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db, scale=1.0)
    provider = NsysSqliteMetricsProvider(str(db))
    skills = provider.list_sql_skills()
    for name in ("nccl_breakdown", "nvtx_kernel_map", "schema_inspect", "thread_utilization"):
        assert name in skills
    assert provider.run_sql_skill("nccl_breakdown", device_id=0, limit=10)
    assert provider.run_sql_skill("schema_inspect", table_like="CUPTI%", limit=100)
    assert provider.run_sql_skill("thread_utilization", limit=10)
    iters = provider.detect_iterations(marker="sample_0", device_id=0)
    assert len(iters) >= 1
    mfu = provider.compute_mfu(step_time_s=0.01, model_flops_per_step=1e12, peak_tflops=100)
    assert "mfu_pct" in mfu
