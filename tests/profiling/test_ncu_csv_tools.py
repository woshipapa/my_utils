from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_ncu_csv_tools_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "ncu"
        / "ncu_csv_tools.py"
    )
    spec = importlib.util.spec_from_file_location("ncu_csv_tools", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ncu_csv_tools"] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_ncu_csv_tools_module()
NcuCsvSkillEngine = _MOD.NcuCsvSkillEngine
analyze_ncu_csv = _MOD.analyze_ncu_csv


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_ncu_csv_engine_long_format(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "long.csv",
        "\n".join(
            [
                "Kernel Name,Metric Name,Metric Value,Metric Unit",
                "k1,gpu__time_duration.sum,10,ns",
                "k1,gpu__time_duration.sum,20,ns",
                "k2,gpu__time_duration.sum,40,ns",
                "k2,sm__throughput.avg.pct_of_peak_sustained_elapsed,50,%",
            ]
        )
        + "\n",
    )
    engine = NcuCsvSkillEngine(csv_path)
    assert "summary" in engine.list_skills()
    summary = engine.run_skill("summary", metric_like="%time%", top_k=10)
    assert isinstance(summary, dict)
    assert int(summary["metric_records"]) == 4
    top = engine.run_skill("top_kernels", metric_like="%time%", top_k=5, score="sum")
    assert isinstance(top, list) and top
    assert top[0]["kernel_name"] == "k2"


def test_ncu_csv_engine_wide_format(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "wide.csv",
        "\n".join(
            [
                "Kernel Name,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,Context",
                "wk1,100,60,1",
                "wk2,200,70,1",
            ]
        )
        + "\n",
    )
    engine = NcuCsvSkillEngine(csv_path)
    summary = engine.run_skill("summary", metric_like="%", top_k=10)
    assert isinstance(summary, dict)
    assert int(summary["metric_records"]) >= 4
    metrics = engine.run_skill("top_metrics", top_k=10)
    assert isinstance(metrics, list) and metrics


def test_ncu_csv_analyze_auto_metric(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "analyze.csv",
        "\n".join(
            [
                "Kernel Name,Metric Name,Metric Value",
                "a,my_duration_metric,10",
                "b,my_duration_metric,40",
                "a,other_metric,3",
            ]
        )
        + "\n",
    )
    payload = analyze_ncu_csv(csv_path, top_k=10)
    assert isinstance(payload, dict)
    assert "selected_metric_like" in payload
    assert str(payload["selected_metric_like"]) in {"my_duration_metric", "%"}
