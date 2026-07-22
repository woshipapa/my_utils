# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


def _load_nccl_tools_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "nccl"
        / "nccl_inspector_tools.py"
    )
    spec = importlib.util.spec_from_file_location("nccl_inspector_tools", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["nccl_inspector_tools"] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_nccl_tools_module()
NcclInspectorSkillEngine = _MOD.NcclInspectorSkillEngine
analyze_nccl_inspector = _MOD.analyze_nccl_inspector
load_nccl_inspector_events = _MOD.load_nccl_inspector_events
load_nccl_prometheus_metrics = _MOD.load_nccl_prometheus_metrics


def _write_jsonl(path: Path) -> Path:
    rows = [
        {
            "header": {
                "id": "0xabc",
                "comm_name": "DP Group 0",
                "rank": 0,
                "n_ranks": 2,
                "nnodes": 1,
            },
            "metadata": {"hostname": "node0", "pid": 100, "dump_timestamp_us": 10},
            "coll_perf": {
                "coll": "AllReduce",
                "coll_sn": 7,
                "coll_msg_size_bytes": 1048576,
                "coll_exec_time_us": 100,
                "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 20.0,
                "coll_busbw_gbs": 40.0,
                "event_trace_ts": {
                    "kernel_events": [
                        {
                            "channel_id": 0,
                            "kernel_start_ts": 1000,
                            "kernel_stop_ts": 1100,
                        },
                        {
                            "channel_id": 1,
                            "kernel_start_ts": 1010,
                            "kernel_stop_ts": 1120,
                        },
                    ]
                },
            },
        },
        {
            "header": {
                "id": "0xabc",
                "comm_name": "DP Group 0",
                "rank": 1,
                "n_ranks": 2,
                "nnodes": 1,
            },
            "metadata": {"hostname": "node1", "pid": 200, "dump_timestamp_us": 11},
            "coll_perf": {
                "coll": "AllReduce",
                "coll_sn": 7,
                "coll_msg_size_bytes": 1048576,
                "coll_exec_time_us": 250,
                "coll_timing_source": "kernel_gpu",
                "coll_algobw_gbs": 10.0,
                "coll_busbw_gbs": 20.0,
            },
        },
        {
            "header": {
                "id": "0xabc",
                "comm_name": "DP Group 0",
                "rank": 0,
                "n_ranks": 2,
                "nnodes": 1,
            },
            "metadata": {"hostname": "node0", "pid": 100, "dump_timestamp_us": 12},
            "p2p_perf": {
                "p2p": "Send",
                "p2p_sn": 8,
                "p2p_peer": 1,
                "p2p_msg_size_bytes": 524288,
                "p2p_exec_time_us": 50,
                "p2p_timing_source": "kernel_gpu",
                "p2p_algobw_gbs": 30.0,
                "p2p_busbw_gbs": 30.0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_nccl_inspector_events_and_skills(tmp_path: Path) -> None:
    jsonl = _write_jsonl(tmp_path / "nccl.jsonl")
    events = load_nccl_inspector_events(str(tmp_path))
    assert len(events) == 3
    assert events[0].kernel_span_us == 120.0

    engine = NcclInspectorSkillEngine(str(jsonl))
    summary = engine.run_skill("summary")
    assert isinstance(summary, dict)
    assert summary["collective_count"] == 2
    assert summary["p2p_count"] == 1

    top_collectives = engine.run_skill("top_collectives", top_k=10)
    assert isinstance(top_collectives, list) and top_collectives
    assert top_collectives[0]["op"] == "AllReduce"
    assert top_collectives[0]["exec_time_us"]["sum"] == 350.0

    skew = engine.run_skill("rank_skew", top_k=10)
    assert isinstance(skew, list) and skew
    assert skew[0]["slow_rank"] == 1
    assert skew[0]["skew_ratio"] == 2.5


def test_nccl_prometheus_and_analyze(tmp_path: Path) -> None:
    jsonl = _write_jsonl(tmp_path / "nccl.jsonl")
    prom = tmp_path / "nccl.prom"
    prom.write_text(
        "\n".join(
            [
                'nccl_bus_bandwidth_gbs{version="v5.1",node="n0",gpu="GPU0",comm_name="DP",n_nodes="1",nranks="2",collective="AllReduce",message_size="1-2MB",algo_proto="Ring_ll"} 123.5',
                'nccl_collective_exec_time_microseconds{version="v5.1",node="n0",gpu="GPU0",comm_name="DP",n_nodes="1",nranks="2",collective="AllReduce",message_size="1-2MB",algo_proto="Ring_ll"} 456',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = load_nccl_prometheus_metrics(str(prom))
    assert len(metrics) == 2
    assert metrics[0].labels["collective"] == "AllReduce"

    payload = analyze_nccl_inspector(str(jsonl), prometheus_path=str(prom), top_k=5)
    assert "prometheus_summary" in payload
    assert payload["summary"]["event_count"] == 3


def test_nccl_inspector_markdown(tmp_path: Path) -> None:
    jsonl = _write_jsonl(tmp_path / "nccl.jsonl")
    payload = analyze_nccl_inspector(str(jsonl), top_k=5)
    text = _MOD.analyze_nccl_inspector_to_markdown(payload)
    assert "NCCL Inspector Analysis" in text
    assert "AllReduce" in text
