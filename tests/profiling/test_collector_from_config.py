# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from my_utils.profiling import MetricsCollector


def test_collector_from_config_with_table_csv(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "external.csv"
    csv_path.write_text("name,value,step\nfoo,1.0,0\nbar,2.0,1\n", encoding="utf-8")

    cfg = {
        "collector": {
            "output_dir": str(tmp_path / "out"),
            "enabled": True,
            "validate_events": True,
            "drop_invalid_events": False,
        },
        "analysis": {"workload_profile": "default"},
        "providers": [
            {
                "type": "table_csv",
                "id": "ext_csv",
                "enabled": True,
                "params": {
                    "csv_path": str(csv_path),
                    "value_column": "value",
                    "name_column": "name",
                    "tag_columns": ["step"],
                    "unit": "ms",
                    "event_name_prefix": "latency.external",
                },
            }
        ],
    }
    cfg_path = tmp_path / "collector.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    collector = MetricsCollector.from_config(str(cfg_path))
    assert collector.list_providers() == ["ext_csv"]
    count = collector.collect(step=3, tags={"run": "ut"})
    assert count == 2

    events = collector.get_events()
    assert len(events) == 2
    assert all(event.provider_id == "ext_csv" for event in events)

    report = collector.analyze(events)
    assert report.summary["total_events"] == 2
    assert "collector" in report.metadata
