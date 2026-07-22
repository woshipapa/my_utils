from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from my_utils.profiling import (
    BaseMetricsProvider,
    MetricEvent,
    MetricsAnalyzer,
    MetricsCollector,
    MetricsReportRenderer,
    NsysSqliteMetricsProvider,
    TableCsvMetricsProvider,
    compare_reports,
    write_diff,
)


class SyntheticDistProvider(BaseMetricsProvider):
    provider_id = "synthetic_dist"

    def __init__(self, *, seed: int = 1234, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._rng = random.Random(seed)
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def get_metrics(self) -> list[MetricEvent]:
        ts = time.time()
        result: list[MetricEvent] = []
        for rank in (0, 1):
            fwd = 25.0 + rank * 5.0 + self._rng.uniform(-1.5, 1.5)
            bwd = 35.0 + rank * 8.0 + self._rng.uniform(-2.0, 2.0)
            mem = 1_000_000_000 + self._step * 25_000_000 + rank * 15_000_000
            comm = 7.0 + rank * 4.5 + self._rng.uniform(-0.8, 0.8)
            tags = {"step": str(self._step), "rank": str(rank)}
            result.extend(
                [
                    MetricEvent(
                        ts,
                        "latency.stage",
                        fwd,
                        "ms",
                        provider_id=self.provider_id,
                        tags={**tags, "stage": "forward"},
                    ),
                    MetricEvent(
                        ts,
                        "latency.stage",
                        bwd,
                        "ms",
                        provider_id=self.provider_id,
                        tags={**tags, "stage": "backward"},
                    ),
                    MetricEvent(
                        ts,
                        "memory.gpu.allocated",
                        float(mem),
                        "bytes",
                        provider_id=self.provider_id,
                        tags=tags,
                    ),
                    MetricEvent(
                        ts,
                        "comm.nccl.all_reduce",
                        comm,
                        "ms",
                        provider_id=self.provider_id,
                        tags=tags,
                    ),
                ]
            )
        return result


def _append_external_csv(csv_path: Path, step: int, rng: random.Random) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["op_name", "latency_ms", "step", "rank"]
        )
        if write_header:
            writer.writeheader()
        for rank in (0, 1):
            writer.writerow(
                {
                    "op_name": "dataloader.next_batch",
                    "latency_ms": f"{3.5 + rank * 0.8 + rng.uniform(-0.3, 0.6):.4f}",
                    "step": str(step),
                    "rank": str(rank),
                }
            )


def run_demo(output_dir: Path, steps: int, seed: int, nsys_sqlite: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ext_csv = output_dir / "external.csv"

    collector = MetricsCollector(
        output_dir=str(output_dir / "run_pretrain"),
        validate_events=True,
        drop_invalid_events=False,
        analyzer=MetricsAnalyzer(workload_profile="pretrain"),
    )
    synthetic = SyntheticDistProvider(seed=seed)
    collector.register_provider(synthetic)
    collector.register_provider(
        TableCsvMetricsProvider(
            csv_path=str(ext_csv),
            value_column="latency_ms",
            name_column="op_name",
            tag_columns=["step", "rank"],
            unit="ms",
            event_name_prefix="latency.external",
            provider_id="external_csv",
        )
    )
    if nsys_sqlite:
        provider = NsysSqliteMetricsProvider(
            sqlite_path=nsys_sqlite, include_osrt=False
        )
        collector.register_provider(provider)
        schema_path = output_dir / "nsys_schema_probe.json"
        schema_path.write_text(
            json.dumps(provider.describe_schema(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for step in range(steps):
        synthetic.set_step(step)
        _append_external_csv(ext_csv, step, rng)
        collector.collect(step=step, tags={"run": "pretrain"})

    report_pretrain = collector.analyze()
    renderer = MetricsReportRenderer()
    pretrain_json = output_dir / "run_pretrain" / "analysis_report.json"
    renderer.write(report_pretrain, str(pretrain_json), fmt="json")
    renderer.write(
        report_pretrain,
        str(output_dir / "run_pretrain" / "analysis_report.md"),
        fmt="markdown",
    )
    renderer.write(
        report_pretrain,
        str(output_dir / "run_pretrain" / "analysis_report.html"),
        fmt="html",
    )

    # Re-analyze same events with another workload profile for diff demo.
    events = collector.get_events()
    report_infer = MetricsAnalyzer(workload_profile="inference").analyze(events)
    infer_json = output_dir / "run_inference" / "analysis_report.json"
    infer_json.parent.mkdir(parents=True, exist_ok=True)
    renderer.write(report_infer, str(infer_json), fmt="json")

    diff = compare_reports(str(pretrain_json), str(infer_json))
    write_diff(diff, str(output_dir / "report_diff.json"))
    (output_dir / "report_diff.md").write_text(
        renderer.diff_to_markdown(diff), encoding="utf-8"
    )

    print("=== P0-P13 end-to-end demo complete ===")
    print(f"output_dir      : {output_dir}")
    print(f"events_collected: {len(events)}")
    print(f"pretrain_score  : {report_pretrain.overall_score}")
    print(f"inference_score : {report_infer.overall_score}")
    print(f"providers       : {collector.list_providers()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0-P13 profiling acceptance demo")
    parser.add_argument("--output-dir", type=str, default="./p0_p13_demo_output")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nsys-sqlite", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        Path(args.output_dir),
        steps=args.steps,
        seed=args.seed,
        nsys_sqlite=args.nsys_sqlite,
    )
