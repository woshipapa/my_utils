from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from my_utils.profiling import (
    BaseMetricsProvider,
    MetricEvent,
    MetricsCollector,
    NcuCsvMetricsProvider,
    TableCsvMetricsProvider,
)


class SyntheticTrainProvider(BaseMetricsProvider):
    """A tiny provider that simulates stage latency and memory metrics."""

    provider_id = "synthetic_train"

    def __init__(self, *, seed: int = 1234, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._rng = random.Random(seed)
        self._step = 0
        self._memory_base = 8.0 * 1024 * 1024 * 1024  # 8GB

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def get_metrics(self) -> list[MetricEvent]:
        ts = time.time()
        step = str(self._step)

        # Synthetic latency model
        forward_ms = 20.0 + self._rng.uniform(-2.0, 2.0)
        backward_ms = 32.0 + self._rng.uniform(-3.0, 3.0)
        # Inject periodic jitter every 10 steps
        if self._step > 0 and self._step % 10 == 0:
            backward_ms *= 1.7

        # Synthetic memory model (slow growth + noise)
        mem_bytes = self._memory_base + self._step * 8_000_000 + int(self._rng.uniform(-2e6, 2e6))

        events = [
            MetricEvent(
                timestamp=ts,
                name="latency.stage",
                value=forward_ms,
                unit="ms",
                provider_id=self.provider_id,
                tags={"step": step, "stage": "forward", "device": "cuda"},
            ),
            MetricEvent(
                timestamp=ts,
                name="latency.stage",
                value=backward_ms,
                unit="ms",
                provider_id=self.provider_id,
                tags={"step": step, "stage": "backward", "device": "cuda"},
            ),
            MetricEvent(
                timestamp=ts,
                name="memory.gpu.allocated",
                value=mem_bytes,
                unit="bytes",
                provider_id=self.provider_id,
                tags={"step": step, "device": "cuda"},
            ),
        ]
        return events


def _append_external_csv_row(csv_path: Path, step: int, rng: random.Random) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["op_name", "latency_ms", "step"])
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "op_name": "dataloader.next_batch",
                "latency_ms": f"{4.0 + rng.uniform(-0.5, 0.8):.4f}",
                "step": str(step),
            }
        )


def _append_ncu_csv_row(csv_path: Path, step: int, rng: random.Random) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Metric Name", "Metric Value", "Metric Unit", "Kernel Name"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "Metric Name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "Metric Value": f"{55.0 + rng.uniform(-8.0, 8.0):.4f}",
                "Metric Unit": "percent",
                "Kernel Name": f"kernel_step_{step % 3}",
            }
        )


def run_demo(output_dir: Path, steps: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    external_csv = output_dir / "external_tool_metrics.csv"
    ncu_csv = output_dir / "ncu_metrics.csv"

    collector = MetricsCollector(output_dir=str(output_dir))

    # 1) Register a custom in-process provider
    synthetic = SyntheticTrainProvider(seed=seed)
    collector.register_provider(synthetic)

    # 2) Register a generic external CSV provider
    collector.register_provider(
        TableCsvMetricsProvider(
            csv_path=str(external_csv),
            value_column="latency_ms",
            name_column="op_name",
            tag_columns=["step"],
            unit="ms",
            event_name_prefix="latency.external",
            provider_id="external_csv",
        )
    )

    # 3) Register an NCU CSV provider
    collector.register_provider(NcuCsvMetricsProvider(csv_path=str(ncu_csv)))

    collector.start()
    for step in range(steps):
        synthetic.set_step(step)
        _append_external_csv_row(external_csv, step=step, rng=rng)
        _append_ncu_csv_row(ncu_csv, step=step, rng=rng)
        collector.collect(step=step, tags={"run": "demo"})
    collector.stop()

    report = collector.analyze()
    json_path = collector.export_report(fmt="json", output_path=str(output_dir / "report.json"), report=report)
    md_path = collector.export_report(fmt="markdown", output_path=str(output_dir / "report.md"), report=report)
    html_path = collector.export_report(fmt="html", output_path=str(output_dir / "report.html"), report=report)

    print("=== Unified Metrics Demo Finished ===")
    print(f"Output dir : {output_dir}")
    print(f"Providers  : {collector.list_providers()}")
    print(f"Events     : {report.summary.get('total_events')}")
    print(f"Findings   : {len(report.findings)}")
    print(f"JSON report: {json_path}")
    print(f"MD report  : {md_path}")
    print(f"HTML report: {html_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified profiling metrics demo")
    parser.add_argument("--output-dir", type=str, default="./demo_metrics_output")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(output_dir=Path(args.output_dir), steps=args.steps, seed=args.seed)
