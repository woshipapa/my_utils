"""Tests for my_utils.profiling.ncu.signal_scan."""

from __future__ import annotations




from _synthetic_loader import signal_scan


class TestSignalScanOverAllMetrics:
    """Reason about metrics no curated rule covers, without inventing noise."""

    def test_saturated_uncatalogued_unit_is_found(self):
        out = signal_scan.scan_all_signals({
            "idc__request_cycles_active.avg.pct_of_peak_sustained_elapsed": 91.0,
        })
        assert any(f.category == "unit_saturated" for f in out["findings"])

    def test_units_with_curated_rules_are_not_duplicated(self):
        out = signal_scan.scan_all_signals({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 95.0,
        })
        assert not [f for f in out["findings"] if f.category == "unit_saturated"]

    def test_bursty_unit_is_explained_rather_than_called_idle(self):
        out = signal_scan.scan_all_signals({
            "tpc__warps_active.avg.pct_of_peak_sustained_active": 96.0,
            "tpc__warps_active.avg.pct_of_peak_sustained_elapsed": 8.0,
        })
        finding = next(f for f in out["findings"] if f.category == "unit_duty_cycle")
        assert "Both" in finding.summary and "correct" in finding.summary

    def test_percentage_above_100_is_a_measurement_fault(self):
        out = signal_scan.scan_all_signals({
            "fbpa__x.avg.pct_of_peak_sustained_elapsed": 118.0,
        })
        finding = next(f for f in out["findings"]
                       if f.category == "measurement_above_physical_limit")
        assert finding.severity == "high"

    def test_hit_rate_is_not_read_as_utilisation(self):
        """A 95% hit rate must not be reported as a saturated unit."""
        out = signal_scan.scan_all_signals({
            "lts__t_sector_hit_rate.pct": 95.0,
        })
        assert not [f for f in out["findings"] if f.category == "unit_saturated"]

    def test_quiet_report_produces_nothing(self):
        out = signal_scan.scan_all_signals({
            "idc__request_cycles_active.avg.pct_of_peak_sustained_elapsed": 12.0,
        })
        assert out["findings"] == []
