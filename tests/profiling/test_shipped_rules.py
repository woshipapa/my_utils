"""Tests for my_utils.profiling.ncu.shipped_rules."""

from __future__ import annotations


import pytest


from _synthetic_loader import ncu_diagnostics, shipped_rules


class TestShippedRuleReconciliation:
    def _rule(self, ident, msg, mtype="warning", stype="", speedup=None):
        return {
            "rule_identifier": ident,
            "rule_message": {"message": msg, "title": msg, "message_type": mtype},
            "speedup_estimation": {"type": stype, "speedup": speedup},
        }

    def test_local_speedup_is_not_promoted_to_kernel_level(self):
        rules = shipped_rules.normalize_shipped_rules(
            [
                self._rule(
                    "UncoalescedGlobalAccess",
                    "excess sectors",
                    stype="LOCAL",
                    speedup=45.0,
                )
            ]
        )
        assert rules[0].speedup_ceiling is None
        findings = shipped_rules.shipped_rules_to_findings(rules)
        assert any("LOCAL" in a for a in findings[0].actions)

    def test_global_speedup_converts_to_a_ceiling(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("SOLBottleneck", "memory bound", stype="GLOBAL", speedup=50.0)]
        )
        assert rules[0].speedup_ceiling == pytest.approx(2.0)

    def test_ok_messages_are_not_raised_as_problems(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("Occupancy", "no issues", mtype="ok")]
        )
        assert rules[0].is_actionable is False
        assert shipped_rules.shipped_rules_to_findings(rules) == []

    def test_bottleneck_disagreement_is_reported(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("SOLBottleneck", "This kernel is memory bound.")]
        )
        out = shipped_rules.reconcile_with_shipped_rules(
            [], rules, our_verdict="compute_bound"
        )
        assert out["conflicts"], "compute-vs-memory disagreement must be raised"
        assert any(f.category == "evidence_conflict" for f in out["findings"])

    def test_agreement_promotes_confidence_and_names_the_rule(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("UncoalescedGlobalAccess", "excess sectors")]
        )
        ours = [
            ncu_diagnostics.Finding(
                category="uncoalesced_global_access",
                title="t",
                summary="s",
                confidence="medium",
            )
        ]
        out = shipped_rules.reconcile_with_shipped_rules(ours, rules)
        promoted = next(
            f for f in out["findings"] if f.category == "uncoalesced_global_access"
        )
        assert promoted.confidence == "high"
        assert "corroborated_by_ncu_rule" in promoted.evidence

    def test_absent_shipped_rules_do_not_weaken_findings(self):
        ours = [
            ncu_diagnostics.Finding(
                category="uncoalesced_global_access", title="t", summary="s"
            )
        ]
        out = shipped_rules.reconcile_with_shipped_rules(ours, [])
        assert out["shipped_rules_available"] is False
        assert out["findings"] == ours

    def test_ncu_only_rules_are_not_dropped(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("SharedMemoryConflicts", "bank conflicts detected")]
        )
        out = shipped_rules.reconcile_with_shipped_rules([], rules)
        assert any(f.source == "ncu_rule" for f in out["findings"])

    def test_malformed_rule_blocks_do_not_raise(self):
        assert shipped_rules.normalize_shipped_rules([None, 42, "x", {}]) != []
