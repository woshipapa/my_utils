# SPDX-License-Identifier: Apache-2.0
"""Reconcile our rule engine against the rules Nsight Compute itself ships.

An ``.ncu-rep`` carries NVIDIA's own findings: every action exposes
``rule_results_as_dicts()``, one entry per rule that fired, each with a message,
a section, focus metrics, and often a speedup estimate. Those findings are
counter-backed and written by the people who defined the counters. Ignoring them
while running our own re-implementation of the same rules would be strange.

Reconciling the two is worth more than either alone:

* **Agreement** raises confidence, and says so with a named corroborating rule.
* **Disagreement** is the valuable output. If our engine says compute-bound and
  NVIDIA's ``SOLBottleneck`` says memory-bound, at least one is wrong, and the
  user needs to know that before acting.
* **NVIDIA-only findings** cover rules we never implemented. They are reported
  with their own identifier so nothing shipped in the report is silently lost.
* **Our-only findings** are marked as uncorroborated, which is not the same as
  wrong: we deliberately implement analyses NVIDIA does not ship (roofline
  placement, wave quantisation, workload expectations).

Rule identifiers are matched *tolerantly*. We have not enumerated every rule
identifier in every Nsight Compute release, and inventing a table of exact
strings would produce silent misses when a spelling differs. Instead each
category is matched by normalised substrings, and anything unmatched is still
surfaced -- as an uncategorised shipped rule, never dropped.

Speedup estimates need care. Nsight Compute reports both ``GLOBAL`` and
``LOCAL`` estimated speedups. A LOCAL estimate is relative to the section it
came from, so a 40% LOCAL win on a section that is 5% of the kernel's runtime is
worth about 2%. We keep the distinction and refuse to promote a LOCAL estimate
into a kernel-level ceiling.
"""

from __future__ import annotations

import re
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ncu_diagnostics import Finding

__all__ = [
    "ShippedRule",
    "normalize_shipped_rules",
    "shipped_rules_to_findings",
    "reconcile_with_shipped_rules",
]


# Rule identifiers are mapped to axes by the shared table in
# ``analyzers.axes``, which both sides of the reconciliation read. Keeping a
# second table here is how the two vocabularies drifted apart in the first place.

# Message types Nsight Compute uses. Only the warning-ish ones become findings
# with real severity; OK/INFO messages are recorded but not raised.
_ACTIONABLE_MESSAGE_TYPES = {"warning", "error", "optimization", "issue"}
_INERT_MESSAGE_TYPES = {"ok", "info", "informational", "none"}


def _norm(text: Any) -> str:
    """Lowercase and strip everything but letters and digits."""
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _classify_identifier(*candidates: Any) -> str:
    """Map a shipped rule onto an axis id, or "" when unrecognised."""
    from ..analyzers.axes import axis_for_shipped_rule  # local: avoid a cycle

    return axis_for_shipped_rule(*candidates)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


@dataclass
class ShippedRule:
    """One finding emitted by Nsight Compute's own rule engine."""

    rule_identifier: str = ""
    rule_name: str = ""
    section_identifier: str = ""
    message: str = ""
    message_title: str = ""
    message_type: str = ""
    speedup: Optional[float] = None
    speedup_type: str = ""  # "global" | "local" | ""
    focus_metrics: Tuple[Dict[str, Any], ...] = ()
    kernel_name: str = ""
    axis: str = ""  # canonical axis id, or "" when unrecognised

    @property
    def is_actionable(self) -> bool:
        """Whether this rule reported a problem rather than a clean result."""
        kind = _norm(self.message_type)
        if kind in {_norm(t) for t in _INERT_MESSAGE_TYPES}:
            return False
        if kind in {_norm(t) for t in _ACTIONABLE_MESSAGE_TYPES}:
            return True
        # Unknown message type: a rule that bothered to attach a speedup
        # estimate found something. Otherwise stay quiet rather than inventing
        # a problem out of an unrecognised enum.
        return self.speedup is not None and self.speedup > 0.0

    @property
    def speedup_ceiling(self) -> Optional[float]:
        """Kernel-level speedup multiplier, or None for LOCAL-only estimates.

        Nsight Compute reports the estimate as a percentage. A GLOBAL estimate
        is relative to the whole kernel and converts directly. A LOCAL estimate
        is relative to its section and cannot be turned into a kernel-level
        number without knowing that section's share of runtime, which the rule
        result does not carry -- so we refuse to guess.
        """
        if self.speedup is None or self.speedup <= 0.0:
            return None
        if _norm(self.speedup_type) != "global":
            return None
        pct = min(float(self.speedup), 99.0)
        return 100.0 / (100.0 - pct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_identifier": self.rule_identifier,
            "rule_name": self.rule_name,
            "section_identifier": self.section_identifier,
            "message": self.message,
            "message_title": self.message_title,
            "message_type": self.message_type,
            "speedup": self.speedup,
            "speedup_type": self.speedup_type,
            "speedup_ceiling": self.speedup_ceiling,
            "focus_metrics": [dict(m) for m in self.focus_metrics],
            "kernel_name": self.kernel_name,
            "axis": self.axis,
            "is_actionable": self.is_actionable,
        }


def normalize_shipped_rules(raw_rules: Iterable[Any]) -> List[ShippedRule]:
    """Normalise ``rule_results_as_dicts()`` output into :class:`ShippedRule`.

    Accepts the dict shape produced by
    :func:`~my_utils.profiling.ncu.ncu_report_tools.collect_rule_results` as well
    as the raw dicts from the ``ncu_report`` Python API, whose keys differ
    slightly between versions. Unknown shapes yield an empty list rather than an
    exception: a malformed rule block must not take down the whole analysis.
    """
    out: List[ShippedRule] = []
    for item in raw_rules or ():
        if not isinstance(item, Mapping):
            continue

        # Two shapes: already flattened by collect_rule_results(), or the raw
        # nested dict with a "rule_message" sub-dict.
        message_block = item.get("rule_message")
        if isinstance(message_block, Mapping):
            message = message_block.get("message", "")
            title = message_block.get("title", "")
            msg_type = message_block.get("message_type", message_block.get("type", ""))
        else:
            message = item.get("rule_message", "")
            title = item.get("rule_message_title", "")
            msg_type = item.get("rule_message_type", "")

        speedup_block = item.get("speedup_estimation")
        if isinstance(speedup_block, Mapping):
            speedup = _to_float(speedup_block.get("speedup"))
            speedup_type = speedup_block.get("type", "")
        else:
            speedup = _to_float(item.get("speedup"))
            speedup_type = item.get("speedup_type", "")

        focus_raw = item.get("focus_metrics") or ()
        focus: List[Dict[str, Any]] = []
        for entry in focus_raw if isinstance(focus_raw, (list, tuple)) else ():
            if isinstance(entry, Mapping):
                focus.append(dict(entry))

        identifier = str(item.get("rule_identifier", "") or "")
        name = str(item.get("rule_name", item.get("name", "")) or "")
        section = str(item.get("section_identifier", "") or "")

        out.append(
            ShippedRule(
                rule_identifier=identifier,
                rule_name=name,
                section_identifier=section,
                message=str(message or ""),
                message_title=str(title or ""),
                message_type=str(msg_type or ""),
                speedup=speedup,
                speedup_type=str(speedup_type or ""),
                focus_metrics=tuple(focus),
                kernel_name=str(item.get("kernel_name", "") or ""),
                axis=_classify_identifier(identifier, name, section),
            )
        )
    return out


def shipped_rules_to_findings(rules: Sequence[ShippedRule]) -> List[Finding]:
    """Turn actionable shipped rules into :class:`Finding` objects.

    Confidence is ``high``: these rules read hardware counters through the tool
    that defined them. The evidence dict names the rule so a reader can go back
    to the Nsight Compute UI and see the same text.
    """
    findings: List[Finding] = []
    for rule in rules:
        if not rule.is_actionable:
            continue

        evidence: Dict[str, Any] = {
            "ncu_rule": rule.rule_identifier or rule.rule_name or "(unnamed)",
            "ncu_section": rule.section_identifier,
            "ncu_message_type": rule.message_type,
        }
        if rule.focus_metrics:
            evidence["focus_metrics"] = [dict(m) for m in rule.focus_metrics[:8]]
        if rule.speedup is not None:
            evidence["ncu_estimated_speedup_pct"] = rule.speedup
            evidence["ncu_speedup_scope"] = rule.speedup_type or "unspecified"

        actions: List[str] = []
        if (
            rule.speedup is not None
            and rule.speedup_ceiling is None
            and rule.speedup > 0
        ):
            # The trap worth naming: a large LOCAL number on a small section.
            actions.append(
                "Nsight Compute reported this speedup as LOCAL to its section, not "
                "for the whole kernel. Weight it by that section's share of runtime "
                "before treating it as a kernel-level win."
            )

        findings.append(
            Finding(
                category=rule.axis or "ncu_shipped_rule",
                title=rule.message_title
                or (rule.rule_identifier or "Nsight Compute rule"),
                summary=rule.message or "(the rule fired but carried no message text)",
                severity="medium",
                confidence="high",
                evidence=evidence,
                actions=tuple(actions),
                speedup_ceiling=rule.speedup_ceiling,
                source="ncu_rule",
            )
        )
    return findings


# Verdict words that contradict each other when they appear in a bottleneck
# claim. Used only for the bottleneck category, where "compute" and "memory"
# are genuinely exclusive; elsewhere two rules firing is not a conflict.
_BOTTLENECK_TERMS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("compute", ("computebound", "smbound", "computethroughput", "compute_bound")),
    (
        "memory",
        ("memorybound", "membound", "drambound", "memorythroughput", "memory_bound"),
    ),
    ("latency", ("latencybound", "latency_bound", "lowutilization", "unusedwarps")),
)


def _bottleneck_terms(text: str) -> set:
    blob = _norm(text)
    hits = set()
    for label, needles in _BOTTLENECK_TERMS:
        if any(n in blob for n in needles):
            hits.add(label)
    return hits


def reconcile_with_shipped_rules(
    our_findings: Sequence[Finding],
    shipped: Sequence[ShippedRule],
    *,
    our_verdict: str = "",
) -> Dict[str, Any]:
    """Cross-check our findings against NVIDIA's, and report the disagreements.

    Returns a dict with:

    ``findings``
        Our findings, with confidence adjusted, plus findings for the rules only
        NVIDIA reported and for every genuine disagreement.
    ``corroborated`` / ``uncorroborated`` / ``ncu_only``
        Category lists, for a report that wants to show the overlap.
    ``conflicts``
        Bottleneck-verdict disagreements, the ones that mean somebody is wrong.
    ``shipped_rules_seen``
        Every normalised shipped rule, actionable or not, so nothing is lost.

    When ``shipped`` is empty the findings pass through untouched and
    ``shipped_rules_available`` is False -- absence of NVIDIA's rules is missing
    corroboration, never evidence that our findings are wrong.
    """
    shipped = list(shipped or ())
    actionable = [r for r in shipped if r.is_actionable]

    if not shipped:
        return {
            "shipped_rules_available": False,
            "findings": list(our_findings),
            "corroborated": [],
            "uncorroborated": [],
            "ncu_only": [],
            "conflicts": [],
            "shipped_rules_seen": [],
            "note": (
                "No Nsight Compute rule results in this report. Findings below are "
                "from our engine only, with no independent corroboration. Collect "
                "with --set full (rules ship with their sections) to cross-check."
            ),
        }

    # Match on axis, not on category name. The two vocabularies do not overlap:
    # we emit "uncoalesced_global_access", NVIDIA emits "UncoalescedGlobalAccess"
    # in a "MemoryWorkloadAnalysis" section. Comparing the raw strings would
    # report every finding as uncorroborated and never fire a single conflict.
    from ..analyzers.axes import axis_for_category

    our_axes = {axis_for_category(f.category) for f in our_findings}
    our_axes.discard("")

    adjusted: List[Finding] = []
    corroborated: List[str] = []
    uncorroborated: List[str] = []

    for finding in our_findings:
        axis = axis_for_category(finding.category)
        matches = [r for r in actionable if axis and r.axis == axis]
        if not matches:
            # Not a demotion: many of our analyses have no NVIDIA counterpart.
            if finding.category in _NEVER_SHIPPED_BY_NVIDIA or not axis:
                adjusted.append(finding)
            else:
                uncorroborated.append(axis)
                adjusted.append(
                    _with_evidence(
                        finding,
                        {
                            "corroboration": (
                                f"No Nsight Compute rule fired on the '{axis}' axis in this "
                                "report. That is weaker support, not a contradiction."
                            ),
                        },
                    )
                )
            continue

        corroborated.append(axis)
        names = [m.rule_identifier or m.rule_name or "(unnamed)" for m in matches]
        promoted = _with_evidence(
            finding,
            {
                "corroborated_by_ncu_rule": names,
                "corroborating_message": matches[0].message[:400],
            },
        )
        # Independent agreement is exactly what should move confidence up.
        if promoted.confidence != "high":
            promoted = _replace_confidence(promoted, "high")
        adjusted.append(promoted)

    # Rules NVIDIA reported that we have no finding for.
    ncu_only_rules = [r for r in actionable if not r.axis or r.axis not in our_axes]
    adjusted.extend(shipped_rules_to_findings(ncu_only_rules))

    conflicts = _bottleneck_conflicts(our_verdict, actionable)
    for conflict in conflicts:
        adjusted.append(
            Finding(
                category="evidence_conflict",
                title="Our bottleneck verdict disagrees with Nsight Compute's own rule",
                summary=conflict["summary"],
                severity="high",
                confidence="high",
                evidence=conflict,
                actions=(
                    "Do not act on either verdict yet. Read the SpeedOfLight section "
                    "directly: check whether Compute and Memory throughput were read "
                    "from the same denominator (_elapsed for both), and whether the "
                    "report has the counters both verdicts depend on.",
                ),
                source="evidence",
            )
        )

    return {
        "shipped_rules_available": True,
        "findings": adjusted,
        "corroborated": sorted(set(corroborated)),
        "uncorroborated": sorted(set(uncorroborated)),
        "ncu_only": sorted({r.axis or "(unmapped)" for r in ncu_only_rules}),
        "conflicts": conflicts,
        "shipped_rules_seen": [r.to_dict() for r in shipped],
        "note": (
            f"{len(actionable)} of {len(shipped)} shipped rules reported a problem; "
            f"{len(set(corroborated))} of our categories were independently confirmed."
        ),
    }


# Analyses we implement that Nsight Compute does not ship a rule for. Their
# absence from the shipped rules says nothing, so they are not marked
# uncorroborated -- doing so would read as doubt where none is warranted.
_NEVER_SHIPPED_BY_NVIDIA = frozenset(
    {
        "roofline",  # placement against an absolute ceiling needs a GPU spec
        "workload_expectation",  # "a GEMM should hit tensor cores" is our rule
        "tile_quantization",
        "wave_quantization",
        "measurement_caveat",
        "evidence_conflict",
        "uninformative_name",
        "coverage",
    }
)


def _bottleneck_conflicts(
    our_verdict: str,
    actionable: Sequence[ShippedRule],
) -> List[Dict[str, Any]]:
    """Find genuine compute-vs-memory disagreements, ignoring vocabulary noise."""
    ours = _bottleneck_terms(our_verdict)
    if not ours:
        return []

    conflicts: List[Dict[str, Any]] = []
    for rule in actionable:
        if rule.axis not in ("compute", "memory_bandwidth"):
            continue
        theirs = _bottleneck_terms(f"{rule.message_title} {rule.message}")
        if not theirs:
            continue
        # Only compute-vs-memory is exclusive. Latency coexists with either:
        # a memory-bound kernel is usually also latency bound.
        if ("compute" in ours and "memory" in theirs and "compute" not in theirs) or (
            "memory" in ours and "compute" in theirs and "memory" not in theirs
        ):
            conflicts.append(
                {
                    "summary": (
                        f"Our engine classified this kernel as {sorted(ours)} bound; "
                        f"Nsight Compute's own rule "
                        f"'{rule.rule_identifier or rule.rule_name}' reported "
                        f"{sorted(theirs)} bound. One of the two is reading the wrong "
                        "number -- most often a Speed-Of-Light comparison that mixes "
                        "the _active and _elapsed denominators."
                    ),
                    "our_verdict": our_verdict,
                    "ncu_rule": rule.rule_identifier or rule.rule_name,
                    "ncu_message": rule.message[:400],
                }
            )
    return conflicts


def _with_evidence(finding: Finding, extra: Mapping[str, Any]) -> Finding:
    """Return a copy of ``finding`` with additional evidence merged in."""
    merged = dict(finding.evidence)
    merged.update(extra)
    # dataclasses.replace, so fields added to Finding later (e.g. the
    # speedup-model annotations) survive reconciliation unchanged.
    return dataclasses.replace(finding, evidence=merged)


def _replace_confidence(finding: Finding, confidence: str) -> Finding:
    return dataclasses.replace(finding, confidence=confidence)
