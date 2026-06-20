"""Evaluation metrics for comparing strategies against labeled sample claims."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldMetrics:
    exact_matches: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.exact_matches / self.total if self.total else 0.0


@dataclass
class StrategyMetrics:
    strategy_name: str
    claim_status: FieldMetrics = field(default_factory=FieldMetrics)
    evidence_standard_met: FieldMetrics = field(default_factory=FieldMetrics)
    issue_type: FieldMetrics = field(default_factory=FieldMetrics)
    object_part: FieldMetrics = field(default_factory=FieldMetrics)
    severity: FieldMetrics = field(default_factory=FieldMetrics)
    risk_flags_jaccard_sum: float = 0.0
    total_rows: int = 0

    @property
    def avg_risk_flags_jaccard(self) -> float:
        return self.risk_flags_jaccard_sum / self.total_rows if self.total_rows else 0.0

    @property
    def composite_score(self) -> float:
        weights = [
            (self.claim_status.accuracy, 0.35),
            (self.evidence_standard_met.accuracy, 0.20),
            (self.issue_type.accuracy, 0.15),
            (self.object_part.accuracy, 0.15),
            (self.severity.accuracy, 0.05),
            (self.avg_risk_flags_jaccard, 0.10),
        ]
        return sum(score * weight for score, weight in weights)


def _parse_flags(value: str) -> set[str]:
    if not value or value.strip().lower() == "none":
        return set()
    return {f.strip() for f in value.split(";") if f.strip() and f.strip() != "none"}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _norm_bool(value: str) -> str:
    return str(value).strip().lower()


def compare_row(predicted: dict[str, str], expected: dict[str, str], metrics: StrategyMetrics) -> None:
    metrics.total_rows += 1

    if predicted.get("claim_status") == expected.get("claim_status"):
        metrics.claim_status.exact_matches += 1
    metrics.claim_status.total += 1

    if _norm_bool(str(predicted.get("evidence_standard_met", ""))) == _norm_bool(
        str(expected.get("evidence_standard_met", ""))
    ):
        metrics.evidence_standard_met.exact_matches += 1
    metrics.evidence_standard_met.total += 1

    if predicted.get("issue_type") == expected.get("issue_type"):
        metrics.issue_type.exact_matches += 1
    metrics.issue_type.total += 1

    if predicted.get("object_part") == expected.get("object_part"):
        metrics.object_part.exact_matches += 1
    metrics.object_part.total += 1

    if predicted.get("severity") == expected.get("severity"):
        metrics.severity.exact_matches += 1
    metrics.severity.total += 1

    metrics.risk_flags_jaccard_sum += jaccard(
        _parse_flags(str(predicted.get("risk_flags", "none"))),
        _parse_flags(str(expected.get("risk_flags", "none"))),
    )


def format_metrics_report(metrics: StrategyMetrics) -> str:
    return (
        f"### {metrics.strategy_name}\n\n"
        f"| Metric | Accuracy |\n|---|---|\n"
        f"| claim_status | {metrics.claim_status.accuracy:.1%} ({metrics.claim_status.exact_matches}/{metrics.claim_status.total}) |\n"
        f"| evidence_standard_met | {metrics.evidence_standard_met.accuracy:.1%} |\n"
        f"| issue_type | {metrics.issue_type.accuracy:.1%} |\n"
        f"| object_part | {metrics.object_part.accuracy:.1%} |\n"
        f"| severity | {metrics.severity.accuracy:.1%} |\n"
        f"| risk_flags Jaccard (avg) | {metrics.avg_risk_flags_jaccard:.1%} |\n"
        f"| **Composite score** | **{metrics.composite_score:.1%}** |\n"
    )
