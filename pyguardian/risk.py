"""
risk.py — Risk & Confidence Engine for PyGuardian.

Responsibilities:
- Evaluate risk levels (LOW, MEDIUM, HIGH) for detected ImpactAssessment records.
- Preserve explicit distinction between Risk (concerningness) and Confidence (evidence strength).
- Generate deterministic, explainable reasons for risk assessments.
- Preserve full underlying provenance (source_url, file_path, line_number, description, evidence).
- Aggregate risk assessments across a project into ProjectRiskReport without calculating arbitrary overall scores.

Design Principles:
- Pure deterministic analysis: NO AI, LLM, RAG, or vector DBs.
- 0 network requests, 0 file system modifications, 0 code execution.
- Completely inert analysis layer.
- Update type (MAJOR, MINOR, PATCH) does NOT dictate risk level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from pyguardian.change_analysis import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_BEHAVIOR_CHANGE,
    CHANGE_TYPE_CHANGED,
    CHANGE_TYPE_DEPRECATED,
    CHANGE_TYPE_MIGRATION,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
)
from pyguardian.compatibility import CompatibilityReport, ImpactAssessment
from pyguardian.project_impact import ProjectImpactReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """
    Risk level indicating how concerning a detected change is.

    Attributes:
        LOW:    Minor or less concerning change (e.g., added API, weak package-level evidence).
        MEDIUM: Meaningful but non-breaking or less conclusive change (e.g., behavior change, deprecation).
        HIGH:   Strong evidence of a potentially breaking API change (e.g., removal, rename, breaking migration).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class RiskAssessment:
    """
    Evidence-driven risk and confidence assessment for a detected dependency impact.

    Attributes:
        package_name: Canonical package name (e.g. 'pandas').
        affected_api: Fully qualified API path affected, or None if package-level.
        risk_level:   Evaluated RiskLevel (LOW, MEDIUM, HIGH).
        confidence:   Evidence confidence rating preserved from ImpactAssessment ('high', 'medium', 'low').
        reasons:      Structured tuple of explainable reasons for the risk classification.
        impact:       Underlying ImpactAssessment maintaining full evidence provenance.
    """

    package_name: str
    affected_api: Optional[str]
    risk_level: RiskLevel
    confidence: str
    reasons: tuple[str, ...]
    impact: ImpactAssessment


@dataclass(frozen=True)
class ProjectRiskReport:
    """
    Aggregated project-level risk report containing individual RiskAssessment records.

    Does NOT calculate speculative overall project risk scores or break probabilities.
    """

    project_path: Path
    assessments: list[RiskAssessment] = field(default_factory=list)

    @property
    def total_assessments(self) -> int:
        """Total number of risk assessments across the project."""
        return len(self.assessments)

    @property
    def high_count(self) -> int:
        """Count of HIGH risk assessments."""
        return sum(1 for a in self.assessments if a.risk_level == RiskLevel.HIGH)

    @property
    def medium_count(self) -> int:
        """Count of MEDIUM risk assessments."""
        return sum(1 for a in self.assessments if a.risk_level == RiskLevel.MEDIUM)

    @property
    def low_count(self) -> int:
        """Count of LOW risk assessments."""
        return sum(1 for a in self.assessments if a.risk_level == RiskLevel.LOW)


# ---------------------------------------------------------------------------
# Risk Evaluation Logic
# ---------------------------------------------------------------------------

def _determine_risk_level(impact: ImpactAssessment) -> RiskLevel:
    """
    Determine risk level based strictly on evidence in ImpactAssessment.

    Rules:
    - Exact API + removed / renamed / migration -> HIGH
    - Exact API + changed / behavior_change / deprecated -> MEDIUM
    - Exact API + added / unknown -> LOW
    - Package-level + removed / renamed / migration / changed / behavior_change -> MEDIUM
    - Package-level + deprecated / added / unknown / weak -> LOW
    """
    change_type = (impact.change_type or "").lower()

    if impact.is_exact_match:
        if change_type in (
            CHANGE_TYPE_REMOVED,
            CHANGE_TYPE_RENAMED,
            CHANGE_TYPE_MIGRATION,
        ):
            return RiskLevel.HIGH
        elif change_type in (
            CHANGE_TYPE_CHANGED,
            CHANGE_TYPE_BEHAVIOR_CHANGE,
            CHANGE_TYPE_DEPRECATED,
        ):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    else:
        # Package-level match (affected_api is None or match_type == "package")
        if change_type in (
            CHANGE_TYPE_REMOVED,
            CHANGE_TYPE_RENAMED,
            CHANGE_TYPE_MIGRATION,
            CHANGE_TYPE_CHANGED,
            CHANGE_TYPE_BEHAVIOR_CHANGE,
        ):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


def _generate_reasons(impact: ImpactAssessment) -> tuple[str, ...]:
    """
    Generate structured, machine-readable explanation strings from evidence.
    """
    reasons: list[str] = []

    # 1. Match type evidence
    if impact.is_exact_match:
        reasons.append("Exact API match")
    else:
        reasons.append("Package-level change")

    # 2. Documented change type evidence
    change_type = (impact.change_type or "").lower()
    if change_type == CHANGE_TYPE_REMOVED:
        if impact.is_exact_match:
            reasons.append("API is documented as removed")
        else:
            reasons.append("Package component is documented as removed")
    elif change_type == CHANGE_TYPE_RENAMED:
        if impact.is_exact_match:
            reasons.append("API is documented as renamed")
        else:
            reasons.append("Package component is documented as renamed")
    elif change_type == CHANGE_TYPE_MIGRATION:
        if impact.is_exact_match:
            reasons.append("API is documented as requiring migration")
        else:
            reasons.append("Package is documented as requiring migration")
    elif change_type == CHANGE_TYPE_DEPRECATED:
        if impact.is_exact_match:
            reasons.append("API is documented as deprecated")
        else:
            reasons.append("Package component is documented as deprecated")
    elif change_type in (CHANGE_TYPE_CHANGED, CHANGE_TYPE_BEHAVIOR_CHANGE):
        if impact.is_exact_match:
            reasons.append("API behavior is documented as changed")
        else:
            reasons.append("Package behavior is documented as changed")
    elif change_type == CHANGE_TYPE_ADDED:
        reasons.append("API is documented as added")
    else:
        reasons.append("Documented package change detected")

    # 3. Version transition evidence
    if impact.current_version is not None or impact.target_version is not None:
        reasons.append("Change occurs within the requested version transition")

    # 4. Confidence evidence
    conf = (impact.confidence or "").lower()
    if conf in ("high", "medium", "low"):
        reasons.append(f"Evidence confidence is {conf}")

    return tuple(reasons)


# ---------------------------------------------------------------------------
# Assessment API
# ---------------------------------------------------------------------------

def assess_impact(impact: ImpactAssessment) -> RiskAssessment:
    """
    Assess risk for a single ImpactAssessment object.

    Pure function: 0 network calls, 0 filesystem access, 0 code execution, 0 mutations.

    Args:
        impact: Underlying ImpactAssessment.

    Returns:
        RiskAssessment containing evaluated risk level, preserved confidence, and reasons.
    """
    risk_level = _determine_risk_level(impact)
    reasons = _generate_reasons(impact)

    return RiskAssessment(
        package_name=impact.package_name,
        affected_api=impact.affected_api,
        risk_level=risk_level,
        confidence=impact.confidence,
        reasons=reasons,
        impact=impact,
    )


def assess_compatibility(report: CompatibilityReport) -> list[RiskAssessment]:
    """
    Assess risk for all impact assessments in a CompatibilityReport.

    Returns an empty list if the report has no impacts.

    Args:
        report: CompatibilityReport from compatibility analysis.

    Returns:
        List of RiskAssessment objects.
    """
    if not report.has_impacts:
        return []

    return [assess_impact(a) for a in report.assessments]


def assess_project_risk(
    project_path: Path,
    reports_or_assessments: Sequence[CompatibilityReport | ProjectImpactReport | RiskAssessment],
) -> ProjectRiskReport:
    """
    Aggregate risk assessments across an entire project.

    Args:
        project_path: Path to the root directory of the project.
        reports_or_assessments: Sequence of CompatibilityReport, ProjectImpactReport, or RiskAssessment objects.

    Returns:
        ProjectRiskReport aggregating all assessments.
    """
    assessments: list[RiskAssessment] = []

    for item in reports_or_assessments:
        if isinstance(item, RiskAssessment):
            assessments.append(item)
        elif isinstance(item, CompatibilityReport):
            assessments.extend(assess_compatibility(item))
        elif isinstance(item, ProjectImpactReport):
            for entry in item.entries:
                if entry.report and entry.report.has_impacts:
                    assessments.extend(assess_compatibility(entry.report))

    return ProjectRiskReport(
        project_path=Path(project_path),
        assessments=assessments,
    )
