"""
migration.py — Migration Intelligence Foundation for PyGuardian.

Responsibilities:
- Convert structured ProjectAnalysis findings and RiskAssessment records into explicit MigrationCandidate objects.
- Assemble a project-level MigrationPlan representing all detected code locations and APIs requiring migration.
- Preserve full provenance (package, API, file, line, current/target version, change_type, risk, confidence, source_url, evidence).
- Maintain deterministic ordering (package scan order -> file path -> line number).

Design Principles:
- Pure deterministic conversion layer: NO AI, LLM, RAG, embeddings, or code generation.
- 0 network requests, 0 filesystem scanning, 0 code execution, 0 mutations.
- Migration intelligence is representation, NOT code modification or patch execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.version import Version

from pyguardian.compatibility import ImpactAssessment
from pyguardian.impact import normalize_package_name
from pyguardian.pipeline import ProjectAnalysis
from pyguardian.risk import RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MigrationCandidate:
    """
    Represents a single concrete code or package location identified for migration.

    Attributes:
        package_name:    Canonical name of the package (e.g. 'pandas').
        affected_api:    Fully qualified API path affected, or None if package-level change.
        file_path:       Source file path where API usage occurs, or None if package-level.
        line_number:     1-based line number of API usage in source file, or None if package-level.
        current_version: Current baseline version of the package.
        target_version:  Target version being evaluated or upgraded to.
        change_type:     Kind of documented change (removed, deprecated, behavior_change, renamed, etc.).
        risk_level:      Evaluated RiskLevel (LOW, MEDIUM, HIGH).
        confidence:      Evidence confidence rating ('high', 'medium', 'low').
        description:     Human-readable description of change from documentation.
        source_url:      URL of authoritative documentation if available.
        evidence:        Underlying RiskAssessment object maintaining complete evidence provenance.
    """

    package_name: str
    affected_api: Optional[str]
    file_path: Optional[Path]
    line_number: Optional[int]
    current_version: Optional[Version]
    target_version: Optional[Version]
    change_type: Optional[str]
    risk_level: RiskLevel
    confidence: str
    description: str
    source_url: Optional[str]
    evidence: Optional[RiskAssessment] = field(default=None)

    @property
    def is_exact_match(self) -> bool:
        """True if candidate stems from an exact API symbol usage location."""
        return self.affected_api is not None and self.file_path is not None

    @property
    def is_package_level(self) -> bool:
        """True if candidate represents a package-level behavior change without specific source file."""
        return self.affected_api is None and self.file_path is None


@dataclass(frozen=True)
class MigrationPlan:
    """
    Aggregated project-level migration plan containing MigrationCandidate records.

    Attributes:
        project_path: Root directory path of the analyzed project.
        candidates:   Ordered list of MigrationCandidate objects.
    """

    project_path: Path
    candidates: list[MigrationCandidate] = field(default_factory=list)

    @property
    def has_candidates(self) -> bool:
        """True if the migration plan contains at least one candidate."""
        return len(self.candidates) > 0

    @property
    def candidate_count(self) -> int:
        """Total number of migration candidates."""
        return len(self.candidates)

    @property
    def high_risk_count(self) -> int:
        """Count of HIGH risk migration candidates."""
        return sum(1 for c in self.candidates if c.risk_level == RiskLevel.HIGH)

    @property
    def medium_risk_count(self) -> int:
        """Count of MEDIUM risk migration candidates."""
        return sum(1 for c in self.candidates if c.risk_level == RiskLevel.MEDIUM)

    @property
    def low_risk_count(self) -> int:
        """Count of LOW risk migration candidates."""
        return sum(1 for c in self.candidates if c.risk_level == RiskLevel.LOW)

    @property
    def packages_affected(self) -> list[str]:
        """
        List of unique package names affected by migration candidates,
        preserving original candidate discovery order.
        """
        seen: set[str] = set()
        pkgs: list[str] = []
        for c in self.candidates:
            if c.package_name not in seen:
                seen.add(c.package_name)
                pkgs.append(c.package_name)
        return pkgs


# ---------------------------------------------------------------------------
# Migration Plan Builder API
# ---------------------------------------------------------------------------

def build_migration_plan(project_analysis: ProjectAnalysis) -> MigrationPlan:
    """
    Derive a structured MigrationPlan from an existing ProjectAnalysis.

    Extracts MigrationCandidate objects exclusively from structured RiskAssessment evidence,
    preserving exact provenance, version transitions, risk levels, and source locations.

    Maintains deterministic ordering:
    1. Package scan order (as discovered in project_scan)
    2. Source file path
    3. Line number

    Args:
        project_analysis: The ProjectAnalysis result from pipeline execution.

    Returns:
        A MigrationPlan object aggregating all derived MigrationCandidate records.
    """
    if not project_analysis.risk_assessments:
        return MigrationPlan(
            project_path=project_analysis.project_path,
            candidates=[],
        )

    # Build index mapping package names to original scan order
    pkg_order: dict[str, int] = {}
    if project_analysis.project_scan and project_analysis.project_scan.results:
        for idx, res in enumerate(project_analysis.project_scan.results):
            norm_name = normalize_package_name(res.name)
            if norm_name not in pkg_order:
                pkg_order[norm_name] = idx

    candidates: list[MigrationCandidate] = []

    for risk in project_analysis.risk_assessments:
        impact = risk.impact
        candidate = MigrationCandidate(
            package_name=risk.package_name,
            affected_api=risk.affected_api,
            file_path=impact.file_path,
            line_number=impact.line_number,
            current_version=impact.current_version,
            target_version=impact.target_version,
            change_type=impact.change_type,
            risk_level=risk.risk_level,
            confidence=risk.confidence,
            description=impact.description,
            source_url=impact.source_url,
            evidence=risk,
        )
        candidates.append(candidate)

    # Sort candidates deterministically:
    # 1. Package order
    # 2. File path (alphabetical)
    # 3. Line number (numerical, placing package-level None as -1)
    def _sort_key(c: MigrationCandidate) -> tuple[int, str, int]:
        norm_pkg = normalize_package_name(c.package_name)
        p_idx = pkg_order.get(norm_pkg, 9999)
        f_path = str(c.file_path) if c.file_path is not None else ""
        l_num = c.line_number if c.line_number is not None else -1
        return (p_idx, f_path, l_num)

    sorted_candidates = sorted(candidates, key=_sort_key)

    return MigrationPlan(
        project_path=project_analysis.project_path,
        candidates=sorted_candidates,
    )
