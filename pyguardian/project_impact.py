"""
project_impact.py — Project-Level Impact Aggregation for PyGuardian.

Responsibilities:
- Aggregate individual dependency CompatibilityReport objects across an entire project.
- Provide high-level project metrics (total dependencies, affected dependencies, impact counts by confidence).
- Preserve full underlying evidence hierarchy (ImpactAssessment, APIChangeMatch, DocumentedChange, source_url).
- Distinguish between analyzed dependencies with zero impacts and dependencies with analysis errors.
- Maintain original dependency ordering.

Design Principles:
- Pure in-memory aggregation: 0 network access, 0 filesystem I/O, 0 code execution.
- Deterministic analysis: NO AI, LLM, RAG, or vector DBs.
- Does NOT perform risk scoring or claim project/dependencies are "safe" or "broken".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from pyguardian.compatibility import CompatibilityReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status Model
# ---------------------------------------------------------------------------

class AnalysisStatus(Enum):
    """Status of compatibility analysis for a dependency."""

    ANALYZED = "analyzed"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DependencyImpactEntry:
    """
    Represents the compatibility analysis status and findings for a single dependency.

    Attributes:
        package_name:  Canonical name of the package.
        status:        AnalysisStatus (ANALYZED or ERROR).
        report:        CompatibilityReport object if analysis completed.
        error_message: Error description if status is ERROR.
    """

    package_name: str
    status: AnalysisStatus
    report: Optional[CompatibilityReport] = None
    error_message: Optional[str] = None

    @property
    def has_impacts(self) -> bool:
        """True if analysis succeeded and detected at least one impact assessment."""
        return self.report is not None and self.report.has_impacts

    @property
    def impact_count(self) -> int:
        """Total number of impact assessments detected for this dependency."""
        return self.report.impact_count if self.report else 0


@dataclass(frozen=True)
class ProjectImpactReport:
    """
    Aggregated compatibility and impact analysis report for an entire project.

    Attributes:
        project_path: Path to the root directory of the project analyzed.
        entries:      Ordered list of DependencyImpactEntry objects.
    """

    project_path: Path
    entries: list[DependencyImpactEntry] = field(default_factory=list)

    @property
    def total_dependencies(self) -> int:
        """Total number of dependencies in the project report."""
        return len(self.entries)

    @property
    def dependencies_with_impacts(self) -> int:
        """Number of dependencies that have at least one impact assessment."""
        return sum(1 for e in self.entries if e.has_impacts)

    @property
    def dependencies_with_errors(self) -> int:
        """Number of dependencies that encountered an analysis error."""
        return sum(1 for e in self.entries if e.status == AnalysisStatus.ERROR)

    @property
    def total_impacts(self) -> int:
        """Total count of impact assessments across all dependencies."""
        return sum(e.impact_count for e in self.entries)

    @property
    def high_confidence_impacts(self) -> int:
        """Total count of high confidence impact assessments."""
        count = 0
        for entry in self.entries:
            if entry.report:
                count += entry.report.high_confidence_count
        return count

    @property
    def medium_confidence_impacts(self) -> int:
        """Total count of medium confidence impact assessments."""
        count = 0
        for entry in self.entries:
            if entry.report:
                count += sum(1 for a in entry.report.assessments if a.confidence.lower() == "medium")
        return count

    @property
    def low_confidence_impacts(self) -> int:
        """Total count of low confidence impact assessments."""
        count = 0
        for entry in self.entries:
            if entry.report:
                count += sum(1 for a in entry.report.assessments if a.confidence.lower() == "low")
        return count


# ---------------------------------------------------------------------------
# Builder & Helpers
# ---------------------------------------------------------------------------

def create_error_entry(package_name: str, error_message: str) -> DependencyImpactEntry:
    """
    Helper to construct a DependencyImpactEntry for a failed dependency analysis.
    """
    return DependencyImpactEntry(
        package_name=package_name,
        status=AnalysisStatus.ERROR,
        report=None,
        error_message=error_message,
    )


def build_project_impact_report(
    project_path: Path,
    reports: Sequence[CompatibilityReport | DependencyImpactEntry],
) -> ProjectImpactReport:
    """
    Build a ProjectImpactReport aggregating individual dependency analysis results.

    Preserves exact input order and retains zero-impact dependencies and error states.

    Args:
        project_path: Path to the project root directory.
        reports:      Sequence of CompatibilityReport or DependencyImpactEntry objects.

    Returns:
        ProjectImpactReport aggregating the project findings.
    """
    entries: list[DependencyImpactEntry] = []

    for item in reports:
        if isinstance(item, DependencyImpactEntry):
            entries.append(item)
        elif isinstance(item, CompatibilityReport):
            entries.append(
                DependencyImpactEntry(
                    package_name=item.package_name,
                    status=AnalysisStatus.ANALYZED,
                    report=item,
                    error_message=None,
                )
            )

    return ProjectImpactReport(project_path=Path(project_path), entries=entries)
