"""
compatibility.py — Compatibility & Impact Analysis for PyGuardian.

Responsibilities:
- Synthesize dependency updates, AST API usages, and documented changes into structured ImpactAssessment objects.
- Perform version-window filtering (current_version < change_version <= target_version).
- Separate version transition classification (UpdateType) from documented code impact.
- Produce traceable, evidence-based CompatibilityReport objects.

Design Principles:
- Pure deterministic analysis: NO AI, LLM, RAG, or vector DBs.
- 0 network requests, 0 file system modifications, 0 code execution.
- Preserves full provenance (source_url, source location, description) on all assessments.
- Does NOT claim project "will break" or calculate speculative risk scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.version import InvalidVersion, Version

from pyguardian.change_analysis import (
    APIChangeMatch,
    DocumentedChange,
    match_api_changes,
)
from pyguardian.packages import UpdateType, classify_update
from pyguardian.source import APIUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImpactAssessment:
    """
    Represents an evidence-based assessment of how a dependency update impacts project API usages.

    Attributes:
        package_name:    Canonical package name (e.g. 'pandas').
        current_version: Current baseline version of the package.
        target_version:  Target version to compare or upgrade to.
        update_type:     Version update classification (MAJOR, MINOR, PATCH, etc.).
        affected_api:    Specific API path affected (e.g. 'pandas.read_csv'), or None if package-level.
        file_path:       Source file path where affected API usage occurs.
        line_number:     Line number of API usage in source file.
        change_type:     Kind of documented change (removed, deprecated, behavior_change, etc.).
        description:     Description of change from authoritative documentation.
        confidence:      Confidence level of impact assessment ('high', 'medium', 'low').
        evidence:        The underlying APIChangeMatch object.
    """

    package_name: str
    current_version: Optional[Version]
    target_version: Optional[Version]
    update_type: UpdateType
    affected_api: Optional[str]
    file_path: Optional[Path]
    line_number: Optional[int]
    change_type: Optional[str]
    description: str
    confidence: str
    evidence: Optional[APIChangeMatch] = None

    @property
    def is_exact_match(self) -> bool:
        """True if assessment stems from an exact API symbol match."""
        return self.evidence is not None and self.evidence.match_type == "exact"

    @property
    def source_url(self) -> Optional[str]:
        """Returns authoritative documentation URL if evidence is present."""
        if self.evidence and self.evidence.documented_change:
            return self.evidence.documented_change.source_url
        return None


@dataclass(frozen=True)
class CompatibilityReport:
    """
    Aggregated compatibility analysis report for a dependency update.

    Attributes:
        package_name:    Canonical package name.
        current_version: Baseline version.
        target_version:  Target version.
        update_type:     Update classification (MAJOR / MINOR / PATCH).
        assessments:     List of ImpactAssessment records.
    """

    package_name: str
    current_version: Optional[Version]
    target_version: Optional[Version]
    update_type: UpdateType
    assessments: list[ImpactAssessment] = field(default_factory=list)

    @property
    def has_impacts(self) -> bool:
        """True if at least one impact assessment was generated."""
        return len(self.assessments) > 0

    @property
    def impact_count(self) -> int:
        """Total number of impact assessments."""
        return len(self.assessments)

    @property
    def high_confidence_count(self) -> int:
        """Number of high confidence impact assessments."""
        return sum(1 for a in self.assessments if a.confidence.lower() == "high")


# ---------------------------------------------------------------------------
# Version Filtering & Analysis
# ---------------------------------------------------------------------------

def _is_change_in_version_window(
    change_version: Optional[Version],
    current_version: Optional[Version],
    target_version: Optional[Version],
) -> bool:
    """
    Check if a documented change version falls strictly within the version window:
    current_version < change_version <= target_version
    """
    if change_version is None:
        # Preserve changes with unknown version conservatively
        return True

    if current_version is None or target_version is None:
        return True

    if current_version == target_version:
        return False

    return current_version < change_version <= target_version


def analyze_compatibility(
    package_name: str,
    current_version: str | Version | None,
    target_version: str | Version | None,
    api_usages: list[APIUsage],
    documented_changes: list[DocumentedChange],
    *,
    matches: Optional[list[APIChangeMatch]] = None,
    update_type: Optional[UpdateType] = None,
) -> CompatibilityReport:
    """
    Perform deterministic compatibility and impact analysis for a package update.

    Args:
        package_name:       Canonical name of the package.
        current_version:    Baseline current version.
        target_version:     Target version to compare.
        api_usages:         Discovered APIUsage records from source code.
        documented_changes: Extracted DocumentedChange records.
        matches:            Optional pre-computed list of APIChangeMatch objects.
        update_type:        Optional UpdateType classification.

    Returns:
        A CompatibilityReport containing matching ImpactAssessment records.
    """
    # Parse version parameters safely
    curr_ver: Optional[Version] = None
    if isinstance(current_version, Version):
        curr_ver = current_version
    elif current_version:
        try:
            curr_ver = Version(str(current_version))
        except InvalidVersion:
            pass

    targ_ver: Optional[Version] = None
    if isinstance(target_version, Version):
        targ_ver = target_version
    elif target_version:
        try:
            targ_ver = Version(str(target_version))
        except InvalidVersion:
            pass

    # Classify update type if not provided
    if update_type is None:
        c_str = str(curr_ver) if curr_ver else ""
        t_str = str(targ_ver) if targ_ver else ""
        utype = classify_update(c_str, t_str)
    else:
        utype = update_type

    # Version-aware filtering of documented changes
    filtered_changes = [
        c
        for c in documented_changes
        if _is_change_in_version_window(c.version, curr_ver, targ_ver)
    ]

    # Perform matching if matches not provided
    if matches is None:
        computed_matches = match_api_changes(api_usages, filtered_changes)
    else:
        computed_matches = [
            m
            for m in matches
            if _is_change_in_version_window(m.documented_change.version, curr_ver, targ_ver)
        ]

    assessments: list[ImpactAssessment] = []

    for match in computed_matches:
        usage = match.api_usage
        change = match.documented_change

        affected_api = usage.api_path if match.match_type == "exact" else None

        assessment = ImpactAssessment(
            package_name=package_name,
            current_version=curr_ver,
            target_version=targ_ver,
            update_type=utype,
            affected_api=affected_api,
            file_path=usage.file_path,
            line_number=usage.line_number,
            change_type=change.change_type,
            description=change.description,
            confidence=match.confidence,
            evidence=match,
        )
        assessments.append(assessment)

    return CompatibilityReport(
        package_name=package_name,
        current_version=curr_ver,
        target_version=targ_ver,
        update_type=utype,
        assessments=assessments,
    )
