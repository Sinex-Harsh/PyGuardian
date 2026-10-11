"""
migration_planner.py — Project-level Migration Planning Layer for PyGuardian.

Architecture:
ProjectAnalysis
      ↓
MigrationPlan / MigrationCandidates
      ↓
MigrationContext
      ↓
LLMProvider / MigrationRecommendation(s)
      ↓
build_project_migration_plan()
      ↓
ProjectMigrationPlan (PlannedMigration records)

Responsibilities:
- Combine deterministic MigrationCandidates and AI MigrationRecommendations into a ProjectMigrationPlan.
- Preserve original deterministic evidence (candidate risk, confidence, URLs, AST lines).
- Classify human review status (PROPOSED, REVIEW_REQUIRED, INSUFFICIENT_EVIDENCE, CONFLICT).
- Detect and flag conflicting AI recommendations without silently resolving or discarding them.
- Preserve deterministic discovery ordering across packages, files, and lines.
- Pure planning layer: 0 filesystem mutations, 0 git operations, 0 patch executions, 0 network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from pyguardian.impact import normalize_package_name
from pyguardian.migration import MigrationCandidate, MigrationPlan
from pyguardian.migration_ai import MigrationRecommendation
from pyguardian.risk import RiskLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MigrationPlannerError(Exception):
    """Raised when migration planning or recommendation matching fails."""
    pass


# ---------------------------------------------------------------------------
# Planning Status Enum
# ---------------------------------------------------------------------------

class MigrationStatus(str, Enum):
    """
    Status classification for a planned migration item.

    PROPOSED:              Clear migration proposal (high confidence, exact API, proposed replacement present, no uncertainties).
    REVIEW_REQUIRED:       Migration requires human developer review (low/medium confidence, missing replacement, uncertainties, package-level).
    INSUFFICIENT_EVIDENCE: Missing recommendation or insufficient evidence.
    CONFLICT:              Multiple conflicting AI recommendations received for the candidate.
    """

    PROPOSED = "proposed"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICT = "conflict"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedMigration:
    """
    Combines a deterministic MigrationCandidate with its AI MigrationRecommendation(s).

    Attributes:
        candidate:       The underlying deterministic MigrationCandidate.
        recommendations: Tuple of matched AI MigrationRecommendation objects.
        status:          Evaluated MigrationStatus classification.
        review_reasons:  Tuple of human-readable explanations for why review or conflict status was assigned.
    """

    candidate: MigrationCandidate
    recommendations: tuple[MigrationRecommendation, ...] = ()
    status: MigrationStatus = MigrationStatus.REVIEW_REQUIRED
    review_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce sequence attributes to immutable tuples."""
        if not isinstance(self.recommendations, tuple):
            object.__setattr__(self, "recommendations", tuple(self.recommendations))
        if not isinstance(self.review_reasons, tuple):
            object.__setattr__(self, "review_reasons", tuple(self.review_reasons))

    @property
    def requires_human_review(self) -> bool:
        """True if human review is required before migrating this item."""
        return self.status != MigrationStatus.PROPOSED

    @property
    def has_recommendation(self) -> bool:
        """True if at least one AI recommendation is attached."""
        return len(self.recommendations) > 0

    @property
    def primary_recommendation(self) -> Optional[MigrationRecommendation]:
        """Return the primary (first) AI recommendation, if available."""
        return self.recommendations[0] if self.recommendations else None

    @property
    def has_conflict(self) -> bool:
        """True if this planned migration has conflicting recommendations."""
        return self.status == MigrationStatus.CONFLICT


@dataclass(frozen=True)
class ProjectMigrationPlan:
    """
    Project-level migration plan aggregating all PlannedMigration items.

    Attributes:
        project_path: Root directory path of the analyzed project.
        migrations:   Tuple of ordered PlannedMigration items.
    """

    project_path: Path
    migrations: tuple[PlannedMigration, ...] = ()

    def __post_init__(self) -> None:
        """Coerce sequence attributes to immutable tuples."""
        if not isinstance(self.migrations, tuple):
            object.__setattr__(self, "migrations", tuple(self.migrations))

    @property
    def has_migrations(self) -> bool:
        """True if the plan contains at least one planned migration."""
        return len(self.migrations) > 0

    @property
    def migration_count(self) -> int:
        """Total number of planned migrations in the project."""
        return len(self.migrations)

    @property
    def packages_affected(self) -> tuple[str, ...]:
        """
        Tuple of unique package names affected by migrations,
        preserving candidate discovery order.
        """
        seen: set[str] = set()
        pkgs: list[str] = []
        for m in self.migrations:
            norm = normalize_package_name(m.candidate.package_name)
            if norm not in seen:
                seen.add(norm)
                pkgs.append(m.candidate.package_name)
        return tuple(pkgs)

    @property
    def high_risk_count(self) -> int:
        """Count of HIGH risk candidates."""
        return sum(1 for m in self.migrations if m.candidate.risk_level == RiskLevel.HIGH)

    @property
    def medium_risk_count(self) -> int:
        """Count of MEDIUM risk candidates."""
        return sum(1 for m in self.migrations if m.candidate.risk_level == RiskLevel.MEDIUM)

    @property
    def low_risk_count(self) -> int:
        """Count of LOW risk candidates."""
        return sum(1 for m in self.migrations if m.candidate.risk_level == RiskLevel.LOW)

    @property
    def requires_review_count(self) -> int:
        """Count of planned migrations requiring human review."""
        return sum(1 for m in self.migrations if m.requires_human_review)

    @property
    def conflict_count(self) -> int:
        """Count of planned migrations with conflicting AI recommendations."""
        return sum(1 for m in self.migrations if m.status == MigrationStatus.CONFLICT)

    @property
    def uncertain_count(self) -> int:
        """Count of migrations with AI uncertainties or non-high AI confidence."""
        count = 0
        for m in self.migrations:
            if any(r.uncertainties or r.confidence != "high" for r in m.recommendations):
                count += 1
        return count


# ---------------------------------------------------------------------------
# Internal Classification Helper
# ---------------------------------------------------------------------------

def _classify_planned_migration(
    candidate: MigrationCandidate,
    matched_recs: list[MigrationRecommendation],
) -> PlannedMigration:
    """
    Classify human review status and aggregate reasons for a candidate + matched recommendations.
    """
    recs_tuple = tuple(matched_recs)

    # Case 1: No recommendation available
    if not matched_recs:
        return PlannedMigration(
            candidate=candidate,
            recommendations=(),
            status=MigrationStatus.INSUFFICIENT_EVIDENCE,
            review_reasons=("No AI migration recommendation provided for candidate",),
        )

    # Case 2: Multiple recommendations for the same candidate
    if len(matched_recs) > 1:
        first = matched_recs[0]
        has_difference = any(
            r.proposed_replacement != first.proposed_replacement
            or r.recommended_action != first.recommended_action
            or r.summary != first.summary
            for r in matched_recs[1:]
        )
        if has_difference:
            return PlannedMigration(
                candidate=candidate,
                recommendations=recs_tuple,
                status=MigrationStatus.CONFLICT,
                review_reasons=("Multiple conflicting AI migration recommendations detected",),
            )

    # Case 3: Single recommendation (or identical recommendations)
    rec = matched_recs[0]
    reasons: list[str] = []

    if rec.confidence.lower() in ("low", "medium"):
        reasons.append(f"AI confidence level is '{rec.confidence}'")

    if not rec.proposed_replacement or not rec.proposed_replacement.strip():
        reasons.append("Missing proposed code replacement")

    if rec.uncertainties:
        reasons.append(f"AI identified uncertainties: {', '.join(rec.uncertainties)}")

    if candidate.is_package_level or candidate.affected_api is None:
        reasons.append("Package-level candidate without specific API usage location")

    if candidate.confidence.lower() == "low":
        reasons.append("Low deterministic evidence confidence")

    if reasons:
        return PlannedMigration(
            candidate=candidate,
            recommendations=recs_tuple,
            status=MigrationStatus.REVIEW_REQUIRED,
            review_reasons=tuple(reasons),
        )

    return PlannedMigration(
        candidate=candidate,
        recommendations=recs_tuple,
        status=MigrationStatus.PROPOSED,
        review_reasons=(),
    )


# ---------------------------------------------------------------------------
# Main Builder API
# ---------------------------------------------------------------------------

def build_project_migration_plan(
    migration_plan: MigrationPlan,
    recommendations: Sequence[MigrationRecommendation] = (),
) -> ProjectMigrationPlan:
    """
    Build a project-level migration plan aggregating candidates and AI recommendations.

    Maintains exact candidate discovery order.
    Matches recommendations by stable candidate identity (package_name, affected_api).
    Does NOT call any LLM and does NOT perform filesystem mutations.

    Args:
        migration_plan:  The MigrationPlan object containing ordered MigrationCandidates.
        recommendations: Sequence of AI MigrationRecommendation objects.

    Returns:
        Deterministic ProjectMigrationPlan.

    Raises:
        MigrationPlannerError: On invalid arguments or unmatched recommendations.
    """
    if not isinstance(migration_plan, MigrationPlan):
        raise MigrationPlannerError(f"Expected MigrationPlan object, got {type(migration_plan).__name__}.")

    # Group recommendations by (normalized_package_name, affected_api)
    rec_map: dict[tuple[str, Optional[str]], list[MigrationRecommendation]] = {}
    for rec in recommendations:
        if not isinstance(rec, MigrationRecommendation):
            raise MigrationPlannerError(f"Expected MigrationRecommendation item, got {type(rec).__name__}.")
        norm_pkg = normalize_package_name(rec.package_name)
        key = (norm_pkg, rec.affected_api)
        if key not in rec_map:
            rec_map[key] = []
        rec_map[key].append(rec)

    # Validate that all recommendations match at least one candidate
    candidate_keys: set[tuple[str, Optional[str]]] = set()
    for c in migration_plan.candidates:
        norm_pkg = normalize_package_name(c.package_name)
        candidate_keys.add((norm_pkg, c.affected_api))

    for (norm_pkg, api), rec_list in rec_map.items():
        if (norm_pkg, api) not in candidate_keys:
            raise MigrationPlannerError(
                f"Recommendation for package '{norm_pkg}' and API '{api}' "
                f"does not match any candidate in the provided MigrationPlan."
            )

    # Map candidates to matching recommendations while preserving exact candidate order
    planned_migrations: list[PlannedMigration] = []

    for candidate in migration_plan.candidates:
        norm_pkg = normalize_package_name(candidate.package_name)
        key = (norm_pkg, candidate.affected_api)
        matched_recs = rec_map.get(key, [])

        planned = _classify_planned_migration(candidate, matched_recs)
        planned_migrations.append(planned)

    return ProjectMigrationPlan(
        project_path=migration_plan.project_path,
        migrations=tuple(planned_migrations),
    )
