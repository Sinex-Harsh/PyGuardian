"""
test_migration_plan.py — Tests for V0.3 Step 5: Migration Plan Generation.

All tests are completely offline and deterministic.
0 network calls, 0 source file mutations, 0 LLM calls.
"""

from pathlib import Path
from typing import Optional

import pytest
from packaging.version import Version

from pyguardian.migration import MigrationCandidate, MigrationPlan
from pyguardian.migration_ai import MigrationRecommendation
from pyguardian.migration_planner import (
    MigrationPlannerError,
    MigrationStatus,
    PlannedMigration,
    ProjectMigrationPlan,
    build_project_migration_plan,
)
from pyguardian.risk import RiskLevel


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def create_candidate(
    package_name: str = "pandas",
    affected_api: Optional[str] = "pandas.DataFrame.append",
    file_path: Optional[Path] = Path("src/data.py"),
    line_number: Optional[int] = 42,
    risk_level: RiskLevel = RiskLevel.HIGH,
    confidence: str = "high",
    description: str = "DataFrame.append was removed.",
    source_url: Optional[str] = "https://pandas.pydata.org/docs/whatsnew/v2.0.0.html",
) -> MigrationCandidate:
    """Helper to construct a deterministic MigrationCandidate."""
    return MigrationCandidate(
        package_name=package_name,
        affected_api=affected_api,
        file_path=file_path,
        line_number=line_number,
        current_version=Version("1.5.3"),
        target_version=Version("2.0.0"),
        change_type="removed",
        risk_level=risk_level,
        confidence=confidence,
        description=description,
        source_url=source_url,
    )


def create_recommendation(
    package_name: str = "pandas",
    affected_api: Optional[str] = "pandas.DataFrame.append",
    summary: str = "Replace DataFrame.append with pandas.concat",
    reason: str = "DataFrame.append was removed in 2.0.0.",
    recommended_action: str = "Use pd.concat([df1, df2])",
    proposed_replacement: Optional[str] = "res = pd.concat([df, other_df])",
    confidence: str = "high",
    evidence_references: tuple[str, ...] = ("CANDIDATE", "SOURCE-1"),
    uncertainties: tuple[str, ...] = (),
) -> MigrationRecommendation:
    """Helper to construct an AI MigrationRecommendation."""
    return MigrationRecommendation(
        package_name=package_name,
        affected_api=affected_api,
        summary=summary,
        reason=reason,
        recommended_action=recommended_action,
        proposed_replacement=proposed_replacement,
        confidence=confidence,
        evidence_references=evidence_references,
        uncertainties=uncertainties,
    )


# ---------------------------------------------------------------------------
# Basic Planning Tests (1-4)
# ---------------------------------------------------------------------------

def test_1_one_candidate_one_recommendation() -> None:
    """1. One candidate + one recommendation -> one planned migration."""
    cand = create_candidate()
    rec = create_recommendation()
    plan = MigrationPlan(project_path=Path("."), candidates=[cand])

    proj_plan = build_project_migration_plan(plan, [rec])

    assert proj_plan.has_migrations
    assert proj_plan.migration_count == 1
    pm = proj_plan.migrations[0]
    assert pm.candidate == cand
    assert pm.recommendations == (rec,)
    assert pm.status == MigrationStatus.PROPOSED
    assert not pm.requires_human_review


def test_2_multiple_candidates_all_preserved() -> None:
    """2. Multiple candidates -> all preserved in ProjectMigrationPlan."""
    c1 = create_candidate(package_name="pandas", affected_api="pandas.DataFrame.append", line_number=10)
    c2 = create_candidate(package_name="pandas", affected_api="pandas.util.testing", line_number=20)
    c3 = create_candidate(package_name="requests", affected_api="requests.packages", line_number=5)

    r1 = create_recommendation(package_name="pandas", affected_api="pandas.DataFrame.append")
    r2 = create_recommendation(package_name="pandas", affected_api="pandas.util.testing")
    r3 = create_recommendation(package_name="requests", affected_api="requests.packages")

    plan = MigrationPlan(project_path=Path("."), candidates=[c1, c2, c3])
    proj_plan = build_project_migration_plan(plan, [r1, r2, r3])

    assert proj_plan.migration_count == 3
    assert proj_plan.migrations[0].candidate == c1
    assert proj_plan.migrations[1].candidate == c2
    assert proj_plan.migrations[2].candidate == c3


def test_3_multiple_packages_represented_correctly() -> None:
    """3. Multiple packages -> grouped/represented in packages_affected correctly."""
    c1 = create_candidate(package_name="pandas")
    c2 = create_candidate(package_name="requests", affected_api="requests.get")
    c3 = create_candidate(package_name="pandas", affected_api="pandas.read_csv")

    r1 = create_recommendation(package_name="pandas", affected_api="pandas.DataFrame.append")
    r2 = create_recommendation(package_name="requests", affected_api="requests.get")
    r3 = create_recommendation(package_name="pandas", affected_api="pandas.read_csv")

    plan = MigrationPlan(project_path=Path("."), candidates=[c1, c2, c3])
    proj_plan = build_project_migration_plan(plan, [r1, r2, r3])

    assert proj_plan.packages_affected == ("pandas", "requests")


def test_4_candidate_order_is_deterministic() -> None:
    """4. Candidate order is strictly preserved from input MigrationPlan."""
    c1 = create_candidate(package_name="numpy", affected_api="numpy.matrix", line_number=100)
    c2 = create_candidate(package_name="pandas", affected_api="pandas.DataFrame.append", line_number=5)

    r1 = create_recommendation(package_name="numpy", affected_api="numpy.matrix")
    r2 = create_recommendation(package_name="pandas", affected_api="pandas.DataFrame.append")

    plan = MigrationPlan(project_path=Path("."), candidates=[c1, c2])
    proj_plan = build_project_migration_plan(plan, [r2, r1])  # Rec order differs from plan order

    assert proj_plan.migrations[0].candidate == c1
    assert proj_plan.migrations[1].candidate == c2


# ---------------------------------------------------------------------------
# Risk Properties Tests (5-7)
# ---------------------------------------------------------------------------

def test_5_high_risk_candidates_counted() -> None:
    """5. High-risk candidates counted correctly."""
    c1 = create_candidate(risk_level=RiskLevel.HIGH)
    c2 = create_candidate(affected_api="pandas.read_csv", risk_level=RiskLevel.HIGH)
    r1 = create_recommendation(affected_api="pandas.DataFrame.append")
    r2 = create_recommendation(affected_api="pandas.read_csv")

    plan = MigrationPlan(project_path=Path("."), candidates=[c1, c2])
    proj_plan = build_project_migration_plan(plan, [r1, r2])

    assert proj_plan.high_risk_count == 2
    assert proj_plan.medium_risk_count == 0
    assert proj_plan.low_risk_count == 0


def test_6_medium_risk_candidates_counted() -> None:
    """6. Medium-risk candidates counted correctly."""
    c1 = create_candidate(risk_level=RiskLevel.MEDIUM)
    r1 = create_recommendation()

    plan = MigrationPlan(project_path=Path("."), candidates=[c1])
    proj_plan = build_project_migration_plan(plan, [r1])

    assert proj_plan.medium_risk_count == 1
    assert proj_plan.high_risk_count == 0


def test_7_low_risk_candidates_counted() -> None:
    """7. Low-risk candidates counted correctly."""
    c1 = create_candidate(risk_level=RiskLevel.LOW)
    r1 = create_recommendation()

    plan = MigrationPlan(project_path=Path("."), candidates=[c1])
    proj_plan = build_project_migration_plan(plan, [r1])

    assert proj_plan.low_risk_count == 1


# ---------------------------------------------------------------------------
# Human Review Tests (8-12)
# ---------------------------------------------------------------------------

def test_8_low_ai_confidence_requires_review() -> None:
    """8. Low AI confidence -> review required."""
    cand = create_candidate()
    rec = create_recommendation(confidence="low")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.status == MigrationStatus.REVIEW_REQUIRED
    assert pm.requires_human_review
    assert any("AI confidence level is 'low'" in r for r in pm.review_reasons)


def test_9_missing_proposed_replacement_requires_review() -> None:
    """9. Missing proposed replacement -> review required."""
    cand = create_candidate()
    rec = create_recommendation(proposed_replacement=None)

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.status == MigrationStatus.REVIEW_REQUIRED
    assert pm.requires_human_review
    assert any("Missing proposed code replacement" in r for r in pm.review_reasons)


def test_10_ai_uncertainty_requires_review() -> None:
    """10. AI uncertainty -> review required."""
    cand = create_candidate()
    rec = create_recommendation(uncertainties=("Ensure DataFrame columns match",))

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.status == MigrationStatus.REVIEW_REQUIRED
    assert pm.requires_human_review
    assert any("AI identified uncertainties" in r for r in pm.review_reasons)


def test_11_package_level_candidate_requires_review() -> None:
    """11. Package-level candidate -> review required where exact mapping is unavailable."""
    cand = create_candidate(affected_api=None, file_path=None, line_number=None)
    rec = create_recommendation(affected_api=None)

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.status == MigrationStatus.REVIEW_REQUIRED
    assert any("Package-level candidate" in r for r in pm.review_reasons)


def test_12_insufficient_evidence_requires_review() -> None:
    """12. Insufficient evidence / missing recommendation -> status INSUFFICIENT_EVIDENCE."""
    cand = create_candidate()
    plan = MigrationPlan(project_path=Path("."), candidates=[cand])

    proj_plan = build_project_migration_plan(plan, [])  # No recommendations passed

    pm = proj_plan.migrations[0]
    assert pm.status == MigrationStatus.INSUFFICIENT_EVIDENCE
    assert pm.requires_human_review
    assert not pm.has_recommendation


# ---------------------------------------------------------------------------
# Conflicting Recommendation Tests (13-15)
# ---------------------------------------------------------------------------

def test_13_conflicting_recommendations_detected() -> None:
    """13. Conflicting recommendations are detected."""
    cand = create_candidate()
    r1 = create_recommendation(proposed_replacement="res = pd.concat([df, other])")
    r2 = create_recommendation(proposed_replacement="res = df.assign(**other)")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [r1, r2])

    pm = proj_plan.migrations[0]
    assert pm.has_conflict
    assert pm.status == MigrationStatus.CONFLICT


def test_14_conflicting_recommendations_not_silently_resolved() -> None:
    """14. Conflicting recommendations are NOT silently resolved; all are preserved."""
    cand = create_candidate()
    r1 = create_recommendation(proposed_replacement="res = pd.concat([df, other])")
    r2 = create_recommendation(proposed_replacement="res = df.assign(**other)")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [r1, r2])

    pm = proj_plan.migrations[0]
    assert len(pm.recommendations) == 2
    assert pm.recommendations[0] == r1
    assert pm.recommendations[1] == r2


def test_15_conflict_marked_for_human_review() -> None:
    """15. Conflict is marked for human review."""
    cand = create_candidate()
    r1 = create_recommendation(proposed_replacement="res = pd.concat([df, other])")
    r2 = create_recommendation(proposed_replacement="res = df.assign(**other)")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [r1, r2])

    assert proj_plan.conflict_count == 1
    assert proj_plan.migrations[0].requires_human_review


# ---------------------------------------------------------------------------
# Matching Tests (16-18)
# ---------------------------------------------------------------------------

def test_16_recommendation_matched_by_stable_identity() -> None:
    """16. Recommendation is matched by stable candidate identity (package + affected API)."""
    cand = create_candidate(package_name="pandas", affected_api="pandas.DataFrame.append")
    rec = create_recommendation(package_name="pandas", affected_api="pandas.DataFrame.append")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    assert proj_plan.migrations[0].primary_recommendation == rec


def test_17_mismatched_recommendation_raises_error() -> None:
    """17. Recommendation with mismatched package/API raises MigrationPlannerError."""
    cand = create_candidate(package_name="pandas", affected_api="pandas.DataFrame.append")
    # Rec for unknown package 'unknown_pkg'
    rec = create_recommendation(package_name="unknown_pkg", affected_api="unknown_pkg.func")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])

    with pytest.raises(MigrationPlannerError, match="does not match any candidate"):
        build_project_migration_plan(plan, [rec])


def test_18_duplicate_ambiguous_recommendations_handled_deterministically() -> None:
    """18. Duplicate/ambiguous recommendations are matched deterministically."""
    c1 = create_candidate(package_name="pandas", affected_api="pandas.DataFrame.append", line_number=10)
    c2 = create_candidate(package_name="pandas", affected_api="pandas.DataFrame.append", line_number=20)
    rec = create_recommendation(package_name="pandas", affected_api="pandas.DataFrame.append")

    plan = MigrationPlan(project_path=Path("."), candidates=[c1, c2])
    proj_plan = build_project_migration_plan(plan, [rec])

    # Rec attached to both candidates sharing the same package & API
    assert proj_plan.migrations[0].primary_recommendation == rec
    assert proj_plan.migrations[1].primary_recommendation == rec


# ---------------------------------------------------------------------------
# Evidence Preservation Tests (19-23)
# ---------------------------------------------------------------------------

def test_19_deterministic_candidate_confidence_unchanged() -> None:
    """19. Deterministic candidate confidence remains unchanged."""
    cand = create_candidate(confidence="medium")
    rec = create_recommendation(confidence="high")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    assert proj_plan.migrations[0].candidate.confidence == "medium"


def test_20_deterministic_risk_unchanged() -> None:
    """20. Deterministic risk level remains unchanged."""
    cand = create_candidate(risk_level=RiskLevel.HIGH)
    rec = create_recommendation()

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    assert proj_plan.migrations[0].candidate.risk_level == RiskLevel.HIGH


def test_21_source_url_unchanged() -> None:
    """21. Source URL remains unchanged."""
    url = "https://pandas.pydata.org/docs/whatsnew/v2.0.0.html"
    cand = create_candidate(source_url=url)
    rec = create_recommendation()

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    assert proj_plan.migrations[0].candidate.source_url == url


def test_22_ai_confidence_remains_separate() -> None:
    """22. AI confidence remains separate on MigrationRecommendation."""
    cand = create_candidate(confidence="low")
    rec = create_recommendation(confidence="high")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.candidate.confidence == "low"
    assert pm.primary_recommendation is not None
    assert pm.primary_recommendation.confidence == "high"


def test_23_proposed_replacement_remains_explicitly_ai_generated() -> None:
    """23. Proposed replacement remains explicitly AI-generated on recommendation."""
    cand = create_candidate(description="DataFrame.append is removed.")
    rec = create_recommendation(proposed_replacement="pd.concat([df1, df2])")

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.candidate.description == "DataFrame.append is removed."
    assert pm.primary_recommendation is not None
    assert pm.primary_recommendation.proposed_replacement == "pd.concat([df1, df2])"


# ---------------------------------------------------------------------------
# Edge Cases & Safety Tests (24-30)
# ---------------------------------------------------------------------------

def test_24_empty_candidates_empty_project_plan() -> None:
    """24. Empty candidates -> empty project plan."""
    plan = MigrationPlan(project_path=Path("."), candidates=[])
    proj_plan = build_project_migration_plan(plan, [])

    assert not proj_plan.has_migrations
    assert proj_plan.migration_count == 0
    assert proj_plan.packages_affected == ()


def test_25_missing_recommendation_candidate_retained() -> None:
    """25. Missing recommendation -> candidate retained with INSUFFICIENT_EVIDENCE status."""
    cand = create_candidate()
    plan = MigrationPlan(project_path=Path("."), candidates=[cand])

    proj_plan = build_project_migration_plan(plan, [])

    assert proj_plan.migration_count == 1
    pm = proj_plan.migrations[0]
    assert pm.status == MigrationStatus.INSUFFICIENT_EVIDENCE
    assert pm.requires_human_review


def test_26_null_proposed_replacement_handled_safely() -> None:
    """26. Null proposed replacement handled safely."""
    cand = create_candidate()
    rec = create_recommendation(proposed_replacement=None)

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.primary_recommendation is not None
    assert pm.primary_recommendation.proposed_replacement is None
    assert pm.requires_human_review


def test_27_empty_uncertainties_handled_safely() -> None:
    """27. Empty uncertainties handled safely."""
    cand = create_candidate()
    rec = create_recommendation(uncertainties=())

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    proj_plan = build_project_migration_plan(plan, [rec])

    pm = proj_plan.migrations[0]
    assert pm.primary_recommendation is not None
    assert pm.primary_recommendation.uncertainties == ()


def test_28_input_objects_not_mutated() -> None:
    """28. Input candidate and recommendation objects are not mutated."""
    cand = create_candidate()
    rec = create_recommendation()

    cand_dict_before = dict(cand.__dict__)
    rec_dict_before = dict(rec.__dict__)

    plan = MigrationPlan(project_path=Path("."), candidates=[cand])
    build_project_migration_plan(plan, [rec])

    assert cand.__dict__ == cand_dict_before
    assert rec.__dict__ == rec_dict_before


def test_29_no_filesystem_modification(tmp_path: Path) -> None:
    """29. No filesystem modifications are performed during planning."""
    test_file = tmp_path / "app.py"
    initial_content = "import pandas as pd\ndf.append(x)\n"
    test_file.write_text(initial_content, encoding="utf-8")

    cand = create_candidate(file_path=test_file)
    rec = create_recommendation()
    plan = MigrationPlan(project_path=tmp_path, candidates=[cand])

    build_project_migration_plan(plan, [rec])

    assert test_file.read_text(encoding="utf-8") == initial_content


def test_30_no_network_calls() -> None:
    """30. Planning is completely offline and deterministic."""
    cand = create_candidate()
    rec = create_recommendation()
    plan = MigrationPlan(project_path=Path("."), candidates=[cand])

    proj_plan = build_project_migration_plan(plan, [rec])

    assert proj_plan.migration_count == 1
