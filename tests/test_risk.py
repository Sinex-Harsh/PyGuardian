"""
test_risk.py — Unit tests for PyGuardian Risk & Confidence Engine.
"""

from __future__ import annotations

from pathlib import Path
from packaging.version import Version

from pyguardian.change_analysis import APIChangeMatch, DocumentedChange
from pyguardian.compatibility import CompatibilityReport, ImpactAssessment
from pyguardian.packages import UpdateType
from pyguardian.project_impact import DependencyImpactEntry, ProjectImpactReport, AnalysisStatus
from pyguardian.risk import (
    ProjectRiskReport,
    RiskAssessment,
    RiskLevel,
    assess_compatibility,
    assess_impact,
    assess_project_risk,
)
from pyguardian.source import APIUsage


def _create_dummy_impact(
    package_name: str = "pandas",
    current_version: str = "2.0.0",
    target_version: str = "2.1.0",
    update_type: UpdateType = UpdateType.MINOR,
    affected_api: str | None = "pandas.read_csv",
    change_type: str = "removed",
    description: str = "pandas.read_csv was removed.",
    confidence: str = "high",
    is_exact: bool = True,
    file_path: str = "app/data.py",
    line_number: int = 42,
    source_url: str = "https://pandas.pydata.org/docs/release/2.1.0.html",
) -> ImpactAssessment:
    """Helper to construct an ImpactAssessment for testing."""
    curr_v = Version(current_version) if current_version else None
    targ_v = Version(target_version) if target_version else None
    usage = APIUsage(
        package_name=package_name,
        api_path=affected_api or f"{package_name}.module",
        file_path=Path(file_path),
        line_number=line_number,
        usage_type="call",
    )
    doc_change = DocumentedChange(
        package_name=package_name,
        version=targ_v,
        api_path=affected_api if is_exact else None,
        change_type=change_type,
        description=description,
        source_url=source_url,
        confidence=confidence,
    )
    match_type = "exact" if is_exact else "package"
    match = APIChangeMatch(
        api_usage=usage,
        documented_change=doc_change,
        match_type=match_type,
        confidence=confidence,
    )
    return ImpactAssessment(
        package_name=package_name,
        current_version=curr_v,
        target_version=targ_v,
        update_type=update_type,
        affected_api=affected_api if is_exact else None,
        file_path=Path(file_path),
        line_number=line_number,
        change_type=change_type,
        description=description,
        confidence=confidence,
        evidence=match,
    )


# ---------------------------------------------------------------------------
# Core Specification Tests (Tests 1-12)
# ---------------------------------------------------------------------------

def test_1_exact_removed_api():
    """Test 1 — Exact removed API produces HIGH risk and preserves confidence."""
    impact = _create_dummy_impact(
        affected_api="pandas.read_csv",
        change_type="removed",
        confidence="high",
        is_exact=True,
    )
    assessment = assess_impact(impact)

    assert assessment.risk_level == RiskLevel.HIGH
    assert assessment.confidence == "high"
    assert assessment.affected_api == "pandas.read_csv"
    assert "API is documented as removed" in assessment.reasons
    assert "Exact API match" in assessment.reasons


def test_2_exact_renamed_api():
    """Test 2 — Exact renamed API produces HIGH risk."""
    impact = _create_dummy_impact(
        affected_api="pandas.read_csv",
        change_type="renamed",
        description="pandas.read_csv was renamed to pandas.read_file.",
        confidence="high",
        is_exact=True,
    )
    assessment = assess_impact(impact)

    assert assessment.risk_level == RiskLevel.HIGH
    assert "API is documented as renamed" in assessment.reasons


def test_3_exact_deprecated_api():
    """Test 3 — Exact deprecated API produces MEDIUM risk (not HIGH)."""
    impact = _create_dummy_impact(
        affected_api="pandas.read_csv",
        change_type="deprecated",
        description="pandas.read_csv is deprecated.",
        confidence="medium",
        is_exact=True,
    )
    assessment = assess_impact(impact)

    assert assessment.risk_level == RiskLevel.MEDIUM
    assert assessment.risk_level != RiskLevel.HIGH
    assert "API is documented as deprecated" in assessment.reasons


def test_4_exact_behavior_change():
    """Test 4 — Exact behavior change produces MEDIUM risk."""
    impact = _create_dummy_impact(
        affected_api="pandas.read_csv",
        change_type="behavior_change",
        description="pandas.read_csv now returns nullable dtypes by default.",
        confidence="medium",
        is_exact=True,
    )
    assessment = assess_impact(impact)

    assert assessment.risk_level == RiskLevel.MEDIUM
    assert "API behavior is documented as changed" in assessment.reasons


def test_5_package_level_behavior_change():
    """Test 5 — Package-level behavior change produces MEDIUM risk and low confidence."""
    impact = _create_dummy_impact(
        affected_api=None,
        change_type="behavior_change",
        description="Global behavior change in pandas index handling.",
        confidence="low",
        is_exact=False,
    )
    assessment = assess_impact(impact)

    assert assessment.risk_level == RiskLevel.MEDIUM
    assert assessment.confidence == "low"
    assert "Package-level change" in assessment.reasons
    assert "Package behavior is documented as changed" in assessment.reasons


def test_6_no_impact():
    """Test 6 — CompatibilityReport with zero impacts produces empty risk list."""
    report = CompatibilityReport(
        package_name="pandas",
        current_version=Version("2.0.0"),
        target_version=Version("2.1.0"),
        update_type=UpdateType.MINOR,
        assessments=[],
    )
    results = assess_compatibility(report)

    assert results == []
    assert len(results) == 0


def test_7_patch_plus_removed_api():
    """Test 7 — Patch version update with exact API removal produces HIGH risk."""
    impact = _create_dummy_impact(
        current_version="2.2.0",
        target_version="2.2.1",
        update_type=UpdateType.PATCH,
        affected_api="pandas.read_csv",
        change_type="removed",
        is_exact=True,
    )
    assessment = assess_impact(impact)

    assert assessment.risk_level == RiskLevel.HIGH
    assert impact.update_type == UpdateType.PATCH


def test_8_major_plus_no_impact():
    """Test 8 — Major version update with no matching API changes produces empty risk list."""
    report = CompatibilityReport(
        package_name="pandas",
        current_version=Version("2.0.0"),
        target_version=Version("3.0.0"),
        update_type=UpdateType.MAJOR,
        assessments=[],
    )
    results = assess_compatibility(report)

    assert results == []


def test_9_reasons_structure_and_evidence():
    """Test 9 — Verify every assessment has non-empty reasons matching evidence."""
    impact = _create_dummy_impact(
        affected_api="requests.get",
        change_type="removed",
        confidence="high",
        is_exact=True,
    )
    assessment = assess_impact(impact)

    assert len(assessment.reasons) >= 1
    assert isinstance(assessment.reasons, tuple)
    for reason in assessment.reasons:
        assert isinstance(reason, str)
        assert len(reason) > 0


def test_10_provenance_retention():
    """Test 10 — Verify full provenance remains accessible through RiskAssessment.impact."""
    url = "https://docs.pytest.org/en/7.4.x/changelog.html"
    impact = _create_dummy_impact(
        package_name="pytest",
        affected_api="pytest.main",
        file_path="tests/runner.py",
        line_number=15,
        description="pytest.main return value changed",
        source_url=url,
    )
    assessment = assess_impact(impact)

    assert assessment.impact is impact
    assert assessment.impact.source_url == url
    assert assessment.impact.file_path == Path("tests/runner.py")
    assert assessment.impact.line_number == 15
    assert assessment.impact.description == "pytest.main return value changed"
    assert assessment.impact.affected_api == "pytest.main"


def test_11_confidence_preservation():
    """Test 11 — Verify confidence is preserved across high, medium, and low ratings."""
    for conf in ("high", "medium", "low"):
        impact = _create_dummy_impact(confidence=conf)
        assessment = assess_impact(impact)
        assert assessment.confidence == conf


def test_12_multiple_assessments():
    """Test 12 — CompatibilityReport with multiple changes evaluates independent risk levels."""
    i1 = _create_dummy_impact(affected_api="pkg.func1", change_type="removed")
    i2 = _create_dummy_impact(affected_api="pkg.func2", change_type="deprecated")
    i3 = _create_dummy_impact(affected_api="pkg.func3", change_type="behavior_change")

    report = CompatibilityReport(
        package_name="pkg",
        current_version=Version("1.0.0"),
        target_version=Version("2.0.0"),
        update_type=UpdateType.MAJOR,
        assessments=[i1, i2, i3],
    )

    results = assess_compatibility(report)
    assert len(results) == 3
    assert results[0].risk_level == RiskLevel.HIGH
    assert results[1].risk_level == RiskLevel.MEDIUM
    assert results[2].risk_level == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Additional Aggregation & Edge Case Tests
# ---------------------------------------------------------------------------

def test_project_risk_report_aggregation():
    """Test project-level risk report aggregation and property counts."""
    i_high = _create_dummy_impact(affected_api="a.b", change_type="removed")
    i_med = _create_dummy_impact(affected_api="a.c", change_type="deprecated")
    i_low = _create_dummy_impact(affected_api="a.d", change_type="added")

    report = CompatibilityReport(
        package_name="a",
        current_version=Version("1.0.0"),
        target_version=Version("2.0.0"),
        update_type=UpdateType.MAJOR,
        assessments=[i_high, i_med, i_low],
    )

    proj_impact = ProjectImpactReport(
        project_path=Path("/tmp/proj"),
        entries=[
            DependencyImpactEntry(
                package_name="a",
                status=AnalysisStatus.ANALYZED,
                report=report,
            )
        ],
    )

    proj_risk = assess_project_risk(Path("/tmp/proj"), [proj_impact])

    assert proj_risk.total_assessments == 3
    assert proj_risk.high_count == 1
    assert proj_risk.medium_count == 1
    assert proj_risk.low_count == 1


def test_added_api_is_low_risk():
    """Verify added API produces LOW risk."""
    impact = _create_dummy_impact(
        affected_api="pandas.read_feather",
        change_type="added",
        is_exact=True,
    )
    assessment = assess_impact(impact)
    assert assessment.risk_level == RiskLevel.LOW
