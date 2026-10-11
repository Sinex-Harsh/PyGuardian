"""
test_migration.py — Unit tests for PyGuardian Migration Intelligence Foundation.
"""

from __future__ import annotations

from pathlib import Path
from packaging.version import Version

from pyguardian.change_analysis import APIChangeMatch, DocumentedChange
from pyguardian.compatibility import CompatibilityReport, ImpactAssessment
from pyguardian.dependencies import Dependency
from pyguardian.migration import (
    MigrationCandidate,
    MigrationPlan,
    build_migration_plan,
)
from pyguardian.packages import PackageInfo, UpdateType
from pyguardian.pipeline import ProjectAnalysis
from pyguardian.project import DependencyResult, ProjectScan
from pyguardian.risk import RiskAssessment, RiskLevel, assess_impact
from pyguardian.source import APIUsage, SourceAnalysis


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_test_risk_assessment(
    package_name: str = "pandas",
    current_version: str = "2.0.0",
    target_version: str = "2.1.0",
    affected_api: str | None = "pandas.read_csv",
    file_path: str | None = "app/data.py",
    line_number: int | None = 42,
    change_type: str = "removed",
    description: str = "pandas.read_csv was removed.",
    confidence: str = "high",
    is_exact: bool = True,
    source_url: str = "https://pandas.pydata.org/docs/release/2.1.0.html",
) -> RiskAssessment:
    """Helper to construct a RiskAssessment for migration testing."""
    curr_v = Version(current_version) if current_version else None
    targ_v = Version(target_version) if target_version else None

    usage = APIUsage(
        package_name=package_name,
        api_path=affected_api or f"{package_name}.module",
        file_path=Path(file_path) if file_path else Path("app/main.py"),
        line_number=line_number or 1,
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
    impact = ImpactAssessment(
        package_name=package_name,
        current_version=curr_v,
        target_version=targ_v,
        update_type=UpdateType.MINOR,
        affected_api=affected_api if is_exact else None,
        file_path=Path(file_path) if file_path else None,
        line_number=line_number,
        change_type=change_type,
        description=description,
        confidence=confidence,
        evidence=match,
    )
    return assess_impact(impact)


def _create_mock_analysis(
    project_path: Path,
    pkg_names: list[str],
    risk_assessments: list[RiskAssessment],
) -> ProjectAnalysis:
    """Helper to construct a mock ProjectAnalysis."""
    req_file = project_path / "requirements.txt"
    results = [
        DependencyResult(
            dependency=Dependency(
                name=name,
                raw_line=f"{name}==1.0.0",
                specifier="==1.0.0",
                source_file=req_file,
                line_number=idx + 1,
            ),
            package_info=PackageInfo(name=name, latest_version=Version("2.0.0")),
            update_type=UpdateType.MAJOR,
        )
        for idx, name in enumerate(pkg_names)
    ]
    scan = ProjectScan(
        project_path=project_path,
        project_name="test_proj",
        requirements_files=[req_file],
        results=results,
    )
    return ProjectAnalysis(
        project_scan=scan,
        source_analysis=SourceAnalysis(project_path=project_path),
        risk_assessments=risk_assessments,
    )


# ---------------------------------------------------------------------------
# Test Suite (Tests 1-15)
# ---------------------------------------------------------------------------

def test_1_no_risk_assessments_zero_candidates(tmp_path: Path):
    """Test 1 — No risk assessments in ProjectAnalysis produces an empty MigrationPlan."""
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [])
    plan = build_migration_plan(analysis)

    assert isinstance(plan, MigrationPlan)
    assert plan.candidate_count == 0
    assert not plan.has_candidates
    assert plan.candidates == []


def test_2_exact_removed_api_high_risk_candidate(tmp_path: Path):
    """Test 2 — One exact removed API produces a HIGH risk migration candidate."""
    risk = _create_test_risk_assessment(
        package_name="pandas",
        affected_api="pandas.read_csv",
        change_type="removed",
        confidence="high",
        is_exact=True,
    )
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    assert plan.candidate_count == 1
    cand = plan.candidates[0]
    assert cand.package_name == "pandas"
    assert cand.affected_api == "pandas.read_csv"
    assert cand.risk_level == RiskLevel.HIGH
    assert cand.is_exact_match


def test_3_exact_renamed_api_preserves_change_type(tmp_path: Path):
    """Test 3 — Exact renamed API preserves 'renamed' change_type on candidate."""
    risk = _create_test_risk_assessment(
        affected_api="pandas.read_csv",
        change_type="renamed",
        description="pandas.read_csv renamed to pandas.read_file",
        is_exact=True,
    )
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.change_type == "renamed"
    assert cand.risk_level == RiskLevel.HIGH


def test_4_deprecated_api_preserves_deprecation_evidence(tmp_path: Path):
    """Test 4 — Deprecated API candidate preserves 'deprecated' change_type and MEDIUM risk."""
    risk = _create_test_risk_assessment(
        affected_api="pandas.read_csv",
        change_type="deprecated",
        description="pandas.read_csv is deprecated",
        confidence="medium",
        is_exact=True,
    )
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.change_type == "deprecated"
    assert cand.risk_level == RiskLevel.MEDIUM


def test_5_changed_api_preserves_description_and_confidence(tmp_path: Path):
    """Test 5 — Behavior change candidate preserves description and confidence level."""
    desc = "pandas.read_csv default engine changed to pyarrow."
    risk = _create_test_risk_assessment(
        affected_api="pandas.read_csv",
        change_type="behavior_change",
        description=desc,
        confidence="medium",
        is_exact=True,
    )
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.change_type == "behavior_change"
    assert cand.description == desc
    assert cand.confidence == "medium"


def test_6_package_level_change(tmp_path: Path):
    """Test 6 — Package-level change has None for API, file_path, line_number, preserving uncertainty."""
    risk = _create_test_risk_assessment(
        package_name="pandas",
        affected_api=None,
        file_path=None,
        line_number=None,
        change_type="behavior_change",
        description="Global behavior change in pandas index indexing.",
        confidence="low",
        is_exact=False,
    )
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.affected_api is None
    assert cand.file_path is None
    assert cand.line_number is None
    assert cand.is_package_level
    assert not cand.is_exact_match
    assert cand.confidence == "low"


def test_7_multiple_candidates_across_packages(tmp_path: Path):
    """Test 7 — Candidates across multiple packages are ordered by project dependency scan order."""
    r_pandas = _create_test_risk_assessment(package_name="pandas", affected_api="pandas.read_csv")
    r_requests = _create_test_risk_assessment(package_name="requests", affected_api="requests.get")
    r_numpy = _create_test_risk_assessment(package_name="numpy", affected_api="numpy.abs")

    # Dependency order: requests, pandas, numpy
    analysis = _create_mock_analysis(
        tmp_path,
        ["requests", "pandas", "numpy"],
        [r_numpy, r_pandas, r_requests],  # passed out-of-order
    )
    plan = build_migration_plan(analysis)

    pkgs = [c.package_name for c in plan.candidates]
    assert pkgs == ["requests", "pandas", "numpy"]


def test_8_multiple_candidates_same_file_ordered_by_line(tmp_path: Path):
    """Test 8 — Candidates in the same file are ordered by line number ascending."""
    r_line42 = _create_test_risk_assessment(file_path="app.py", line_number=42, affected_api="pandas.read_csv")
    r_line10 = _create_test_risk_assessment(file_path="app.py", line_number=10, affected_api="pandas.DataFrame")

    analysis = _create_mock_analysis(tmp_path, ["pandas"], [r_line42, r_line10])
    plan = build_migration_plan(analysis)

    lines = [c.line_number for c in plan.candidates]
    assert lines == [10, 42]


def test_9_current_and_target_versions_preserved(tmp_path: Path):
    """Test 9 — Verify current_version and target_version are preserved on candidate."""
    risk = _create_test_risk_assessment(current_version="1.5.3", target_version="2.1.0")
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.current_version == Version("1.5.3")
    assert cand.target_version == Version("2.1.0")


def test_10_source_url_and_provenance_preserved(tmp_path: Path):
    """Test 10 — Verify source_url and underlying evidence RiskAssessment are reachable."""
    url = "https://pandas.pydata.org/docs/release/2.1.0.html"
    risk = _create_test_risk_assessment(source_url=url)
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.source_url == url
    assert cand.evidence is risk


def test_11_risk_level_and_confidence_preserved_exactly(tmp_path: Path):
    """Test 11 — Risk level and confidence ratings match RiskAssessment exactly."""
    risk = _create_test_risk_assessment(change_type="removed", confidence="high")
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])
    plan = build_migration_plan(analysis)

    cand = plan.candidates[0]
    assert cand.risk_level == RiskLevel.HIGH
    assert cand.confidence == "high"


def test_12_no_impact_dependency_no_candidate(tmp_path: Path):
    """Test 12 — Dependency with no risk assessment does not generate a candidate."""
    analysis = _create_mock_analysis(tmp_path, ["pandas", "requests"], [])
    plan = build_migration_plan(analysis)

    assert plan.candidate_count == 0
    assert plan.packages_affected == []


def test_13_candidate_ordering_is_deterministic(tmp_path: Path):
    """Test 13 — Running build_migration_plan repeatedly produces identical candidate lists."""
    r1 = _create_test_risk_assessment(package_name="pandas", file_path="b.py", line_number=20)
    r2 = _create_test_risk_assessment(package_name="pandas", file_path="a.py", line_number=5)
    r3 = _create_test_risk_assessment(package_name="pandas", file_path="a.py", line_number=1)

    analysis = _create_mock_analysis(tmp_path, ["pandas"], [r1, r2, r3])

    plan1 = build_migration_plan(analysis)
    plan2 = build_migration_plan(analysis)

    key1 = [(c.package_name, str(c.file_path), c.line_number) for c in plan1.candidates]
    key2 = [(c.package_name, str(c.file_path), c.line_number) for c in plan2.candidates]

    assert key1 == key2
    assert key1 == [("pandas", "a.py", 1), ("pandas", "a.py", 5), ("pandas", "b.py", 20)]


def test_14_migration_plan_properties(tmp_path: Path):
    """Test 14 — Verify candidate counts, risk counts, packages_affected, and has_candidates properties."""
    r_high = _create_test_risk_assessment(package_name="pandas", change_type="removed")
    r_med = _create_test_risk_assessment(package_name="requests", change_type="deprecated")

    analysis = _create_mock_analysis(tmp_path, ["pandas", "requests"], [r_high, r_med])
    plan = build_migration_plan(analysis)

    assert plan.has_candidates
    assert plan.candidate_count == 2
    assert plan.high_risk_count == 1
    assert plan.medium_risk_count == 1
    assert plan.low_risk_count == 0
    assert plan.packages_affected == ["pandas", "requests"]


def test_15_input_project_analysis_not_mutated(tmp_path: Path):
    """Test 15 — Input ProjectAnalysis object is not mutated by build_migration_plan."""
    risk = _create_test_risk_assessment()
    analysis = _create_mock_analysis(tmp_path, ["pandas"], [risk])

    original_assessments = list(analysis.risk_assessments)
    original_results = list(analysis.project_scan.results)

    plan = build_migration_plan(analysis)

    assert analysis.risk_assessments == original_assessments
    assert analysis.project_scan.results == original_results
    assert plan.candidate_count == 1
