"""
test_project_impact.py — Unit tests for Project-Level Impact Report (V0.2 Step 8).

Key testing principle:
    All tests are fully deterministic with zero live network or file system I/O.
"""

from pathlib import Path
import pytest
from packaging.version import Version

from pyguardian.change_analysis import (
    CHANGE_TYPE_BEHAVIOR_CHANGE,
    CHANGE_TYPE_DEPRECATED,
    CHANGE_TYPE_REMOVED,
    DocumentedChange,
)
from pyguardian.compatibility import (
    CompatibilityReport,
    ImpactAssessment,
    analyze_compatibility,
)
from pyguardian.packages import UpdateType
from pyguardian.project_impact import (
    AnalysisStatus,
    DependencyImpactEntry,
    ProjectImpactReport,
    build_project_impact_report,
    create_error_entry,
)
from pyguardian.source import APIUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(pkg: str, assessments: list[ImpactAssessment]) -> CompatibilityReport:
    return CompatibilityReport(
        package_name=pkg,
        current_version=Version("1.0.0"),
        target_version=Version("2.0.0"),
        update_type=UpdateType.MAJOR,
        assessments=assessments,
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestProjectImpactReport:

    def test_multiple_dependencies_aggregation(self, tmp_path):
        u1 = APIUsage("pandas", "pandas.read_csv", Path("a.py"), 10, "call")
        u2 = APIUsage("pandas", "pandas.DataFrame", Path("b.py"), 20, "call")
        c1 = DocumentedChange("pandas", Version("2.0.0"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "read_csv deprecated", "http://url1", "high")
        c2 = DocumentedChange("pandas", Version("2.0.0"), "pandas.DataFrame", CHANGE_TYPE_REMOVED, "DataFrame removed", "http://url2", "high")
        r_pandas = analyze_compatibility("pandas", "1.0.0", "2.0.0", [u1, u2], [c1, c2])

        r_requests = analyze_compatibility("requests", "2.0.0", "3.0.0", [], [])

        u3 = APIUsage("numpy", "numpy.array", Path("c.py"), 30, "call")
        c3 = DocumentedChange("numpy", Version("2.0.0"), "numpy.array", CHANGE_TYPE_BEHAVIOR_CHANGE, "array behavior changed", "http://url3", "medium")
        r_numpy = analyze_compatibility("numpy", "1.0.0", "2.0.0", [u3], [c3])

        proj_report = build_project_impact_report(tmp_path, [r_pandas, r_requests, r_numpy])

        assert proj_report.total_dependencies == 3
        assert proj_report.dependencies_with_impacts == 2
        assert proj_report.total_impacts == 3
        assert proj_report.dependencies_with_errors == 0

    def test_zero_impact_dependency_preserved(self, tmp_path):
        r_requests = analyze_compatibility("requests", "2.0.0", "3.0.0", [], [])

        proj_report = build_project_impact_report(tmp_path, [r_requests])

        assert proj_report.total_dependencies == 1
        assert proj_report.dependencies_with_impacts == 0
        assert proj_report.total_impacts == 0

        entry = proj_report.entries[0]
        assert entry.package_name == "requests"
        assert entry.status == AnalysisStatus.ANALYZED
        assert entry.has_impacts is False
        assert entry.report is not None

    def test_confidence_aggregation(self, tmp_path):
        u1 = APIUsage("pkg", "pkg.a", Path("x.py"), 1, "call")
        u2 = APIUsage("pkg", "pkg.b", Path("x.py"), 2, "call")
        u3 = APIUsage("pkg", "pkg.c", Path("x.py"), 3, "call")

        c1 = DocumentedChange("pkg", Version("2.0.0"), "pkg.a", CHANGE_TYPE_REMOVED, "a removed", "http://url1", "high")
        c2 = DocumentedChange("pkg", Version("2.0.0"), "pkg.b", CHANGE_TYPE_DEPRECATED, "b deprecated", "http://url2", "high")
        c3 = DocumentedChange("pkg", Version("2.0.0"), "pkg.c", CHANGE_TYPE_BEHAVIOR_CHANGE, "c changed", "http://url3", "medium")

        r_pkg = analyze_compatibility("pkg", "1.0.0", "2.0.0", [u1, u2, u3], [c1, c2, c3])

        u_low = APIUsage("pkg_low", "pkg_low.x", Path("y.py"), 1, "call")
        c_low = DocumentedChange("pkg_low", Version("2.0.0"), None, CHANGE_TYPE_BEHAVIOR_CHANGE, "pkg changed", "http://url4", "low")
        r_low = analyze_compatibility("pkg_low", "1.0.0", "2.0.0", [u_low], [c_low])

        proj_report = build_project_impact_report(tmp_path, [r_pkg, r_low])

        assert proj_report.total_impacts == 4
        assert proj_report.high_confidence_impacts == 2
        assert proj_report.medium_confidence_impacts == 1
        assert proj_report.low_confidence_impacts == 1

    def test_evidence_preservation_through_project_report(self, tmp_path):
        u = APIUsage("pandas", "pandas.read_csv", Path("app/data.py"), 42, "call")
        c = DocumentedChange("pandas", Version("3.0.0"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "read_csv deprecated", "https://pandas.pydata.org/docs", "high")

        r = analyze_compatibility("pandas", "2.2.0", "3.0.0", [u], [c])
        proj_report = build_project_impact_report(tmp_path, [r])

        assessment = proj_report.entries[0].report.assessments[0]
        assert assessment.source_url == "https://pandas.pydata.org/docs"
        assert assessment.description == "read_csv deprecated"
        assert assessment.affected_api == "pandas.read_csv"
        assert assessment.file_path == Path("app/data.py")
        assert assessment.line_number == 42

    def test_dependency_ordering_preserved(self, tmp_path):
        r1 = analyze_compatibility("pandas", "1.0.0", "2.0.0", [], [])
        r2 = analyze_compatibility("requests", "1.0.0", "2.0.0", [], [])
        r3 = analyze_compatibility("numpy", "1.0.0", "2.0.0", [], [])

        proj_report = build_project_impact_report(tmp_path, [r1, r2, r3])

        names = [e.package_name for e in proj_report.entries]
        assert names == ["pandas", "requests", "numpy"]

    def test_no_false_safe_attribute_exposed(self, tmp_path):
        r = analyze_compatibility("requests", "2.0.0", "3.0.0", [], [])
        proj_report = build_project_impact_report(tmp_path, [r])

        entry = proj_report.entries[0]
        assert not hasattr(entry, "is_safe")
        assert not hasattr(entry, "risk_free")
        assert not hasattr(entry, "guaranteed_compatible")

    def test_analysis_error_handling(self, tmp_path):
        r_good = analyze_compatibility("pandas", "1.0.0", "2.0.0", [], [])
        err_entry = create_error_entry("broken-pkg", "Documentation lookup failed (HTTP 404)")

        proj_report = build_project_impact_report(tmp_path, [r_good, err_entry])

        assert proj_report.total_dependencies == 2
        assert proj_report.dependencies_with_errors == 1
        assert proj_report.dependencies_with_impacts == 0
        assert proj_report.total_impacts == 0

        err_item = proj_report.entries[1]
        assert err_item.package_name == "broken-pkg"
        assert err_item.status == AnalysisStatus.ERROR
        assert err_item.error_message == "Documentation lookup failed (HTTP 404)"
        assert err_item.report is None

    def test_all_dependencies_impacted(self, tmp_path):
        u1 = APIUsage("pandas", "pandas.read_csv", Path("a.py"), 1, "call")
        c1 = DocumentedChange("pandas", Version("2.0.0"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "d", "http://u1")
        r1 = analyze_compatibility("pandas", "1.0.0", "2.0.0", [u1], [c1])

        u2 = APIUsage("numpy", "numpy.array", Path("b.py"), 1, "call")
        c2 = DocumentedChange("numpy", Version("2.0.0"), "numpy.array", CHANGE_TYPE_REMOVED, "r", "http://u2")
        r2 = analyze_compatibility("numpy", "1.0.0", "2.0.0", [u2], [c2])

        proj_report = build_project_impact_report(tmp_path, [r1, r2])

        assert proj_report.total_dependencies == 2
        assert proj_report.dependencies_with_impacts == 2
        assert proj_report.total_impacts == 2

    def test_empty_project(self, tmp_path):
        proj_report = build_project_impact_report(tmp_path, [])

        assert proj_report.total_dependencies == 0
        assert proj_report.dependencies_with_impacts == 0
        assert proj_report.total_impacts == 0
        assert proj_report.high_confidence_impacts == 0
        assert proj_report.entries == []

    def test_mixed_confidence_and_zero_impacts(self, tmp_path):
        # pandas -> HIGH
        u_p = APIUsage("pandas", "pandas.read_csv", Path("a.py"), 1, "call")
        c_p = DocumentedChange("pandas", Version("2.0.0"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "dep", "http://u1", "high")
        r_pandas = analyze_compatibility("pandas", "1.0.0", "2.0.0", [u_p], [c_p])

        # numpy -> MEDIUM
        u_n = APIUsage("numpy", "numpy.array", Path("b.py"), 1, "call")
        c_n = DocumentedChange("numpy", Version("2.0.0"), "numpy.array", CHANGE_TYPE_BEHAVIOR_CHANGE, "changed", "http://u2", "medium")
        r_numpy = analyze_compatibility("numpy", "1.0.0", "2.0.0", [u_n], [c_n])

        # requests -> LOW (package-level)
        u_r = APIUsage("requests", "requests.get", Path("c.py"), 1, "call")
        c_r = DocumentedChange("requests", Version("3.0.0"), None, CHANGE_TYPE_BEHAVIOR_CHANGE, "pkg changed", "http://u3", "low")
        r_requests = analyze_compatibility("requests", "2.0.0", "3.0.0", [u_r], [c_r])

        # flask -> 0 impacts
        r_flask = analyze_compatibility("flask", "1.0.0", "2.0.0", [], [])

        proj_report = build_project_impact_report(tmp_path, [r_pandas, r_numpy, r_requests, r_flask])

        assert proj_report.total_dependencies == 4
        assert proj_report.dependencies_with_impacts == 3
        assert proj_report.total_impacts == 3
        assert proj_report.high_confidence_impacts == 1
        assert proj_report.medium_confidence_impacts == 1
        assert proj_report.low_confidence_impacts == 1
