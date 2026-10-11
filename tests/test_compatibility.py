"""
test_compatibility.py — Unit tests for Compatibility & Impact Analysis (V0.2 Step 7).

Key testing principle:
    All tests are fully deterministic with zero live network or file system dependencies.
"""

from pathlib import Path
import pytest
from packaging.version import Version

from pyguardian.change_analysis import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_BEHAVIOR_CHANGE,
    CHANGE_TYPE_CHANGED,
    CHANGE_TYPE_DEPRECATED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    DocumentedChange,
)
from pyguardian.compatibility import (
    CompatibilityReport,
    ImpactAssessment,
    analyze_compatibility,
)
from pyguardian.packages import UpdateType
from pyguardian.source import APIUsage


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestAnalyzeCompatibility:

    def test_exact_api_impact(self):
        usage = APIUsage("pandas", "pandas.read_csv", Path("app/data_loader.py"), 42, "call")
        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.read_csv",
            change_type=CHANGE_TYPE_DEPRECATED,
            description="read_csv is deprecated.",
            source_url="https://example.org/docs",
            confidence="high",
        )

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.2.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[change],
        )

        assert report.has_impacts is True
        assert report.impact_count == 1
        assert report.high_confidence_count == 1
        assert report.update_type == UpdateType.MAJOR

        assessment = report.assessments[0]
        assert assessment.package_name == "pandas"
        assert assessment.current_version == Version("2.2.0")
        assert assessment.target_version == Version("3.0.0")
        assert assessment.affected_api == "pandas.read_csv"
        assert assessment.file_path == Path("app/data_loader.py")
        assert assessment.line_number == 42
        assert assessment.change_type == CHANGE_TYPE_DEPRECATED
        assert assessment.confidence == "high"
        assert assessment.is_exact_match is True
        assert assessment.source_url == "https://example.org/docs"

    def test_package_level_impact(self):
        usage = APIUsage("pandas", "pandas.DataFrame", Path("analysis.py"), 20, "call")
        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path=None,
            change_type=CHANGE_TYPE_BEHAVIOR_CHANGE,
            description="IO behavior changed significantly.",
            source_url="https://example.org/docs",
            confidence="low",
        )

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.2.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[change],
        )

        assert report.has_impacts is True
        assert report.impact_count == 1
        assessment = report.assessments[0]
        assert assessment.affected_api is None
        assert assessment.confidence == "low"
        assert assessment.is_exact_match is False

    def test_no_impact(self):
        usage = APIUsage("pandas", "pandas.concat", Path("app.py"), 10, "call")
        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.read_csv",
            change_type=CHANGE_TYPE_DEPRECATED,
            description="read_csv is deprecated.",
            source_url="https://example.org/docs",
        )

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.2.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[change],
        )

        assert report.has_impacts is False
        assert report.impact_count == 0
        assert report.assessments == []

    def test_multiple_locations_preserved(self):
        u1 = APIUsage("pandas", "pandas.DataFrame", Path("a.py"), 10, "call")
        u2 = APIUsage("pandas", "pandas.DataFrame", Path("b.py"), 20, "call")
        u3 = APIUsage("pandas", "pandas.DataFrame", Path("c.py"), 30, "call")

        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.DataFrame",
            change_type=CHANGE_TYPE_REMOVED,
            description="DataFrame was removed.",
            source_url="https://example.org/docs",
        )

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.2.0",
            target_version="3.0.0",
            api_usages=[u1, u2, u3],
            documented_changes=[change],
        )

        assert report.impact_count == 3
        paths = [a.file_path for a in report.assessments]
        assert paths == [Path("a.py"), Path("b.py"), Path("c.py")]
        lines = [a.line_number for a in report.assessments]
        assert lines == [10, 20, 30]

    def test_multiple_documented_changes_neither_overwritten(self):
        usage = APIUsage("pandas", "pandas.read_csv", Path("app.py"), 15, "call")
        c1 = DocumentedChange(
            package_name="pandas",
            version=Version("2.5.0"),
            api_path="pandas.read_csv",
            change_type=CHANGE_TYPE_DEPRECATED,
            description="read_csv is deprecated in 2.5.",
            source_url="https://example.org/v2.5",
        )
        c2 = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.read_csv",
            change_type=CHANGE_TYPE_REMOVED,
            description="read_csv removed in 3.0.",
            source_url="https://example.org/v3.0",
        )

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.0.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[c1, c2],
        )

        assert report.impact_count == 2
        change_types = [a.change_type for a in report.assessments]
        assert change_types == [CHANGE_TYPE_DEPRECATED, CHANGE_TYPE_REMOVED]

    def test_version_filtering_window(self):
        usage = APIUsage("pandas", "pandas.read_csv", Path("app.py"), 10, "call")

        c_past = DocumentedChange("pandas", Version("1.9.0"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "old", "http://url1")
        c_in_range1 = DocumentedChange("pandas", Version("2.5.0"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "mid1", "http://url2")
        c_in_range2 = DocumentedChange("pandas", Version("3.0.0"), "pandas.read_csv", CHANGE_TYPE_REMOVED, "mid2", "http://url3")
        c_future = DocumentedChange("pandas", Version("3.1.0"), "pandas.read_csv", CHANGE_TYPE_RENAMED, "future", "http://url4")

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.0.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[c_past, c_in_range1, c_in_range2, c_future],
        )

        assert report.impact_count == 2
        descs = [a.description for a in report.assessments]
        assert descs == ["mid1", "mid2"]

    def test_unknown_version_handled_conservatively(self):
        usage = APIUsage("pandas", "pandas.read_csv", Path("app.py"), 10, "call")
        c_unknown = DocumentedChange("pandas", None, "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "unknown ver", "http://url")

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.0.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[c_unknown],
        )

        assert report.impact_count == 1
        assert report.assessments[0].description == "unknown ver"

    def test_major_update_without_api_evidence_yields_no_impacts(self):
        usage = APIUsage("pandas", "pandas.concat", Path("app.py"), 10, "call")
        change = DocumentedChange("pandas", Version("3.0.0"), "pandas.read_csv", CHANGE_TYPE_REMOVED, "read_csv removed", "http://url")

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.0.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[change],
        )

        assert report.update_type == UpdateType.MAJOR
        assert report.has_impacts is False
        assert report.impact_count == 0

    def test_patch_update_with_documented_api_change(self):
        usage = APIUsage("pandas", "pandas.read_csv", Path("app.py"), 10, "call")
        change = DocumentedChange("pandas", Version("2.2.1"), "pandas.read_csv", CHANGE_TYPE_DEPRECATED, "read_csv deprecated in patch", "http://url")

        report = analyze_compatibility(
            package_name="pandas",
            current_version="2.2.0",
            target_version="2.2.1",
            api_usages=[usage],
            documented_changes=[change],
        )

        assert report.update_type == UpdateType.PATCH
        assert report.has_impacts is True
        assert report.impact_count == 1

    def test_provenance_preservation(self):
        usage = APIUsage("requests", "requests.get", Path("net.py"), 5, "call")
        change = DocumentedChange("requests", Version("3.0.0"), "requests.get", CHANGE_TYPE_CHANGED, "requests.get params changed", "https://requests.readthedocs.io")

        report = analyze_compatibility(
            package_name="requests",
            current_version="2.31.0",
            target_version="3.0.0",
            api_usages=[usage],
            documented_changes=[change],
        )

        assessment = report.assessments[0]
        assert assessment.source_url == "https://requests.readthedocs.io"
        assert assessment.description == "requests.get params changed"
        assert assessment.file_path == Path("net.py")
        assert assessment.line_number == 5
