"""
test_pipeline.py — Unit tests for PyGuardian Unified Analysis Pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

from pyguardian.change_analysis import DocumentedChange
from pyguardian.change_sources import ChangeDocument, ChangeSource
from pyguardian.compatibility import CompatibilityReport, ImpactAssessment
from pyguardian.dependencies import Dependency
from pyguardian.packages import PackageInfo, UpdateType
from pyguardian.pipeline import ProjectAnalysis, analyze_project
from pyguardian.project import DependencyResult, ProjectScan
from pyguardian.project_impact import AnalysisStatus
from pyguardian.risk import RiskLevel
from pyguardian.source import APIUsage, ImportUsage, SourceAnalysis


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def _mock_pypi_dict(package_name: str, latest_version: str = "2.1.0", doc_url: str = "") -> dict:
    urls = {}
    if doc_url:
        urls["Changelog"] = doc_url
    else:
        urls["Changelog"] = f"https://docs.{package_name}.org/changelog.html"

    return {
        "info": {
            "name": package_name,
            "version": latest_version,
            "project_urls": urls,
        },
        "releases": {
            latest_version: [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}],
        },
    }


def _make_mock_requests(doc_text: str = "", doc_url: str = ""):
    """Helper to mock requests.get for both PyPI API and documentation requests."""
    def side_effect(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        if "pypi.org" in url:
            parts = url.split("/")
            pkg_name = parts[-2] if len(parts) >= 2 else "pandas"
            mock_resp.json.return_value = _mock_pypi_dict(pkg_name, "2.1.0", doc_url)
            mock_resp.text = '{"info": {}}'
        else:
            mock_resp.json.return_value = {}
            mock_resp.text = doc_text

        return mock_resp

    return side_effect


# ---------------------------------------------------------------------------
# Pipeline Tests (1-10)
# ---------------------------------------------------------------------------

def test_1_empty_project(tmp_path: Path):
    """Test 1 — Empty project (no requirements file) returns valid ProjectAnalysis with scan error handled."""
    analysis = analyze_project(tmp_path)

    assert isinstance(analysis, ProjectAnalysis)
    assert analysis.project_path == tmp_path
    assert analysis.total_dependencies == 0
    assert not analysis.has_impacts
    assert len(analysis.compatibility_reports) == 0
    assert len(analysis.risk_assessments) == 0
    assert not analysis.project_scan.is_successful


def test_2_single_dependency_pipeline(tmp_path: Path):
    """Test 2 — Single dependency flows through full pipeline from scan to risk assessment."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\n", encoding="utf-8")

    src_file = tmp_path / "app.py"
    src_file.write_text("import pandas as pd\npd.read_csv('data.csv')\n", encoding="utf-8")

    doc_text = "Version 2.1.0: `pandas.read_csv` was removed."
    doc_url = "https://docs.pandas.org/changelog.html"

    with patch("requests.get", side_effect=_make_mock_requests(doc_text, doc_url)):
        analysis = analyze_project(tmp_path)

        assert analysis.total_dependencies == 1
        assert len(analysis.compatibility_reports) == 1
        assert len(analysis.risk_assessments) == 1

        risk = analysis.risk_assessments[0]
        assert risk.package_name == "pandas"
        assert risk.affected_api == "pandas.read_csv"
        assert risk.risk_level == RiskLevel.HIGH


def test_3_multiple_dependencies_ordering(tmp_path: Path):
    """Test 3 — Multiple dependencies remain in exact input order."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\nrequests==2.28.0\nnumpy==1.24.0\n", encoding="utf-8")

    with patch("requests.get", side_effect=_make_mock_requests()):
        analysis = analyze_project(tmp_path)

        comp_names = [r.package_name for r in analysis.compatibility_reports]
        impact_names = [e.package_name for e in analysis.project_impact.entries]

        assert comp_names == ["pandas", "requests", "numpy"]
        assert impact_names == ["pandas", "requests", "numpy"]


def test_4_dependency_with_no_impact(tmp_path: Path):
    """Test 4 — Dependency with no breaking change yields has_impacts=False and no fake risk."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("requests==2.28.0\n", encoding="utf-8")

    src_file = tmp_path / "app.py"
    src_file.write_text("import requests\nrequests.get('https://example.com')\n", encoding="utf-8")

    doc_text = "Documentation updated for requests in version 2.29.0."

    with patch("requests.get", side_effect=_make_mock_requests(doc_text)):
        analysis = analyze_project(tmp_path)

        assert len(analysis.compatibility_reports) == 1
        assert not analysis.compatibility_reports[0].has_impacts
        assert len(analysis.risk_assessments) == 0


def test_5_dependency_with_impact_reachability(tmp_path: Path):
    """Test 5 — Verify both ImpactAssessment and RiskAssessment are reachable from ProjectAnalysis."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\n", encoding="utf-8")

    src_file = tmp_path / "app.py"
    src_file.write_text("import pandas as pd\npd.read_csv('data.csv')\n", encoding="utf-8")

    doc_text = "Version 2.1.0: `pandas.read_csv` was removed."
    doc_url = "https://pandas.pydata.org/docs/whatsnew/v2.1.0.html"

    with patch("requests.get", side_effect=_make_mock_requests(doc_text, doc_url)):
        analysis = analyze_project(tmp_path)

        comp_report = analysis.compatibility_reports[0]
        assert comp_report.has_impacts
        impact = comp_report.assessments[0]
        risk = analysis.risk_assessments[0]

        assert risk.impact is impact
        assert impact.affected_api == "pandas.read_csv"


def test_6_one_dependency_fails(tmp_path: Path):
    """Test 6 — One failing dependency preserves errors without failing the entire pipeline."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\nbroken-pkg==1.0.0\nrequests==2.28.0\n", encoding="utf-8")

    def side_effect(url, **kwargs):
        mock_resp = MagicMock()
        if "broken-pkg" in url:
            mock_resp.status_code = 404
            import requests as req_module
            mock_resp.raise_for_status.side_effect = req_module.exceptions.HTTPError(response=mock_resp)
            return mock_resp

        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _mock_pypi_dict("pkg", "2.1.0")
        mock_resp.text = "Documentation text"
        return mock_resp

    with patch("requests.get", side_effect=side_effect):
        analysis = analyze_project(tmp_path)

        assert analysis.total_dependencies == 3
        entries = analysis.project_impact.entries
        assert len(entries) == 3

        assert entries[0].status == AnalysisStatus.ANALYZED
        assert entries[1].status == AnalysisStatus.ERROR
        assert "not found" in (entries[1].error_message or "").lower() or "http 404" in (entries[1].error_message or "").lower()
        assert entries[2].status == AnalysisStatus.ANALYZED


def test_7_source_parse_error(tmp_path: Path):
    """Test 7 — Source file syntax error is recorded while valid files are processed."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\n", encoding="utf-8")

    valid_file = tmp_path / "good.py"
    valid_file.write_text("import pandas as pd\npd.read_csv('x')\n", encoding="utf-8")

    broken_file = tmp_path / "bad.py"
    broken_file.write_text("def broken_syntax(\n", encoding="utf-8")

    with patch("requests.get", side_effect=_make_mock_requests()):
        analysis = analyze_project(tmp_path)

        assert len(analysis.source_analysis.parse_errors) == 1
        assert analysis.source_analysis.parse_errors[0].file_path == broken_file
        assert len(analysis.source_analysis.api_usages) >= 1


def test_8_no_documented_changes(tmp_path: Path):
    """Test 8 — Dependency with no discovered change sources produces empty assessment list."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("numpy==1.24.0\n", encoding="utf-8")

    def side_effect(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "info": {"name": "numpy", "version": "1.25.0"},
            "releases": {"1.25.0": []},
        }
        mock_resp.text = ""
        return mock_resp

    with patch("requests.get", side_effect=side_effect):
        analysis = analyze_project(tmp_path)

        assert len(analysis.compatibility_reports) == 1
        assert not analysis.compatibility_reports[0].has_impacts
        assert len(analysis.risk_assessments) == 0


def test_9_exact_api_impact_provenance(tmp_path: Path):
    """Test 9 — Exact API impact carries full provenance (URL, file, line, risk, confidence)."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\n", encoding="utf-8")

    src_file = tmp_path / "main.py"
    src_file.write_text("import pandas as pd\n\npd.read_csv('test.csv')\n", encoding="utf-8")

    doc_text = "Version 2.1.0:\n`pandas.read_csv` was removed."
    doc_url = "https://pandas.pydata.org/docs/whatsnew/v2.1.0.html"

    with patch("requests.get", side_effect=_make_mock_requests(doc_text, doc_url)):
        analysis = analyze_project(tmp_path)

        assert len(analysis.risk_assessments) == 1
        risk = analysis.risk_assessments[0]

        assert risk.package_name == "pandas"
        assert risk.affected_api == "pandas.read_csv"
        assert risk.risk_level == RiskLevel.HIGH
        assert risk.confidence == "high"
        assert risk.impact.file_path == src_file
        assert risk.impact.line_number == 3
        assert risk.impact.source_url == doc_url


def test_10_no_duplicate_responsibility(tmp_path: Path):
    """Test 10 — Verify pipeline calls public module entry points rather than duplicating logic."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==2.0.0\n", encoding="utf-8")

    dep_res = DependencyResult(
        dependency=Dependency(
            name="pandas",
            raw_line="pandas==2.0.0",
            specifier="==2.0.0",
            source_file=req_file,
            line_number=1,
        ),
        package_info=PackageInfo(name="pandas", latest_version=None),
        update_type=UpdateType.MINOR,
    )

    with patch("pyguardian.pipeline.scan_project") as mock_scan, \
         patch("pyguardian.pipeline.analyze_source_tree") as mock_source, \
         patch("pyguardian.pipeline.map_dependencies_to_usage") as mock_map, \
         patch("pyguardian.pipeline.build_project_impact_report") as mock_impact, \
         patch("pyguardian.pipeline.assess_project_risk") as mock_risk:

        mock_scan.return_value = ProjectScan(
            project_path=tmp_path,
            project_name="test",
            requirements_files=[req_file],
            results=[dep_res],
        )
        mock_source.return_value = SourceAnalysis(project_path=tmp_path)
        mock_map.return_value = []
        mock_impact.return_value = MagicMock()
        mock_risk.return_value = MagicMock(assessments=[])

        analyze_project(tmp_path)

        mock_scan.assert_called_once_with(tmp_path)
        mock_source.assert_called_once_with(tmp_path, ignored_dirs=None)
        mock_map.assert_called_once()
        mock_impact.assert_called_once()
        mock_risk.assert_called_once()
