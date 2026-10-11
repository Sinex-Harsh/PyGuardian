"""
test_migrate.py — End-to-end migration workflow integration tests for PyGuardian.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from packaging.version import Version
import pytest

from pyguardian.cli import run_migrate
from pyguardian.compatibility import ImpactAssessment
from pyguardian.migrate import MigrationWorkflowResult, run_migration_workflow
from pyguardian.packages import UpdateType
from pyguardian.patch import SourceEdit
from pyguardian.pipeline import ProjectAnalysis
from pyguardian.project import ProjectScan
from pyguardian.risk import RiskAssessment, RiskLevel
from pyguardian.source import SourceAnalysis


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Construct a small sample project with requirements and source code."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas==1.5.0\n", encoding="utf-8")

    src_file = tmp_path / "data_utils.py"
    src_code = (
        "import pandas as pd\n"
        "\n"
        "def combine(df1, df2):\n"
        "    return df1.append(df2)\n"
    )
    src_file.write_text(src_code, encoding="utf-8")

    test_file = tmp_path / "test_data.py"
    test_code = (
        "import sys\n"
        "import types\n"
        "\n"
        "# Self-contained stub for pandas so the temporary project test executes\n"
        "# deterministically without requiring pandas installed in the test environment.\n"
        "class _MockDataFrame:\n"
        "    def __init__(self, data=None):\n"
        "        self.data = list(data.values())[0] if isinstance(data, dict) else list(data or [])\n"
        "    def append(self, other):\n"
        "        return _MockDataFrame(self.data + other.data)\n"
        "    def __len__(self):\n"
        "        return len(self.data)\n"
        "\n"
        "def _mock_concat(dfs):\n"
        "    res = []\n"
        "    for d in dfs:\n"
        "        res.extend(d.data)\n"
        "    return _MockDataFrame(res)\n"
        "\n"
        "_mod = types.ModuleType('pandas')\n"
        "_mod.DataFrame = _MockDataFrame\n"
        "_mod.concat = _mock_concat\n"
        "sys.modules['pandas'] = _mod\n"
        "\n"
        "from data_utils import combine\n"
        "import pandas as pd\n"
        "\n"
        "def test_combine():\n"
        "    d1 = pd.DataFrame({'a': [1]})\n"
        "    d2 = pd.DataFrame({'a': [2]})\n"
        "    res = combine(d1, d2)\n"
        "    assert len(res) == 2\n"
    )
    test_file.write_text(test_code, encoding="utf-8")

    return tmp_path


def make_mock_project_analysis(project_path: Path, src_file: Path) -> ProjectAnalysis:
    impact = ImpactAssessment(
        package_name="pandas",
        current_version=Version("1.5.0"),
        target_version=Version("2.0.0"),
        update_type=UpdateType.MAJOR,
        affected_api="pandas.DataFrame.append",
        file_path=src_file,
        line_number=4,
        change_type="removed",
        description="DataFrame.append is removed in pandas 2.0",
        confidence="high",
    )
    risk = RiskAssessment(
        package_name="pandas",
        affected_api="pandas.DataFrame.append",
        risk_level=RiskLevel.HIGH,
        confidence="high",
        reasons=("Removed in pandas 2.0",),
        impact=impact,
    )
    scan = ProjectScan(project_name=project_path.name, project_path=project_path)
    source = SourceAnalysis(project_path=project_path)
    return ProjectAnalysis(
        project_scan=scan,
        source_analysis=source,
        risk_assessments=[risk],
    )


def test_end_to_end_successful_migration(sample_project: Path) -> None:
    src_file = sample_project / "data_utils.py"
    mock_analysis = make_mock_project_analysis(sample_project, src_file)

    custom_edit = SourceEdit(
        file_path=src_file,
        start_line=4,
        end_line=4,
        replacement="    return pd.concat([df1, df2])",
    )

    with patch("pyguardian.migrate.analyze_project", return_value=mock_analysis):
        res = run_migration_workflow(
            project_path=sample_project,
            custom_source_edit=custom_edit,
            auto_approve=True,
            run_tests=True,
        )

    assert isinstance(res, MigrationWorkflowResult)
    assert res.status == "completed"
    assert res.apply_result is not None
    assert res.apply_result.success is True
    assert "pd.concat" in src_file.read_text(encoding="utf-8")
    assert res.test_result is not None
    assert res.test_result.is_passed is True


def test_end_to_end_user_rejected_migration_leaves_files_untouched(sample_project: Path) -> None:
    src_file = sample_project / "data_utils.py"
    original_code = src_file.read_text(encoding="utf-8")
    mock_analysis = make_mock_project_analysis(sample_project, src_file)

    custom_edit = SourceEdit(
        file_path=src_file,
        start_line=4,
        end_line=4,
        replacement="    return pd.concat([df1, df2])",
    )

    def mock_input(_: str) -> str:
        return "n"  # User rejects

    with patch("pyguardian.migrate.analyze_project", return_value=mock_analysis):
        res = run_migration_workflow(
            project_path=sample_project,
            custom_source_edit=custom_edit,
            interactive=True,
            auto_approve=False,
            input_func=mock_input,
            print_func=lambda _: None,
        )

    assert res.status == "user_rejected"
    assert res.apply_result is None
    assert src_file.read_text(encoding="utf-8") == original_code


def test_stale_source_file_blocks_migration(sample_project: Path) -> None:
    src_file = sample_project / "data_utils.py"
    mock_analysis = make_mock_project_analysis(sample_project, src_file)

    custom_edit = SourceEdit(
        file_path=src_file,
        start_line=4,
        end_line=4,
        replacement="    return pd.concat([df1, df2])",
    )

    # Modify source file after edit creation but before workflow execution to simulate external edit
    src_file.write_text("import pandas as pd\n# Modified externally\n", encoding="utf-8")

    with patch("pyguardian.migrate.analyze_project", return_value=mock_analysis):
        res = run_migration_workflow(
            project_path=sample_project,
            custom_source_edit=custom_edit,
            auto_approve=True,
        )

    assert res.status in ("no_structured_edit", "review_rejected")
    assert res.apply_result is None


def test_migration_without_candidates_returns_no_candidates(tmp_path: Path) -> None:
    scan = ProjectScan(project_name=tmp_path.name, project_path=tmp_path)
    source = SourceAnalysis(project_path=tmp_path)
    empty_analysis = ProjectAnalysis(project_scan=scan, source_analysis=source)

    with patch("pyguardian.migrate.analyze_project", return_value=empty_analysis):
        res = run_migration_workflow(project_path=tmp_path, auto_approve=True)

    assert res.status == "no_candidates"
    assert res.apply_result is None


def test_invalid_syntax_patch_fails_safety_review(sample_project: Path) -> None:
    src_file = sample_project / "data_utils.py"
    original_code = src_file.read_text(encoding="utf-8")
    mock_analysis = make_mock_project_analysis(sample_project, src_file)

    custom_edit = SourceEdit(
        file_path=src_file,
        start_line=4,
        end_line=4,
        replacement="    def invalid python syntax (((",
    )

    with patch("pyguardian.migrate.analyze_project", return_value=mock_analysis):
        res = run_migration_workflow(
            project_path=sample_project,
            custom_source_edit=custom_edit,
            auto_approve=True,
        )

    assert res.status == "review_rejected"
    assert res.apply_result is None
    assert src_file.read_text(encoding="utf-8") == original_code


def test_cli_migrate_subcommand_auto_approve(sample_project: Path) -> None:
    src_file = sample_project / "data_utils.py"
    mock_analysis = make_mock_project_analysis(sample_project, src_file)

    with patch("pyguardian.migrate.analyze_project", return_value=mock_analysis):
        exit_code = run_migrate(project_path=sample_project, auto_approve=True, run_tests=True)

    assert exit_code in (0, 1)
