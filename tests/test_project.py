"""
test_project.py — Tests for the project orchestration layer.

What we test here:
- scan_project() coordinates all subsystems correctly.
- Project-level errors (bad path, no requirements file) are captured in
  ProjectScan.scan_error without raising.
- Individual dependency failures do NOT abort the scan.
- DependencyResult correctly exposes combined state from Dependency + PackageInfo.
- detect_project_name() derives a name from the directory.

Mocking strategy:
    We mock `pyguardian.project.get_package_info` — the PyPI call — so
    that tests are fast, deterministic, and network-independent.

    We do NOT mock `parse_requirements_file` or `find_requirements_files`.
    These are pure filesystem operations we control via `tmp_path`. Mocking
    them would test the mock, not our code.

    This means our tests exercise the real parsing logic AND the real
    orchestration logic, with only the network layer replaced.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from packaging.version import Version

from pyguardian.project import (
    scan_project,
    detect_project_name,
    ProjectScan,
    DependencyResult,
)
from pyguardian.packages import PackageInfo, UpdateType
from pyguardian.dependencies import Dependency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_project(tmp_path: Path, requirements_content: str) -> Path:
    """
    Create a minimal fake Python project directory with a requirements.txt.

    Returns the project root path.
    """
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    (project_dir / "requirements.txt").write_text(
        requirements_content, encoding="utf-8"
    )
    return project_dir


def make_success_info(name: str, version: str) -> PackageInfo:
    """Build a successful PackageInfo result for mocking."""
    return PackageInfo(name=name, latest_version=Version(version))


def make_failed_info(name: str, error: str = "PyPI error") -> PackageInfo:
    """Build a failed PackageInfo result for mocking."""
    return PackageInfo(name=name, fetch_error=error)


# ---------------------------------------------------------------------------
# Tests: detect_project_name
# ---------------------------------------------------------------------------

class TestDetectProjectName:

    def test_returns_directory_name(self, tmp_path):
        """Should return the final component of the path."""
        project_dir = tmp_path / "Factory Quantum Optimizer"
        project_dir.mkdir()
        assert detect_project_name(project_dir) == "Factory Quantum Optimizer"

    def test_returns_simple_name(self, tmp_path):
        """Works for plain single-word directory names."""
        project_dir = tmp_path / "myapp"
        project_dir.mkdir()
        assert detect_project_name(project_dir) == "myapp"

    def test_does_not_require_directory_to_exist(self, tmp_path):
        """
        detect_project_name is a naming utility, not a validator.
        It should work even if the path doesn't exist yet.
        """
        imaginary_path = tmp_path / "nonexistent_project"
        assert detect_project_name(imaginary_path) == "nonexistent_project"


# ---------------------------------------------------------------------------
# Tests: ProjectScan properties
# ---------------------------------------------------------------------------

class TestProjectScan:
    """
    Unit tests for the ProjectScan dataclass and its computed properties.
    These do not call scan_project() — they test the data object directly.
    """

    def _make_dep_result(
        self,
        name: str,
        specifier: str = "==1.0.0",
        update_type: UpdateType = UpdateType.UP_TO_DATE,
        fetch_error: str = None,
        parse_error: str = None,
    ) -> DependencyResult:
        dep = Dependency(
            name=name,
            raw_line=f"{name}{specifier}",
            specifier=specifier,
            parse_error=parse_error,
        )
        if parse_error:
            pkg_info = None
        elif fetch_error:
            pkg_info = PackageInfo(name=name, fetch_error=fetch_error)
        else:
            pkg_info = PackageInfo(name=name, latest_version=Version("1.0.0"))
        return DependencyResult(
            dependency=dep,
            package_info=pkg_info,
            update_type=update_type,
        )

    def test_is_successful_when_no_scan_error(self, tmp_path):
        scan = ProjectScan(
            project_path=tmp_path,
            project_name="test",
        )
        assert scan.is_successful is True

    def test_is_not_successful_when_scan_error_set(self, tmp_path):
        scan = ProjectScan(
            project_path=tmp_path,
            project_name="test",
            scan_error="something went wrong",
        )
        assert scan.is_successful is False

    def test_total_dependencies_count(self, tmp_path):
        results = [
            self._make_dep_result("numpy"),
            self._make_dep_result("pandas"),
        ]
        scan = ProjectScan(
            project_path=tmp_path, project_name="test", results=results
        )
        assert scan.total_dependencies == 2

    def test_updates_available_filters_correctly(self, tmp_path):
        results = [
            self._make_dep_result("numpy", update_type=UpdateType.MAJOR),
            self._make_dep_result("pandas", update_type=UpdateType.MINOR),
            self._make_dep_result("requests", update_type=UpdateType.UP_TO_DATE),
            self._make_dep_result("flask", update_type=UpdateType.PATCH),
        ]
        scan = ProjectScan(
            project_path=tmp_path, project_name="test", results=results
        )
        updates = scan.updates_available
        names = [r.name for r in updates]
        assert "numpy" in names
        assert "pandas" in names
        assert "flask" in names
        assert "requests" not in names

    def test_errored_filters_parse_errors(self, tmp_path):
        results = [
            self._make_dep_result("numpy"),
            self._make_dep_result("bad", parse_error="invalid line"),
        ]
        scan = ProjectScan(
            project_path=tmp_path, project_name="test", results=results
        )
        assert len(scan.errored) == 1
        assert scan.errored[0].name == "bad"

    def test_errored_filters_fetch_errors(self, tmp_path):
        results = [
            self._make_dep_result("numpy"),
            self._make_dep_result("private", fetch_error="HTTP 404"),
        ]
        scan = ProjectScan(
            project_path=tmp_path, project_name="test", results=results
        )
        assert len(scan.errored) == 1
        assert scan.errored[0].name == "private"


# ---------------------------------------------------------------------------
# Tests: DependencyResult properties
# ---------------------------------------------------------------------------

class TestDependencyResult:

    def test_name_delegates_to_dependency(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        result = DependencyResult(dependency=dep)
        assert result.name == "numpy"

    def test_current_version_from_specifier(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        result = DependencyResult(dependency=dep)
        assert result.current_version == "1.26.4"

    def test_current_version_empty_when_no_specifier(self):
        dep = Dependency(name="numpy", raw_line="numpy", specifier="")
        result = DependencyResult(dependency=dep)
        assert result.current_version == ""

    def test_latest_version_from_package_info(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        info = PackageInfo(name="numpy", latest_version=Version("2.3.3"))
        result = DependencyResult(dependency=dep, package_info=info)
        assert result.latest_version == "2.3.3"

    def test_latest_version_empty_when_no_package_info(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        result = DependencyResult(dependency=dep, package_info=None)
        assert result.latest_version == ""

    def test_latest_version_empty_when_fetch_failed(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        info = PackageInfo(name="numpy", fetch_error="timeout")
        result = DependencyResult(dependency=dep, package_info=info)
        assert result.latest_version == ""

    def test_has_error_false_when_clean(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        info = PackageInfo(name="numpy", latest_version=Version("2.3.3"))
        result = DependencyResult(
            dependency=dep, package_info=info, update_type=UpdateType.MAJOR
        )
        assert result.has_error is False

    def test_has_error_true_when_parse_error(self):
        dep = Dependency(
            name="bad", raw_line="!!!bad", specifier="", parse_error="invalid"
        )
        result = DependencyResult(dependency=dep)
        assert result.has_error is True

    def test_has_error_true_when_fetch_error(self):
        dep = Dependency(name="pkg", raw_line="pkg==1.0.0", specifier="==1.0.0")
        info = PackageInfo(name="pkg", fetch_error="HTTP 404")
        result = DependencyResult(dependency=dep, package_info=info)
        assert result.has_error is True

    def test_error_message_returns_parse_error_first(self):
        dep = Dependency(
            name="bad", raw_line="!!!bad", specifier="", parse_error="invalid syntax"
        )
        result = DependencyResult(dependency=dep)
        assert result.error_message == "invalid syntax"

    def test_error_message_returns_fetch_error(self):
        dep = Dependency(name="pkg", raw_line="pkg==1.0.0", specifier="==1.0.0")
        info = PackageInfo(name="pkg", fetch_error="HTTP 404 not found")
        result = DependencyResult(dependency=dep, package_info=info)
        assert result.error_message == "HTTP 404 not found"

    def test_error_message_none_when_no_error(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        info = PackageInfo(name="numpy", latest_version=Version("2.3.3"))
        result = DependencyResult(
            dependency=dep, package_info=info, update_type=UpdateType.MAJOR
        )
        assert result.error_message is None


# ---------------------------------------------------------------------------
# Tests: scan_project — path validation
# ---------------------------------------------------------------------------

class TestScanProjectPathValidation:

    def test_nonexistent_path_returns_scan_error(self, tmp_path):
        """A path that doesn't exist should produce a scan_error, not raise."""
        missing = tmp_path / "does_not_exist"
        scan = scan_project(missing)

        assert scan.is_successful is False
        assert scan.scan_error is not None
        assert scan.results == []

    def test_file_instead_of_directory_returns_scan_error(self, tmp_path):
        """Passing a file path instead of a directory should produce scan_error."""
        a_file = tmp_path / "somefile.txt"
        a_file.write_text("content")

        scan = scan_project(a_file)

        assert scan.is_successful is False
        assert scan.scan_error is not None
        assert "directory" in scan.scan_error.lower()
        assert scan.results == []

    def test_project_name_still_derived_on_bad_path(self, tmp_path):
        """Even when path is invalid, the project name should be readable."""
        missing = tmp_path / "MyProject"
        scan = scan_project(missing)

        assert scan.project_name == "MyProject"

    def test_project_path_preserved_in_result(self, tmp_path):
        """The original path should always be echoed back in the result."""
        missing = tmp_path / "does_not_exist"
        scan = scan_project(missing)

        assert scan.project_path == missing


# ---------------------------------------------------------------------------
# Tests: scan_project — missing requirements file
# ---------------------------------------------------------------------------

class TestScanProjectMissingRequirements:

    def test_empty_project_returns_scan_error(self, tmp_path):
        """A directory with no requirements.txt should set scan_error."""
        empty_project = tmp_path / "empty_project"
        empty_project.mkdir()

        scan = scan_project(empty_project)

        assert scan.is_successful is False
        assert scan.scan_error is not None
        assert "requirements" in scan.scan_error.lower()
        assert scan.results == []

    def test_project_name_and_path_present_even_on_missing_reqs(self, tmp_path):
        """Metadata should always be populated even when scan_error is set."""
        project_dir = tmp_path / "My App"
        project_dir.mkdir()

        scan = scan_project(project_dir)

        assert scan.project_name == "My App"
        assert scan.project_path == project_dir


# ---------------------------------------------------------------------------
# Tests: scan_project — successful scans (with mocked PyPI)
# ---------------------------------------------------------------------------

class TestScanProjectSuccess:
    """
    These tests verify that scan_project() orchestrates the full pipeline
    correctly: parsing → PyPI lookup → classification → assembly.

    PyPI calls are mocked so tests are network-independent.
    We patch 'pyguardian.project.get_package_info' — the name as used
    inside project.py — not 'pyguardian.packages.get_package_info'.
    """

    @patch("pyguardian.project.get_package_info")
    def test_single_dependency_success(self, mock_get_info, tmp_path):
        """A project with one dependency should produce one DependencyResult."""
        project_dir = make_project(tmp_path, "numpy==1.26.4\n")
        mock_get_info.return_value = make_success_info("numpy", "2.3.3")

        scan = scan_project(project_dir)

        assert scan.is_successful is True
        assert scan.scan_error is None
        assert len(scan.results) == 1

        result = scan.results[0]
        assert result.name == "numpy"
        assert result.current_version == "1.26.4"
        assert result.latest_version == "2.3.3"
        assert result.update_type == UpdateType.MAJOR
        assert result.has_error is False

    @patch("pyguardian.project.get_package_info")
    def test_multiple_dependencies(self, mock_get_info, tmp_path):
        """Multiple dependencies should all appear in results."""
        content = "numpy==1.26.4\npandas==2.2.0\nrequests==2.31.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),
            make_success_info("pandas", "2.3.2"),
            make_success_info("requests", "2.32.5"),
        ]

        scan = scan_project(project_dir)

        assert scan.is_successful is True
        assert len(scan.results) == 3

        names = [r.name for r in scan.results]
        assert "numpy" in names
        assert "pandas" in names
        assert "requests" in names

    @patch("pyguardian.project.get_package_info")
    def test_update_types_classified_correctly(self, mock_get_info, tmp_path):
        """Each dependency should receive the correct UpdateType."""
        content = "numpy==1.26.4\npandas==2.2.0\nrequests==2.31.0\nflask==3.0.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),     # MAJOR: 1.x → 2.x
            make_success_info("pandas", "2.3.2"),    # MINOR: 2.2 → 2.3
            make_success_info("requests", "2.31.5"), # PATCH: 2.31.0 → 2.31.5
            make_success_info("flask", "3.0.0"),     # UP_TO_DATE
        ]

        scan = scan_project(project_dir)
        by_name = {r.name: r.update_type for r in scan.results}

        assert by_name["numpy"] == UpdateType.MAJOR
        assert by_name["pandas"] == UpdateType.MINOR
        assert by_name["requests"] == UpdateType.PATCH
        assert by_name["flask"] == UpdateType.UP_TO_DATE

    @patch("pyguardian.project.get_package_info")
    def test_project_name_is_directory_name(self, mock_get_info, tmp_path):
        """project_name should match the directory name."""
        project_dir = tmp_path / "Factory Quantum Optimizer"
        project_dir.mkdir()
        (project_dir / "requirements.txt").write_text("numpy==1.26.4\n")
        mock_get_info.return_value = make_success_info("numpy", "2.3.3")

        scan = scan_project(project_dir)

        assert scan.project_name == "Factory Quantum Optimizer"

    @patch("pyguardian.project.get_package_info")
    def test_requirements_files_listed_in_result(self, mock_get_info, tmp_path):
        """scan.requirements_files should list the found requirements file."""
        project_dir = make_project(tmp_path, "numpy==1.26.4\n")
        mock_get_info.return_value = make_success_info("numpy", "2.3.3")

        scan = scan_project(project_dir)

        assert len(scan.requirements_files) == 1
        assert scan.requirements_files[0].name == "requirements.txt"

    @patch("pyguardian.project.get_package_info")
    def test_updates_available_computed_correctly(self, mock_get_info, tmp_path):
        """scan.updates_available should contain only packages with pending updates."""
        content = "numpy==1.26.4\nflask==3.0.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),  # MAJOR — has update
            make_success_info("flask", "3.0.0"),  # UP_TO_DATE — no update
        ]

        scan = scan_project(project_dir)

        assert len(scan.updates_available) == 1
        assert scan.updates_available[0].name == "numpy"


# ---------------------------------------------------------------------------
# Tests: scan_project — partial failures (critical behavior)
# ---------------------------------------------------------------------------

class TestScanProjectPartialFailures:
    """
    These tests verify the most important safety property:
    A single failing dependency must NEVER abort the entire scan.
    """

    @patch("pyguardian.project.get_package_info")
    def test_one_failing_pypi_does_not_abort_scan(self, mock_get_info, tmp_path):
        """
        If one PyPI lookup fails, the other dependencies should still be
        scanned and the full results returned.

        This is the most critical error-containment test.
        """
        content = "numpy==1.26.4\nprivatepkg==1.0.0\nrequests==2.31.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),
            make_failed_info("privatepkg", "Package not found on PyPI (HTTP 404)"),
            make_success_info("requests", "2.32.5"),
        ]

        scan = scan_project(project_dir)

        # Scan itself should succeed — this is not a project-level error
        assert scan.is_successful is True
        assert scan.scan_error is None

        # All three dependencies should be in the results
        assert len(scan.results) == 3

        names = {r.name for r in scan.results}
        assert "numpy" in names
        assert "privatepkg" in names
        assert "requests" in names

    @patch("pyguardian.project.get_package_info")
    def test_failed_dependency_has_error_state(self, mock_get_info, tmp_path):
        """The failed dependency result should carry the error information."""
        content = "numpy==1.26.4\nprivatepkg==1.0.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),
            make_failed_info("privatepkg", "HTTP 404 not found"),
        ]

        scan = scan_project(project_dir)

        failed = next(r for r in scan.results if r.name == "privatepkg")
        assert failed.has_error is True
        assert failed.error_message is not None
        assert "404" in failed.error_message
        assert failed.update_type == UpdateType.UNKNOWN

    @patch("pyguardian.project.get_package_info")
    def test_successful_dependencies_unaffected_by_one_failure(
        self, mock_get_info, tmp_path
    ):
        """Successful dependencies should have correct data despite one failure."""
        content = "numpy==1.26.4\nprivatepkg==1.0.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),
            make_failed_info("privatepkg", "timeout"),
        ]

        scan = scan_project(project_dir)

        numpy_result = next(r for r in scan.results if r.name == "numpy")
        assert numpy_result.has_error is False
        assert numpy_result.update_type == UpdateType.MAJOR
        assert numpy_result.latest_version == "2.3.3"

    @patch("pyguardian.project.get_package_info")
    def test_invalid_dependency_line_included_in_results(
        self, mock_get_info, tmp_path
    ):
        """
        A malformed requirements line should appear in results with has_error=True
        but should NOT cause a PyPI lookup attempt (can't look up '!!!bad').
        """
        content = "numpy==1.26.4\n!!!invalid_line\nrequests==2.31.0\n"
        project_dir = make_project(tmp_path, content)

        # Only 2 PyPI calls should be made — the invalid line is skipped
        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),
            make_success_info("requests", "2.32.5"),
        ]

        scan = scan_project(project_dir)

        assert len(scan.results) == 3  # all three lines captured

        invalid = next(r for r in scan.results if r.has_error and r.dependency.parse_error)
        assert invalid.dependency.parse_error is not None
        assert invalid.package_info is None  # no PyPI lookup attempted

        # PyPI was called exactly twice — not three times
        assert mock_get_info.call_count == 2

    @patch("pyguardian.project.get_package_info")
    def test_errored_property_collects_all_error_types(
        self, mock_get_info, tmp_path
    ):
        """scan.errored should collect both parse errors and PyPI failures."""
        content = "numpy==1.26.4\n!!!bad_line\nprivatepkg==1.0.0\n"
        project_dir = make_project(tmp_path, content)

        mock_get_info.side_effect = [
            make_success_info("numpy", "2.3.3"),
            make_failed_info("privatepkg", "HTTP 404"),
        ]

        scan = scan_project(project_dir)

        assert len(scan.errored) == 2
        errored_names = {r.name for r in scan.errored}
        # The invalid line uses the raw line text as name placeholder
        assert "privatepkg" in errored_names

    @patch("pyguardian.project.get_package_info")
    def test_comments_and_blanks_not_in_results(self, mock_get_info, tmp_path):
        """Comment lines and blank lines should produce zero DependencyResult entries."""
        content = (
            "# This is a comment\n"
            "\n"
            "numpy==1.26.4\n"
            "# another comment\n"
            "\n"
        )
        project_dir = make_project(tmp_path, content)
        mock_get_info.return_value = make_success_info("numpy", "2.3.3")

        scan = scan_project(project_dir)

        # Only numpy, not the comments or blank lines
        assert len(scan.results) == 1
        assert scan.results[0].name == "numpy"
