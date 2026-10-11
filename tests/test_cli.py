"""
test_cli.py — Tests for the CLI presentation layer.

Testing philosophy for CLI:
    The CLI's job is to:
        1. Parse arguments correctly.
        2. Call scan_project() with the right path.
        3. Render the ProjectScan output in a human-readable way.
        4. Return the correct exit code.

    We test these four concerns separately.

    We do NOT test Rich's own rendering engine — only our usage of it.

Output capture strategy:
    Rich writes to a Console. For tests, we inject a Console backed by a
    StringIO buffer. Rich detects that StringIO is not a terminal and writes
    plain text without ANSI escape codes. We can then assert on the text.

    Helper: make_test_console() returns (Console, buffer) pairs.

Mocking strategy:
    We mock 'pyguardian.cli.scan_project' — the import inside cli.py.
    Tests for run_scan() never touch the real filesystem or the network.

    We do NOT mock render functions — we test them end-to-end through
    run_scan() to catch integration issues between the render pipeline.
"""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version
from rich.console import Console

from pyguardian.cli import (
    _build_argument_parser,
    _render_update_type,
    _resolve_project_path,
    render_scan_report,
    run_scan,
)
from pyguardian.dependencies import Dependency
from pyguardian.packages import PackageInfo, UpdateType
from pyguardian.project import DependencyResult, ProjectScan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_console() -> tuple[Console, StringIO]:
    """
    Create a Rich Console backed by a StringIO buffer for output capture.

    Rich detects that StringIO is not a TTY and disables ANSI escape codes,
    so the buffer contains readable plain text.
    """
    buffer = StringIO()
    console = Console(file=buffer, highlight=False, width=120)
    return console, buffer


def make_dep_result(
    name: str,
    specifier: str = "==1.0.0",
    latest: str = "2.0.0",
    update_type: UpdateType = UpdateType.MAJOR,
    fetch_error: str = None,
    parse_error: str = None,
) -> DependencyResult:
    """Build a DependencyResult for use in tests."""
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
        pkg_info = PackageInfo(name=name, latest_version=Version(latest))
    return DependencyResult(
        dependency=dep,
        package_info=pkg_info,
        update_type=update_type,
    )


def make_successful_scan(
    tmp_path: Path,
    results: list[DependencyResult] = None,
    project_name: str = "Test Project",
) -> ProjectScan:
    """Build a successful ProjectScan for use in tests."""
    req_file = tmp_path / "requirements.txt"
    if not req_file.exists():
        req_file.write_text("numpy==1.26.4\n")
    return ProjectScan(
        project_path=tmp_path,
        project_name=project_name,
        requirements_files=[req_file],
        results=results or [],
    )


def make_failed_scan(tmp_path: Path, error: str = "No requirements file") -> ProjectScan:
    """Build a failed ProjectScan for use in tests."""
    return ProjectScan(
        project_path=tmp_path,
        project_name="Test Project",
        scan_error=error,
    )


# ---------------------------------------------------------------------------
# Tests: _resolve_project_path
# ---------------------------------------------------------------------------

class TestResolveProjectPath:

    def test_none_returns_cwd(self):
        """None input should return the current working directory."""
        result = _resolve_project_path(None)
        assert result == Path.cwd()

    def test_string_path_converted(self, tmp_path):
        """A string path should be converted to a Path object."""
        result = _resolve_project_path(str(tmp_path))
        assert result == tmp_path

    def test_returns_path_object(self):
        """Return type must always be pathlib.Path."""
        result = _resolve_project_path(None)
        assert isinstance(result, Path)

    def test_path_string_preserved(self, tmp_path):
        """The path value should be converted to a Path matching input."""
        target = tmp_path / "some" / "project"
        result = _resolve_project_path(str(target))
        assert result == target


# ---------------------------------------------------------------------------
# Tests: _render_update_type
# ---------------------------------------------------------------------------

class TestRenderUpdateType:
    """
    Tests for the update type → Rich Text conversion.

    These are pure rendering tests — no mocking needed.
    """

    def test_major_label(self):
        text = _render_update_type(UpdateType.MAJOR)
        assert text.plain == "MAJOR"

    def test_minor_label(self):
        text = _render_update_type(UpdateType.MINOR)
        assert text.plain == "MINOR"

    def test_patch_label(self):
        text = _render_update_type(UpdateType.PATCH)
        assert text.plain == "PATCH"

    def test_up_to_date_label(self):
        text = _render_update_type(UpdateType.UP_TO_DATE)
        assert "up to date" in text.plain

    def test_unknown_label(self):
        text = _render_update_type(UpdateType.UNKNOWN)
        assert text.plain == "—"

    def test_returns_rich_text(self):
        from rich.text import Text
        result = _render_update_type(UpdateType.MAJOR)
        assert isinstance(result, Text)


# ---------------------------------------------------------------------------
# Tests: _build_argument_parser
# ---------------------------------------------------------------------------

class TestArgumentParser:
    """
    Test that the argument parser is configured correctly.

    We call parse_args() directly — no sys.exit() involved here.
    """

    def test_bare_command_has_no_subcommand(self):
        """pyguardian → command is None."""
        parser = _build_argument_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_scan_subcommand_no_path(self):
        """pyguardian scan → command is 'scan', path is None."""
        parser = _build_argument_parser()
        args = parser.parse_args(["scan"])
        assert args.command == "scan"
        assert args.path is None

    def test_scan_subcommand_with_path(self, tmp_path):
        """pyguardian scan /some/path → path is set."""
        parser = _build_argument_parser()
        args = parser.parse_args(["scan", str(tmp_path)])
        assert args.command == "scan"
        assert args.path == str(tmp_path)

    def test_version_flag_exits(self):
        """--version should call sys.exit (via argparse action='version')."""
        parser = _build_argument_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Tests: render_scan_report — output content
# ---------------------------------------------------------------------------

class TestRenderScanReport:
    """
    Tests that render_scan_report() produces the expected textual content.

    We inject a Console backed by StringIO and inspect the buffer.
    """

    def test_header_always_present(self, tmp_path):
        """PyGuardian header should appear in all output."""
        scan = make_successful_scan(tmp_path)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "PyGuardian" in output

    def test_project_name_in_output(self, tmp_path):
        """The project name should appear in the output."""
        scan = make_successful_scan(tmp_path, project_name="My Cool App")
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        assert "My Cool App" in buffer.getvalue()

    def test_package_name_in_output(self, tmp_path):
        """Each package name should appear in the table output."""
        results = [make_dep_result("numpy", latest="2.3.3")]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        assert "numpy" in buffer.getvalue()

    def test_version_strings_in_output(self, tmp_path):
        """Current and latest versions should appear in the output."""
        results = [make_dep_result("numpy", specifier="==1.26.4", latest="2.3.3")]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "1.26.4" in output
        assert "2.3.3" in output

    def test_update_type_label_in_output(self, tmp_path):
        """The MAJOR update type label should appear."""
        results = [make_dep_result("numpy", update_type=UpdateType.MAJOR)]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        assert "MAJOR" in buffer.getvalue()

    def test_scan_error_displayed_on_failure(self, tmp_path):
        """When scan fails, the scan_error message should appear."""
        scan = make_failed_scan(tmp_path, error="No requirements.txt found")
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "No requirements.txt found" in output

    def test_failed_scan_shows_failure_indicator(self, tmp_path):
        """A failed scan should show a clear failure message."""
        scan = make_failed_scan(tmp_path)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        assert "Scan failed" in buffer.getvalue()

    def test_zero_dependencies_shows_message(self, tmp_path):
        """A scan with 0 results should show the empty message."""
        scan = make_successful_scan(tmp_path, results=[])
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "No dependencies found" in output

    def test_pypi_failure_shown_as_warning(self, tmp_path):
        """A dependency with a PyPI fetch error should show a warning."""
        results = [
            make_dep_result("privatepkg", fetch_error="HTTP 404 not found")
        ]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "privatepkg" in output
        assert "PyPI lookup failed" in output

    def test_parse_error_dependency_shown(self, tmp_path):
        """A dependency with a parse error should appear in the table."""
        results = [
            make_dep_result("bad", parse_error="invalid requirement string")
        ]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "Parse error" in output

    def test_summary_shows_updates_count(self, tmp_path):
        """The summary line should show the number of updates available."""
        results = [
            make_dep_result("numpy", update_type=UpdateType.MAJOR),
            make_dep_result("flask", update_type=UpdateType.UP_TO_DATE),
        ]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        assert "1 update available" in buffer.getvalue()

    def test_summary_all_up_to_date(self, tmp_path):
        """If nothing needs updating, show 'up to date' in summary."""
        results = [make_dep_result("flask", update_type=UpdateType.UP_TO_DATE)]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        assert "up to date" in buffer.getvalue()

    def test_multiple_packages_all_appear(self, tmp_path):
        """All packages in the scan should appear in the table."""
        results = [
            make_dep_result("numpy"),
            make_dep_result("pandas"),
            make_dep_result("requests"),
        ]
        scan = make_successful_scan(tmp_path, results=results)
        console, buffer = make_test_console()

        render_scan_report(scan, console)

        output = buffer.getvalue()
        assert "numpy" in output
        assert "pandas" in output
        assert "requests" in output


# ---------------------------------------------------------------------------
# Tests: run_scan — exit codes and call delegation
# ---------------------------------------------------------------------------

class TestRunScan:
    """
    Tests for run_scan() — the bridge between argparse and rendering.

    We mock pyguardian.cli.scan_project to avoid filesystem and network I/O.
    """

    @patch("pyguardian.cli.scan_project")
    def test_successful_scan_returns_exit_code_0(self, mock_scan, tmp_path):
        """A successful scan should return exit code 0."""
        mock_scan.return_value = make_successful_scan(tmp_path)
        console, _ = make_test_console()

        exit_code = run_scan(tmp_path, console)

        assert exit_code == 0

    @patch("pyguardian.cli.scan_project")
    def test_failed_scan_returns_exit_code_1(self, mock_scan, tmp_path):
        """A scan that fails at the project level should return exit code 1."""
        mock_scan.return_value = make_failed_scan(tmp_path)
        console, _ = make_test_console()

        exit_code = run_scan(tmp_path, console)

        assert exit_code == 1

    @patch("pyguardian.cli.scan_project")
    def test_scan_called_with_correct_path(self, mock_scan, tmp_path):
        """run_scan must pass the project_path to scan_project unchanged."""
        mock_scan.return_value = make_successful_scan(tmp_path)
        console, _ = make_test_console()

        run_scan(tmp_path, console)

        mock_scan.assert_called_once_with(tmp_path)

    @patch("pyguardian.cli.scan_project")
    def test_updates_available_still_exits_0(self, mock_scan, tmp_path):
        """
        Having updates available is NOT a failure.
        Exit code 0 must be returned even when packages are out of date.
        """
        results = [make_dep_result("numpy", update_type=UpdateType.MAJOR)]
        mock_scan.return_value = make_successful_scan(tmp_path, results=results)
        console, _ = make_test_console()

        exit_code = run_scan(tmp_path, console)

        assert exit_code == 0

    @patch("pyguardian.cli.scan_project")
    def test_invalid_path_exits_1(self, mock_scan, tmp_path):
        """An invalid project path should result in exit code 1."""
        missing = tmp_path / "does_not_exist"
        mock_scan.return_value = ProjectScan(
            project_path=missing,
            project_name="does_not_exist",
            scan_error="Path does not exist",
        )
        console, _ = make_test_console()

        exit_code = run_scan(missing, console)

        assert exit_code == 1

    @patch("pyguardian.cli.scan_project")
    def test_missing_requirements_exits_1(self, mock_scan, tmp_path):
        """A project with no requirements.txt should give exit code 1."""
        mock_scan.return_value = make_failed_scan(
            tmp_path, error="No requirements.txt found"
        )
        console, _ = make_test_console()

        exit_code = run_scan(tmp_path, console)

        assert exit_code == 1

    @patch("pyguardian.cli.scan_project")
    def test_one_pypi_failure_still_exits_0(self, mock_scan, tmp_path):
        """
        A single PyPI failure (e.g. private package 404) is NOT a project-
        level failure. The scan is still considered successful; exit code 0.
        """
        results = [
            make_dep_result("numpy"),
            make_dep_result("privatepkg", fetch_error="HTTP 404"),
        ]
        mock_scan.return_value = make_successful_scan(tmp_path, results=results)
        console, _ = make_test_console()

        exit_code = run_scan(tmp_path, console)

        assert exit_code == 0

    @patch("pyguardian.cli.scan_project")
    def test_zero_dependencies_exits_0(self, mock_scan, tmp_path):
        """An empty requirements.txt is not a failure — exit code 0."""
        mock_scan.return_value = make_successful_scan(tmp_path, results=[])
        console, _ = make_test_console()

        exit_code = run_scan(tmp_path, console)

        assert exit_code == 0


# ---------------------------------------------------------------------------
# Tests: current-directory scan behaviour (argument routing)
# ---------------------------------------------------------------------------

class TestCurrentDirectoryScan:
    """
    Verify that bare 'pyguardian' and 'pyguardian scan' route to CWD.
    We test through the parser + _resolve_project_path, not main(),
    to avoid sys.exit() calls.
    """

    def test_bare_command_resolves_to_cwd(self):
        """No subcommand → should target the current working directory."""
        # The bare-command logic in main() does:  project_path = Path.cwd()
        # We verify _resolve_project_path(None) == Path.cwd() as a proxy.
        assert _resolve_project_path(None) == Path.cwd()

    def test_scan_with_no_path_resolves_to_cwd(self):
        """pyguardian scan (no path) → CWD."""
        parser = _build_argument_parser()
        args = parser.parse_args(["scan"])
        resolved = _resolve_project_path(args.path)  # args.path is None
        assert resolved == Path.cwd()

    def test_scan_with_explicit_path_resolves_correctly(self, tmp_path):
        """pyguardian scan <path> → that exact path."""
        parser = _build_argument_parser()
        args = parser.parse_args(["scan", str(tmp_path)])
        resolved = _resolve_project_path(args.path)
        assert resolved == tmp_path


# ---------------------------------------------------------------------------
# Tests: real end-to-end CLI smoke test (with mocked PyPI)
# ---------------------------------------------------------------------------

class TestEndToEndSmoke:
    """
    Smoke test: exercise the full path from run_scan() down through
    render_scan_report(), using real file I/O for parsing but mocking
    PyPI so the test is network-independent.

    This is the closest to an integration test we can write without
    actually hitting PyPI.
    """

    @patch("pyguardian.project.get_package_info")
    def test_real_project_scan_produces_output(self, mock_get_info, tmp_path):
        """
        A real requirements.txt on disk + mocked PyPI should produce
        a readable report with the package name visible in output.
        """
        # Create a real project directory
        project_dir = tmp_path / "SmokeTestProject"
        project_dir.mkdir()
        (project_dir / "requirements.txt").write_text(
            "numpy==1.26.4\nrequests==2.31.0\n",
            encoding="utf-8",
        )

        mock_get_info.side_effect = [
            PackageInfo(name="numpy",    latest_version=Version("2.3.3")),
            PackageInfo(name="requests", latest_version=Version("2.32.5")),
        ]

        console, buffer = make_test_console()
        exit_code = run_scan(project_dir, console)
        output = buffer.getvalue()

        # Verify exit code
        assert exit_code == 0

        # Verify key content
        assert "SmokeTestProject" in output
        assert "numpy" in output
        assert "requests" in output
        assert "1.26.4" in output
        assert "2.31.0" in output
        assert "2.3.3" in output
        assert "2.32.5" in output
        assert "MAJOR" in output
        assert "MINOR" in output

    @patch("pyguardian.project.get_package_info")
    def test_real_missing_requirements_produces_error_output(
        self, mock_get_info, tmp_path
    ):
        """
        A real project directory with no requirements.txt should produce
        an error report and exit code 1.
        """
        project_dir = tmp_path / "EmptyProject"
        project_dir.mkdir()
        # No requirements.txt created

        console, buffer = make_test_console()
        exit_code = run_scan(project_dir, console)
        output = buffer.getvalue()

        assert exit_code == 1
        assert "Scan failed" in output
        # PyPI should not have been called
        mock_get_info.assert_not_called()
