"""
cli.py — Command-line interface for PyGuardian.

This is the presentation layer only. All business logic lives in project.py,
dependencies.py, and packages.py.

Responsibilities:
- Parse command-line arguments.
- Resolve the target project path.
- Call scan_project() from project.py.
- Render the ProjectScan result using Rich.
- Return appropriate exit codes.

What this module does NOT do:
- Parse requirement files.   (dependencies.py)
- Query PyPI.                (packages.py)
- Orchestrate the scan.      (project.py)

Layering rule:
    cli.py calls project.py.
    cli.py NEVER calls dependencies.py or packages.py directly.

Entry point (defined in pyproject.toml):
    [project.scripts]
    pyguardian = "pyguardian.cli:main"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from pyguardian import __version__
from pyguardian.migrate import run_migration_workflow
from pyguardian.packages import UpdateType
from pyguardian.project import DependencyResult, ProjectScan, scan_project


# ---------------------------------------------------------------------------
# Update type rendering
# ---------------------------------------------------------------------------

# Maps each UpdateType to a (display label, Rich style) pair.
# Centralising these means a single edit changes the visual everywhere.
_UPDATE_DISPLAY = {
    UpdateType.MAJOR:       ("MAJOR",         "bold red"),
    UpdateType.MINOR:       ("MINOR",         "bold yellow"),
    UpdateType.PATCH:       ("PATCH",         "bold green"),
    UpdateType.UP_TO_DATE:  ("✓ up to date",  "dim green"),
    UpdateType.UNKNOWN:     ("—",             "dim"),
}


def _render_update_type(update_type: UpdateType) -> Text:
    """Return a styled Rich Text object for the given UpdateType."""
    label, style = _UPDATE_DISPLAY.get(update_type, ("?", ""))
    return Text(label, style=style)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _render_header(console: Console) -> None:
    """Print the PyGuardian header banner."""
    console.print()
    console.print(
        Rule(
            f"[bold cyan]🛡  PyGuardian[/bold cyan]  "
            f"[dim]v{__version__}[/dim]"
        )
    )
    console.print()


def _render_project_info(scan: ProjectScan, console: Console) -> None:
    """Print the project metadata block."""
    console.print(f"  [bold]Project:[/bold]  {scan.project_name}")
    console.print(f"  [bold]Path:[/bold]     {scan.project_path}")
    if scan.requirements_files:
        files_str = ", ".join(f.name for f in scan.requirements_files)
        console.print(f"  [bold]File:[/bold]     {files_str}")
    console.print()


def _render_dependency_table(scan: ProjectScan, console: Console) -> None:
    """
    Render the dependency results as a Rich Table.

    Each row represents one DependencyResult. Rows with errors show a
    warning note instead of version or update information.
    """
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=False,
        expand=False,
        pad_edge=True,
    )

    table.add_column("Package",  style="bold white", no_wrap=True, min_width=15)
    table.add_column("Current",  style="white",      no_wrap=True, min_width=10)
    table.add_column("Latest",   style="white",      no_wrap=True, min_width=10)
    table.add_column("Update",   no_wrap=True,       min_width=14)
    table.add_column("Note",     style="dim",        no_wrap=False)

    for result in scan.results:
        _add_table_row(table, result)

    console.print(table)


def _add_table_row(table: Table, result: DependencyResult) -> None:
    """Add a single DependencyResult row to the table."""
    if not result.dependency.is_valid:
        # Malformed line in requirements.txt — we couldn't even parse the name.
        table.add_row(
            result.name,
            "—",
            "—",
            Text("—", style="dim"),
            Text("⚠ Parse error", style="yellow"),
        )
        return

    if result.package_info is not None and not result.package_info.is_available:
        # PyPI lookup failed — we have a name and declared version, but no
        # latest version. Still show what we know.
        table.add_row(
            result.name,
            result.current_version or "—",
            "—",
            Text("—", style="dim"),
            Text("⚠ PyPI lookup failed", style="yellow"),
        )
        return

    # Clean result.
    table.add_row(
        result.name,
        result.current_version or "—",
        result.latest_version or "—",
        _render_update_type(result.update_type),
        "",
    )


def _render_summary(scan: ProjectScan, console: Console) -> None:
    """Print the one-line summary beneath the dependency table."""
    n_updates = len(scan.updates_available)
    n_errors  = len(scan.errored)
    n_total   = scan.total_dependencies

    parts: list[str] = []

    if n_updates > 0:
        noun = "update" if n_updates == 1 else "updates"
        parts.append(f"[bold yellow]{n_updates} {noun} available[/bold yellow]")
    else:
        parts.append("[dim]All dependencies up to date[/dim]")

    if n_errors > 0:
        noun = "error" if n_errors == 1 else "errors"
        parts.append(f"[bold red]{n_errors} {noun}[/bold red]")

    dep_noun = "dependency" if n_total == 1 else "dependencies"
    parts.append(f"[dim]{n_total} {dep_noun} scanned[/dim]")

    console.print("  " + "   ·   ".join(parts))
    console.print()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_scan_report(scan: ProjectScan, console: Console) -> None:
    """
    Render the complete scan result to the console.

    Handles all states of ProjectScan:
    - Scan failed at project level (invalid path, no requirements file).
    - Scan succeeded with zero dependencies.
    - Scan succeeded with dependencies (possibly including errors/updates).

    Args:
        scan:    The ProjectScan result from scan_project().
        console: The Rich Console to write to. Injected for testability.
    """
    _render_header(console)
    _render_project_info(scan, console)

    if not scan.is_successful:
        console.print("  [bold red]✗  Scan failed[/bold red]")
        console.print()
        # scan_error is already a developer-friendly message from project.py.
        for line in scan.scan_error.splitlines():
            console.print(f"  [red]{line}[/red]")
        console.print()
        return

    if scan.total_dependencies == 0:
        console.print(
            "  [dim]No dependencies found in the requirements file.[/dim]"
        )
        console.print()
        return

    _render_dependency_table(scan, console)
    console.print()
    _render_summary(scan, console)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_project_path(path_str: Optional[str]) -> Path:
    """
    Resolve a user-supplied path string to a pathlib.Path.

    Defaults to the current working directory if path_str is None.

    Validation is intentionally left to scan_project() — this function
    simply converts a string to a Path object.
    """
    if path_str is None:
        return Path.cwd()
    return Path(path_str)


# ---------------------------------------------------------------------------
# Core run functions
# ---------------------------------------------------------------------------

def run_scan(project_path: Path, console: Console) -> int:
    """
    Execute a full project scan and render the report to the console.

    Separated from main() so that tests can call it directly with an
    injected Console backed by StringIO, without going through argparse
    or sys.exit().

    Args:
        project_path: The project directory to scan.
        console:      The Rich Console to write to.

    Returns:
        0 — scan completed successfully (updates may still be available).
        1 — scan failed at the project level (invalid path, no reqs file…).
    """
    scan = scan_project(project_path)
    render_scan_report(scan, console)
    return 0 if scan.is_successful else 1


def run_migrate(
    project_path: Path,
    auto_approve: bool = False,
    run_tests: bool = True,
    console: Optional[Console] = None,
) -> int:
    """
    Execute end-to-end migration workflow from CLI.

    Args:
        project_path: Path to target Python project directory.
        auto_approve: True to automatically approve proposed patch without prompt.
        run_tests:    True to execute project pytest after migration application.
        console:      Rich console for rendering output.

    Returns:
        0 — migration completed successfully or cleanly rejected.
        1 — safety review failed or tests failed.
    """
    c = console or Console(highlight=False)
    _render_header(c)

    res = run_migration_workflow(
        project_path=project_path,
        interactive=not auto_approve,
        auto_approve=auto_approve,
        run_tests=run_tests,
    )

    if res.status == "no_candidates":
        c.print("  [dim]No migration candidates found.[/dim]\n")
        return 0

    if res.status == "no_recommendation":
        c.print("  [bold yellow]Migration recommendation unavailable.[/bold yellow]")
        c.print("  [dim]No files changed.[/dim]\n")
        return 0

    if res.status == "no_structured_edit":
        c.print("  [bold yellow]Migration cannot be safely converted into a structured source edit.[/bold yellow]")
        c.print("  [dim]No files changed.[/dim]\n")
        return 0

    if res.status == "review_rejected":
        c.print("  [bold red]Safety review failed.[/bold red]")
        if res.message:
            c.print(f"  [dim]{res.message}[/dim]")
        c.print("  [dim]No files changed.[/dim]\n")
        return 1

    if res.status == "user_rejected":
        c.print("  [bold yellow]Migration rejected. No files changed.[/bold yellow]\n")
        return 0

    if res.status == "completed" and res.apply_result and res.apply_result.success:
        c.print(f"  [bold green]Migration applied to {res.apply_result.file_path.name}.[/bold green]")
        if res.test_result is not None:
            if res.test_result.is_passed:
                c.print("  [bold green]Tests: PASS[/bold green]\n")
                return 0
            else:
                c.print("  [bold red]Tests: FAIL[/bold red]\n")
                return 1
        c.print()
        return 0

    c.print(f"  [bold red]Migration workflow failed: {res.message}[/bold red]\n")
    return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct and return the argument parser.

    Separated from main() so tests can call parser.parse_args([...]) without
    triggering sys.exit().
    """
    parser = argparse.ArgumentParser(
        prog="pyguardian",
        description="🛡  PyGuardian — Python Dependency Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pyguardian                        Scan the current directory\n"
            "  pyguardian scan                   Scan the current directory\n"
            '  pyguardian scan "E:\\MyProject"   Scan a specific project\n'
            "  pyguardian migrate                Run interactive migration on current directory\n"
            "  pyguardian migrate --yes          Run migration with auto-approval\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"PyGuardian {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a project's dependencies for available updates.",
        description=(
            "Scan a Python project and report available dependency updates.\n"
            "Reads requirements.txt from the project directory."
        ),
    )
    scan_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="<path>",
        help=(
            "Path to the Python project to scan. "
            "Defaults to the current directory."
        ),
    )

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Run end-to-end dependency migration workflow.",
        description=(
            "Analyze project dependencies, generate a proposed patch, review diff, "
            "require approval, apply patch, and run project tests."
        ),
    )
    migrate_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="<path>",
        help="Path to Python project. Defaults to current directory.",
    )
    migrate_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Automatically approve the proposed patch without interactive prompt.",
    )
    migrate_parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip running project tests after applying patch.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point. Called by the 'pyguardian' console script.

    Parses arguments, resolves the target path, and delegates to run_scan() or run_migrate().
    Ensures that no raw Python traceback reaches the user during normal usage.
    """
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

    parser = _build_argument_parser()
    args = parser.parse_args()

    console = Console(highlight=False)

    try:
        if args.command is None:
            # Bare `pyguardian` → scan current directory.
            project_path = Path.cwd()
            exit_code = run_scan(project_path, console)
            sys.exit(exit_code)

        elif args.command == "scan":
            project_path = _resolve_project_path(getattr(args, "path", None))
            exit_code = run_scan(project_path, console)
            sys.exit(exit_code)

        elif args.command == "migrate":
            project_path = _resolve_project_path(getattr(args, "path", None))
            exit_code = run_migrate(
                project_path=project_path,
                auto_approve=args.yes,
                run_tests=not args.no_test,
                console=console,
            )
            sys.exit(exit_code)

        else:
            parser.print_help()
            sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)

    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[bold red]Unexpected error:[/bold red] {exc}")
        console.print("[dim]Please report this as a bug.[/dim]")
        sys.exit(1)
