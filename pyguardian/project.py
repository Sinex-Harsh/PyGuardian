"""
project.py — Project orchestration layer.

Responsibilities:
- Accept a filesystem path and build a complete picture of the project's
  dependency health.
- Coordinate dependency parsing, PyPI lookup, and update classification.
- Return a single structured result (ProjectScan) that higher layers
  (cli.py) can consume without knowing any implementation details.

What this module does NOT do:
- Display anything to the terminal. (That is cli.py's job.)
- Parse requirement strings. (That is dependencies.py's job.)
- Make HTTP requests. (That is packages.py's job.)
- Analyse source code. (That is a future analyzer.py's job.)

Design note — layering:
    cli.py
       ↓  calls
    project.py
       ↓  delegates to
    dependencies.py   packages.py

    cli.py must NEVER import directly from dependencies.py or packages.py.
    All business logic is mediated through project.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pyguardian.dependencies import (
    Dependency,
    find_requirements_files,
    parse_requirements_file,
)
from pyguardian.packages import (
    PackageInfo,
    UpdateType,
    classify_update,
    get_package_info,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class DependencyResult:
    """
    The combined result for a single dependency after a full scan.

    This is the object that was identified as 'missing' in the architecture
    review. It ties together all three pieces of information that the CLI
    needs to display one row in the report table:

    Attributes:
        dependency:   The parsed dependency entry (name, specifier, source
                      file, line number). Always present.
        package_info: PyPI metadata for this package. None only if the
                      dependency itself had a parse error (in which case we
                      can't even look it up). Otherwise always present, though
                      it may carry a fetch_error if the lookup failed.
        update_type:  The semantic classification of the update. UNKNOWN when
                      we don't have enough information to classify.
    """

    dependency: Dependency
    package_info: Optional[PackageInfo] = field(default=None)
    update_type: UpdateType = field(default=UpdateType.UNKNOWN)

    @property
    def name(self) -> str:
        """Convenience accessor for the package name."""
        return self.dependency.name

    @property
    def current_version(self) -> str:
        """
        The version string declared in the requirements file.

        Uses Dependency.declared_version, which extracts the bare version
        from the specifier (e.g. '==1.26.4' → '1.26.4'). Returns '' if no
        version was pinned or if the dependency had a parse error.
        """
        return self.dependency.declared_version

    @property
    def latest_version(self) -> str:
        """
        The latest available version on PyPI.

        Returns '' if the PyPI lookup failed or was not performed.
        """
        if self.package_info is None:
            return ""
        return self.package_info.latest_version_str

    @property
    def has_error(self) -> bool:
        """
        True if this dependency result carries any kind of error.

        Covers both parse errors (malformed line in requirements.txt) and
        PyPI lookup failures (network, 404, etc.).
        """
        if not self.dependency.is_valid:
            return True
        if self.package_info is not None and not self.package_info.is_available:
            return True
        return False

    @property
    def error_message(self) -> Optional[str]:
        """
        The first error message found, or None if the result is clean.

        Callers can use this to display a single diagnostic without needing
        to know which layer produced the error.
        """
        if self.dependency.parse_error:
            return self.dependency.parse_error
        if self.package_info is not None and self.package_info.fetch_error:
            return self.package_info.fetch_error
        return None


@dataclass
class ProjectScan:
    """
    The complete result of scanning a project's dependency health.

    This is the single object returned by scan_project() and consumed by
    cli.py. It contains everything needed to render the full report without
    any further calls to dependencies.py, packages.py, or the network.

    Attributes:
        project_path:       The directory that was scanned.
        project_name:       A human-readable name for the project (derived
                            from the directory name in V0.1).
        requirements_files: The requirements files that were found and
                            parsed. May be empty if scan_error is set.
        results:            One DependencyResult per parsed dependency line.
                            May be empty if no requirements files were found.
        scan_error:         Set when the scan could not complete at the
                            project level — e.g., the path doesn't exist,
                            isn't a directory, or has no requirements files.
                            When set, results will be empty.
    """

    project_path: Path
    project_name: str
    requirements_files: list[Path] = field(default_factory=list)
    results: list[DependencyResult] = field(default_factory=list)
    scan_error: Optional[str] = field(default=None)

    @property
    def is_successful(self) -> bool:
        """True when the scan completed without a project-level error."""
        return self.scan_error is None

    @property
    def total_dependencies(self) -> int:
        """Total number of dependency entries parsed (valid or not)."""
        return len(self.results)

    @property
    def updates_available(self) -> list[DependencyResult]:
        """
        Dependencies where an update is available.

        Returns only results that were cleanly scanned and have a newer
        version on PyPI (MAJOR, MINOR, or PATCH).
        """
        updateable = {UpdateType.MAJOR, UpdateType.MINOR, UpdateType.PATCH}
        return [r for r in self.results if r.update_type in updateable]

    @property
    def errored(self) -> list[DependencyResult]:
        """Dependencies that produced any kind of error during scanning."""
        return [r for r in self.results if r.has_error]


# ---------------------------------------------------------------------------
# Project detection helpers
# ---------------------------------------------------------------------------

def detect_project_name(project_path: Path) -> str:
    """
    Derive a human-readable project name from the project directory.

    V0.1 strategy: use the directory name.
    Future versions may read from pyproject.toml [project] name, setup.py,
    or other metadata.

    Args:
        project_path: The root directory of the project.

    Returns:
        A non-empty string. Falls back to the raw path string if needed.
    """
    name = project_path.name
    if not name:
        # This happens when project_path is a filesystem root (e.g. "C:\").
        return str(project_path)
    return name


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def scan_project(project_path: Path) -> ProjectScan:
    """
    Perform a complete dependency health scan of a Python project.

    This is the single entry point that cli.py calls. It orchestrates the
    full pipeline:

        1. Validate the project path.
        2. Detect the project name.
        3. Find requirements files.
        4. Parse each requirements file.
        5. For every valid dependency: query PyPI + classify the update.
        6. Assemble and return a ProjectScan.

    Individual dependency failures (parse errors, PyPI failures) are captured
    inside DependencyResult objects and do NOT abort the scan. Only
    project-level failures (invalid path, no requirements files found) set
    scan_error on the returned ProjectScan.

    Args:
        project_path: Path to the root directory of the Python project to
                      scan. May be absolute or relative.

    Returns:
        A ProjectScan object. Always returns — never raises for expected
        failure conditions.
    """
    logger.info("Starting project scan: %s", project_path)

    # ---- 1. Validate the project path ------------------------------------
    if not project_path.exists():
        msg = (
            f"Project path does not exist: {project_path}\n"
            f"Check that the path is correct."
        )
        logger.error(msg)
        return ProjectScan(
            project_path=project_path,
            project_name=detect_project_name(project_path),
            scan_error=msg,
        )

    if not project_path.is_dir():
        msg = (
            f"Project path is not a directory: {project_path}\n"
            f"PyGuardian expects a directory, not a file."
        )
        logger.error(msg)
        return ProjectScan(
            project_path=project_path,
            project_name=detect_project_name(project_path),
            scan_error=msg,
        )

    # ---- 2. Detect project name ------------------------------------------
    project_name = detect_project_name(project_path)
    logger.debug("Project name: %s", project_name)

    # ---- 3. Find requirements files --------------------------------------
    req_files = find_requirements_files(project_path)
    logger.debug("Requirements files found: %s", req_files)

    if not req_files:
        msg = (
            f"No supported dependency files found in: {project_path}\n"
            f"PyGuardian currently supports: requirements.txt\n"
            f"Make sure the file exists in the project root."
        )
        logger.warning(msg)
        return ProjectScan(
            project_path=project_path,
            project_name=project_name,
            requirements_files=[],
            scan_error=msg,
        )

    # ---- 4. Parse all requirements files ---------------------------------
    # Collect all Dependency objects across all discovered files.
    all_dependencies: list[Dependency] = []

    for req_file in req_files:
        logger.debug("Parsing: %s", req_file)
        try:
            deps = parse_requirements_file(req_file)
            all_dependencies.extend(deps)
            logger.debug("Parsed %d entries from %s", len(deps), req_file)
        except FileNotFoundError:
            # Should not happen — we found this file moments ago — but be
            # defensive against race conditions (e.g. file deleted mid-scan).
            msg = (
                f"Requirements file disappeared during scan: {req_file}\n"
                f"The file may have been deleted. Try scanning again."
            )
            logger.error(msg)
            return ProjectScan(
                project_path=project_path,
                project_name=project_name,
                requirements_files=req_files,
                scan_error=msg,
            )
        except PermissionError:
            msg = (
                f"Permission denied reading requirements file: {req_file}\n"
                f"Check that you have read access to the file."
            )
            logger.error(msg)
            return ProjectScan(
                project_path=project_path,
                project_name=project_name,
                requirements_files=req_files,
                scan_error=msg,
            )

    # ---- 5. For each dependency: PyPI lookup + update classification -----
    results: list[DependencyResult] = []

    for dep in all_dependencies:
        if not dep.is_valid:
            # Dependency had a parse error. We cannot look it up on PyPI.
            # Capture it in the results with no package_info so the CLI can
            # flag it as a problem line rather than silently skipping it.
            logger.debug(
                "Skipping PyPI lookup for invalid dependency: %r (error: %s)",
                dep.name,
                dep.parse_error,
            )
            results.append(
                DependencyResult(
                    dependency=dep,
                    package_info=None,
                    update_type=UpdateType.UNKNOWN,
                )
            )
            continue

        # Query PyPI — this never raises; failures are in package_info.
        logger.debug("Querying PyPI for: %s", dep.name)
        info = get_package_info(dep.name)

        # Classify the update using the bridge we built in the review step.
        # dep.declared_version extracts "1.26.4" from specifier "==1.26.4".
        # info.latest_version_str gives us "2.3.3".
        # Both are plain strings — exactly what classify_update() expects.
        update_type = classify_update(dep.declared_version, info.latest_version_str)

        logger.debug(
            "%s: current=%r latest=%r update_type=%s",
            dep.name,
            dep.declared_version,
            info.latest_version_str,
            update_type.value,
        )

        results.append(
            DependencyResult(
                dependency=dep,
                package_info=info,
                update_type=update_type,
            )
        )

    logger.info(
        "Scan complete: %d dependencies, %d updates available, %d errors",
        len(results),
        len([r for r in results if r.update_type in {UpdateType.MAJOR, UpdateType.MINOR, UpdateType.PATCH}]),
        len([r for r in results if r.has_error]),
    )

    return ProjectScan(
        project_path=project_path,
        project_name=project_name,
        requirements_files=req_files,
        results=results,
    )
