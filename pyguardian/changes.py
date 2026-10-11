"""
changes.py — Package release history and change intelligence foundation.

Responsibilities:
- Query PyPI metadata for release history of packages.
- Filter releases occurring between a baseline `current_version` and `target_version`.
- Sort version releases strictly using `packaging.version.Version`.
- Handle pre-releases, unparseable version strings, and missing release dates safely.
- Extract project URLs (documentation, repository, changelog) from PyPI metadata.
- Return structured `ChangeInfo` and `ReleaseInfo` objects without throwing exceptions.

Design Principles:
- Does NOT execute or inspect project source code.
- Does NOT perform AI/RAG analysis, patch generation, or risk scoring.
- Handles network/PyPI errors gracefully, returning structured error data in `ChangeInfo`.
- Uses packaging.version.Version objects for all version range operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from packaging.version import Version, InvalidVersion

from pyguardian.packages import REQUEST_TIMEOUT_SECONDS, fetch_pypi_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReleaseInfo:
    """
    Represents metadata for a single version release of a package.

    Attributes:
        version:      The Version object for this release.
        release_date: Optional UTC datetime timestamp when the release was uploaded to PyPI.
    """

    version: Version
    release_date: Optional[datetime] = field(default=None)

    @property
    def version_str(self) -> str:
        """Returns the version as a plain string."""
        return str(self.version)


@dataclass(frozen=True)
class ChangeInfo:
    """
    Represents the collected release history and metadata between two versions.

    Attributes:
        package_name:    Canonical name of the package.
        current_version: Baseline current version (Version object).
        target_version:  Target version to compare against (Version object).
        releases:        Sorted list of ReleaseInfo objects where current_version < v <= target_version.
        source_url:      Optional project repository, documentation, or homepage URL.
        fetch_error:     Human-readable error description if lookup or parsing failed.
    """

    package_name: str
    current_version: Version
    target_version: Version
    releases: list[ReleaseInfo] = field(default_factory=list)
    source_url: Optional[str] = field(default=None)
    fetch_error: Optional[str] = field(default=None)

    @property
    def is_available(self) -> bool:
        """Returns True if change metadata was fetched successfully without error."""
        return self.fetch_error is None

    @property
    def release_count(self) -> int:
        """Number of intermediate/target releases in the version range."""
        return len(self.releases)

    @property
    def target_found(self) -> bool:
        """
        True if target_version was explicitly found in the release metadata.
        If current_version == target_version, returns True.
        """
        if self.current_version == self.target_version:
            return True
        return any(rel.version == self.target_version for rel in self.releases)


def _extract_source_url(info_dict: dict) -> Optional[str]:
    """
    Extract a repository, documentation, or homepage URL from PyPI info metadata.
    """
    if not isinstance(info_dict, dict):
        return None

    project_urls = info_dict.get("project_urls") or {}
    if isinstance(project_urls, dict):
        # Check priority keys
        for key in ["Changelog", "Changes", "Release Notes", "Source", "Source Code", "Repository", "Homepage", "Home"]:
            for p_key, p_val in project_urls.items():
                if p_key and p_key.lower() == key.lower() and p_val and isinstance(p_val, str):
                    return p_val.strip()

    home_page = info_dict.get("home_page")
    if home_page and isinstance(home_page, str) and home_page.strip():
        return home_page.strip()

    return None


def _parse_release_date(file_list: list[dict]) -> Optional[datetime]:
    """
    Extract the earliest upload timestamp from a list of release files.
    """
    if not isinstance(file_list, list) or not file_list:
        return None

    dates: list[datetime] = []
    for f in file_list:
        if not isinstance(f, dict):
            continue
        ts_str = f.get("upload_time_iso_8601") or f.get("upload_time")
        if not ts_str or not isinstance(ts_str, str):
            continue
        try:
            clean_str = ts_str.rstrip("Z")
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dates.append(dt)
        except ValueError:
            continue

    return min(dates) if dates else None


def get_change_info(
    package_name: str,
    current_version: str | Version,
    target_version: str | Version,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> ChangeInfo:
    """
    Retrieve change and release history for a package between current_version and target_version.

    Args:
        package_name:    The pip/PyPI package name (e.g. "pandas").
        current_version: Baseline current version (str or Version).
        target_version:  Target version to upgrade/compare to (str or Version).
        timeout:         Network request timeout in seconds.

    Returns:
        A ChangeInfo object containing matching ReleaseInfo records and metadata.
    """
    # Parse version parameters
    try:
        current_ver = current_version if isinstance(current_version, Version) else Version(str(current_version))
    except (InvalidVersion, TypeError) as exc:
        msg = f"Invalid current version string for '{package_name}': {current_version!r} — {exc}"
        logger.error(msg)
        dummy_ver = Version("0.0.0")
        return ChangeInfo(
            package_name=package_name,
            current_version=dummy_ver,
            target_version=dummy_ver,
            fetch_error=msg,
        )

    try:
        target_ver = target_version if isinstance(target_version, Version) else Version(str(target_version))
    except (InvalidVersion, TypeError) as exc:
        msg = f"Invalid target version string for '{package_name}': {target_version!r} — {exc}"
        logger.error(msg)
        return ChangeInfo(
            package_name=package_name,
            current_version=current_ver,
            target_version=current_ver,
            fetch_error=msg,
        )

    # If current == target, return empty releases cleanly
    if current_ver == target_ver:
        data, _ = fetch_pypi_json(package_name, timeout=timeout)
        source_url = _extract_source_url(data.get("info", {})) if data else None
        return ChangeInfo(
            package_name=package_name,
            current_version=current_ver,
            target_version=target_ver,
            releases=[],
            source_url=source_url,
        )

    # Fetch PyPI metadata
    data, error = fetch_pypi_json(package_name, timeout=timeout)
    if error or data is None:
        return ChangeInfo(
            package_name=package_name,
            current_version=current_ver,
            target_version=target_ver,
            fetch_error=error,
        )

    info_dict = data.get("info", {})
    canonical_name = info_dict.get("name", package_name)
    source_url = _extract_source_url(info_dict)
    raw_releases = data.get("releases", {})

    if not isinstance(raw_releases, dict):
        msg = f"PyPI response for '{package_name}' contained malformed release data."
        logger.error(msg)
        return ChangeInfo(
            package_name=canonical_name,
            current_version=current_ver,
            target_version=target_ver,
            source_url=source_url,
            fetch_error=msg,
        )

    matching_releases: list[ReleaseInfo] = []
    found_target_in_pypi = False

    for ver_str, file_list in raw_releases.items():
        try:
            rel_ver = Version(ver_str)
        except InvalidVersion:
            logger.warning("Skipping invalid release version '%s' for package '%s'", ver_str, package_name)
            continue

        if rel_ver == target_ver:
            found_target_in_pypi = True

        # Range condition: current_ver < rel_ver <= target_ver
        if current_ver < rel_ver <= target_ver:
            rel_date = _parse_release_date(file_list)
            matching_releases.append(ReleaseInfo(version=rel_ver, release_date=rel_date))

    # Sort releases by version ascending
    matching_releases.sort(key=lambda r: r.version)

    fetch_error: Optional[str] = None
    if not found_target_in_pypi:
        fetch_error = (
            f"Target version '{target_ver}' was not found in release history for package '{canonical_name}'."
        )

    return ChangeInfo(
        package_name=canonical_name,
        current_version=current_ver,
        target_version=target_ver,
        releases=matching_releases,
        source_url=source_url,
        fetch_error=fetch_error,
    )
