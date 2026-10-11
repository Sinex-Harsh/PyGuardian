"""
packages.py — PyPI package metadata retrieval and version intelligence.

Responsibilities:
- Query PyPI's JSON API for package metadata.
- Extract the latest available version for a package.
- Classify the update type (MAJOR / MINOR / PATCH / UP_TO_DATE / UNKNOWN).
- Handle all network and API failure scenarios gracefully.

V0.1 scope:
- Latest version only (no changelog, no release history yet).
- Single-package queries (no batch optimisation yet).

Design note on error handling:
  Rather than raising exceptions on network failures, functions here return
  result objects that carry the error information. This allows the scanner
  to continue processing the remaining packages when one lookup fails,
  and lets the caller decide how to surface the problem to the user.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from packaging.version import Version, InvalidVersion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"

# How long to wait for a PyPI response before giving up.
# 10 seconds is generous — PyPI is fast. Prevents tool from hanging on poor
# connectivity.
REQUEST_TIMEOUT_SECONDS = 10


# ---------------------------------------------------------------------------
# Update classification
# ---------------------------------------------------------------------------

class UpdateType(enum.Enum):
    """
    Classifies the nature of a version update.

    MAJOR — Breaking changes likely (X.y.z → (X+n).y.z).
    MINOR — New features, backwards-compatible (x.Y.z → x.(Y+n).z).
    PATCH — Bug fixes only (x.y.Z → x.y.(Z+n)).
    UP_TO_DATE — Current version equals latest.
    UNKNOWN — Could not determine (missing or unparseable versions).
    """

    MAJOR = "MAJOR"
    MINOR = "MINOR"
    PATCH = "PATCH"
    UP_TO_DATE = "UP_TO_DATE"
    UNKNOWN = "UNKNOWN"


def classify_update(current_version_str: str, latest_version_str: str) -> UpdateType:
    """
    Classify the update type between two version strings.

    Uses packaging.version.Version for correct semantic version comparison.
    String comparison is intentionally avoided (e.g. "2.10" > "2.9" works
    correctly with packaging.Version but NOT with plain Python string
    comparison where "2.10" < "2.9" because "1" < "9").

    Args:
        current_version_str: The currently declared/installed version string.
        latest_version_str:  The latest available version string from PyPI.

    Returns:
        An UpdateType enum value.

    Examples:
        classify_update("1.26.4", "2.3.3")  → UpdateType.MAJOR
        classify_update("2.2.0", "2.3.2")   → UpdateType.MINOR
        classify_update("2.31.0", "2.31.5") → UpdateType.PATCH
        classify_update("2.3.3", "2.3.3")   → UpdateType.UP_TO_DATE
        classify_update("", "2.3.3")        → UpdateType.UNKNOWN
    """
    if not current_version_str or not latest_version_str:
        return UpdateType.UNKNOWN

    try:
        current = Version(current_version_str)
        latest = Version(latest_version_str)
    except InvalidVersion:
        logger.warning(
            "Could not parse version strings for comparison: "
            "current=%r, latest=%r",
            current_version_str,
            latest_version_str,
        )
        return UpdateType.UNKNOWN

    if current >= latest:
        return UpdateType.UP_TO_DATE

    # Compare the major, minor, micro (patch) components individually.
    # packaging.Version exposes these as integer attributes.
    if latest.major > current.major:
        return UpdateType.MAJOR
    if latest.minor > current.minor:
        return UpdateType.MINOR
    return UpdateType.PATCH


# ---------------------------------------------------------------------------
# Package metadata result
# ---------------------------------------------------------------------------

@dataclass
class PackageInfo:
    """
    Holds PyPI metadata for a single package, as retrieved for V0.1.

    Attributes:
        name:           The canonical package name as returned by PyPI.
        latest_version: The latest stable version (as a Version object).
                        None if the lookup failed.
        fetch_error:    Human-readable error message if the lookup failed.
                        None if the lookup was successful.
    """

    name: str
    latest_version: Optional[Version] = field(default=None)
    fetch_error: Optional[str] = field(default=None)

    @property
    def is_available(self) -> bool:
        """True when the PyPI lookup succeeded and we have a latest version."""
        return self.latest_version is not None and self.fetch_error is None

    @property
    def latest_version_str(self) -> str:
        """Latest version as a plain string, or empty string if unavailable."""
        if self.latest_version is None:
            return ""
        return str(self.latest_version)


# ---------------------------------------------------------------------------
# PyPI client
# ---------------------------------------------------------------------------

def fetch_pypi_json(
    package_name: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Fetch raw JSON metadata dictionary from PyPI for a package.

    Returns:
        A tuple of (json_data_dict, error_message).
        On success: (data_dict, None)
        On error:   (None, human_readable_error_string)
    """
    url = PYPI_JSON_URL.format(package=package_name)
    logger.debug("Querying PyPI: %s", url)

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        msg = (
            f"Request to PyPI timed out after {timeout}s for package '{package_name}'.\n"
            f"Check your internet connection and try again."
        )
        logger.warning(msg)
        return None, msg
    except requests.exceptions.ConnectionError:
        msg = (
            f"Could not connect to PyPI for package '{package_name}'.\n"
            f"Check your internet connection."
        )
        logger.warning(msg)
        return None, msg
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 404:
            msg = (
                f"Package '{package_name}' was not found on PyPI (HTTP 404).\n"
                f"The package may be private, misspelled, or not published on PyPI."
            )
        elif status_code == 429:
            msg = (
                f"PyPI rate-limited this request for package '{package_name}' (HTTP 429).\n"
                f"Wait a moment and try again."
            )
        else:
            msg = (
                f"PyPI returned HTTP {status_code} for package '{package_name}'.\n"
                f"PyPI may be experiencing issues. Try again later."
            )
        logger.warning(msg)
        return None, msg
    except requests.exceptions.RequestException as exc:
        msg = (
            f"Unexpected network error while querying PyPI for "
            f"'{package_name}': {exc}"
        )
        logger.error(msg)
        return None, msg

    try:
        data = response.json()
        if not isinstance(data, dict):
            msg = f"PyPI returned invalid non-dict JSON for package '{package_name}'."
            logger.error(msg)
            return None, msg
        return data, None
    except ValueError as exc:
        msg = (
            f"PyPI returned non-JSON response for package '{package_name}': {exc}"
        )
        logger.error(msg)
        return None, msg


def get_package_info(
    package_name: str,
    *,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> PackageInfo:
    """
    Query PyPI's JSON API for the latest version of a package.

    This function is the single point of contact with PyPI in V0.1.
    It handles all expected failure modes and always returns a PackageInfo
    object — it never raises exceptions for network or API issues.

    Args:
        package_name: The pip/PyPI package name (e.g. "numpy", "requests").
        timeout:      HTTP request timeout in seconds. Exposed for testing.

    Returns:
        A PackageInfo object. Check `.is_available` before using `.latest_version`.

    Examples:
        info = get_package_info("numpy")
        if info.is_available:
            print(info.latest_version_str)   # "2.3.3"
        else:
            print(info.fetch_error)
    """
    data, error = fetch_pypi_json(package_name, timeout=timeout)
    if error or data is None:
        return PackageInfo(name=package_name, fetch_error=error)

    # Extract the latest version from the `info` block.
    try:
        canonical_name = data["info"]["name"]
        latest_version_str = data["info"]["version"]
    except (KeyError, TypeError) as exc:
        msg = (
            f"PyPI response for '{package_name}' was missing expected fields: {exc}\n"
            f"The API response structure may have changed."
        )
        logger.error(msg)
        return PackageInfo(name=package_name, fetch_error=msg)

    # Parse the version string with packaging.version.Version.
    # This should not fail for well-formed PyPI data, but we guard anyway.
    try:
        latest_version = Version(latest_version_str)
    except InvalidVersion as exc:
        msg = (
            f"PyPI returned an unparseable version string for "
            f"'{package_name}': {latest_version_str!r} — {exc}"
        )
        logger.error(msg)
        return PackageInfo(name=package_name, fetch_error=msg)

    logger.debug(
        "PyPI lookup succeeded: %s → latest=%s", canonical_name, latest_version
    )
    return PackageInfo(name=canonical_name, latest_version=latest_version)
