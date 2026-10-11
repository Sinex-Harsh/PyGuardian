"""
test_packages.py — Tests for PyPI package metadata retrieval and version logic.

Key testing principle:
    We NEVER make real network requests in this test suite.
    All HTTP calls are intercepted using unittest.mock.patch, which replaces
    `requests.get` with a controlled fake that we fully control.

    This means tests:
    - Run in milliseconds (no network latency).
    - Pass whether or not the internet is available.
    - Pass whether or not PyPI is up.
    - Are repeatable regardless of what PyPI currently returns.

Pattern used:
    @patch("pyguardian.packages.requests.get")
    def test_something(self, mock_get):
        mock_get.return_value = _make_response(...)
        result = get_package_info("numpy")
        assert result.latest_version_str == "2.3.3"

Why patch "pyguardian.packages.requests.get" and NOT "requests.get"?
    Because Python's patch replaces the name *where it is used*, not where it
    is defined. Our code imports requests in packages.py, so that's the
    namespace we must patch. Patching "requests.get" directly would have no
    effect on already-imported references.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from packaging.version import Version
import requests

from pyguardian.packages import (
    get_package_info,
    classify_update,
    UpdateType,
    PackageInfo,
    PYPI_JSON_URL,
)


# ---------------------------------------------------------------------------
# Helpers — fake HTTP responses
# ---------------------------------------------------------------------------

def _make_response(
    json_data: dict,
    status_code: int = 200,
    raise_for_status_exc: Exception = None,
) -> MagicMock:
    """
    Build a mock requests.Response object.

    Args:
        json_data:            Data the .json() method will return.
        status_code:          HTTP status code.
        raise_for_status_exc: If set, .raise_for_status() will raise this.

    Returns:
        A MagicMock configured to behave like a requests.Response.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data

    if raise_for_status_exc is not None:
        mock_response.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_response.raise_for_status.return_value = None  # no-op

    return mock_response


def _make_pypi_payload(name: str, version: str) -> dict:
    """
    Build a minimal PyPI JSON response payload.

    Only includes the fields that get_package_info() actually reads.
    Real PyPI responses are much larger; we only need what our code touches.
    """
    return {
        "info": {
            "name": name,
            "version": version,
            "summary": f"Test package {name}",
        },
        "releases": {},
        "urls": [],
    }


# ---------------------------------------------------------------------------
# Tests: classify_update
# ---------------------------------------------------------------------------

class TestClassifyUpdate:
    """
    Tests for the pure-logic version classification function.

    No mocking needed here — classify_update is purely deterministic,
    takes two strings, returns an UpdateType. No network involved.
    """

    def test_major_update(self):
        """Bumping the leftmost non-zero segment is MAJOR."""
        assert classify_update("1.26.4", "2.3.3") == UpdateType.MAJOR

    def test_major_update_large_jump(self):
        """A jump of several major versions is still MAJOR."""
        assert classify_update("1.0.0", "5.0.0") == UpdateType.MAJOR

    def test_minor_update(self):
        """Same major, higher minor is MINOR."""
        assert classify_update("2.2.0", "2.3.2") == UpdateType.MINOR

    def test_minor_update_same_major_higher_minor(self):
        """Regression check: 2.10 > 2.9 must work correctly."""
        # String comparison would fail here: "2.10" < "2.9" alphabetically.
        # packaging.Version handles this correctly.
        assert classify_update("2.9.0", "2.10.0") == UpdateType.MINOR

    def test_patch_update(self):
        """Same major and minor, higher patch is PATCH."""
        assert classify_update("2.31.0", "2.31.5") == UpdateType.PATCH

    def test_up_to_date(self):
        """Identical versions means UP_TO_DATE."""
        assert classify_update("2.3.3", "2.3.3") == UpdateType.UP_TO_DATE

    def test_current_ahead_of_latest(self):
        """If current > latest (e.g. pre-release pinned), UP_TO_DATE."""
        # This can happen when a user pins a pre-release that later gets yanked
        # or when their pinned version is newer than PyPI's "stable" latest.
        assert classify_update("3.0.0rc1", "2.3.3") == UpdateType.UP_TO_DATE

    def test_empty_current_version(self):
        """Empty current version string → UNKNOWN."""
        assert classify_update("", "2.3.3") == UpdateType.UNKNOWN

    def test_empty_latest_version(self):
        """Empty latest version string → UNKNOWN."""
        assert classify_update("1.0.0", "") == UpdateType.UNKNOWN

    def test_both_empty(self):
        """Both empty → UNKNOWN."""
        assert classify_update("", "") == UpdateType.UNKNOWN

    def test_invalid_current_version(self):
        """Unparseable current version string → UNKNOWN."""
        assert classify_update("not-a-version", "2.3.3") == UpdateType.UNKNOWN

    def test_invalid_latest_version(self):
        """Unparseable latest version string → UNKNOWN."""
        assert classify_update("1.0.0", "totally-invalid") == UpdateType.UNKNOWN


# ---------------------------------------------------------------------------
# Tests: get_package_info — success cases
# ---------------------------------------------------------------------------

class TestGetPackageInfoSuccess:

    @patch("pyguardian.packages.requests.get")
    def test_returns_latest_version(self, mock_get):
        """A valid PyPI response should return the correct latest version."""
        payload = _make_pypi_payload("numpy", "2.3.3")
        mock_get.return_value = _make_response(payload)

        result = get_package_info("numpy")

        assert result.is_available is True
        assert result.fetch_error is None
        assert result.latest_version == Version("2.3.3")
        assert result.latest_version_str == "2.3.3"

    @patch("pyguardian.packages.requests.get")
    def test_uses_canonical_name_from_pypi(self, mock_get):
        """The name in PackageInfo should come from PyPI, not our input."""
        # PyPI normalizes package names; e.g. "Requests" → "requests"
        payload = _make_pypi_payload("requests", "2.32.5")
        mock_get.return_value = _make_response(payload)

        result = get_package_info("Requests")  # note: capitalised input

        assert result.name == "requests"  # PyPI's canonical name

    @patch("pyguardian.packages.requests.get")
    def test_queries_correct_pypi_url(self, mock_get):
        """We must be requesting the correct PyPI URL for the package."""
        payload = _make_pypi_payload("pandas", "2.3.2")
        mock_get.return_value = _make_response(payload)

        get_package_info("pandas")

        expected_url = PYPI_JSON_URL.format(package="pandas")
        mock_get.assert_called_once_with(expected_url, timeout=10)

    @patch("pyguardian.packages.requests.get")
    def test_version_is_packaging_version_object(self, mock_get):
        """latest_version should be a packaging.Version instance, not a string."""
        payload = _make_pypi_payload("flask", "3.1.0")
        mock_get.return_value = _make_response(payload)

        result = get_package_info("flask")

        assert isinstance(result.latest_version, Version)


# ---------------------------------------------------------------------------
# Tests: get_package_info — failure cases
# ---------------------------------------------------------------------------

class TestGetPackageInfoFailures:

    @patch("pyguardian.packages.requests.get")
    def test_package_not_found_404(self, mock_get):
        """HTTP 404 should return PackageInfo with a descriptive fetch_error."""
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=404))
        mock_get.return_value = _make_response({}, 404, raise_for_status_exc=http_error)

        result = get_package_info("nonexistent-package-xyz-123")

        assert result.is_available is False
        assert result.latest_version is None
        assert result.fetch_error is not None
        assert "404" in result.fetch_error

    @patch("pyguardian.packages.requests.get")
    def test_rate_limited_429(self, mock_get):
        """HTTP 429 should return PackageInfo with a rate-limit error message."""
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=429))
        mock_get.return_value = _make_response({}, 429, raise_for_status_exc=http_error)

        result = get_package_info("numpy")

        assert result.is_available is False
        assert "429" in result.fetch_error

    @patch("pyguardian.packages.requests.get")
    def test_server_error_500(self, mock_get):
        """HTTP 5xx should return PackageInfo with an error, not raise."""
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=500))
        mock_get.return_value = _make_response({}, 500, raise_for_status_exc=http_error)

        result = get_package_info("numpy")

        assert result.is_available is False
        assert "500" in result.fetch_error

    @patch("pyguardian.packages.requests.get")
    def test_network_timeout(self, mock_get):
        """A Timeout exception should be caught and returned as fetch_error."""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = get_package_info("numpy")

        assert result.is_available is False
        assert result.fetch_error is not None
        assert "timed out" in result.fetch_error.lower()

    @patch("pyguardian.packages.requests.get")
    def test_no_internet_connection_error(self, mock_get):
        """A ConnectionError should be caught and returned as fetch_error."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = get_package_info("numpy")

        assert result.is_available is False
        assert result.fetch_error is not None
        assert "connect" in result.fetch_error.lower()

    @patch("pyguardian.packages.requests.get")
    def test_malformed_json_response(self, mock_get):
        """If response.json() raises ValueError, we get a fetch_error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_get.return_value = mock_response

        result = get_package_info("numpy")

        assert result.is_available is False
        assert result.fetch_error is not None

    @patch("pyguardian.packages.requests.get")
    def test_missing_info_key_in_response(self, mock_get):
        """PyPI response missing 'info' key should produce fetch_error."""
        # Simulate an unexpected API response shape
        payload = {"unexpected_key": "unexpected_value"}
        mock_get.return_value = _make_response(payload)

        result = get_package_info("numpy")

        assert result.is_available is False
        assert result.fetch_error is not None

    @patch("pyguardian.packages.requests.get")
    def test_missing_version_key_in_info(self, mock_get):
        """PyPI response with 'info' but no 'version' should produce fetch_error."""
        payload = {"info": {"name": "numpy"}}  # 'version' key missing
        mock_get.return_value = _make_response(payload)

        result = get_package_info("numpy")

        assert result.is_available is False
        assert result.fetch_error is not None

    @patch("pyguardian.packages.requests.get")
    def test_never_raises_on_any_network_error(self, mock_get):
        """
        get_package_info must NEVER propagate an exception to the caller
        for any network-related failure. It always returns a PackageInfo.
        """
        mock_get.side_effect = requests.exceptions.RequestException("unexpected")

        # This must NOT raise
        result = get_package_info("numpy")

        assert isinstance(result, PackageInfo)
        assert result.is_available is False


# ---------------------------------------------------------------------------
# Tests: PackageInfo dataclass
# ---------------------------------------------------------------------------

class TestPackageInfo:

    def test_is_available_true_when_version_present(self):
        info = PackageInfo(name="numpy", latest_version=Version("2.3.3"))
        assert info.is_available is True

    def test_is_available_false_when_fetch_error_set(self):
        info = PackageInfo(name="numpy", fetch_error="something went wrong")
        assert info.is_available is False

    def test_is_available_false_when_version_is_none(self):
        info = PackageInfo(name="numpy")  # no version, no error
        assert info.is_available is False

    def test_latest_version_str_returns_string(self):
        info = PackageInfo(name="numpy", latest_version=Version("2.3.3"))
        assert info.latest_version_str == "2.3.3"

    def test_latest_version_str_returns_empty_string_when_none(self):
        info = PackageInfo(name="numpy")
        assert info.latest_version_str == ""
