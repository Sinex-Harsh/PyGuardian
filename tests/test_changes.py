"""
test_changes.py — Unit tests for PyGuardian's package change intelligence (pyguardian/changes.py).

Testing Principles:
- Pure mock unit tests: NO real network calls.
- HTTP requests are mocked at the `pyguardian.packages.requests.get` boundary.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from packaging.version import Version
import requests

from pyguardian.changes import (
    ChangeInfo,
    ReleaseInfo,
    get_change_info,
    _extract_source_url,
    _parse_release_date,
)


# ---------------------------------------------------------------------------
# Helpers — Mock Payloads
# ---------------------------------------------------------------------------

def _make_response(
    json_data: dict,
    status_code: int = 200,
    raise_for_status_exc: Exception = None,
) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data

    if raise_for_status_exc is not None:
        mock_response.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_response.raise_for_status.return_value = None

    return mock_response


def _make_releases_payload(
    name: str = "pandas",
    releases_dict: dict = None,
    project_urls: dict = None,
    home_page: str = None,
) -> dict:
    if releases_dict is None:
        releases_dict = {
            "2.1.0": [{"upload_time_iso_8601": "2023-09-01T12:00:00Z"}],
            "2.2.0": [{"upload_time_iso_8601": "2024-01-15T10:00:00Z"}],
            "2.2.1": [{"upload_time_iso_8601": "2024-02-01T10:00:00Z"}],
            "2.2.2": [{"upload_time_iso_8601": "2024-02-15T10:00:00Z"}],
            "2.3.0": [{"upload_time_iso_8601": "2024-03-01T10:00:00Z"}],
            "3.0.0": [{"upload_time_iso_8601": "2024-06-01T10:00:00Z"}],
        }

    info = {"name": name, "version": "3.0.0"}
    if project_urls is not None:
        info["project_urls"] = project_urls
    if home_page is not None:
        info["home_page"] = home_page

    return {
        "info": info,
        "releases": releases_dict,
    }


# ---------------------------------------------------------------------------
# Unit tests: _extract_source_url & _parse_release_date
# ---------------------------------------------------------------------------

class TestChangeHelpers:

    def test_extract_source_url_priority(self):
        info = {
            "home_page": "https://homepage.com",
            "project_urls": {
                "Homepage": "https://homepage.com",
                "Changelog": "https://github.com/org/repo/blob/main/CHANGELOG.md",
                "Source Code": "https://github.com/org/repo",
            },
        }
        url = _extract_source_url(info)
        assert url == "https://github.com/org/repo/blob/main/CHANGELOG.md"

    def test_extract_source_url_fallback_homepage(self):
        info = {"home_page": "https://pandas.pydata.org"}
        url = _extract_source_url(info)
        assert url == "https://pandas.pydata.org"

    def test_extract_source_url_none(self):
        assert _extract_source_url({}) is None

    def test_parse_release_date_valid(self):
        files = [{"upload_time_iso_8601": "2024-02-15T10:20:30Z"}]
        dt = _parse_release_date(files)
        assert dt is not None
        assert dt == datetime(2024, 2, 15, 10, 20, 30, tzinfo=timezone.utc)

    def test_parse_release_date_empty(self):
        assert _parse_release_date([]) is None


# ---------------------------------------------------------------------------
# Unit tests: Version Range & Matching (get_change_info)
# ---------------------------------------------------------------------------

class TestGetChangeInfoRanges:

    @patch("pyguardian.packages.requests.get")
    def test_current_equals_target(self, mock_get):
        """2.2.0 → 2.2.0 should return empty releases range."""
        mock_get.return_value = _make_response(_make_releases_payload())

        change = get_change_info("pandas", "2.2.0", "2.2.0")

        assert change.is_available is True
        assert change.fetch_error is None
        assert change.release_count == 0
        assert change.releases == []
        assert change.target_found is True

    @patch("pyguardian.packages.requests.get")
    def test_patch_range(self, mock_get):
        """2.2.0 → 2.2.2 should include 2.2.1 and 2.2.2."""
        mock_get.return_value = _make_response(_make_releases_payload())

        change = get_change_info("pandas", "2.2.0", "2.2.2")

        assert change.is_available is True
        assert change.release_count == 2
        versions = [r.version_str for r in change.releases]
        assert versions == ["2.2.1", "2.2.2"]
        assert change.target_found is True

    @patch("pyguardian.packages.requests.get")
    def test_minor_range(self, mock_get):
        """2.1.0 → 2.3.0 should include 2.2.0, 2.2.1, 2.2.2, 2.3.0 in order."""
        mock_get.return_value = _make_response(_make_releases_payload())

        change = get_change_info("pandas", "2.1.0", "2.3.0")

        assert change.is_available is True
        assert change.release_count == 4
        versions = [r.version_str for r in change.releases]
        assert versions == ["2.2.0", "2.2.1", "2.2.2", "2.3.0"]
        assert change.target_found is True

    @patch("pyguardian.packages.requests.get")
    def test_major_range(self, mock_get):
        """2.2.0 → 3.0.0 should include all releases up to 3.0.0."""
        mock_get.return_value = _make_response(_make_releases_payload())

        change = get_change_info("pandas", "2.2.0", "3.0.0")

        assert change.is_available is True
        versions = [r.version_str for r in change.releases]
        assert versions == ["2.2.1", "2.2.2", "2.3.0", "3.0.0"]

    @patch("pyguardian.packages.requests.get")
    def test_releases_are_sorted_by_version(self, mock_get):
        """Unsorted JSON release keys must be returned in version-sorted order."""
        # Unsorted releases in dictionary
        unordered = {
            "2.10.0": [{"upload_time_iso_8601": "2024-05-01T10:00:00Z"}],
            "2.9.0": [{"upload_time_iso_8601": "2024-04-01T10:00:00Z"}],
            "2.8.5": [{"upload_time_iso_8601": "2024-03-01T10:00:00Z"}],
        }
        payload = _make_releases_payload("requests", unordered)
        mock_get.return_value = _make_response(payload)

        change = get_change_info("requests", "2.8.0", "2.10.0")

        assert change.is_available is True
        versions = [r.version_str for r in change.releases]
        # packaging.Version sort correctly places 2.8.5 < 2.9.0 < 2.10.0
        assert versions == ["2.8.5", "2.9.0", "2.10.0"]

    @patch("pyguardian.packages.requests.get")
    def test_prerelease_versions_handling(self, mock_get):
        """Pre-release versions like 2.3.0rc1 should be handled correctly by Version."""
        releases_dict = {
            "2.2.0": [],
            "2.3.0rc1": [{"upload_time_iso_8601": "2024-02-15T10:00:00Z"}],
            "2.3.0": [{"upload_time_iso_8601": "2024-03-01T10:00:00Z"}],
        }
        mock_get.return_value = _make_response(_make_releases_payload("pandas", releases_dict))

        change = get_change_info("pandas", "2.2.0", "2.3.0")

        assert change.is_available is True
        versions = [r.version_str for r in change.releases]
        assert "2.3.0rc1" in versions
        assert "2.3.0" in versions

    @patch("pyguardian.packages.requests.get")
    def test_invalid_release_version_key_skipped(self, mock_get):
        """Unparseable release keys should be skipped safely without crashing."""
        releases_dict = {
            "2.2.0": [],
            "invalid-version-string-xyz": [],
            "2.2.1": [],
        }
        mock_get.return_value = _make_response(_make_releases_payload("pandas", releases_dict))

        change = get_change_info("pandas", "2.1.0", "2.2.1")

        assert change.is_available is True
        versions = [r.version_str for r in change.releases]
        assert versions == ["2.2.0", "2.2.1"]


# ---------------------------------------------------------------------------
# Unit tests: Error & Edge Case Handling
# ---------------------------------------------------------------------------

class TestGetChangeInfoErrors:

    @patch("pyguardian.packages.requests.get")
    def test_http_404_not_found(self, mock_get):
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=404))
        mock_get.return_value = _make_response({}, 404, raise_for_status_exc=http_error)

        change = get_change_info("nonexistent-pkg", "1.0.0", "2.0.0")

        assert change.is_available is False
        assert change.fetch_error is not None
        assert "404" in change.fetch_error

    @patch("pyguardian.packages.requests.get")
    def test_network_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()

        change = get_change_info("pandas", "1.0.0", "2.0.0")

        assert change.is_available is False
        assert change.fetch_error is not None
        assert "timed out" in change.fetch_error.lower()

    @patch("pyguardian.packages.requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        change = get_change_info("pandas", "1.0.0", "2.0.0")

        assert change.is_available is False
        assert change.fetch_error is not None
        assert "connect" in change.fetch_error.lower()

    @patch("pyguardian.packages.requests.get")
    def test_malformed_json(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        change = get_change_info("pandas", "1.0.0", "2.0.0")

        assert change.is_available is False
        assert change.fetch_error is not None

    @patch("pyguardian.packages.requests.get")
    def test_target_not_found_in_releases(self, mock_get):
        """If target version doesn't exist in release metadata, handle explicitly."""
        mock_get.return_value = _make_response(_make_releases_payload())

        # Target 9.9.9 does not exist in payload
        change = get_change_info("pandas", "2.2.0", "9.9.9")

        assert change.is_available is False
        assert change.target_found is False
        assert change.fetch_error is not None
        assert "9.9.9" in change.fetch_error

    def test_invalid_current_version_string(self):
        change = get_change_info("pandas", "invalid-version", "2.0.0")

        assert change.is_available is False
        assert change.fetch_error is not None
        assert "Invalid current version" in change.fetch_error

    def test_invalid_target_version_string(self):
        change = get_change_info("pandas", "1.0.0", "invalid-target")

        assert change.is_available is False
        assert change.fetch_error is not None
        assert "Invalid target version" in change.fetch_error
