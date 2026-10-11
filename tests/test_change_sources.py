"""
test_change_sources.py — Tests for Change-Source Intelligence (V0.2 Step 4).

Key testing principle:
    All HTTP requests (PyPI metadata & document fetch) are intercepted using unittest.mock.patch.
    No live network connections are made.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from pyguardian.change_sources import (
    SOURCE_TYPE_CHANGELOG,
    SOURCE_TYPE_DOCUMENTATION,
    SOURCE_TYPE_MIGRATION_GUIDE,
    SOURCE_TYPE_RELEASE_NOTES,
    SOURCE_TYPE_REPOSITORY,
    ChangeDocument,
    ChangeSource,
    classify_source_type,
    discover_change_sources,
    fetch_change_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    text: str = "",
    status_code: int = 200,
    raise_for_status_exc: Exception = None,
) -> MagicMock:
    """Build a mock requests.Response object for document fetching."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text

    if raise_for_status_exc is not None:
        mock_resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        mock_resp.raise_for_status.return_value = None

    return mock_resp


# ---------------------------------------------------------------------------
# Tests: classify_source_type
# ---------------------------------------------------------------------------

class TestClassifySourceType:

    def test_migration_guide_classification(self):
        assert classify_source_type("Migration Guide", "https://example.org/migration") == SOURCE_TYPE_MIGRATION_GUIDE
        assert classify_source_type("Upgrading to v3", "https://example.org/upgrade") == SOURCE_TYPE_MIGRATION_GUIDE

    def test_release_notes_classification(self):
        assert classify_source_type("Release Notes", "https://github.com/org/repo/releases") == SOURCE_TYPE_RELEASE_NOTES
        assert classify_source_type("What's New", "https://example.org/whatsnew") == SOURCE_TYPE_RELEASE_NOTES

    def test_changelog_classification(self):
        assert classify_source_type("Changelog", "https://example.org/changelog.html") == SOURCE_TYPE_CHANGELOG
        assert classify_source_type("Changes", "https://github.com/org/repo/blob/main/CHANGES.md") == SOURCE_TYPE_CHANGELOG

    def test_repository_classification(self):
        assert classify_source_type("Source Code", "https://github.com/pandas-dev/pandas") == SOURCE_TYPE_REPOSITORY
        assert classify_source_type("Repository", "https://gitlab.com/org/repo") == SOURCE_TYPE_REPOSITORY

    def test_documentation_classification(self):
        assert classify_source_type("Documentation", "https://pandas.pydata.org/docs/") == SOURCE_TYPE_DOCUMENTATION
        assert classify_source_type("Homepage", "https://pandas.pydata.org") == SOURCE_TYPE_DOCUMENTATION

    def test_fallback_does_not_misclassify_repository_as_changelog(self):
        # A repository URL with key "Code" is a repository, not changelog
        stype = classify_source_type("Code", "https://github.com/foo/bar")
        assert stype == SOURCE_TYPE_REPOSITORY


# ---------------------------------------------------------------------------
# Tests: discover_change_sources
# ---------------------------------------------------------------------------

class TestDiscoverChangeSources:

    def test_discovers_and_ranks_sources_from_pypi_data(self):
        pypi_data = {
            "info": {
                "name": "pandas",
                "home_page": "https://pandas.pydata.org",
                "project_urls": {
                    "Documentation": "https://pandas.pydata.org/docs/",
                    "Source Code": "https://github.com/pandas-dev/pandas",
                    "Release Notes": "https://github.com/pandas-dev/pandas/releases",
                    "Migration Guide": "https://pandas.pydata.org/docs/migration.html",
                },
            }
        }

        sources = discover_change_sources("pandas", pypi_data=pypi_data)

        # 4 from project_urls + 1 from home_page = 5 distinct URLs
        assert len(sources) == 5
        # Verify priority ordering
        assert sources[0].source_type == SOURCE_TYPE_MIGRATION_GUIDE
        assert sources[0].url == "https://pandas.pydata.org/docs/migration.html"

        assert sources[1].source_type == SOURCE_TYPE_RELEASE_NOTES
        assert sources[1].url == "https://github.com/pandas-dev/pandas/releases"

        assert sources[2].source_type == SOURCE_TYPE_DOCUMENTATION
        assert sources[3].source_type == SOURCE_TYPE_DOCUMENTATION
        assert sources[4].source_type == SOURCE_TYPE_REPOSITORY

        # All sources from PyPI metadata are marked as official
        assert all(s.is_official for s in sources)

    def test_deduplicates_identical_urls(self):
        pypi_data = {
            "info": {
                "name": "foo",
                "home_page": "https://example.org/",
                "project_urls": {
                    "Homepage": "https://example.org",
                    "Documentation": "https://example.org/",
                },
            }
        }

        sources = discover_change_sources("foo", pypi_data=pypi_data)
        assert len(sources) == 1
        assert sources[0].url in ["https://example.org", "https://example.org/"]

    def test_handles_missing_or_empty_metadata(self):
        sources = discover_change_sources("empty-pkg", pypi_data={"info": {}})
        assert sources == []

        sources_invalid = discover_change_sources("invalid-pkg", pypi_data={})
        assert sources_invalid == []

    @patch("pyguardian.change_sources.fetch_pypi_json")
    def test_fetches_pypi_json_when_not_provided(self, mock_fetch):
        mock_fetch.return_value = (
            {
                "info": {
                    "name": "requests",
                    "project_urls": {
                        "Changelog": "https://github.com/psf/requests/blob/main/HISTORY.md"
                    },
                }
            },
            None,
        )

        sources = discover_change_sources("requests")
        assert len(sources) == 1
        assert sources[0].source_type == SOURCE_TYPE_CHANGELOG
        assert sources[0].url == "https://github.com/psf/requests/blob/main/HISTORY.md"
        mock_fetch.assert_called_once_with("requests", timeout=10.0)

    @patch("pyguardian.change_sources.fetch_pypi_json")
    def test_handles_pypi_fetch_failure_gracefully(self, mock_fetch):
        mock_fetch.return_value = (None, "404 Not Found")

        sources = discover_change_sources("unknown-package")
        assert sources == []

    def test_does_not_construct_arbitrary_urls(self):
        # Even if package name is "numpy", we do not invent "https://github.com/numpy/numpy"
        pypi_data = {
            "info": {
                "name": "numpy",
                "home_page": "https://numpy.org",
            }
        }
        sources = discover_change_sources("numpy", pypi_data=pypi_data)
        urls = [s.url for s in sources]
        assert "https://numpy.org" in urls
        assert not any("github.com" in u for u in urls)


# ---------------------------------------------------------------------------
# Tests: fetch_change_document
# ---------------------------------------------------------------------------

class TestFetchChangeDocument:

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_success_markdown(self, mock_get):
        content = "# Release Notes 3.0.0\n\n* Fixed critical bug in dataframe indexing."
        mock_get.return_value = _make_response(text=content, status_code=200)

        source = ChangeSource(
            package_name="pandas",
            url="https://example.org/release_notes.md",
            source_type=SOURCE_TYPE_RELEASE_NOTES,
            is_official=True,
        )

        doc = fetch_change_document(source)

        assert doc.package_name == "pandas"
        assert doc.source == source
        assert doc.content == content
        assert doc.title == "Release Notes 3.0.0"
        assert doc.is_success is True
        assert doc.fetch_error is None
        assert isinstance(doc.fetched_at, datetime)

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_success_html_title(self, mock_get):
        content = "<html><head><title>Pandas Migration Guide</title></head><body>Details</body></html>"
        mock_get.return_value = _make_response(text=content, status_code=200)

        source = ChangeSource(
            package_name="pandas",
            url="https://example.org/migration.html",
            source_type=SOURCE_TYPE_MIGRATION_GUIDE,
        )

        doc = fetch_change_document(source)

        assert doc.is_success is True
        assert doc.title == "Pandas Migration Guide"

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_404_error(self, mock_get):
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=404))
        mock_get.return_value = _make_response("", status_code=404, raise_for_status_exc=http_error)

        source = ChangeSource(package_name="foo", url="https://example.org/missing", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source)

        assert doc.is_success is False
        assert doc.fetch_error is not None
        assert "404" in doc.fetch_error

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_403_error(self, mock_get):
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=403))
        mock_get.return_value = _make_response("", status_code=403, raise_for_status_exc=http_error)

        source = ChangeSource(package_name="foo", url="https://example.org/forbidden", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source)

        assert doc.is_success is False
        assert "403" in doc.fetch_error

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_429_error(self, mock_get):
        http_error = requests.exceptions.HTTPError(response=MagicMock(status_code=429))
        mock_get.return_value = _make_response("", status_code=429, raise_for_status_exc=http_error)

        source = ChangeSource(package_name="foo", url="https://example.org/rate-limit", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source)

        assert doc.is_success is False
        assert "429" in doc.fetch_error

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()

        source = ChangeSource(package_name="foo", url="https://example.org/slow", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source, timeout=5.0)

        assert doc.is_success is False
        assert "timed out" in doc.fetch_error

    @patch("pyguardian.change_sources.requests.get")
    def test_fetch_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        source = ChangeSource(package_name="foo", url="https://example.org/down", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source)

        assert doc.is_success is False
        assert "connect" in doc.fetch_error.lower()

    @patch("pyguardian.change_sources.requests.get")
    def test_empty_document_handling(self, mock_get):
        mock_get.return_value = _make_response(text="   \n  ", status_code=200)

        source = ChangeSource(package_name="foo", url="https://example.org/empty", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source)

        assert doc.is_success is False
        assert "empty document" in doc.fetch_error.lower()

    def test_invalid_url_scheme_handling(self):
        source = ChangeSource(package_name="foo", url="file:///etc/passwd", source_type=SOURCE_TYPE_CHANGELOG)
        doc = fetch_change_document(source)

        assert doc.is_success is False
        assert "Only http/https supported" in doc.fetch_error
