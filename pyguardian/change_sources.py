"""
change_sources.py — Change-Source Intelligence for PyGuardian.

Responsibilities:
- Discover authoritative documentation sources (changelogs, release notes, migration guides)
  from PyPI package metadata.
- Classify candidate change sources by source type and official status.
- Retrieve change documentation over HTTP safely and deterministically without arbitrary web crawling.
- Represent candidate change sources (`ChangeSource`) and fetched documents (`ChangeDocument`)
  as structured, immutable dataclasses.

Design Principles:
- Does NOT execute or crawl arbitrary websites or search engines.
- Does NOT perform AI/LLM/RAG processing, vector embeddings, or code patch generation.
- Treats retrieved documentation strictly as untrusted text.
- Never raises exceptions during documentation retrieval; returns structured error state instead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from pyguardian.packages import REQUEST_TIMEOUT_SECONDS, fetch_pypi_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source Types
# ---------------------------------------------------------------------------

SOURCE_TYPE_MIGRATION_GUIDE = "migration_guide"
SOURCE_TYPE_RELEASE_NOTES = "release_notes"
SOURCE_TYPE_CHANGELOG = "changelog"
SOURCE_TYPE_DOCUMENTATION = "documentation"
SOURCE_TYPE_REPOSITORY = "repository"

VALID_SOURCE_TYPES = {
    SOURCE_TYPE_MIGRATION_GUIDE,
    SOURCE_TYPE_RELEASE_NOTES,
    SOURCE_TYPE_CHANGELOG,
    SOURCE_TYPE_DOCUMENTATION,
    SOURCE_TYPE_REPOSITORY,
}

SOURCE_TYPE_PRIORITY = {
    SOURCE_TYPE_MIGRATION_GUIDE: 1,
    SOURCE_TYPE_RELEASE_NOTES: 2,
    SOURCE_TYPE_CHANGELOG: 3,
    SOURCE_TYPE_DOCUMENTATION: 4,
    SOURCE_TYPE_REPOSITORY: 5,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChangeSource:
    """
    Represents a candidate location for authoritative change documentation.

    Attributes:
        package_name: Canonical name of the package.
        url:          Absolute URL of the change source.
        source_type:  Type of source (migration_guide, release_notes, changelog, documentation, repository).
        is_official:  True if the source is from authoritative metadata (e.g. PyPI metadata).
    """

    package_name: str
    url: str
    source_type: str
    is_official: bool = True


@dataclass(frozen=True)
class ChangeDocument:
    """
    Represents retrieved documentation content from a ChangeSource.

    Attributes:
        package_name: Canonical name of the package.
        source:       The ChangeSource from which this document was fetched.
        title:        Extracted title or header of the document, if available.
        content:      Raw text or HTML content retrieved from the URL.
        fetched_at:   Timestamp when the fetch was performed (UTC).
        fetch_error:  Human-readable error description if retrieval failed.
    """

    package_name: str
    source: ChangeSource
    title: Optional[str]
    content: str
    fetched_at: datetime
    fetch_error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """True if retrieval succeeded with non-empty content and no error."""
        return self.fetch_error is None and bool(self.content.strip())


# ---------------------------------------------------------------------------
# Source Discovery & Classification
# ---------------------------------------------------------------------------

def classify_source_type(key: str, url: str) -> str:
    """
    Determine the source_type for a given project metadata URL key and URL string.
    """
    key_lower = key.lower().strip()
    url_lower = url.lower().strip()

    # Explicit Migration Guide
    if any(kw in key_lower for kw in ["migration", "upgrade guide", "upgrading"]):
        return SOURCE_TYPE_MIGRATION_GUIDE

    # Explicit Release Notes
    if any(kw in key_lower for kw in ["release note", "release-note", "releases", "what's new", "whats new"]):
        return SOURCE_TYPE_RELEASE_NOTES

    # Explicit Changelog
    if any(kw in key_lower for kw in ["changelog", "change log", "changes", "history", "news"]):
        return SOURCE_TYPE_CHANGELOG

    # Explicit Repository / Source Code
    if any(kw in key_lower for kw in ["repository", "source code", "source", "code", "github", "gitlab", "bitbucket"]):
        return SOURCE_TYPE_REPOSITORY

    # Explicit Documentation or Homepage
    if any(kw in key_lower for kw in ["documentation", "docs", "doc", "homepage", "home"]):
        return SOURCE_TYPE_DOCUMENTATION

    # Fallback inspection of URL path if key is ambiguous
    if any(kw in url_lower for kw in ["migration", "upgrade"]):
        return SOURCE_TYPE_MIGRATION_GUIDE
    if any(kw in url_lower for kw in ["release-notes", "releases", "whatsnew"]):
        return SOURCE_TYPE_RELEASE_NOTES
    if any(kw in url_lower for kw in ["changelog", "changes", "history"]):
        return SOURCE_TYPE_CHANGELOG
    if any(domain in url_lower for domain in ["github.com", "gitlab.com", "bitbucket.org"]):
        return SOURCE_TYPE_REPOSITORY

    return SOURCE_TYPE_DOCUMENTATION


def _normalize_url(url: str) -> str:
    """Strip trailing slashes for clean deduplication."""
    return url.strip().rstrip("/")


def discover_change_sources(
    package_name: str,
    *,
    pypi_data: Optional[dict] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> list[ChangeSource]:
    """
    Discover authoritative candidate change sources for a package using PyPI metadata.

    Args:
        package_name: Canonical package name (e.g. "pandas").
        pypi_data:    Optional PyPI JSON metadata dictionary. If not provided,
                      it will be fetched via `fetch_pypi_json`.
        timeout:      Network timeout in seconds if PyPI metadata fetch is needed.

    Returns:
        List of ChangeSource objects sorted by authoritativeness:
        1. migration_guide
        2. release_notes
        3. changelog
        4. documentation
        5. repository
    """
    if pypi_data is None:
        data, error = fetch_pypi_json(package_name, timeout=timeout)
        if error or data is None:
            logger.warning("Could not fetch PyPI metadata for source discovery: %s", error)
            return []
        pypi_data = data

    info_dict = pypi_data.get("info", {})
    if not isinstance(info_dict, dict):
        return []

    canonical_name = info_dict.get("name", package_name)
    raw_urls: list[tuple[str, str]] = []

    # Process project_urls
    project_urls = info_dict.get("project_urls")
    if isinstance(project_urls, dict):
        for key, val in project_urls.items():
            if isinstance(key, str) and isinstance(val, str) and val.strip():
                raw_urls.append((key.strip(), val.strip()))

    # Process home_page
    home_page = info_dict.get("home_page")
    if isinstance(home_page, str) and home_page.strip():
        raw_urls.append(("Homepage", home_page.strip()))

    seen_urls: set[str] = set()
    sources: list[ChangeSource] = []

    for key, url in raw_urls:
        if not (url.startswith("http://") or url.startswith("https://")):
            continue

        norm_url = _normalize_url(url)
        if norm_url in seen_urls:
            continue
        seen_urls.add(norm_url)

        stype = classify_source_type(key, url)
        sources.append(
            ChangeSource(
                package_name=canonical_name,
                url=url,
                source_type=stype,
                is_official=True,
            )
        )

    # Sort sources by priority
    sources.sort(key=lambda s: SOURCE_TYPE_PRIORITY.get(s.source_type, 99))
    return sources


# ---------------------------------------------------------------------------
# Document Retrieval
# ---------------------------------------------------------------------------

def _extract_document_title(content: str) -> Optional[str]:
    """
    Extract a simple document title from HTML <title> tag or Markdown H1.
    Does not use heavy HTML/DOM parsers.
    """
    html_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if html_match:
        clean_title = html_match.group(1).strip()
        if clean_title:
            return clean_title

    for line in content.splitlines():
        line_str = line.strip()
        if line_str.startswith("# "):
            return line_str[2:].strip()

    return None


def fetch_change_document(
    source: ChangeSource,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> ChangeDocument:
    """
    Retrieve document content for a given ChangeSource over HTTP.

    Does not raise network exceptions. Returns a ChangeDocument containing
    the content on success or human-readable fetch_error on failure.

    Args:
        source:  The ChangeSource candidate to fetch.
        timeout: Network timeout in seconds.

    Returns:
        A ChangeDocument object.
    """
    now = datetime.now(timezone.utc)

    if not (source.url.startswith("http://") or source.url.startswith("https://")):
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=f"Invalid URL schema for source '{source.url}'. Only http/https supported.",
        )

    try:
        response = requests.get(
            source.url,
            timeout=timeout,
            headers={"User-Agent": "PyGuardian/0.2 (Change-Source Intelligence)"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        msg = f"Request to '{source.url}' timed out after {timeout}s."
        logger.warning(msg)
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=msg,
        )
    except requests.exceptions.ConnectionError:
        msg = f"Could not connect to documentation host at '{source.url}'."
        logger.warning(msg)
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=msg,
        )
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 404:
            msg = f"Documentation resource not found (HTTP 404) at '{source.url}'."
        elif status_code == 403:
            msg = f"Access forbidden (HTTP 403) at '{source.url}'."
        elif status_code == 429:
            msg = f"Rate limit exceeded (HTTP 429) at '{source.url}'."
        else:
            msg = f"HTTP {status_code} error while fetching documentation from '{source.url}'."
        logger.warning(msg)
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=msg,
        )
    except requests.exceptions.RequestException as exc:
        msg = f"Network error while fetching documentation from '{source.url}': {exc}"
        logger.error(msg)
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=msg,
        )

    try:
        content_text = response.text or ""
    except Exception as exc:
        msg = f"Failed to decode response content from '{source.url}': {exc}"
        logger.error(msg)
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=msg,
        )

    if not content_text.strip():
        msg = f"Retrieved empty document from '{source.url}'."
        logger.warning(msg)
        return ChangeDocument(
            package_name=source.package_name,
            source=source,
            title=None,
            content="",
            fetched_at=now,
            fetch_error=msg,
        )

    title = _extract_document_title(content_text)
    return ChangeDocument(
        package_name=source.package_name,
        source=source,
        title=title,
        content=content_text,
        fetched_at=now,
        fetch_error=None,
    )
