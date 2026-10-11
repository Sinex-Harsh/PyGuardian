"""
test_change_analysis.py — Tests for API ↔ Change Matching (V0.2 Step 6).

Key testing principle:
    All tests are fully deterministic with zero live network or external tool dependencies.
"""

from pathlib import Path
import pytest
from packaging.version import Version

from pyguardian.change_analysis import (
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_BEHAVIOR_CHANGE,
    CHANGE_TYPE_DEPRECATED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    MATCH_TYPE_EXACT,
    MATCH_TYPE_PACKAGE,
    APIChangeMatch,
    DocumentedChange,
    extract_documented_changes,
    match_api_changes,
)
from pyguardian.change_sources import ChangeDocument, ChangeSource
from pyguardian.source import APIUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, package_name: str = "pandas", url: str = "https://example.org/changelog.md") -> ChangeDocument:
    source = ChangeSource(package_name=package_name, url=url, source_type="changelog")
    from datetime import datetime, timezone
    return ChangeDocument(
        package_name=package_name,
        source=source,
        title="Release Notes",
        content=text,
        fetched_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests: DocumentedChange Extraction
# ---------------------------------------------------------------------------

class TestChangeExtraction:

    def test_explicit_removed_api(self):
        text = "DataFrame was removed in version 3.0.0."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.package_name == "pandas"
        assert change.api_path == "pandas.DataFrame"
        assert change.change_type == CHANGE_TYPE_REMOVED
        assert change.version == Version("3.0.0")

    def test_explicit_deprecation(self):
        text = "read_csv is deprecated."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.api_path == "pandas.read_csv"
        assert change.change_type == CHANGE_TYPE_DEPRECATED

    def test_explicit_rename(self):
        text = "old_api was renamed to new_api."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.api_path == "pandas.old_api"
        assert change.change_type == CHANGE_TYPE_RENAMED

    def test_explicit_addition(self):
        text = "Added new_api."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.api_path == "pandas.new_api"
        assert change.change_type == CHANGE_TYPE_ADDED

    def test_behavior_change_without_api(self):
        text = "IO behavior changed significantly."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.api_path is None
        assert change.change_type == CHANGE_TYPE_BEHAVIOR_CHANGE
        assert change.confidence == "low"

    def test_package_qualified_api(self):
        text = "pandas.read_csv behavior changed."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.api_path == "pandas.read_csv"
        assert change.change_type == CHANGE_TYPE_BEHAVIOR_CHANGE

    def test_version_extraction_context(self):
        text = "Changes in version 3.0.0\nDataFrame was removed."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        assert len(changes) == 1
        change = changes[0]
        assert change.version == Version("3.0.0")
        assert change.api_path == "pandas.DataFrame"

    def test_ambiguous_text_no_false_positive(self):
        text = "The documentation changed in March."
        doc = _make_doc(text)
        changes = extract_documented_changes(doc)

        # Meta-doc line ignored, no false API change created
        assert changes == []


# ---------------------------------------------------------------------------
# Tests: API ↔ Change Matching
# ---------------------------------------------------------------------------

class TestAPIMatching:

    def test_exact_api_match(self):
        usage = APIUsage(
            package_name="pandas",
            api_path="pandas.read_csv",
            file_path=Path("app.py"),
            line_number=10,
            usage_type="call",
        )
        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.read_csv",
            change_type=CHANGE_TYPE_DEPRECATED,
            description="read_csv is deprecated.",
            source_url="https://example.org/docs",
            confidence="high",
        )

        matches = match_api_changes([usage], [change])

        assert len(matches) == 1
        match = matches[0]
        assert match.api_usage == usage
        assert match.documented_change == change
        assert match.match_type == MATCH_TYPE_EXACT
        assert match.confidence == "high"

    def test_package_level_change_match(self):
        usage = APIUsage(
            package_name="pandas",
            api_path="pandas.DataFrame",
            file_path=Path("analysis.py"),
            line_number=42,
            usage_type="call",
        )
        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path=None,
            change_type=CHANGE_TYPE_BEHAVIOR_CHANGE,
            description="IO behavior changed significantly.",
            source_url="https://example.org/docs",
            confidence="low",
        )

        matches = match_api_changes([usage], [change])

        assert len(matches) == 1
        match = matches[0]
        assert match.match_type == MATCH_TYPE_PACKAGE
        assert match.confidence == "low"
        assert match.documented_change.api_path is None

    def test_no_match(self):
        usage = APIUsage(
            package_name="pandas",
            api_path="pandas.concat",
            file_path=Path("app.py"),
            line_number=15,
            usage_type="call",
        )
        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.read_csv",
            change_type=CHANGE_TYPE_REMOVED,
            description="read_csv was removed.",
            source_url="https://example.org/docs",
        )

        matches = match_api_changes([usage], [change])
        assert matches == []

    def test_multiple_usages_matched_to_one_change(self):
        u1 = APIUsage("pandas", "pandas.DataFrame", Path("app.py"), 10, "call")
        u2 = APIUsage("pandas", "pandas.DataFrame", Path("analysis.py"), 25, "call")
        u3 = APIUsage("pandas", "pandas.DataFrame", Path("reports.py"), 40, "call")

        change = DocumentedChange(
            package_name="pandas",
            version=Version("3.0.0"),
            api_path="pandas.DataFrame",
            change_type=CHANGE_TYPE_REMOVED,
            description="DataFrame was removed.",
            source_url="https://example.org/docs",
        )

        matches = match_api_changes([u1, u2, u3], [change])

        assert len(matches) == 3
        files = [m.api_usage.file_path for m in matches]
        assert files == [Path("app.py"), Path("analysis.py"), Path("reports.py")]
        assert all(m.match_type == MATCH_TYPE_EXACT for m in matches)
