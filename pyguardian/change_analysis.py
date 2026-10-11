"""
change_analysis.py — API ↔ Change Matching for PyGuardian.

Responsibilities:
- Parse raw change documentation into structured `DocumentedChange` objects.
- Normalize API paths, version numbers, and change types deterministically.
- Match project `APIUsage` records against `DocumentedChange` objects.
- Express connections as structured `APIChangeMatch` potential matches with confidence scoring.

Design Principles:
- Pure static / deterministic analysis: NO AI, LLM, RAG, or vector DBs.
- Safe heuristics: prefers `unknown` / `api_path = None` over false positives.
- Does NOT perform risk scoring or claim code is "definitely broken".
- Preserves full source provenance (`source_url`) on all matched changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from packaging.version import InvalidVersion, Version

from pyguardian.change_sources import ChangeDocument
from pyguardian.source import APIUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Change Types
# ---------------------------------------------------------------------------

CHANGE_TYPE_ADDED = "added"
CHANGE_TYPE_CHANGED = "changed"
CHANGE_TYPE_DEPRECATED = "deprecated"
CHANGE_TYPE_REMOVED = "removed"
CHANGE_TYPE_RENAMED = "renamed"
CHANGE_TYPE_BEHAVIOR_CHANGE = "behavior_change"
CHANGE_TYPE_MIGRATION = "migration"
CHANGE_TYPE_UNKNOWN = "unknown"

VALID_CHANGE_TYPES = {
    CHANGE_TYPE_ADDED,
    CHANGE_TYPE_CHANGED,
    CHANGE_TYPE_DEPRECATED,
    CHANGE_TYPE_REMOVED,
    CHANGE_TYPE_RENAMED,
    CHANGE_TYPE_BEHAVIOR_CHANGE,
    CHANGE_TYPE_MIGRATION,
    CHANGE_TYPE_UNKNOWN,
}

# ---------------------------------------------------------------------------
# Match Types
# ---------------------------------------------------------------------------

MATCH_TYPE_EXACT = "exact"
MATCH_TYPE_PACKAGE = "package"
MATCH_TYPE_UNRESOLVED = "unresolved"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentedChange:
    """
    Represents a normalized documented change for a package.

    Attributes:
        package_name: Canonical package name (e.g. 'pandas').
        version:      Target package version where change occurred, if known.
        api_path:     Fully qualified API path (e.g. 'pandas.DataFrame'), or None if package-wide.
        change_type:  Type of change (removed, deprecated, renamed, changed, added, etc.).
        description:  Human-readable description / line from documentation.
        source_url:   URL of the authoritative change source.
        confidence:   Confidence rating of extraction ('high', 'medium', 'low').
    """

    package_name: str
    version: Optional[Version]
    api_path: Optional[str]
    change_type: str
    description: str
    source_url: str
    confidence: str = "medium"


@dataclass(frozen=True)
class APIChangeMatch:
    """
    Represents a potential match between a project API usage and a documented change.

    Attributes:
        api_usage:         The APIUsage object from AST analysis.
        documented_change: The DocumentedChange object extracted from documentation.
        match_type:        Level of match ('exact', 'package', 'unresolved').
        confidence:        Confidence level of the match ('high', 'medium', 'low').
    """

    api_usage: APIUsage
    documented_change: DocumentedChange
    match_type: str
    confidence: str


# ---------------------------------------------------------------------------
# Extraction Regex Patterns
# ---------------------------------------------------------------------------

VERSION_HEADER_REGEX = re.compile(
    r"(?:version|v|release|what's new in|changes in|upgrading to)\s*(\d+\.\d+(?:\.\d+)?(?:[a-zA-Z0-9\.\-]+)?)",
    re.IGNORECASE,
)

VERSION_STANDALONE_REGEX = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")

META_DOCS_REGEX = re.compile(
    r"\b(documentation changed|doc changed|docs changed|removed example|removed docs|removed documentation|typo|fix typo|spelling)\b",
    re.IGNORECASE,
)

KEYWORD_PATTERNS = [
    (
        re.compile(
            r"\b(removed|no longer available|deleted|dropped|was removed|has been removed)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_REMOVED,
    ),
    (
        re.compile(
            r"\b(deprecated|deprecation|will be removed|is deprecated|has been deprecated)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_DEPRECATED,
    ),
    (
        re.compile(
            r"\b(renamed|rename|replaced by|was renamed)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_RENAMED,
    ),
    (
        re.compile(
            r"\b(added|introduced|new api|new feature)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_ADDED,
    ),
    (
        re.compile(
            r"\b(migration|upgrade instructions|migrate from|how to migrate)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_MIGRATION,
    ),
    (
        re.compile(
            r"\b(behavior changed|behaviour changed|changed behavior|now behaves|breaking change)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_BEHAVIOR_CHANGE,
    ),
    (
        re.compile(
            r"\b(changed|updated|modified)\b",
            re.IGNORECASE,
        ),
        CHANGE_TYPE_CHANGED,
    ),
]


def _extract_api_symbol(line: str, pkg_name: str) -> Optional[str]:
    """
    Extract candidate API symbol or method name from documentation line.
    """
    patterns = [
        # Backticked code symbols: `pandas.read_csv`, `read_csv`, `DataFrame`
        re.compile(r"`([a-zA-Z_][a-zA-Z0-9_\.]*)`"),
        # Package-qualified path in text: pandas.read_csv, requests.get
        re.compile(r"\b(" + re.escape(pkg_name) + r"\.[a-zA-Z_][a-zA-Z0-9_\.]*)\b"),
        # Function calls with parentheses: read_csv(), DataFrame()
        re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)"),
        # Symbol before change keyword: "DataFrame was removed", "read_csv is deprecated"
        re.compile(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:was|is|has been)\s+(?:removed|deprecated|renamed|added|changed)\b",
            re.IGNORECASE,
        ),
        # Symbol after action keyword: "Added new_api", "Added new_api."
        re.compile(
            r"\b(?:added|introduced|new api|renamed to)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(line)
        if match:
            sym = match.group(1).strip()
            if sym.lower() in {
                "the",
                "a",
                "an",
                "this",
                "that",
                "it",
                "new",
                "old",
                "io",
                "api",
                "version",
                "docs",
                "documentation",
            }:
                continue
            return sym

    return None


# ---------------------------------------------------------------------------
# Change Extraction Pipeline
# ---------------------------------------------------------------------------

def extract_documented_changes(
    doc: ChangeDocument | str,
    *,
    package_name: Optional[str] = None,
    source_url: Optional[str] = None,
) -> list[DocumentedChange]:
    """
    Extract normalized DocumentedChange records from a ChangeDocument or raw documentation text.

    Args:
        doc: ChangeDocument instance or raw text string.
        package_name: Optional package name override.
        source_url: Optional source URL override.

    Returns:
        List of DocumentedChange objects.
    """
    if isinstance(doc, ChangeDocument):
        content = doc.content
        pkg_name = package_name or doc.package_name
        url = source_url or doc.source.url
    else:
        content = str(doc)
        pkg_name = package_name or "unknown"
        url = source_url or ""

    if not content or not content.strip():
        return []

    changes: list[DocumentedChange] = []
    current_version: Optional[Version] = None

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    for line in lines:
        # Check for version markers in section headers / text
        ver_match = VERSION_HEADER_REGEX.search(line)
        if ver_match:
            v_str = ver_match.group(1).rstrip(".,:;)")
            try:
                current_version = Version(v_str)
            except InvalidVersion:
                pass
        elif VERSION_STANDALONE_REGEX.search(line) and ("version" in line.lower() or "release" in line.lower() or line.startswith("#")):
            m = VERSION_STANDALONE_REGEX.search(line)
            if m:
                v_str = m.group(1).rstrip(".,:;)")
                try:
                    current_version = Version(v_str)
                except InvalidVersion:
                    pass

        # Filter meta-documentation false positives
        if META_DOCS_REGEX.search(line):
            continue

        # Detect change keyword
        detected_change_type: Optional[str] = None
        for pattern, ctype in KEYWORD_PATTERNS:
            if pattern.search(line):
                detected_change_type = ctype
                break

        if not detected_change_type:
            continue

        # Extract API symbol if present
        api_symbol = _extract_api_symbol(line, pkg_name)
        api_path: Optional[str] = None
        confidence = "medium"

        if api_symbol:
            if api_symbol.startswith(f"{pkg_name}."):
                api_path = api_symbol
            elif "." not in api_symbol:
                api_path = f"{pkg_name}.{api_symbol}"
            else:
                api_path = api_symbol

            if detected_change_type in (
                CHANGE_TYPE_REMOVED,
                CHANGE_TYPE_DEPRECATED,
                CHANGE_TYPE_RENAMED,
                CHANGE_TYPE_ADDED,
            ):
                confidence = "high"
            else:
                confidence = "medium"
        else:
            api_path = None
            confidence = "low"

        changes.append(
            DocumentedChange(
                package_name=pkg_name,
                version=current_version,
                api_path=api_path,
                change_type=detected_change_type,
                description=line,
                source_url=url,
                confidence=confidence,
            )
        )

    return changes


# ---------------------------------------------------------------------------
# API ↔ Change Matching
# ---------------------------------------------------------------------------

def match_api_changes(
    api_usages: list[APIUsage],
    documented_changes: list[DocumentedChange],
) -> list[APIChangeMatch]:
    """
    Match AST API symbol usages against normalized documented package changes.

    Args:
        api_usages: List of APIUsage objects from source AST analysis.
        documented_changes: List of DocumentedChange objects from documentation analysis.

    Returns:
        List of APIChangeMatch objects.
    """
    matches: list[APIChangeMatch] = []

    for usage in api_usages:
        for change in documented_changes:
            # Match package name
            if usage.package_name.lower() != change.package_name.lower():
                continue

            if change.api_path is not None:
                # Rule 1: Exact API path match
                if usage.api_path == change.api_path:
                    confidence = "high" if change.confidence == "high" else "medium"
                    matches.append(
                        APIChangeMatch(
                            api_usage=usage,
                            documented_change=change,
                            match_type=MATCH_TYPE_EXACT,
                            confidence=confidence,
                        )
                    )
            else:
                # Rule 2: Package-level documented change match
                matches.append(
                    APIChangeMatch(
                        api_usage=usage,
                        documented_change=change,
                        match_type=MATCH_TYPE_PACKAGE,
                        confidence="low",
                    )
                )

    return matches
