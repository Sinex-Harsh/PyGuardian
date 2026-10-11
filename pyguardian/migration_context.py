"""
migration_context.py — Migration Evidence & Context Builder for PyGuardian.

Responsibilities:
- Build a focused, structured, deterministic context for reasoning about a MigrationCandidate.
- Extract bounded source code excerpts around affected line locations.
- Link relevant AST API usages, imports, documented changes, and change documents.
- Maintain strict context size discipline without reading entire repositories.
- Preserve full provenance (source_url, versions, risk, confidence, file locations).

Design Principles:
- Pure deterministic analysis: NO AI, LLM, RAG, embeddings, or code generation.
- 0 network requests, 0 code execution, 0 Git operations, 0 mutations.
- Treats source files strictly as data.
- Security-first: never reads credentials, private keys, or .env files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from packaging.version import Version

from pyguardian.change_analysis import DocumentedChange
from pyguardian.change_sources import ChangeDocument
from pyguardian.impact import normalize_package_name
from pyguardian.migration import MigrationCandidate, MigrationPlan
from pyguardian.pipeline import ProjectAnalysis
from pyguardian.source import APIUsage, ImportUsage

logger = logging.getLogger(__name__)

# Default number of lines before and after the affected line to include in source context
DEFAULT_CONTEXT_LINES: int = 8

# Disallowed file names and extensions for security
DISALLOWED_FILE_NAMES: set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

DISALLOWED_EXTENSIONS: set[str] = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceContext:
    """
    Focused source code excerpt around an affected API usage line.

    Attributes:
        file_path:  Path to the source file.
        start_line: 1-based start line of excerpt (inclusive).
        end_line:   1-based end line of excerpt (inclusive).
        content:    Code content string within [start_line, end_line].
    """

    file_path: Optional[Path]
    start_line: Optional[int]
    end_line: Optional[int]
    content: Optional[str]

    @property
    def line_count(self) -> int:
        """Total number of lines in this context excerpt."""
        if self.start_line is None or self.end_line is None:
            return 0
        return max(0, self.end_line - self.start_line + 1)


@dataclass(frozen=True)
class MigrationContext:
    """
    Focused, structured evidence context for reasoning about a single MigrationCandidate.

    Attributes:
        candidate:          The MigrationCandidate being evaluated.
        source_context:     Extracted source window around affected line, or None if unavailable/package-level.
        relevant_usages:    Tuple of relevant APIUsage records for the package / API.
        relevant_imports:   Tuple of relevant ImportUsage records for the package in source files.
        relevant_changes:   Tuple of relevant DocumentedChange records.
        relevant_documents: Tuple of relevant ChangeDocument records associated with candidate.
        project_path:       Root path of the project.
    """

    candidate: MigrationCandidate
    source_context: Optional[SourceContext]
    relevant_usages: tuple[APIUsage, ...] = ()
    relevant_imports: tuple[ImportUsage, ...] = ()
    relevant_changes: tuple[DocumentedChange, ...] = ()
    relevant_documents: tuple[ChangeDocument, ...] = ()
    project_path: Path = field(default_factory=lambda: Path("."))

    @property
    def package_name(self) -> str:
        """Name of the package for this candidate."""
        return self.candidate.package_name

    @property
    def affected_api(self) -> Optional[str]:
        """Affected API symbol path, if exact match."""
        return self.candidate.affected_api

    @property
    def current_version(self) -> Optional[Version]:
        """Baseline current package version."""
        return self.candidate.current_version

    @property
    def target_version(self) -> Optional[Version]:
        """Target upgrade package version."""
        return self.candidate.target_version

    @property
    def has_source_context(self) -> bool:
        """True if source excerpt is present."""
        return self.source_context is not None and self.source_context.content is not None


# ---------------------------------------------------------------------------
# Path Helper
# ---------------------------------------------------------------------------

def _paths_match(
    p1: Optional[Path | str],
    p2: Optional[Path | str],
    project_path: Optional[Path] = None,
) -> bool:
    """Check if two file paths refer to the same file location."""
    if p1 is None or p2 is None:
        return False
    path1 = Path(p1)
    path2 = Path(p2)
    if path1 == path2:
        return True
    s1 = path1.as_posix().rstrip("/")
    s2 = path2.as_posix().rstrip("/")
    if s1 == s2 or s1.endswith("/" + s2) or s2.endswith("/" + s1):
        return True
    try:
        resolved1 = (
            (project_path / path1 if project_path and not path1.is_absolute() else path1)
            .resolve()
        )
        resolved2 = (
            (project_path / path2 if project_path and not path2.is_absolute() else path2)
            .resolve()
        )
        return resolved1 == resolved2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Source Context Builder
# ---------------------------------------------------------------------------

def build_source_context(
    file_path: Optional[Path | str],
    line_number: Optional[int],
    *,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    project_path: Optional[Path | str] = None,
) -> Optional[SourceContext]:
    """
    Extract a bounded window of source code lines around an affected line.

    Args:
        file_path:     Path to the source file (absolute or relative to project_path).
        line_number:   1-based target line number.
        context_lines: Number of lines before and after to include (default 8).
        project_path:  Optional project root directory.

    Returns:
        SourceContext containing extracted lines, or None if unavailable.
    """
    if file_path is None or line_number is None:
        return None

    path_obj = Path(file_path)

    # Security guard: refuse sensitive credentials or env files
    if (
        path_obj.name.lower() in DISALLOWED_FILE_NAMES
        or path_obj.suffix.lower() in DISALLOWED_EXTENSIONS
    ):
        logger.warning("Refusing to read sensitive file: %s", path_obj)
        return None

    # Resolve target file path
    target_path: Path
    if path_obj.is_absolute():
        target_path = path_obj
    elif project_path is not None:
        target_path = Path(project_path) / path_obj
    else:
        target_path = path_obj

    if not target_path.exists() or not target_path.is_file():
        return None

    try:
        content = target_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        logger.warning("Could not read source file %s: %s", target_path, err)
        return None

    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines == 0 or line_number < 1 or line_number > total_lines:
        return None

    start_line = max(1, line_number - context_lines)
    end_line = min(total_lines, line_number + context_lines)

    extracted_lines = lines[start_line - 1 : end_line]
    extracted_content = "\n".join(extracted_lines)

    return SourceContext(
        file_path=path_obj,
        start_line=start_line,
        end_line=end_line,
        content=extracted_content,
    )


# ---------------------------------------------------------------------------
# Evidence Extractors
# ---------------------------------------------------------------------------

def extract_relevant_usages(
    candidate: MigrationCandidate,
    project_analysis: ProjectAnalysis,
) -> tuple[APIUsage, ...]:
    """
    Collect relevant APIUsage records from AST analysis for a candidate.

    For exact API candidates, exact API matches are prioritized.
    For package-level candidates, package-wide usages are gathered.
    """
    norm_pkg = normalize_package_name(candidate.package_name)
    all_api_usages = (
        project_analysis.source_analysis.api_usages
        if project_analysis.source_analysis
        else []
    )

    matched: list[APIUsage] = []
    seen: set[tuple[str, str, int, str]] = set()

    def _add_usage(u: APIUsage) -> None:
        key = (str(u.file_path), u.api_path, u.line_number, u.usage_type)
        if key not in seen:
            seen.add(key)
            matched.append(u)

    # 1. Include candidate's own direct usage from evidence if present
    if (
        candidate.evidence
        and candidate.evidence.impact
        and candidate.evidence.impact.evidence
        and candidate.evidence.impact.evidence.api_usage
    ):
        _add_usage(candidate.evidence.impact.evidence.api_usage)

    if candidate.is_exact_match and candidate.affected_api is not None:
        # Prioritize exact matches for affected_api
        # First from same file if candidate.file_path is known
        for u in all_api_usages:
            if (
                normalize_package_name(u.package_name) == norm_pkg
                and u.api_path == candidate.affected_api
            ):
                if candidate.file_path and _paths_match(
                    u.file_path, candidate.file_path, project_analysis.project_path
                ):
                    _add_usage(u)

        # Then other occurrences across the project
        for u in all_api_usages:
            if (
                normalize_package_name(u.package_name) == norm_pkg
                and u.api_path == candidate.affected_api
            ):
                _add_usage(u)
    else:
        # Package-level candidate: collect package-level usages without claiming exact match
        for u in all_api_usages:
            if normalize_package_name(u.package_name) == norm_pkg:
                _add_usage(u)

    return tuple(matched)


def extract_relevant_imports(
    candidate: MigrationCandidate,
    project_analysis: ProjectAnalysis,
) -> tuple[ImportUsage, ...]:
    """
    Collect relevant ImportUsage records for the candidate's package.

    If candidate specifies a file_path, restricts imports to that file.
    """
    norm_pkg = normalize_package_name(candidate.package_name)
    all_imports = (
        project_analysis.source_analysis.usages
        if project_analysis.source_analysis
        else []
    )

    matched: list[ImportUsage] = []
    seen: set[tuple[str, str, Optional[str], Optional[str], int]] = set()

    for imp in all_imports:
        if normalize_package_name(imp.package_name) != norm_pkg:
            continue

        if candidate.file_path is not None:
            if not _paths_match(
                imp.file_path, candidate.file_path, project_analysis.project_path
            ):
                continue

        key = (
            str(imp.file_path),
            imp.package_name,
            imp.imported_name,
            imp.alias,
            imp.line_number,
        )
        if key not in seen:
            seen.add(key)
            matched.append(imp)

    # Sort deterministically by line number
    matched.sort(key=lambda x: (str(x.file_path), x.line_number))
    return tuple(matched)


def extract_relevant_changes(
    candidate: MigrationCandidate,
    project_analysis: ProjectAnalysis,
) -> tuple[DocumentedChange, ...]:
    """
    Extract relevant DocumentedChange records already linked to this candidate or package.
    """
    norm_pkg = normalize_package_name(candidate.package_name)
    changes: list[DocumentedChange] = []
    seen: set[tuple[str, Optional[str], Optional[str], str]] = set()

    def _add_change(c: DocumentedChange) -> None:
        key = (
            normalize_package_name(c.package_name),
            str(c.version),
            c.api_path,
            c.description,
        )
        if key not in seen:
            seen.add(key)
            changes.append(c)

    # 1. Direct evidence from candidate
    if (
        candidate.evidence
        and candidate.evidence.impact
        and candidate.evidence.impact.evidence
        and candidate.evidence.impact.evidence.documented_change
    ):
        _add_change(candidate.evidence.impact.evidence.documented_change)

    # 2. Existing compatibility reports in analysis
    if project_analysis.compatibility_reports:
        for report in project_analysis.compatibility_reports:
            if normalize_package_name(report.package_name) != norm_pkg:
                continue
            for assessment in report.assessments:
                if assessment.evidence and assessment.evidence.documented_change:
                    doc_chg = assessment.evidence.documented_change
                    if candidate.affected_api is not None:
                        if (
                            doc_chg.api_path == candidate.affected_api
                            or assessment.description == candidate.description
                        ):
                            _add_change(doc_chg)
                    else:
                        _add_change(doc_chg)

    return tuple(changes)


def extract_relevant_documents(
    candidate: MigrationCandidate,
    project_analysis: ProjectAnalysis,
    documents: Optional[Sequence[ChangeDocument]] = None,
) -> tuple[ChangeDocument, ...]:
    """
    Extract relevant ChangeDocument objects already associated with candidate's package.
    """
    if not documents:
        return ()

    norm_pkg = normalize_package_name(candidate.package_name)
    relevant_docs: list[ChangeDocument] = []
    seen_urls: set[str] = set()

    for doc in documents:
        if normalize_package_name(doc.package_name) == norm_pkg:
            url = doc.source.url if doc.source else ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                relevant_docs.append(doc)
            elif not url and doc not in relevant_docs:
                relevant_docs.append(doc)

    return tuple(relevant_docs)


# ---------------------------------------------------------------------------
# Main Builder API
# ---------------------------------------------------------------------------

def build_migration_context(
    candidate: MigrationCandidate,
    project_analysis: ProjectAnalysis,
    *,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    documents: Optional[Sequence[ChangeDocument]] = None,
) -> MigrationContext:
    """
    Build a focused, structured MigrationContext for a single MigrationCandidate.

    Gathers the smallest relevant set of evidence (bounded source excerpt,
    API usages, imports, documented changes, official documents, metadata)
    needed to reason about the migration without scanning or dumping the entire repository.

    Args:
        candidate:        The MigrationCandidate being evaluated.
        project_analysis: Unified ProjectAnalysis from pipeline.
        context_lines:    Number of surrounding source lines (default 8).
        documents:        Optional sequence of authoritative ChangeDocument objects.

    Returns:
        A deterministic, immutable MigrationContext.
    """
    proj_path = project_analysis.project_path if project_analysis else Path(".")

    source_ctx: Optional[SourceContext] = None
    if candidate.file_path is not None and candidate.line_number is not None:
        source_ctx = build_source_context(
            file_path=candidate.file_path,
            line_number=candidate.line_number,
            context_lines=context_lines,
            project_path=proj_path,
        )

    usages = extract_relevant_usages(candidate, project_analysis)
    imports = extract_relevant_imports(candidate, project_analysis)
    changes = extract_relevant_changes(candidate, project_analysis)
    docs = extract_relevant_documents(candidate, project_analysis, documents=documents)

    return MigrationContext(
        candidate=candidate,
        source_context=source_ctx,
        relevant_usages=usages,
        relevant_imports=imports,
        relevant_changes=changes,
        relevant_documents=docs,
        project_path=proj_path,
    )


def build_migration_contexts(
    migration_plan: MigrationPlan,
    project_analysis: ProjectAnalysis,
    *,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    documents: Optional[Sequence[ChangeDocument]] = None,
) -> list[MigrationContext]:
    """
    Build structured MigrationContext records for all candidates in a MigrationPlan.

    Preserves exact candidate order.

    Args:
        migration_plan:   The MigrationPlan containing candidates.
        project_analysis: Unified ProjectAnalysis from pipeline.
        context_lines:    Number of surrounding source lines (default 8).
        documents:        Optional sequence of authoritative ChangeDocument objects.

    Returns:
        List of MigrationContext records corresponding to each candidate.
    """
    return [
        build_migration_context(
            candidate=candidate,
            project_analysis=project_analysis,
            context_lines=context_lines,
            documents=documents,
        )
        for candidate in migration_plan.candidates
    ]
