"""
patch.py — Safe Proposed Patch Generation Layer for PyGuardian.

Architecture:
ProjectMigrationPlan / PlannedMigration / MigrationRecommendation
      ↓
Structured SourceEdit
      ↓
Patch Generation (generate_patch)
      ↓
ProposedPatch (in-memory artifact: original content, proposed content, unified diff)
      ↓
PatchSet (optional aggregation)

Responsibilities:
- Convert approved migration intelligence + explicit structured source edits into in-memory proposed code patches.
- Anchor all patches to exact, verified original source file content.
- Produce deterministic standard unified diffs via Python's `difflib`.
- Strictly validate source files, edit ranges, security-sensitive filenames, and line numbers.
- Perform ZERO filesystem mutations, ZERO git operations, ZERO network/LLM calls.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from pyguardian.migration import MigrationCandidate
from pyguardian.migration_ai import MigrationRecommendation
from pyguardian.migration_planner import PlannedMigration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Security Sensitive Patterns
# ---------------------------------------------------------------------------

SECURITY_SENSITIVE_EXACT_NAMES: set[str] = {
    ".env",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials",
    "secrets",
}

SECURITY_SENSITIVE_EXTENSIONS: set[str] = {
    ".pem",
    ".key",
}

SECURITY_SENSITIVE_PREFIXES: tuple[str, ...] = (
    ".env",
    "credentials.",
    "secrets.",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "id_ecdsa",
)


def is_security_sensitive_file(file_path: Path) -> bool:
    """
    Check if a target file path corresponds to a security-sensitive file
    such as private keys, environment files, or credential stores.
    """
    name = file_path.name.lower()
    if name in SECURITY_SENSITIVE_EXACT_NAMES:
        return True
    if file_path.suffix.lower() in SECURITY_SENSITIVE_EXTENSIONS:
        return True
    for prefix in SECURITY_SENSITIVE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PatchGenerationError(Exception):
    """Raised when proposed patch generation or validation fails."""
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceEdit:
    """
    Represents an explicit, structured edit to be applied to a specific source file.

    Attributes:
        file_path:  Path to the target source file.
        start_line: 1-based start line number of the range to replace.
        end_line:   1-based end line number of the range to replace (inclusive).
        replacement: String replacement content for the specified range.
    """

    file_path: Path
    start_line: int
    end_line: int
    replacement: str

    def __post_init__(self) -> None:
        if not isinstance(self.file_path, Path):
            object.__setattr__(self, "file_path", Path(self.file_path))


@dataclass(frozen=True)
class ProposedPatch:
    """
    In-memory representation of a proposed source code patch.

    Attributes:
        file_path:             Target file path.
        original_content:      Exact original content of the target file at generation time.
        proposed_content:      Resulting file content if the patch were to be applied.
        diff:                  Deterministic unified diff (a/ -> b/).
        package_name:          Associated package name.
        affected_api:          Affected API symbol path, or None.
        description:           Human-readable description of the patch.
        source_recommendation: Matched AI MigrationRecommendation provenance, if available.
    """

    file_path: Path
    original_content: str
    proposed_content: str
    diff: str
    package_name: str
    affected_api: Optional[str]
    description: str
    source_recommendation: Optional[MigrationRecommendation] = None

    def __post_init__(self) -> None:
        if not isinstance(self.file_path, Path):
            object.__setattr__(self, "file_path", Path(self.file_path))

    @property
    def is_no_op(self) -> bool:
        """True if the proposed content is identical to the original content."""
        return self.original_content == self.proposed_content


@dataclass(frozen=True)
class PatchSet:
    """
    Container aggregating multiple ProposedPatch objects for project-level review.

    Attributes:
        patches: Tuple of ordered ProposedPatch objects.
    """

    patches: tuple[ProposedPatch, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.patches, tuple):
            object.__setattr__(self, "patches", tuple(self.patches))

    @property
    def patch_count(self) -> int:
        """Total number of proposed patches in the set."""
        return len(self.patches)

    @property
    def has_patches(self) -> bool:
        """True if the patch set contains at least one patch."""
        return len(self.patches) > 0

    @property
    def files_changed(self) -> tuple[Path, ...]:
        """Tuple of unique target file paths modified by the patches, preserving order."""
        seen: set[Path] = set()
        files: list[Path] = []
        for p in self.patches:
            if p.file_path not in seen:
                seen.add(p.file_path)
                files.append(p.file_path)
        return tuple(files)


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def validate_source_edit(source_edit: SourceEdit) -> bool:
    """
    Validate basic structural attributes of a SourceEdit object.
    Raises PatchGenerationError on invalid inputs.
    """
    if not isinstance(source_edit, SourceEdit):
        raise PatchGenerationError(f"Expected SourceEdit instance, got {type(source_edit).__name__}.")

    if is_security_sensitive_file(source_edit.file_path):
        raise PatchGenerationError(
            f"Security error: Cannot target security-sensitive file '{source_edit.file_path.name}'."
        )

    if source_edit.start_line < 1:
        raise PatchGenerationError(f"Invalid start_line {source_edit.start_line}: must be 1 or greater.")

    if source_edit.start_line > source_edit.end_line:
        raise PatchGenerationError(
            f"Invalid line range: start_line ({source_edit.start_line}) is greater than end_line ({source_edit.end_line})."
        )

    if source_edit.replacement is None or not isinstance(source_edit.replacement, str):
        raise PatchGenerationError("Replacement content must be a non-null string.")

    if len(source_edit.replacement) == 0:
        raise PatchGenerationError("Empty replacement content is rejected.")

    return True


def validate_patch(patch: ProposedPatch) -> bool:
    """
    Validate that a ProposedPatch is internally consistent and safe.
    Re-generates expected diff from original_content and proposed_content to ensure zero mismatch.
    """
    if not isinstance(patch, ProposedPatch):
        raise PatchGenerationError(f"Expected ProposedPatch instance, got {type(patch).__name__}.")

    if is_security_sensitive_file(patch.file_path):
        raise PatchGenerationError(f"Security error: Patch targets security-sensitive file '{patch.file_path.name}'.")

    # Verify diff matches original -> proposed
    expected_diff_lines = list(
        difflib.unified_diff(
            patch.original_content.splitlines(keepends=True),
            patch.proposed_content.splitlines(keepends=True),
            fromfile=f"a/{patch.file_path.as_posix()}",
            tofile=f"b/{patch.file_path.as_posix()}",
            lineterm="\n",
        )
    )
    expected_diff = "".join(expected_diff_lines)

    if patch.diff != expected_diff:
        raise PatchGenerationError(
            "Patch diff mismatch: stored diff does not correspond exactly to original_content -> proposed_content."
        )

    return True


# ---------------------------------------------------------------------------
# Core Patch Generator API
# ---------------------------------------------------------------------------

def generate_patch(
    target: Optional[Any] = None,
    source_edit: Optional[Any] = None,
    arg3: Optional[Any] = None,
    *,
    candidate: Optional[MigrationCandidate] = None,
    recommendation: Optional[MigrationRecommendation] = None,
    migration: Optional[PlannedMigration] = None,
    package_name: Optional[str] = None,
    affected_api: Optional[str] = None,
    description: Optional[str] = None,
) -> ProposedPatch:
    """
    Generate an in-memory ProposedPatch from a structured SourceEdit and migration metadata.

    Does NOT modify project files.
    Does NOT invoke LLM providers.
    Enforces security-sensitive path restrictions and line-range validation.

    Accepts multiple signature patterns:
    - generate_patch(migration: PlannedMigration, source_edit: SourceEdit)
    - generate_patch(candidate: MigrationCandidate, recommendation: MigrationRecommendation, source_edit: SourceEdit)
    - generate_patch(source_edit: SourceEdit, candidate=..., recommendation=...)

    Raises:
        PatchGenerationError: If source_edit is missing, unreadable, out of bounds, security-sensitive,
                              or if file path mismatch occurs.
    """
    actual_source_edit: Optional[SourceEdit] = None
    actual_candidate: Optional[MigrationCandidate] = candidate
    actual_recommendation: Optional[MigrationRecommendation] = recommendation
    actual_migration: Optional[PlannedMigration] = migration

    # Disambiguate positional parameters
    for arg in (target, source_edit, arg3):
        if arg is None:
            continue
        if isinstance(arg, SourceEdit):
            actual_source_edit = arg
        elif isinstance(arg, PlannedMigration):
            actual_migration = arg
        elif isinstance(arg, MigrationCandidate):
            actual_candidate = arg
        elif isinstance(arg, MigrationRecommendation):
            actual_recommendation = arg

    if actual_source_edit is None:
        raise PatchGenerationError(
            "Patch generation unavailable: structured SourceEdit is required. "
            "Free-form text replacement without an explicit source edit is rejected for safety."
        )

    # Validate structural fields of source_edit
    validate_source_edit(actual_source_edit)

    file_path = actual_source_edit.file_path

    # Check file existence and readability
    if not file_path.exists():
        raise PatchGenerationError(f"Target file not found: '{file_path}'.")
    if not file_path.is_file():
        raise PatchGenerationError(f"Target path is not a regular file: '{file_path}'.")

    try:
        original_content = file_path.read_text(encoding="utf-8")
    except Exception as err:
        raise PatchGenerationError(f"Failed to read source file '{file_path}': {err}") from err

    # Validate candidate file path match if candidate is present
    matched_candidate = actual_migration.candidate if actual_migration else actual_candidate
    if matched_candidate is not None and matched_candidate.file_path is not None:
        try:
            cand_file = matched_candidate.file_path.resolve()
            edit_file = file_path.resolve()
            if cand_file != edit_file:
                raise PatchGenerationError(
                    f"Candidate target file '{matched_candidate.file_path}' does not match "
                    f"SourceEdit file path '{file_path}'."
                )
        except OSError:
            # Fall back to string comparison if resolve fails
            if str(matched_candidate.file_path) != str(file_path):
                raise PatchGenerationError(
                    f"Candidate target file '{matched_candidate.file_path}' does not match "
                    f"SourceEdit file path '{file_path}'."
                )

    # Validate line numbers against file contents
    lines = original_content.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines == 0:
        raise PatchGenerationError(f"File '{file_path}' is empty (0 lines). Cannot generate patch.")

    start_line = actual_source_edit.start_line
    end_line = actual_source_edit.end_line

    if end_line > total_lines:
        raise PatchGenerationError(
            f"Invalid end_line {end_line}: source file '{file_path}' has only {total_lines} lines."
        )

    # Build proposed content in memory
    prefix_lines = lines[: start_line - 1]
    suffix_lines = lines[end_line :]

    replaced_block = lines[start_line - 1 : end_line]
    last_replaced_line = replaced_block[-1] if replaced_block else ""
    ending = ""
    if last_replaced_line.endswith("\r\n"):
        ending = "\r\n"
    elif last_replaced_line.endswith("\n"):
        ending = "\n"

    replacement_str = actual_source_edit.replacement
    if ending and not (replacement_str.endswith("\n") or replacement_str.endswith("\r\n")):
        replacement_str += ending

    repl_lines = replacement_str.splitlines(keepends=True)
    proposed_lines = prefix_lines + repl_lines + suffix_lines
    proposed_content = "".join(proposed_lines)

    # Extract provenance fields
    pkg_name = package_name or ""
    aff_api = affected_api
    desc = description or ""
    src_rec = actual_recommendation

    if actual_migration is not None:
        cand = actual_migration.candidate
        pkg_name = pkg_name or cand.package_name
        aff_api = aff_api or cand.affected_api
        if not desc:
            if actual_migration.primary_recommendation:
                desc = actual_migration.primary_recommendation.summary
            else:
                desc = cand.description
        if src_rec is None:
            src_rec = actual_migration.primary_recommendation

    elif actual_candidate is not None:
        pkg_name = pkg_name or actual_candidate.package_name
        aff_api = aff_api or actual_candidate.affected_api
        desc = desc or actual_candidate.description

    if actual_recommendation is not None:
        pkg_name = pkg_name or actual_recommendation.package_name
        if aff_api is None:
            aff_api = actual_recommendation.affected_api
        desc = desc or actual_recommendation.summary
        if src_rec is None:
            src_rec = actual_recommendation

    if not desc:
        desc = f"Proposed migration patch for {pkg_name or 'project'}"

    # Generate standard unified diff
    diff_lines = list(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            proposed_content.splitlines(keepends=True),
            fromfile=f"a/{file_path.as_posix()}",
            tofile=f"b/{file_path.as_posix()}",
            lineterm="\n",
        )
    )
    diff = "".join(diff_lines)

    patch = ProposedPatch(
        file_path=file_path,
        original_content=original_content,
        proposed_content=proposed_content,
        diff=diff,
        package_name=pkg_name,
        affected_api=aff_api,
        description=desc,
        source_recommendation=src_rec,
    )

    validate_patch(patch)
    return patch


def create_patch_set(patches: Sequence[ProposedPatch] = ()) -> PatchSet:
    """
    Construct a PatchSet container aggregating ProposedPatch items.
    """
    for p in patches:
        if not isinstance(p, ProposedPatch):
            raise PatchGenerationError(f"Expected ProposedPatch item, got {type(p).__name__}.")
    return PatchSet(patches=tuple(patches))
