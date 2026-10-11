"""
diff_review.py — Safe Deterministic Diff Review Layer for PyGuardian.

Architecture:
ProposedPatch
      ↓
review_patch(patch)
      ↓
DiffReviewResult (status: PASS / REVIEW_REQUIRED / REJECTED, findings, review_reasons)

Responsibilities:
- Perform deterministic pre-apply review of ProposedPatch artifacts.
- Enforce target file existence, readability, and security restrictions.
- Detect stale source files where disk content differs from patch original content.
- Validate stored patch diff against regenerated deterministic diff.
- Validate Python syntax using ast.parse() for .py target files.
- Reject no-op patches and detect suspicious/excessive code modifications.
- ZERO filesystem mutations, ZERO network/LLM calls, ZERO git operations.
"""

from __future__ import annotations

import ast
import difflib
import logging
from dataclasses import dataclass
from enum import Enum

from pyguardian.patch import ProposedPatch, is_security_sensitive_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Statuses
# ---------------------------------------------------------------------------

class ReviewStatus(str, Enum):
    """Overall review outcome classification for a proposed patch."""
    PASS = "pass"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


class ReviewFindingCode(str, Enum):
    """Specific finding classifications for diff review."""
    STALE_SOURCE = "stale_source"
    SECURITY_SENSITIVE_PATH = "security_sensitive_path"
    DIFF_MISMATCH = "diff_mismatch"
    NO_CHANGES = "no_changes"
    INVALID_PYTHON_SYNTAX = "invalid_python_syntax"
    BROAD_CHANGE_SCOPE = "broad_change_scope"
    SUSPICIOUS_MODIFICATION = "suspicious_modification"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewFinding:
    """
    Individual finding recorded during diff review.

    Attributes:
        code:     Categorized ReviewFindingCode.
        message:  Human-readable summary of the finding.
        severity: Severity rating ('error', 'warning', 'info').
    """

    code: ReviewFindingCode
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class DiffReviewResult:
    """
    Aggregated deterministic review outcome for a ProposedPatch.

    Attributes:
        patch:          The ProposedPatch object evaluated.
        status:         Evaluated ReviewStatus classification.
        findings:       Tuple of ReviewFinding objects.
        review_reasons: Tuple of human-readable reason strings explaining status.
    """

    patch: ProposedPatch
    status: ReviewStatus
    findings: tuple[ReviewFinding, ...] = ()
    review_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.findings, tuple):
            object.__setattr__(self, "findings", tuple(self.findings))
        if not isinstance(self.review_reasons, tuple):
            object.__setattr__(self, "review_reasons", tuple(self.review_reasons))

    @property
    def is_passed(self) -> bool:
        """True if diff review passed cleanly with no blocking issues."""
        return self.status == ReviewStatus.PASS

    @property
    def is_rejected(self) -> bool:
        """True if diff review rejected the patch."""
        return self.status == ReviewStatus.REJECTED


# ---------------------------------------------------------------------------
# Core Review API
# ---------------------------------------------------------------------------

SUSPICIOUS_CALL_NAMES: set[str] = {
    "eval",
    "exec",
    "os.system",
    "subprocess.Popen",
    "shutil.rmtree",
}


def _check_suspicious_additions(original_content: str, proposed_content: str) -> list[str]:
    """
    Check if proposed_content introduces dynamic code execution or file deletion constructs
    not present in original_content.
    """
    additions: list[str] = []
    for fn in SUSPICIOUS_CALL_NAMES:
        if fn in proposed_content and fn not in original_content:
            additions.append(fn)
    return additions


def review_patch(patch: ProposedPatch) -> DiffReviewResult:
    """
    Deterministically review a ProposedPatch for safety, integrity, and syntax.

    Does NOT modify files on disk.
    Does NOT call LLMs or external network APIs.

    Args:
        patch: The ProposedPatch to evaluate.

    Returns:
        DiffReviewResult containing status, findings, and review reasons.
    """
    if not isinstance(patch, ProposedPatch):
        raise TypeError(f"Expected ProposedPatch instance, got {type(patch).__name__}.")

    findings: list[ReviewFinding] = []
    reasons: list[str] = []

    file_path = patch.file_path

    # Check 1: Security-sensitive file path
    if is_security_sensitive_file(file_path):
        findings.append(
            ReviewFinding(
                code=ReviewFindingCode.SECURITY_SENSITIVE_PATH,
                message=f"Target file '{file_path.name}' is security-sensitive.",
                severity="error",
            )
        )
        reasons.append("Security-sensitive file path is blocked from patching.")

    # Check 2: Target file exists and is readable
    if not file_path.exists() or not file_path.is_file():
        findings.append(
            ReviewFinding(
                code=ReviewFindingCode.STALE_SOURCE,
                message=f"Target file '{file_path}' does not exist on disk.",
                severity="error",
            )
        )
        reasons.append("Target file missing on disk.")
    else:
        try:
            current_content = file_path.read_text(encoding="utf-8")
            if current_content != patch.original_content:
                findings.append(
                    ReviewFinding(
                        code=ReviewFindingCode.STALE_SOURCE,
                        message=f"Target file '{file_path.name}' has been modified on disk since patch generation.",
                        severity="error",
                    )
                )
                reasons.append("File content on disk differs from patch original_content (stale patch).")
        except Exception as err:
            findings.append(
                ReviewFinding(
                    code=ReviewFindingCode.STALE_SOURCE,
                    message=f"Failed to read target file '{file_path}': {err}",
                    severity="error",
                )
            )
            reasons.append("Target file unreadable.")

    # Check 3: Stored diff validation against regenerated diff
    expected_diff_lines = list(
        difflib.unified_diff(
            patch.original_content.splitlines(keepends=True),
            patch.proposed_content.splitlines(keepends=True),
            fromfile=f"a/{file_path.as_posix()}",
            tofile=f"b/{file_path.as_posix()}",
            lineterm="\n",
        )
    )
    expected_diff = "".join(expected_diff_lines)
    if patch.diff != expected_diff:
        findings.append(
            ReviewFinding(
                code=ReviewFindingCode.DIFF_MISMATCH,
                message="Stored unified diff does not match regenerated diff.",
                severity="error",
            )
        )
        reasons.append("Diff mismatch detected.")

    # Check 4: No-op patch check
    if patch.is_no_op or patch.original_content == patch.proposed_content:
        findings.append(
            ReviewFinding(
                code=ReviewFindingCode.NO_CHANGES,
                message="Proposed patch contains zero changes.",
                severity="error",
            )
        )
        reasons.append("No-op patch (0 changes).")

    # Check 5: Python syntax validation for .py files
    if file_path.suffix.lower() == ".py" and patch.proposed_content:
        try:
            ast.parse(patch.proposed_content, filename=str(file_path))
        except SyntaxError as exc:
            findings.append(
                ReviewFinding(
                    code=ReviewFindingCode.INVALID_PYTHON_SYNTAX,
                    message=f"Proposed patch produces Python syntax error at line {exc.lineno}: {exc.msg}",
                    severity="error",
                )
            )
            reasons.append(f"Invalid Python syntax: {exc.msg}")

    # Check 6: Suspicious modifications
    suspicious_added = _check_suspicious_additions(patch.original_content, patch.proposed_content)
    if suspicious_added:
        added_str = ", ".join(suspicious_added)
        findings.append(
            ReviewFinding(
                code=ReviewFindingCode.SUSPICIOUS_MODIFICATION,
                message=f"Patch introduces sensitive/suspicious constructs: {added_str}",
                severity="error",
            )
        )
        reasons.append(f"Suspicious constructs introduced ({added_str}).")

    # Check 7: Broad change scope
    orig_lines = len(patch.original_content.splitlines())
    prop_lines = len(patch.proposed_content.splitlines())
    if orig_lines > 10 and (orig_lines - prop_lines) > (orig_lines // 2):
        findings.append(
            ReviewFinding(
                code=ReviewFindingCode.BROAD_CHANGE_SCOPE,
                message=f"Patch deletes more than 50% of file lines ({orig_lines} -> {prop_lines}).",
                severity="warning",
            )
        )
        reasons.append("Broad change scope: excessive line deletions detected.")

    # Classify overall status
    has_errors = any(f.severity == "error" for f in findings)
    has_warnings = any(f.severity == "warning" for f in findings)

    if has_errors:
        status = ReviewStatus.REJECTED
    elif has_warnings:
        status = ReviewStatus.REVIEW_REQUIRED
    else:
        status = ReviewStatus.PASS

    return DiffReviewResult(
        patch=patch,
        status=status,
        findings=tuple(findings),
        review_reasons=tuple(reasons),
    )
