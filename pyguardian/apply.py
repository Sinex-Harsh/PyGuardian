"""
apply.py — Safe Patch Application Layer for PyGuardian.

Architecture:
ProposedPatch + ApprovalResult (approved=True)
      ↓
Pre-write Stale Source & Review Guard
      ↓
apply_patch(patch, approved)
      ↓
ApplyPatchResult (success: bool, file_path: Path, message: str, patch: ProposedPatch)

Responsibilities:
- Write approved ProposedPatch content safely to target file on disk.
- Require explicit human approval flag (approved=True).
- Re-read target file immediately before writing to verify zero disk modifications (stale source guard).
- Re-run deterministic review_patch() immediately before writing.
- Restrict writing strictly to patch.file_path.
- Perform NO shell commands, NO git operations, NO LLM calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pyguardian.diff_review import ReviewStatus, review_patch
from pyguardian.patch import ProposedPatch, is_security_sensitive_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ApplyPatchError(Exception):
    """Raised when patch application fails or safety pre-checks fail."""
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApplyPatchResult:
    """
    Outcome of attempting to apply a ProposedPatch to disk.

    Attributes:
        success:   True ONLY if file was successfully written.
        file_path: Path to the target file.
        message:   Human-readable summary of the application outcome.
        patch:     The ProposedPatch object processed.
    """

    success: bool
    file_path: Path
    message: str
    patch: ProposedPatch


# ---------------------------------------------------------------------------
# Application API
# ---------------------------------------------------------------------------

def apply_patch(patch: ProposedPatch, approved: bool = False) -> ApplyPatchResult:
    """
    Safely write a ProposedPatch to disk following explicit human approval and safety re-checks.

    Safety pre-conditions enforced immediately before writing:
    1. approved MUST be True.
    2. Target file MUST exist and NOT be security-sensitive.
    3. Target file on disk MUST match patch.original_content (stale source check).
    4. Deterministic review_patch(patch) MUST evaluate to PASS or REVIEW_REQUIRED (not REJECTED).

    Args:
        patch:    The ProposedPatch to apply.
        approved: Explicit human approval boolean flag.

    Returns:
        ApplyPatchResult indicating success status and outcome message.
    """
    if not isinstance(patch, ProposedPatch):
        raise ApplyPatchError(f"Expected ProposedPatch instance, got {type(patch).__name__}.")

    file_path = patch.file_path

    # Safety Guard 1: Require explicit approval
    if not approved:
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message="Patch application rejected: explicit human approval was not granted.",
            patch=patch,
        )

    # Safety Guard 2: Security-sensitive file restriction
    if is_security_sensitive_file(file_path):
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message=f"Security violation: Target file '{file_path.name}' is security-sensitive.",
            patch=patch,
        )

    # Safety Guard 3: Target file existence check
    if not file_path.exists() or not file_path.is_file():
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message=f"Target file missing or not a regular file: '{file_path}'.",
            patch=patch,
        )

    # Safety Guard 4: Immediate pre-write stale source check
    try:
        current_disk_content = file_path.read_text(encoding="utf-8")
    except Exception as err:
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message=f"Failed to read target file '{file_path}' prior to application: {err}",
            patch=patch,
        )

    if current_disk_content != patch.original_content:
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message=f"Stale source error: File '{file_path.name}' has been modified on disk since patch generation.",
            patch=patch,
        )

    # Safety Guard 5: Immediate pre-write deterministic review
    review = review_patch(patch)
    if review.status == ReviewStatus.REJECTED:
        reasons_str = "; ".join(review.review_reasons) if review.review_reasons else "Failed safety review"
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message=f"Safety review failed immediately before application: {reasons_str}",
            patch=patch,
        )

    # Execute File Mutation
    try:
        file_path.write_text(patch.proposed_content, encoding="utf-8")
    except Exception as err:
        return ApplyPatchResult(
            success=False,
            file_path=file_path,
            message=f"Failed to write proposed content to '{file_path}': {err}",
            patch=patch,
        )

    return ApplyPatchResult(
        success=True,
        file_path=file_path,
        message=f"Successfully applied proposed patch to '{file_path.name}'.",
        patch=patch,
    )
