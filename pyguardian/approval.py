"""
approval.py — Explicit Human Approval Layer for PyGuardian.

Architecture:
DiffReviewResult
      ↓
request_approval(review, interactive, auto_approve)
      ↓
ApprovalResult (approved: bool, patch: ProposedPatch, reason: str)

Responsibilities:
- Provide explicit human approval check before patch application.
- Guarantee default decision is ALWAYS REJECT (False).
- Reject automatic approvals unless explicitly enabled by caller options.
- Render interactive CLI approval prompt asking `Apply this patch? [y/N]:`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from pyguardian.diff_review import DiffReviewResult, ReviewStatus
from pyguardian.patch import ProposedPatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovalResult:
    """
    Result of a human approval decision for a proposed patch.

    Attributes:
        approved: True ONLY if explicitly approved by user or auto-approve flag.
        patch:    The ProposedPatch being evaluated.
        reason:   Human-readable explanation of approval decision.
    """

    approved: bool
    patch: ProposedPatch
    reason: str


# ---------------------------------------------------------------------------
# Approval API
# ---------------------------------------------------------------------------

def request_approval(
    review: DiffReviewResult,
    interactive: bool = False,
    auto_approve: bool = False,
    input_func: Optional[Callable[[str], str]] = None,
    print_func: Optional[Callable[[str], None]] = None,
) -> ApprovalResult:
    """
    Request human approval for a patch following deterministic safety review.

    Default decision is ALWAYS REJECT (approved=False).
    Pressing Enter or typing any string other than 'y' or 'yes' yields REJECT.

    Args:
        review:       DiffReviewResult from review_patch().
        interactive:  True to prompt user via console input.
        auto_approve: True to bypass interactive prompt (explicit CLI flag only).
        input_func:   Optional custom input function for testing/injection.
        print_func:   Optional custom print function for testing/injection.

    Returns:
        ApprovalResult indicating approved (bool), patch, and reason.
    """
    if not isinstance(review, DiffReviewResult):
        raise TypeError(f"Expected DiffReviewResult instance, got {type(review).__name__}.")

    patch = review.patch
    printer = print_func or print
    getter = input_func or input

    # Rule 1: Rejected review status cannot be approved
    if review.status == ReviewStatus.REJECTED:
        reasons_str = "; ".join(review.review_reasons) if review.review_reasons else "Failed safety review"
        return ApprovalResult(
            approved=False,
            patch=patch,
            reason=f"Patch rejected by safety review: {reasons_str}",
        )

    # Rule 2: Explicit auto-approval flag passed
    if auto_approve:
        return ApprovalResult(
            approved=True,
            patch=patch,
            reason="Explicit auto-approval flag enabled.",
        )

    # Rule 3: Non-interactive mode without auto_approve defaults to REJECT
    if not interactive:
        return ApprovalResult(
            approved=False,
            patch=patch,
            reason="Non-interactive mode defaults to REJECT without explicit approval flag.",
        )

    # Interactive CLI Prompt
    printer("\nProposed Migration Patch:")
    printer(f"File: {patch.file_path}")
    printer(f"Package: {patch.package_name}")
    printer(f"Affected API: {patch.affected_api or 'N/A'}")
    printer(f"Description: {patch.description}")
    printer("\n--- Unified Diff ---")
    printer(patch.diff)
    printer("--------------------")

    try:
        response = getter("Apply this patch? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            return ApprovalResult(
                approved=True,
                patch=patch,
                reason="User explicitly approved patch.",
            )
        else:
            return ApprovalResult(
                approved=False,
                patch=patch,
                reason="User rejected patch.",
            )
    except (EOFError, KeyboardInterrupt):
        return ApprovalResult(
            approved=False,
            patch=patch,
            reason="Approval prompt cancelled by user.",
        )
