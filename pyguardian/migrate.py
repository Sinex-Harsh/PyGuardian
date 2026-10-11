"""
migrate.py — End-to-End Migration Workflow Orchestration Layer for PyGuardian.

Architecture:
PyGuardian
   ↓
analyze_project() -> build_migration_plan()
   ↓
Select candidate -> build_migration_context()
   ↓
analyze_migration() [LLM Provider or reasoning]
   ↓
Structured SourceEdit
   ↓
generate_patch() -> ProposedPatch
   ↓
review_patch() -> DiffReviewResult
   ↓
request_approval() -> ApprovalResult
   ↓
apply_patch() -> ApplyPatchResult
   ↓
run_project_tests() -> TestResult

Responsibilities:
- Orchestrate thin end-to-end migration pipeline.
- Reuse existing deterministic modules (analyze_project, build_migration_plan, generate_patch).
- Preserve existing structured SourceEdit safety boundary: LLMs NEVER write directly to disk.
- Fail safely at any step without modifying source files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from pyguardian.apply import ApplyPatchResult, apply_patch
from pyguardian.approval import ApprovalResult, request_approval
from pyguardian.diff_review import DiffReviewResult, ReviewStatus, review_patch
from pyguardian.llm import LLMProvider
from pyguardian.migration import MigrationCandidate, build_migration_plan
from pyguardian.migration_ai import MigrationRecommendation, analyze_migration
from pyguardian.migration_context import build_migration_context
from pyguardian.patch import PatchGenerationError, ProposedPatch, SourceEdit, generate_patch
from pyguardian.pipeline import analyze_project
from pyguardian.test_runner import TestResult, run_project_tests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow Result Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MigrationWorkflowResult:
    """
    Complete result object returned by the end-to-end migration workflow.

    Attributes:
        status:         Outcome status string:
                        'no_candidates', 'no_recommendation', 'no_structured_edit',
                        'review_rejected', 'user_rejected', 'apply_failed', 'completed'.
        message:        Human-readable summary.
        project_path:   Target project path analyzed.
        candidate:      Selected MigrationCandidate, if any.
        recommendation: Produced MigrationRecommendation, if any.
        patch:          Generated ProposedPatch, if any.
        review:         Produced DiffReviewResult, if any.
        approval:       Produced ApprovalResult, if any.
        apply_result:   Produced ApplyPatchResult, if any.
        test_result:    Produced TestResult, if any.
    """

    status: str
    message: str
    project_path: Path
    candidate: Optional[MigrationCandidate] = None
    recommendation: Optional[MigrationRecommendation] = None
    patch: Optional[ProposedPatch] = None
    review: Optional[DiffReviewResult] = None
    approval: Optional[ApprovalResult] = None
    apply_result: Optional[ApplyPatchResult] = None
    test_result: Optional[TestResult] = None

    @property
    def is_successful(self) -> bool:
        """True if patch was applied and tests passed (or were run)."""
        return self.status == "completed" and self.apply_result is not None and self.apply_result.success


# ---------------------------------------------------------------------------
# Main Orchestration API
# ---------------------------------------------------------------------------

def run_migration_workflow(
    project_path: Path,
    provider: Optional[LLMProvider] = None,
    custom_source_edit: Optional[SourceEdit] = None,
    interactive: bool = False,
    auto_approve: bool = False,
    run_tests: bool = True,
    input_func: Optional[Callable[[str], str]] = None,
    print_func: Optional[Callable[[str], None]] = None,
) -> MigrationWorkflowResult:
    """
    Execute the end-to-end PyGuardian dependency migration workflow.

    Workflow:
    1. Analyze project -> Build migration plan
    2. Select migration candidate
    3. Build migration context
    4. Query LLM provider for reasoning (or derive recommendation)
    5. Construct structured SourceEdit
    6. Generate ProposedPatch (in-memory diff)
    7. Perform deterministic Diff Review
    8. Request explicit Human Approval
    9. Apply patch to disk (only if approved and passed review)
    10. Run project tests (pytest)

    Args:
        project_path:       Path to Python project root directory.
        provider:           Optional LLMProvider for AI reasoning.
        custom_source_edit: Optional explicit SourceEdit override.
        interactive:        True to enable interactive console approval prompt.
        auto_approve:       True to auto-approve patch (CLI -y flag).
        run_tests:          True to run pytest after application.
        input_func:         Optional custom input getter function for testing.
        print_func:         Optional custom printer function for testing.

    Returns:
        MigrationWorkflowResult object capturing overall outcome and step artifacts.
    """
    path_obj = Path(project_path)

    # Step 1: Analyze project & build migration plan
    project_analysis = analyze_project(path_obj)
    migration_plan = build_migration_plan(project_analysis)

    if not migration_plan.has_candidates:
        return MigrationWorkflowResult(
            status="no_candidates",
            message="No migration candidates found.",
            project_path=path_obj,
        )

    # Step 2: Select candidate (prefer candidate matching custom_source_edit or exact file usage)
    selected_candidate: Optional[MigrationCandidate] = None
    if custom_source_edit is not None and custom_source_edit.file_path is not None:
        for cand in migration_plan.candidates:
            if cand.file_path is not None:
                try:
                    if cand.file_path.resolve() == custom_source_edit.file_path.resolve():
                        selected_candidate = cand
                        break
                except OSError:
                    pass

    if selected_candidate is None:
        for cand in migration_plan.candidates:
            if cand.file_path is not None and cand.line_number is not None:
                selected_candidate = cand
                break

    if selected_candidate is None:
        selected_candidate = migration_plan.candidates[0]

    # Step 3: Build migration context & query AI recommendation if provider available
    context = build_migration_context(candidate=selected_candidate, project_analysis=project_analysis)
    recommendation: Optional[MigrationRecommendation] = None

    if provider is not None:
        try:
            recommendation = analyze_migration(context, provider)
        except Exception as exc:
            logger.warning(f"AI migration reasoning failed: {exc}")
            return MigrationWorkflowResult(
                status="no_recommendation",
                message=f"Migration recommendation unavailable: {exc}",
                project_path=path_obj,
                candidate=selected_candidate,
            )
    else:
        # Construct fallback recommendation from candidate details
        rec_action = f"Update usage of {selected_candidate.affected_api or selected_candidate.package_name}"
        recommendation = MigrationRecommendation(
            package_name=selected_candidate.package_name,
            affected_api=selected_candidate.affected_api,
            summary=selected_candidate.description,
            reason="Dependency version change detected.",
            recommended_action=rec_action,
            proposed_replacement=None,
            confidence="medium",
            evidence_references=("CANDIDATE",),
            uncertainties=(),
        )

    # Step 4: Construct structured SourceEdit
    source_edit: Optional[SourceEdit] = custom_source_edit

    if source_edit is None:
        if (
            selected_candidate.file_path is not None
            and selected_candidate.line_number is not None
            and recommendation.proposed_replacement
            and recommendation.proposed_replacement.strip()
        ):
            source_edit = SourceEdit(
                file_path=selected_candidate.file_path,
                start_line=selected_candidate.line_number,
                end_line=selected_candidate.line_number,
                replacement=recommendation.proposed_replacement,
            )

    if source_edit is None:
        return MigrationWorkflowResult(
            status="no_structured_edit",
            message="Migration cannot be safely converted into a structured source edit. No files changed.",
            project_path=path_obj,
            candidate=selected_candidate,
            recommendation=recommendation,
        )

    # Step 5: Generate ProposedPatch
    try:
        patch = generate_patch(
            candidate=selected_candidate,
            recommendation=recommendation,
            source_edit=source_edit,
        )
    except PatchGenerationError as exc:
        return MigrationWorkflowResult(
            status="no_structured_edit",
            message=f"Patch generation failed: {exc}. No files changed.",
            project_path=path_obj,
            candidate=selected_candidate,
            recommendation=recommendation,
        )

    # Step 6: Review ProposedPatch
    review = review_patch(patch)
    if review.status == ReviewStatus.REJECTED:
        reasons_str = "; ".join(review.review_reasons) if review.review_reasons else "Failed safety review"
        return MigrationWorkflowResult(
            status="review_rejected",
            message=f"Safety review failed: {reasons_str}. No files changed.",
            project_path=path_obj,
            candidate=selected_candidate,
            recommendation=recommendation,
            patch=patch,
            review=review,
        )

    # Step 7: Request Approval
    approval = request_approval(
        review=review,
        interactive=interactive,
        auto_approve=auto_approve,
        input_func=input_func,
        print_func=print_func,
    )

    if not approval.approved:
        return MigrationWorkflowResult(
            status="user_rejected",
            message="Migration rejected. No files changed.",
            project_path=path_obj,
            candidate=selected_candidate,
            recommendation=recommendation,
            patch=patch,
            review=review,
            approval=approval,
        )

    # Step 8: Apply patch
    apply_res = apply_patch(patch, approved=True)
    if not apply_res.success:
        return MigrationWorkflowResult(
            status="apply_failed",
            message=f"Failed to apply patch: {apply_res.message}",
            project_path=path_obj,
            candidate=selected_candidate,
            recommendation=recommendation,
            patch=patch,
            review=review,
            approval=approval,
            apply_result=apply_res,
        )

    # Step 9: Run tests
    test_res: Optional[TestResult] = None
    if run_tests:
        test_res = run_project_tests(path_obj)

    return MigrationWorkflowResult(
        status="completed",
        message="Migration applied.",
        project_path=path_obj,
        candidate=selected_candidate,
        recommendation=recommendation,
        patch=patch,
        review=review,
        approval=approval,
        apply_result=apply_res,
        test_result=test_res,
    )
