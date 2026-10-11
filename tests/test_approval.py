"""
test_approval.py — Unit tests for Human Approval Layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyguardian.approval import ApprovalResult, request_approval
from pyguardian.diff_review import DiffReviewResult, ReviewStatus, review_patch
from pyguardian.patch import SourceEdit, generate_patch


@pytest.fixture
def sample_review(tmp_path: Path) -> DiffReviewResult:
    file_path = tmp_path / "main.py"
    file_path.write_text("def run():\n    pass\n", encoding="utf-8")
    edit = SourceEdit(file_path=file_path, start_line=2, end_line=2, replacement="    return 42")
    patch = generate_patch(source_edit=edit)
    return review_patch(patch)


def test_rejected_review_cannot_be_approved(sample_review: DiffReviewResult, tmp_path: Path) -> None:
    # Delete file to force review rejection
    sample_review.patch.file_path.unlink()
    rejected_review = review_patch(sample_review.patch)

    res = request_approval(rejected_review, interactive=True, auto_approve=True)
    assert res.approved is False
    assert "rejected" in res.reason.lower()


def test_non_interactive_defaults_to_rejection(sample_review: DiffReviewResult) -> None:
    res = request_approval(sample_review, interactive=False, auto_approve=False)
    assert res.approved is False
    assert "non-interactive" in res.reason.lower()


def test_auto_approve_flag_grants_approval(sample_review: DiffReviewResult) -> None:
    res = request_approval(sample_review, interactive=False, auto_approve=True)
    assert res.approved is True
    assert "auto-approval" in res.reason.lower()


def test_interactive_yes_response_grants_approval(sample_review: DiffReviewResult) -> None:
    printed: list[str] = []

    def mock_print(msg: str) -> None:
        printed.append(msg)

    def mock_input(prompt: str) -> str:
        return "y"

    res = request_approval(
        sample_review,
        interactive=True,
        auto_approve=False,
        input_func=mock_input,
        print_func=mock_print,
    )
    assert res.approved is True
    assert any("--- Unified Diff ---" in line for line in printed)


def test_interactive_full_yes_word_grants_approval(sample_review: DiffReviewResult) -> None:
    res = request_approval(
        sample_review,
        interactive=True,
        auto_approve=False,
        input_func=lambda _: "YES",
        print_func=lambda _: None,
    )
    assert res.approved is True


def test_interactive_enter_key_defaults_to_rejection(sample_review: DiffReviewResult) -> None:
    res = request_approval(
        sample_review,
        interactive=True,
        auto_approve=False,
        input_func=lambda _: "",  # User pressed Enter
        print_func=lambda _: None,
    )
    assert res.approved is False
    assert "rejected" in res.reason.lower()


def test_interactive_no_response_rejects(sample_review: DiffReviewResult) -> None:
    res = request_approval(
        sample_review,
        interactive=True,
        auto_approve=False,
        input_func=lambda _: "n",
        print_func=lambda _: None,
    )
    assert res.approved is False


def test_interactive_prompt_cancellation_rejects(sample_review: DiffReviewResult) -> None:
    def mock_input(_: str) -> str:
        raise KeyboardInterrupt()

    res = request_approval(
        sample_review,
        interactive=True,
        auto_approve=False,
        input_func=mock_input,
        print_func=lambda _: None,
    )
    assert res.approved is False
    assert "cancelled" in res.reason.lower()
