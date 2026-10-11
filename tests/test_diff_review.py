"""
test_diff_review.py — Unit tests for Diff Review Layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyguardian.diff_review import (
    DiffReviewResult,
    ReviewFindingCode,
    ReviewStatus,
    review_patch,
)
from pyguardian.patch import ProposedPatch, SourceEdit, generate_patch


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "app.py"
    content = "import pandas as pd\n\ndef run(df, df2):\n    return df.append(df2)\n"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_valid_patch_passes_diff_review(sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=sample_file,
        start_line=4,
        end_line=4,
        replacement="    return pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit, package_name="pandas")
    result = review_patch(patch)

    assert result.status == ReviewStatus.PASS
    assert result.is_passed is True
    assert len(result.findings) == 0


def test_missing_target_file_yields_stale_source_rejected(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.py"
    file_path = tmp_path / "existing.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    edit = SourceEdit(file_path=file_path, start_line=1, end_line=1, replacement="x = 2")
    patch = generate_patch(source_edit=edit)

    # Delete the file after patch generation to simulate stale file removal
    file_path.unlink()

    result = review_patch(patch)
    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.STALE_SOURCE for f in result.findings)


def test_modified_file_on_disk_yields_stale_source_rejected(sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=sample_file,
        start_line=4,
        end_line=4,
        replacement="    return pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    # Modify file on disk after patch generation
    sample_file.write_text("import pandas as pd\n# Modified line\n", encoding="utf-8")

    result = review_patch(patch)
    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.STALE_SOURCE for f in result.findings)


def test_security_sensitive_path_yields_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "app.py"
    file_path.write_text("x = 1\n", encoding="utf-8")
    edit = SourceEdit(file_path=file_path, start_line=1, end_line=1, replacement="x = 2")
    patch = generate_patch(source_edit=edit)

    # Create a spoofed patch object targeting .env
    env_patch = ProposedPatch(
        file_path=tmp_path / ".env",
        original_content="SECRET=1\n",
        proposed_content="SECRET=2\n",
        diff=patch.diff,
        package_name="test",
        affected_api=None,
        description="test",
    )

    result = review_patch(env_patch)
    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.SECURITY_SENSITIVE_PATH for f in result.findings)


def test_stored_diff_mismatch_yields_rejected(sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=sample_file,
        start_line=4,
        end_line=4,
        replacement="    return pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    corrupted_patch = ProposedPatch(
        file_path=patch.file_path,
        original_content=patch.original_content,
        proposed_content=patch.proposed_content,
        diff="corrupted diff content",
        package_name=patch.package_name,
        affected_api=patch.affected_api,
        description=patch.description,
    )

    result = review_patch(corrupted_patch)
    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.DIFF_MISMATCH for f in result.findings)


def test_no_op_patch_yields_rejected(sample_file: Path) -> None:
    content = sample_file.read_text(encoding="utf-8")
    no_op_patch = ProposedPatch(
        file_path=sample_file,
        original_content=content,
        proposed_content=content,
        diff="",
        package_name="pandas",
        affected_api=None,
        description="no-op",
    )

    result = review_patch(no_op_patch)
    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.NO_CHANGES for f in result.findings)


def test_invalid_python_syntax_in_proposed_content_yields_rejected(sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=sample_file,
        start_line=4,
        end_line=4,
        replacement="    def invalid syntax ( ( (",
    )
    patch = generate_patch(source_edit=edit)
    result = review_patch(patch)

    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.INVALID_PYTHON_SYNTAX for f in result.findings)


def test_suspicious_eval_addition_yields_rejected(sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=sample_file,
        start_line=4,
        end_line=4,
        replacement="    return eval('df.append(df2)')",
    )
    patch = generate_patch(source_edit=edit)
    result = review_patch(patch)

    assert result.status == ReviewStatus.REJECTED
    assert any(f.code == ReviewFindingCode.SUSPICIOUS_MODIFICATION for f in result.findings)


def test_broad_line_deletion_scope_yields_review_required(tmp_path: Path) -> None:
    file_path = tmp_path / "long_file.py"
    content = "\n".join([f"x_{i} = {i}" for i in range(20)]) + "\n"
    file_path.write_text(content, encoding="utf-8")

    # Replace 20 lines with 1 line
    edit = SourceEdit(file_path=file_path, start_line=1, end_line=19, replacement="x_0 = 0")
    patch = generate_patch(source_edit=edit)
    result = review_patch(patch)

    assert result.status == ReviewStatus.REVIEW_REQUIRED
    assert any(f.code == ReviewFindingCode.BROAD_CHANGE_SCOPE for f in result.findings)
