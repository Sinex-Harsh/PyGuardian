"""
test_apply.py — Unit tests for Safe Patch Application Layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyguardian.apply import ApplyPatchResult, apply_patch
from pyguardian.patch import ProposedPatch, SourceEdit, generate_patch


@pytest.fixture
def target_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "data.py"
    file_path.write_text("def process(df, df2):\n    return df.append(df2)\n", encoding="utf-8")
    return file_path


def test_apply_patch_unapproved_fails(target_file: Path) -> None:
    edit = SourceEdit(file_path=target_file, start_line=2, end_line=2, replacement="    return pd.concat([df, df2])")
    patch = generate_patch(source_edit=edit)

    res = apply_patch(patch, approved=False)
    assert res.success is False
    assert "explicit human approval was not granted" in res.message


def test_apply_patch_approved_writes_proposed_content(target_file: Path) -> None:
    edit = SourceEdit(file_path=target_file, start_line=2, end_line=2, replacement="    return pd.concat([df, df2])")
    patch = generate_patch(source_edit=edit)

    res = apply_patch(patch, approved=True)
    assert res.success is True
    assert target_file.read_text(encoding="utf-8") == patch.proposed_content


def test_pre_write_stale_source_check_catches_disk_modification(target_file: Path) -> None:
    edit = SourceEdit(file_path=target_file, start_line=2, end_line=2, replacement="    return pd.concat([df, df2])")
    patch = generate_patch(source_edit=edit)

    # Modify file on disk immediately before applying
    target_file.write_text("def process(df, df2):\n    # Stale change\n", encoding="utf-8")

    res = apply_patch(patch, approved=True)
    assert res.success is False
    assert "Stale source error" in res.message


def test_security_sensitive_file_application_is_blocked(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=val\n", encoding="utf-8")
    patch = ProposedPatch(
        file_path=env_file,
        original_content="KEY=val\n",
        proposed_content="KEY=new\n",
        diff="--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-KEY=val\n+KEY=new\n",
        package_name="test",
        affected_api=None,
        description="test",
    )

    res = apply_patch(patch, approved=True)
    assert res.success is False
    assert "Security violation" in res.message


def test_missing_file_application_fails(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.py"
    patch = ProposedPatch(
        file_path=missing_path,
        original_content="x = 1\n",
        proposed_content="x = 2\n",
        diff="",
        package_name="test",
        affected_api=None,
        description="test",
    )

    res = apply_patch(patch, approved=True)
    assert res.success is False
    assert "missing" in res.message.lower()


def test_unrelated_files_remain_untouched(target_file: Path, tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.py"
    unrelated_content = "import os\n"
    unrelated.write_text(unrelated_content, encoding="utf-8")

    edit = SourceEdit(file_path=target_file, start_line=2, end_line=2, replacement="    return pd.concat([df, df2])")
    patch = generate_patch(source_edit=edit)

    apply_patch(patch, approved=True)

    assert unrelated.read_text(encoding="utf-8") == unrelated_content


def test_failed_review_blocks_application(target_file: Path) -> None:
    edit = SourceEdit(file_path=target_file, start_line=2, end_line=2, replacement="    def invalid syntax (((")
    patch = generate_patch(source_edit=edit)

    res = apply_patch(patch, approved=True)
    assert res.success is False
    assert "Safety review failed" in res.message
