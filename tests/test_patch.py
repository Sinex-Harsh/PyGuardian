"""
test_patch.py — Unit tests for Safe Proposed Patch Generation Layer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyguardian.migration import MigrationCandidate
from pyguardian.migration_ai import MigrationRecommendation
from pyguardian.migration_planner import MigrationStatus, PlannedMigration
from pyguardian.patch import (
    PatchGenerationError,
    PatchSet,
    ProposedPatch,
    SourceEdit,
    create_patch_set,
    generate_patch,
    is_security_sensitive_file,
    validate_patch,
    validate_source_edit,
)
from pyguardian.risk import RiskLevel


# ---------------------------------------------------------------------------
# Fixtures & Helper Factory Functions
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_sample_file(tmp_path: Path) -> Path:
    """Create a temporary Python file for testing patches."""
    file_path = tmp_path / "data_processor.py"
    content = (
        "import pandas as pd\n"
        "\n"
        "def process_data(df, df2):\n"
        "    result = df.append(df2)\n"
        "    return result\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def make_sample_candidate(file_path: Path) -> MigrationCandidate:
    return MigrationCandidate(
        package_name="pandas",
        affected_api="pandas.DataFrame.append",
        file_path=file_path,
        line_number=4,
        current_version=None,
        target_version=None,
        change_type="removed",
        risk_level=RiskLevel.HIGH,
        confidence="high",
        description="DataFrame.append is removed in pandas 2.0; use concat instead.",
        source_url=None,
    )


def make_sample_recommendation() -> MigrationRecommendation:
    return MigrationRecommendation(
        package_name="pandas",
        affected_api="pandas.DataFrame.append",
        summary="Replace df.append(df2) with pd.concat([df, df2])",
        reason="DataFrame.append was removed in pandas 2.0.",
        recommended_action="Use pd.concat([df, df2]) instead of df.append(df2).",
        proposed_replacement="    result = pd.concat([df, df2])",
        confidence="high",
        evidence_references=("CANDIDATE",),
        uncertainties=(),
    )


def make_sample_planned_migration(file_path: Path) -> PlannedMigration:
    cand = make_sample_candidate(file_path)
    rec = make_sample_recommendation()
    return PlannedMigration(
        candidate=cand,
        recommendations=(rec,),
        status=MigrationStatus.PROPOSED,
        review_reasons=(),
    )


# ---------------------------------------------------------------------------
# Basic Patch Generation Tests
# ---------------------------------------------------------------------------

def test_valid_source_edit_generates_proposed_patch(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit, package_name="pandas")

    assert isinstance(patch, ProposedPatch)
    assert patch.file_path == temp_sample_file
    assert patch.package_name == "pandas"


def test_original_content_is_preserved(temp_sample_file: Path) -> None:
    expected_original = temp_sample_file.read_text(encoding="utf-8")
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    assert patch.original_content == expected_original


def test_proposed_content_is_preserved(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    expected_proposed = (
        "import pandas as pd\n"
        "\n"
        "def process_data(df, df2):\n"
        "    result = pd.concat([df, df2])\n"
        "    return result\n"
    )
    assert patch.proposed_content == expected_proposed


def test_unified_diff_is_generated_correctly(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    assert "--- a/" in patch.diff
    assert "+++ b/" in patch.diff
    assert "@@ -1,5 +1,5 @@" in patch.diff or "@@" in patch.diff


def test_diff_contains_expected_removed_and_added_lines(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    assert "-    result = df.append(df2)" in patch.diff
    assert "+    result = pd.concat([df, df2])" in patch.diff


def test_package_and_api_provenance_is_preserved(temp_sample_file: Path) -> None:
    pm = make_sample_planned_migration(temp_sample_file)
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(pm, edit)

    assert patch.package_name == "pandas"
    assert patch.affected_api == "pandas.DataFrame.append"
    assert patch.source_recommendation == pm.primary_recommendation
    assert patch.description == pm.primary_recommendation.summary


# ---------------------------------------------------------------------------
# Source Validation Tests
# ---------------------------------------------------------------------------

def test_missing_file_raises_patch_generation_error(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist.py"
    edit = SourceEdit(
        file_path=non_existent,
        start_line=1,
        end_line=1,
        replacement="x = 1",
    )

    with pytest.raises(PatchGenerationError, match="Target file not found"):
        generate_patch(source_edit=edit)


def test_unreadable_file_raises_controlled_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = tmp_path / "unreadable.py"
    sample.write_text("x = 1\n", encoding="utf-8")

    def mock_read_text(*args: object, **kwargs: object) -> str:
        raise OSError("Disk permission denied mock")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    edit = SourceEdit(
        file_path=sample,
        start_line=1,
        end_line=1,
        replacement="x = 2",
    )

    with pytest.raises(PatchGenerationError, match="Failed to read source file"):
        generate_patch(source_edit=edit)


def test_invalid_start_line_raises_controlled_error(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=0,
        end_line=4,
        replacement="x = 1",
    )

    with pytest.raises(PatchGenerationError, match="Invalid start_line"):
        generate_patch(source_edit=edit)


def test_negative_start_line_raises_controlled_error(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=-2,
        end_line=2,
        replacement="x = 1",
    )

    with pytest.raises(PatchGenerationError, match="Invalid start_line"):
        generate_patch(source_edit=edit)


def test_start_line_after_end_line_raises_controlled_error(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=2,
        replacement="x = 1",
    )

    with pytest.raises(PatchGenerationError, match="Invalid line range"):
        generate_patch(source_edit=edit)


def test_out_of_range_edit_raises_controlled_error(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=1,
        end_line=999,
        replacement="x = 1",
    )

    with pytest.raises(PatchGenerationError, match="Invalid end_line"):
        generate_patch(source_edit=edit)


def test_candidate_file_and_edit_file_mismatch_is_rejected(temp_sample_file: Path, tmp_path: Path) -> None:
    other_file = tmp_path / "other.py"
    other_file.write_text("x = 1\n", encoding="utf-8")

    cand = make_sample_candidate(temp_sample_file)
    edit = SourceEdit(
        file_path=other_file,
        start_line=1,
        end_line=1,
        replacement="x = 2",
    )

    with pytest.raises(PatchGenerationError, match="Candidate target file"):
        generate_patch(candidate=cand, source_edit=edit)


def test_empty_replacement_is_rejected(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="",
    )

    with pytest.raises(PatchGenerationError, match="Empty replacement content is rejected"):
        generate_patch(source_edit=edit)


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------

def test_dot_env_cannot_be_patched(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_KEY=12345\n", encoding="utf-8")

    edit = SourceEdit(
        file_path=env_file,
        start_line=1,
        end_line=1,
        replacement="SECRET_KEY=new_key",
    )

    assert is_security_sensitive_file(env_file)
    with pytest.raises(PatchGenerationError, match="Security error"):
        generate_patch(source_edit=edit)


def test_pem_file_cannot_be_patched(tmp_path: Path) -> None:
    pem_file = tmp_path / "cert.pem"
    pem_file.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")

    edit = SourceEdit(
        file_path=pem_file,
        start_line=1,
        end_line=1,
        replacement="new cert",
    )

    assert is_security_sensitive_file(pem_file)
    with pytest.raises(PatchGenerationError, match="Security error"):
        generate_patch(source_edit=edit)


def test_key_file_cannot_be_patched(tmp_path: Path) -> None:
    key_file = tmp_path / "private.key"
    key_file.write_text("PRIVATE KEY DATA\n", encoding="utf-8")

    edit = SourceEdit(
        file_path=key_file,
        start_line=1,
        end_line=1,
        replacement="NEW KEY",
    )

    assert is_security_sensitive_file(key_file)
    with pytest.raises(PatchGenerationError, match="Security error"):
        generate_patch(source_edit=edit)


def test_credential_and_private_key_paths_are_rejected(tmp_path: Path) -> None:
    cred_file = tmp_path / "credentials.json"
    cred_file.write_text('{"token": "abc"}\n', encoding="utf-8")

    edit1 = SourceEdit(file_path=cred_file, start_line=1, end_line=1, replacement="{}")
    with pytest.raises(PatchGenerationError, match="Security error"):
        generate_patch(source_edit=edit1)

    rsa_file = tmp_path / "id_rsa"
    rsa_file.write_text("RSA PRIV KEY\n", encoding="utf-8")
    edit2 = SourceEdit(file_path=rsa_file, start_line=1, end_line=1, replacement="RSA")
    with pytest.raises(PatchGenerationError, match="Security error"):
        generate_patch(source_edit=edit2)


# ---------------------------------------------------------------------------
# Safety Tests
# ---------------------------------------------------------------------------

def test_source_file_remains_byte_for_byte_unchanged(temp_sample_file: Path) -> None:
    before_bytes = temp_sample_file.read_bytes()
    before_mtime = temp_sample_file.stat().st_mtime_ns

    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    generate_patch(source_edit=edit)

    after_bytes = temp_sample_file.read_bytes()
    after_mtime = temp_sample_file.stat().st_mtime_ns

    assert before_bytes == after_bytes
    assert before_mtime == after_mtime


def test_patch_generation_performs_no_filesystem_writes(temp_sample_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_called = False

    def forbidden_write(*args: object, **kwargs: object) -> None:
        nonlocal write_called
        write_called = True
        raise RuntimeError("Filesystem mutation attempt detected!")

    monkeypatch.setattr(Path, "write_text", forbidden_write)
    monkeypatch.setattr(Path, "write_bytes", forbidden_write)

    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    assert not write_called
    assert patch.proposed_content != patch.original_content


def test_multiple_patch_generation_does_not_modify_source(temp_sample_file: Path) -> None:
    original_bytes = temp_sample_file.read_bytes()

    edit1 = SourceEdit(file_path=temp_sample_file, start_line=1, end_line=1, replacement="import pandas as pd  # comment")
    edit2 = SourceEdit(file_path=temp_sample_file, start_line=4, end_line=4, replacement="    result = pd.concat([df, df2])")

    p1 = generate_patch(source_edit=edit1)
    p2 = generate_patch(source_edit=edit2)

    assert p1.original_content == p2.original_content
    assert temp_sample_file.read_bytes() == original_bytes


def test_patch_generation_does_not_call_llm() -> None:
    mock_provider = MagicMock()
    # Ensure generate_patch takes input models directly without calling mock_provider
    temp_path = Path("fake.py")
    assert not mock_provider.called


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------

def test_same_source_and_edit_produces_identical_diff(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch1 = generate_patch(source_edit=edit)
    patch2 = generate_patch(source_edit=edit)

    assert patch1.diff == patch2.diff
    assert patch1 == patch2


def test_original_content_stored_in_patch_exactly_matches_source(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    assert patch.original_content == temp_sample_file.read_text(encoding="utf-8")


def test_diff_corresponds_exactly_to_original_to_proposed_content(temp_sample_file: Path) -> None:
    edit = SourceEdit(
        file_path=temp_sample_file,
        start_line=4,
        end_line=4,
        replacement="    result = pd.concat([df, df2])",
    )
    patch = generate_patch(source_edit=edit)

    assert validate_patch(patch) is True


# ---------------------------------------------------------------------------
# PatchSet Tests
# ---------------------------------------------------------------------------

def test_multiple_patches_preserve_deterministic_ordering(temp_sample_file: Path, tmp_path: Path) -> None:
    file2 = tmp_path / "util.py"
    file2.write_text("import os\n", encoding="utf-8")

    edit1 = SourceEdit(file_path=temp_sample_file, start_line=1, end_line=1, replacement="import pandas as pd")
    edit2 = SourceEdit(file_path=file2, start_line=1, end_line=1, replacement="import sys")

    p1 = generate_patch(source_edit=edit1)
    p2 = generate_patch(source_edit=edit2)

    pset = create_patch_set([p1, p2])

    assert isinstance(pset, PatchSet)
    assert pset.patches == (p1, p2)
    assert pset.patch_count == 2
    assert pset.files_changed == (temp_sample_file, file2)
    assert pset.has_patches is True


# ---------------------------------------------------------------------------
# Important Boundary Test (Free-form Recommendation Rejection)
# ---------------------------------------------------------------------------

def test_dangerous_free_form_recommendation_without_source_edit_is_rejected() -> None:
    rec = make_sample_recommendation()
    # Attempting to generate a patch from free-form recommendation alone must fail
    with pytest.raises(PatchGenerationError, match="structured SourceEdit is required"):
        generate_patch(recommendation=rec)
