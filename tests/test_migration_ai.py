"""
test_migration_ai.py — Tests for V0.3 Step 4: AI Migration Reasoning.

All tests are completely offline and use a FakeLLMProvider.
0 network calls, 0 source file mutations, 0 external LLM dependencies.
"""

import json
from pathlib import Path
from typing import Optional

import pytest
from packaging.version import Version

from pyguardian.change_analysis import DocumentedChange
from pyguardian.llm import LLMConfig, LLMProvider, LLMProviderError
from pyguardian.migration import MigrationCandidate
from pyguardian.migration_ai import (
    MigrationAIError,
    MigrationRecommendation,
    analyze_migration,
    build_migration_prompt,
    get_context_evidence_ids,
)
from pyguardian.migration_context import MigrationContext, SourceContext
from pyguardian.risk import RiskLevel
from pyguardian.source import APIUsage, ImportUsage


# ---------------------------------------------------------------------------
# Test Fixtures & Mocks
# ---------------------------------------------------------------------------

class FakeLLMProvider(LLMProvider):
    """Fake LLM provider for completely offline, deterministic testing."""

    def __init__(
        self,
        response_text: Optional[str] = None,
        raise_error: Optional[Exception] = None,
    ) -> None:
        self._config = LLMConfig(model="fake-model", base_url="http://localhost:11434")
        self.response_text = response_text
        self.raise_error = raise_error
        self.last_prompt: Optional[str] = None

    @property
    def config(self) -> LLMConfig:
        return self._config

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self.raise_error is not None:
            raise self.raise_error
        if self.response_text is None:
            raise LLMProviderError("No response_text configured in FakeLLMProvider.")
        return self.response_text


def create_sample_candidate(
    package_name: str = "pandas",
    affected_api: Optional[str] = "pandas.DataFrame.append",
    file_path: Optional[Path] = Path("src/data.py"),
    line_number: Optional[int] = 42,
    confidence: str = "high",
) -> MigrationCandidate:
    """Helper to build a clean test MigrationCandidate."""
    return MigrationCandidate(
        package_name=package_name,
        affected_api=affected_api,
        file_path=file_path,
        line_number=line_number,
        current_version=Version("1.5.3"),
        target_version=Version("2.0.0"),
        change_type="removed",
        risk_level=RiskLevel.HIGH,
        confidence=confidence,
        description="DataFrame.append was removed in pandas 2.0. Use concat instead.",
        source_url="https://pandas.pydata.org/docs/whatsnew/v2.0.0.html",
    )


def create_sample_context(
    candidate: Optional[MigrationCandidate] = None,
    include_source: bool = True,
    include_usage: bool = True,
    include_import: bool = True,
    include_change: bool = True,
) -> MigrationContext:
    """Helper to build a structured MigrationContext for testing."""
    cand = candidate or create_sample_candidate()
    source_ctx = (
        SourceContext(
            file_path=Path("src/data.py"),
            start_line=35,
            end_line=50,
            content="df = pd.DataFrame()\nres = df.append(other_df)",
        )
        if include_source
        else None
    )

    usages = (
        (
            APIUsage(
                file_path=Path("src/data.py"),
                package_name="pandas",
                api_path="pandas.DataFrame.append",
                line_number=42,
                usage_type="attribute_access",
            ),
        )
        if include_usage
        else ()
    )

    imports = (
        (
            ImportUsage(
                file_path=Path("src/data.py"),
                package_name="pandas",
                imported_name=None,
                alias="pd",
                line_number=1,
                import_type="import",
            ),
        )
        if include_import
        else ()
    )

    changes = (
        (
            DocumentedChange(
                package_name="pandas",
                version=Version("2.0.0"),
                change_type="removed",
                api_path="pandas.DataFrame.append",
                description="DataFrame.append is removed.",
                source_url="https://pandas.pydata.org/docs/whatsnew/v2.0.0.html",
            ),
        )
        if include_change
        else ()
    )

    return MigrationContext(
        candidate=cand,
        source_context=source_ctx,
        relevant_usages=usages,
        relevant_imports=imports,
        relevant_changes=changes,
        project_path=Path("."),
    )


def build_valid_response_json(
    package_name: str = "pandas",
    affected_api: Optional[str] = "pandas.DataFrame.append",
    summary: str = "Replace DataFrame.append with pandas.concat",
    reason: str = "DataFrame.append was removed in pandas 2.0.0.",
    recommended_action: str = "Use pandas.concat([df1, df2]) instead.",
    proposed_replacement: Optional[str] = "res = pd.concat([df, other_df])",
    confidence: str = "high",
    evidence_references: Optional[list[str]] = None,
    uncertainties: Optional[list[str]] = None,
) -> str:
    """Helper to produce valid JSON completion strings for testing."""
    data = {
        "package_name": package_name,
        "affected_api": affected_api,
        "summary": summary,
        "reason": reason,
        "recommended_action": recommended_action,
        "proposed_replacement": proposed_replacement,
        "confidence": confidence,
        "evidence_references": evidence_references if evidence_references is not None else ["CANDIDATE", "SOURCE-1", "USAGE-1"],
        "uncertainties": uncertainties if uncertainties is not None else [],
    }
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_evidence_id_extraction() -> None:
    """Test get_context_evidence_ids extracts exact expected IDs."""
    ctx = create_sample_context()
    ids = get_context_evidence_ids(ctx)

    assert "CANDIDATE" in ids
    assert "SOURCE-1" in ids
    assert "USAGE-1" in ids
    assert "IMPORT-1" in ids
    assert "CHANGE-1" in ids


def test_build_migration_prompt_contents() -> None:
    """Test build_migration_prompt produces structured deterministic prompt."""
    ctx = create_sample_context()
    prompt = build_migration_prompt(ctx)

    assert "pandas" in prompt
    assert "pandas.DataFrame.append" in prompt
    assert "CANDIDATE DETAILS" in prompt
    assert "[ID: SOURCE-1]" in prompt
    assert "```python" in prompt
    assert "df.append(other_df)" in prompt
    assert "[ID: USAGE-1]" in prompt
    assert "[ID: IMPORT-1]" in prompt
    assert "[ID: CHANGE-1]" in prompt
    assert "VALID EVIDENCE IDS" in prompt
    assert "Required JSON Schema" in prompt


def test_analyze_migration_valid_json() -> None:
    """Test analyze_migration succeeds with valid JSON response from provider."""
    ctx = create_sample_context()
    response_json = build_valid_response_json()
    provider = FakeLLMProvider(response_text=response_json)

    rec = analyze_migration(ctx, provider)

    assert isinstance(rec, MigrationRecommendation)
    assert rec.package_name == "pandas"
    assert rec.affected_api == "pandas.DataFrame.append"
    assert rec.summary == "Replace DataFrame.append with pandas.concat"
    assert rec.reason == "DataFrame.append was removed in pandas 2.0.0."
    assert rec.recommended_action == "Use pandas.concat([df1, df2]) instead."
    assert rec.proposed_replacement == "res = pd.concat([df, other_df])"
    assert rec.confidence == "high"
    assert rec.evidence_references == ("CANDIDATE", "SOURCE-1", "USAGE-1")
    assert rec.uncertainties == ()


def test_analyze_migration_markdown_wrapped_json() -> None:
    """Test analyze_migration parses JSON wrapped in markdown code fences."""
    ctx = create_sample_context()
    raw_json = build_valid_response_json()
    fenced_response = f"Here is the recommendation:\n```json\n{raw_json}\n```\nHope this helps!"
    provider = FakeLLMProvider(response_text=fenced_response)

    rec = analyze_migration(ctx, provider)

    assert rec.package_name == "pandas"
    assert rec.confidence == "high"


def test_analyze_migration_null_proposed_replacement() -> None:
    """Test analyze_migration accepts null/None proposed_replacement."""
    ctx = create_sample_context()
    response_json = build_valid_response_json(proposed_replacement=None)
    provider = FakeLLMProvider(response_text=response_json)

    rec = analyze_migration(ctx, provider)

    assert rec.proposed_replacement is None


def test_analyze_migration_malformed_json() -> None:
    """Test analyze_migration raises MigrationAIError on malformed JSON."""
    ctx = create_sample_context()
    provider = FakeLLMProvider(response_text="Not valid JSON at all {abc: 123")

    with pytest.raises(MigrationAIError, match="Failed to parse LLM response as JSON"):
        analyze_migration(ctx, provider)


def test_analyze_migration_missing_fields() -> None:
    """Test analyze_migration raises MigrationAIError when required fields are missing."""
    ctx = create_sample_context()
    # Missing 'reason' field
    data = {
        "package_name": "pandas",
        "summary": "Summary text",
        "recommended_action": "Action text",
        "confidence": "high",
        "evidence_references": ["CANDIDATE"],
        "uncertainties": [],
    }
    provider = FakeLLMProvider(response_text=json.dumps(data))

    with pytest.raises(MigrationAIError, match="Missing required field in LLM JSON response: 'reason'"):
        analyze_migration(ctx, provider)


def test_analyze_migration_wrong_field_types() -> None:
    """Test analyze_migration raises MigrationAIError on wrong field data types."""
    ctx = create_sample_context()
    # 'summary' is integer instead of string
    data = {
        "package_name": "pandas",
        "summary": 12345,
        "reason": "Reason text",
        "recommended_action": "Action text",
        "confidence": "high",
        "evidence_references": ["CANDIDATE"],
        "uncertainties": [],
    }
    provider = FakeLLMProvider(response_text=json.dumps(data))

    with pytest.raises(MigrationAIError, match="Field 'summary' must be a non-empty string"):
        analyze_migration(ctx, provider)


def test_analyze_migration_invalid_confidence() -> None:
    """Test analyze_migration raises MigrationAIError for invalid confidence values."""
    ctx = create_sample_context()
    response_json = build_valid_response_json(confidence="extremely_high")
    provider = FakeLLMProvider(response_text=response_json)

    with pytest.raises(MigrationAIError, match="Field 'confidence' must be one of"):
        analyze_migration(ctx, provider)


def test_analyze_migration_invalid_evidence_ids() -> None:
    """Test analyze_migration raises MigrationAIError if model references fabricated evidence IDs."""
    ctx = create_sample_context()
    # FABRICATED-99 is not in valid context evidence IDs
    response_json = build_valid_response_json(evidence_references=["CANDIDATE", "FABRICATED-99"])
    provider = FakeLLMProvider(response_text=response_json)

    with pytest.raises(MigrationAIError, match="Model referenced invalid or fabricated evidence ID 'FABRICATED-99'"):
        analyze_migration(ctx, provider)


def test_analyze_migration_provider_error() -> None:
    """Test analyze_migration wraps LLMProviderError into MigrationAIError."""
    ctx = create_sample_context()
    provider = FakeLLMProvider(raise_error=LLMProviderError("Connection refused to Ollama."))

    with pytest.raises(MigrationAIError, match="LLM provider failed during migration generation"):
        analyze_migration(ctx, provider)


def test_deterministic_confidence_preservation() -> None:
    """
    Test that running AI migration reasoning NEVER mutates or overwrites
    deterministic candidate confidence or risk assessment.
    """
    candidate = create_sample_candidate(confidence="medium")
    ctx = create_sample_context(candidate=candidate)

    response_json = build_valid_response_json(confidence="low")
    provider = FakeLLMProvider(response_text=response_json)

    rec = analyze_migration(ctx, provider)

    # AI recommendation has its own confidence
    assert rec.confidence == "low"
    # Deterministic candidate confidence remains completely untouched
    assert ctx.candidate.confidence == "medium"


def test_no_source_file_mutation(tmp_path: Path) -> None:
    """Test that analyze_migration makes 0 file modifications to project files."""
    test_file = tmp_path / "sample.py"
    initial_content = "import pandas as pd\ndf.append(other)\n"
    test_file.write_text(initial_content, encoding="utf-8")

    candidate = create_sample_candidate(file_path=test_file)
    ctx = create_sample_context(candidate=candidate)

    response_json = build_valid_response_json()
    provider = FakeLLMProvider(response_text=response_json)

    rec = analyze_migration(ctx, provider)

    assert rec is not None
    # File content on disk MUST remain unchanged
    assert test_file.read_text(encoding="utf-8") == initial_content
