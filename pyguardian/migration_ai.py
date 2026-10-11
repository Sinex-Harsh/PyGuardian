"""
migration_ai.py — AI Migration Reasoning for PyGuardian.

Architecture:
MigrationContext
    ↓
build_migration_prompt(context)
    ↓
LLMProvider.generate(prompt)
    ↓
JSON Parsing & Validation
    ↓
MigrationRecommendation

Responsibilities:
- Build deterministic prompts for AI migration reasoning from MigrationContext.
- Call LLMProvider interface to query language models.
- Parse and strictly validate structured JSON completions into MigrationRecommendation objects.
- Enforce strict evidence-reference validation against context evidence IDs.
- Preserve separation between deterministic evidence, AI reasoning, and proposed code replacements.
- Never modify source files or overwrite deterministic confidence scores.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from pyguardian.llm import LLMProvider, LLMProviderError
from pyguardian.migration_context import MigrationContext

logger = logging.getLogger(__name__)

# Allowed values for AI confidence
VALID_CONFIDENCE_VALUES: set[str] = {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MigrationAIError(Exception):
    """Raised when AI migration prompt building, provider call, parsing, or validation fails."""
    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MigrationRecommendation:
    """
    Structured, validated recommendation produced by AI migration reasoning.

    Attributes:
        package_name:         Canonical package name being analyzed.
        affected_api:         Affected API symbol path, or None.
        summary:              Brief summary of the migration recommendation.
        reason:               Detailed AI reasoning explaining why migration is needed/recommended.
        recommended_action:   Actionable steps for developers (e.g. "Replace X with Y").
        proposed_replacement: Optional code snippet or API replacement string, or None.
        confidence:          AI confidence level ('high', 'medium', 'low').
        evidence_references:  Tuple of evidence IDs explicitly referenced from context.
        uncertainties:        Tuple of identified uncertainties or limitations in reasoning.
    """

    package_name: str
    affected_api: Optional[str]
    summary: str
    reason: str
    recommended_action: str
    proposed_replacement: Optional[str]
    confidence: str
    evidence_references: tuple[str, ...]
    uncertainties: tuple[str, ...]

    def __post_init__(self) -> None:
        """Coerce sequence attributes to immutable tuples."""
        if not isinstance(self.evidence_references, tuple):
            object.__setattr__(self, "evidence_references", tuple(self.evidence_references))
        if not isinstance(self.uncertainties, tuple):
            object.__setattr__(self, "uncertainties", tuple(self.uncertainties))


# ---------------------------------------------------------------------------
# Evidence ID Extraction Helper
# ---------------------------------------------------------------------------

def get_context_evidence_ids(context: MigrationContext) -> set[str]:
    """
    Extract the exact set of valid evidence IDs supplied in a MigrationContext.

    Evidence IDs:
    - "CANDIDATE": Candidate metadata
    - "SOURCE-1": Bounded source excerpt (if source_context present)
    - "USAGE-1", "USAGE-2", ... for relevant_usages
    - "IMPORT-1", "IMPORT-2", ... for relevant_imports
    - "CHANGE-1", "CHANGE-2", ... for relevant_changes
    - "DOC-1", "DOC-2", ... for relevant_documents

    Args:
        context: The MigrationContext being processed.

    Returns:
        Set of valid evidence ID strings.
    """
    ids: set[str] = {"CANDIDATE"}

    if context.has_source_context and context.source_context is not None:
        ids.add("SOURCE-1")

    for idx in range(1, len(context.relevant_usages) + 1):
        ids.add(f"USAGE-{idx}")

    for idx in range(1, len(context.relevant_imports) + 1):
        ids.add(f"IMPORT-{idx}")

    for idx in range(1, len(context.relevant_changes) + 1):
        ids.add(f"CHANGE-{idx}")

    for idx in range(1, len(context.relevant_documents) + 1):
        ids.add(f"DOC-{idx}")

    return ids


# ---------------------------------------------------------------------------
# Deterministic Prompt Builder
# ---------------------------------------------------------------------------

def build_migration_prompt(context: MigrationContext) -> str:
    """
    Construct a deterministic prompt for AI migration analysis from a MigrationContext.

    Args:
        context: The MigrationContext containing candidate details and evidence.

    Returns:
        Formatted string prompt for the LLM.

    Raises:
        MigrationAIError: If context is invalid.
    """
    if not isinstance(context, MigrationContext):
        raise MigrationAIError(f"Expected MigrationContext object, got {type(context).__name__}.")

    candidate = context.candidate
    valid_ids = get_context_evidence_ids(context)

    sections: list[str] = []

    sections.append(
        "You are an expert Python migration assistant for PyGuardian. "
        "Analyze the provided dependency update evidence and provide structured migration guidance."
    )

    # Candidate section
    sections.append(
        f"--- CANDIDATE DETAILS [ID: CANDIDATE] ---\n"
        f"Package: {candidate.package_name}\n"
        f"Affected API: {candidate.affected_api or 'N/A (Package-level)'}\n"
        f"Current Version: {candidate.current_version or 'Unknown'}\n"
        f"Target Version: {candidate.target_version or 'Unknown'}\n"
        f"Change Type: {candidate.change_type or 'Unknown'}\n"
        f"Risk Level: {candidate.risk_level.value}\n"
        f"Deterministic Confidence: {candidate.confidence}\n"
        f"Description: {candidate.description}"
    )

    # Source Context
    if context.has_source_context and context.source_context is not None:
        sc = context.source_context
        file_str = str(sc.file_path) if sc.file_path else "Unknown"
        sections.append(
            f"--- SOURCE EXCERPT [ID: SOURCE-1] ---\n"
            f"File: {file_str} (Lines {sc.start_line}-{sc.end_line})\n"
            f"```python\n{sc.content}\n```"
        )

    # Documented Changes
    if context.relevant_changes:
        chg_lines = []
        for idx, chg in enumerate(context.relevant_changes, 1):
            chg_lines.append(
                f"[ID: CHANGE-{idx}] {chg.package_name} v{chg.version} - {chg.change_type}: "
                f"{chg.api_path or 'package'} -> {chg.description}"
            )
        sections.append("--- DOCUMENTED CHANGES ---\n" + "\n".join(chg_lines))

    # Relevant Usages
    if context.relevant_usages:
        usage_lines = []
        for idx, u in enumerate(context.relevant_usages, 1):
            usage_lines.append(
                f"[ID: USAGE-{idx}] {u.file_path}:{u.line_number} -> {u.api_path} ({u.usage_type})"
            )
        sections.append("--- RELEVANT API USAGES ---\n" + "\n".join(usage_lines))

    # Relevant Imports
    if context.relevant_imports:
        import_lines = []
        for idx, imp in enumerate(context.relevant_imports, 1):
            import_lines.append(
                f"[ID: IMPORT-{idx}] {imp.file_path}:{imp.line_number} -> import {imp.package_name}"
                + (f".{imp.imported_name}" if imp.imported_name else "")
                + (f" as {imp.alias}" if imp.alias else "")
            )
        sections.append("--- RELEVANT IMPORTS ---\n" + "\n".join(import_lines))

    # Change Documents
    if context.relevant_documents:
        doc_lines = []
        for idx, doc in enumerate(context.relevant_documents, 1):
            doc_lines.append(
                f"[ID: DOC-{idx}] Document: {doc.package_name} ({len(doc.changes)} documented changes)"
            )
        sections.append("--- CHANGE DOCUMENTS ---\n" + "\n".join(doc_lines))

    # Valid Evidence IDs list
    id_list_str = ", ".join(sorted(valid_ids))
    sections.append(
        f"--- VALID EVIDENCE IDS ---\n"
        f"You MUST only reference evidence IDs from this list: [{id_list_str}]"
    )

    # Instructions
    instructions = (
        "--- INSTRUCTIONS ---\n"
        "Respond ONLY with a valid JSON object matching the schema below. "
        "Do NOT include markdown formatting or extra text outside the JSON object.\n\n"
        "Required JSON Schema:\n"
        "{\n"
        f'  "package_name": "{candidate.package_name}",\n'
        '  "affected_api": "string or null",\n'
        '  "summary": "Short single-sentence summary",\n'
        '  "reason": "Detailed explanation of why migration is required",\n'
        '  "recommended_action": "Actionable instructions for developers",\n'
        '  "proposed_replacement": "Code replacement snippet or null",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "evidence_references": ["ID1", "ID2"],\n'
        '  "uncertainties": ["Uncertainty item 1"]\n'
        "}\n\n"
        "Rules:\n"
        "1. 'confidence' MUST be exactly one of: 'high', 'medium', 'low'.\n"
        "2. 'evidence_references' MUST ONLY contain valid evidence IDs listed above.\n"
        "3. 'package_name' MUST match the package name.\n"
        "4. 'proposed_replacement' should be a string or null if no replacement code applies."
    )
    sections.append(instructions)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# JSON Response Extractor & Strict Validator
# ---------------------------------------------------------------------------

def _extract_json_text(text: str) -> str:
    """
    Extract JSON object substring from completion text.

    Supports bare JSON strings or Markdown fenced code blocks (```json ... ```).
    """
    if not isinstance(text, str):
        raise MigrationAIError(f"LLM response must be a string, got {type(text).__name__}.")

    clean = text.strip()

    if "```" in clean:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if match:
            return match.group(1).strip()

    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1]

    return clean


def _validate_and_parse_response(
    raw_response: str,
    context: MigrationContext,
) -> MigrationRecommendation:
    """
    Extract, parse, and strictly validate the JSON response from the LLM.

    Args:
        raw_response: Raw text returned by LLM provider.
        context:      The MigrationContext supplied to the model.

    Returns:
        MigrationRecommendation object.

    Raises:
        MigrationAIError: On JSON decoding failure, missing fields, type errors,
                          invalid confidence values, or invalid evidence references.
    """
    json_str = _extract_json_text(raw_response)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise MigrationAIError(f"Failed to parse LLM response as JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise MigrationAIError(f"Expected JSON object in LLM response, got {type(data).__name__}.")

    required_keys = [
        "package_name",
        "summary",
        "reason",
        "recommended_action",
        "confidence",
        "evidence_references",
        "uncertainties",
    ]
    for key in required_keys:
        if key not in data:
            raise MigrationAIError(f"Missing required field in LLM JSON response: '{key}'.")

    pkg_name = data["package_name"]
    if not isinstance(pkg_name, str) or not pkg_name.strip():
        raise MigrationAIError(f"Field 'package_name' must be a non-empty string, got {pkg_name!r}.")

    affected_api = data.get("affected_api")
    if affected_api is not None and not isinstance(affected_api, str):
        raise MigrationAIError(
            f"Field 'affected_api' must be a string or null, got {type(affected_api).__name__}."
        )

    summary = data["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise MigrationAIError(f"Field 'summary' must be a non-empty string, got {summary!r}.")

    reason = data["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise MigrationAIError(f"Field 'reason' must be a non-empty string, got {reason!r}.")

    action = data["recommended_action"]
    if not isinstance(action, str) or not action.strip():
        raise MigrationAIError(f"Field 'recommended_action' must be a non-empty string, got {action!r}.")

    replacement = data.get("proposed_replacement")
    if replacement is not None and not isinstance(replacement, str):
        raise MigrationAIError(
            f"Field 'proposed_replacement' must be a string or null, got {type(replacement).__name__}."
        )

    confidence = data["confidence"]
    if not isinstance(confidence, str) or confidence.lower().strip() not in VALID_CONFIDENCE_VALUES:
        raise MigrationAIError(
            f"Field 'confidence' must be one of {sorted(VALID_CONFIDENCE_VALUES)}, got {confidence!r}."
        )
    clean_confidence = confidence.lower().strip()

    ev_refs = data["evidence_references"]
    if not isinstance(ev_refs, (list, tuple)):
        raise MigrationAIError(
            f"Field 'evidence_references' must be a list of strings, got {type(ev_refs).__name__}."
        )

    valid_evidence_ids = get_context_evidence_ids(context)
    parsed_refs: list[str] = []
    for ref in ev_refs:
        if not isinstance(ref, str):
            raise MigrationAIError(f"Items in 'evidence_references' must be strings, got {type(ref).__name__}.")
        ref_clean = ref.strip()
        if ref_clean not in valid_evidence_ids:
            raise MigrationAIError(
                f"Model referenced invalid or fabricated evidence ID '{ref_clean}'. "
                f"Allowed evidence IDs for this context: {sorted(valid_evidence_ids)}"
            )
        parsed_refs.append(ref_clean)

    uncert = data["uncertainties"]
    if not isinstance(uncert, (list, tuple)):
        raise MigrationAIError(
            f"Field 'uncertainties' must be a list of strings, got {type(uncert).__name__}."
        )

    parsed_uncert: list[str] = []
    for item in uncert:
        if not isinstance(item, str):
            raise MigrationAIError(f"Items in 'uncertainties' must be strings, got {type(item).__name__}.")
        parsed_uncert.append(item.strip())

    return MigrationRecommendation(
        package_name=pkg_name.strip(),
        affected_api=affected_api.strip() if affected_api else None,
        summary=summary.strip(),
        reason=reason.strip(),
        recommended_action=action.strip(),
        proposed_replacement=replacement if replacement is not None else None,
        confidence=clean_confidence,
        evidence_references=tuple(parsed_refs),
        uncertainties=tuple(parsed_uncert),
    )


# ---------------------------------------------------------------------------
# Main Reasoning API
# ---------------------------------------------------------------------------

def analyze_migration(
    context: MigrationContext,
    provider: LLMProvider,
) -> MigrationRecommendation:
    """
    Perform AI migration reasoning for a MigrationContext using an LLMProvider.

    Workflow:
    1. Build deterministic prompt from MigrationContext.
    2. Dispatch prompt to LLMProvider.generate().
    3. Extract, parse, and strictly validate the JSON response.
    4. Validate evidence references against context's supplied evidence IDs.
    5. Construct and return a MigrationRecommendation.

    Args:
        context:  The MigrationContext providing candidate evidence.
        provider: The LLMProvider instance to generate reasoning.

    Returns:
        Validated MigrationRecommendation.

    Raises:
        MigrationAIError: If prompt generation, provider call, parsing, or validation fails.
    """
    if not isinstance(context, MigrationContext):
        raise MigrationAIError(f"Expected MigrationContext object, got {type(context).__name__}.")

    if not isinstance(provider, LLMProvider):
        raise MigrationAIError(f"Expected LLMProvider instance, got {type(provider).__name__}.")

    prompt = build_migration_prompt(context)

    try:
        raw_response = provider.generate(prompt)
    except LLMProviderError as exc:
        raise MigrationAIError(f"LLM provider failed during migration generation: {exc}") from exc
    except Exception as exc:
        raise MigrationAIError(f"Unexpected error from LLM provider: {exc}") from exc

    return _validate_and_parse_response(raw_response, context)
