"""
PyGuardian — Python Dependency Intelligence & Auto-Migration Assistant.

Version 0.1 — Dependency Scanner.
"""

__version__ = "0.3.0"
__author__ = "PyGuardian"

from pyguardian.apply import ApplyPatchError, ApplyPatchResult, apply_patch
from pyguardian.approval import ApprovalResult, request_approval
from pyguardian.diff_review import DiffReviewResult, ReviewFinding, ReviewFindingCode, ReviewStatus, review_patch
from pyguardian.llm import (
    LLMConfig,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    OllamaProvider,
    create_llm_provider,
)
from pyguardian.migrate import MigrationWorkflowResult, run_migration_workflow
from pyguardian.migration import MigrationCandidate, MigrationPlan, build_migration_plan
from pyguardian.migration_ai import (
    MigrationAIError,
    MigrationRecommendation,
    analyze_migration,
    build_migration_prompt,
)
from pyguardian.migration_context import (
    MigrationContext,
    SourceContext,
    build_migration_context,
    build_migration_contexts,
)
from pyguardian.migration_planner import (
    MigrationPlannerError,
    MigrationStatus,
    PlannedMigration,
    ProjectMigrationPlan,
    build_project_migration_plan,
)
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
from pyguardian.pipeline import ProjectAnalysis, analyze_project
from pyguardian.test_runner import TestResult, TestStatus, run_project_tests

__all__ = [
    "ApplyPatchError",
    "ApplyPatchResult",
    "ApprovalResult",
    "DiffReviewResult",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderError",
    "MigrationAIError",
    "MigrationCandidate",
    "MigrationContext",
    "MigrationPlan",
    "MigrationPlannerError",
    "MigrationRecommendation",
    "MigrationStatus",
    "MigrationWorkflowResult",
    "OllamaProvider",
    "PatchGenerationError",
    "PatchSet",
    "PlannedMigration",
    "ProjectAnalysis",
    "ProjectMigrationPlan",
    "ProposedPatch",
    "ReviewFinding",
    "ReviewFindingCode",
    "ReviewStatus",
    "SourceContext",
    "SourceEdit",
    "TestResult",
    "TestStatus",
    "analyze_migration",
    "analyze_project",
    "apply_patch",
    "build_migration_context",
    "build_migration_contexts",
    "build_migration_plan",
    "build_migration_prompt",
    "build_project_migration_plan",
    "create_llm_provider",
    "create_patch_set",
    "generate_patch",
    "is_security_sensitive_file",
    "request_approval",
    "review_patch",
    "run_migration_workflow",
    "run_project_tests",
    "validate_patch",
    "validate_source_edit",
]
