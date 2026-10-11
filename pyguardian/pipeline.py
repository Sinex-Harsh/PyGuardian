"""
pipeline.py — Unified Analysis Pipeline for PyGuardian.

Responsibilities:
- Orchestrate end-to-end dependency compatibility and risk analysis across a Python project.
- Coordinate existing deterministic intelligence modules in sequence:
  1. Project Scan (project.py)
  2. Source AST Analysis (source.py)
  3. Dependency ↔ Usage Mapping (impact.py)
  4. Per-Dependency Change Intelligence & Extraction (changes.py, change_sources.py, change_analysis.py)
  5. Compatibility Analysis (compatibility.py)
  6. Project Impact Aggregation (project_impact.py)
  7. Risk & Confidence Assessment (risk.py)
- Return a single unified ProjectAnalysis result preserving all intermediate findings and provenance.
- Handle partial dependency failures and source parse errors gracefully without failing the entire scan.

Design Principles:
- Thin orchestration layer: NO AST parsing, NO PyPI HTTP logic, NO risk/compatibility rules.
- Minimizes redundant PyPI network calls by passing PyPI metadata downstream.
- Sequential, deterministic execution preserving exact dependency input order.
- NO AI, LLM, RAG, embeddings, code modification, or watch mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from pyguardian.change_analysis import (
    DocumentedChange,
    extract_documented_changes,
)
from pyguardian.change_sources import (
    ChangeDocument,
    ChangeSource,
    discover_change_sources,
    fetch_change_document,
)
from pyguardian.changes import ChangeInfo, get_change_info
from pyguardian.compatibility import (
    CompatibilityReport,
    ImpactAssessment,
    analyze_compatibility,
)
from pyguardian.dependencies import Dependency
from pyguardian.impact import (
    DependencyUsage,
    map_dependencies_to_usage,
    normalize_package_name,
)
from pyguardian.packages import (
    PackageInfo,
    fetch_pypi_json,
)
from pyguardian.project import (
    DependencyResult,
    ProjectScan,
    scan_project,
)
from pyguardian.project_impact import (
    AnalysisStatus,
    DependencyImpactEntry,
    ProjectImpactReport,
    build_project_impact_report,
    create_error_entry,
)
from pyguardian.risk import (
    ProjectRiskReport,
    RiskAssessment,
    RiskLevel,
    assess_compatibility,
    assess_project_risk,
)
from pyguardian.source import (
    APIUsage,
    SourceAnalysis,
    analyze_source_tree,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified Result Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectAnalysis:
    """
    Unified end-to-end dependency compatibility and risk analysis report for a project.

    Attributes:
        project_scan:          The complete ProjectScan from dependency discovery.
        source_analysis:       The SourceAnalysis containing AST import/API usages and parse errors.
        dependency_usage:      Mapped list of DependencyUsage objects (dependency ↔ import usages).
        compatibility_reports: List of CompatibilityReport objects for analyzed dependencies.
        project_impact:        Aggregated ProjectImpactReport across all dependencies.
        risk_assessments:      List of RiskAssessment objects across all analyzed dependencies.
    """

    project_scan: ProjectScan
    source_analysis: SourceAnalysis
    dependency_usage: list[DependencyUsage] = field(default_factory=list)
    compatibility_reports: list[CompatibilityReport] = field(default_factory=list)
    project_impact: ProjectImpactReport = field(
        default_factory=lambda: ProjectImpactReport(project_path=Path("."))
    )
    risk_assessments: list[RiskAssessment] = field(default_factory=list)

    @property
    def project_path(self) -> Path:
        """The project root directory analyzed."""
        return self.project_scan.project_path

    @property
    def total_dependencies(self) -> int:
        """Total number of dependencies in the project scan."""
        return self.project_scan.total_dependencies

    @property
    def has_impacts(self) -> bool:
        """True if any dependency has detected compatibility impacts."""
        return self.project_impact.dependencies_with_impacts > 0

    @property
    def high_risk_count(self) -> int:
        """Count of HIGH risk assessments."""
        return sum(1 for r in self.risk_assessments if r.risk_level == RiskLevel.HIGH)

    @property
    def medium_risk_count(self) -> int:
        """Count of MEDIUM risk assessments."""
        return sum(1 for r in self.risk_assessments if r.risk_level == RiskLevel.MEDIUM)

    @property
    def low_risk_count(self) -> int:
        """Count of LOW risk assessments."""
        return sum(1 for r in self.risk_assessments if r.risk_level == RiskLevel.LOW)


# ---------------------------------------------------------------------------
# Pipeline Orchestration API
# ---------------------------------------------------------------------------

def analyze_project(
    project_path: str | Path,
    *,
    ignored_dirs: Optional[Sequence[str]] = None,
    fetch_docs: bool = True,
) -> ProjectAnalysis:
    """
    Run end-to-end PyGuardian dependency compatibility and risk analysis for a project.

    Orchestrates:
    1. Project Scan (scan_project)
    2. Source AST Analysis (analyze_source_tree)
    3. Dependency ↔ Usage Mapping (map_dependencies_to_usage)
    4. Per-Dependency Change Intelligence & Extraction (get_change_info, discover_change_sources, fetch_change_document)
    5. Compatibility Analysis (analyze_compatibility)
    6. Project Impact Aggregation (build_project_impact_report)
    7. Risk Assessment (assess_project_risk)

    Args:
        project_path: Path to the root directory of the Python project to analyze.
        ignored_dirs: Optional sequence of directory names to ignore during source analysis.
        fetch_docs:   Whether to fetch documentation for change extraction (default True).

    Returns:
        A unified ProjectAnalysis object containing all intermediate and final results.
    """
    path_obj = Path(project_path)
    logger.info("Starting unified project analysis: %s", path_obj)

    # Stage 1 — Project Scan
    project_scan = scan_project(path_obj)

    # Stage 2 — Source Analysis
    source_analysis = analyze_source_tree(path_obj, ignored_dirs=ignored_dirs)

    # Stage 3 — Dependency ↔ Usage Mapping
    declared_deps = [res.dependency for res in project_scan.results]
    dep_usages = map_dependencies_to_usage(declared_deps, source_analysis)

    # If project scan failed overall (e.g. invalid directory or no reqs files), return early cleanly
    if not project_scan.is_successful or not project_scan.results:
        empty_impact = build_project_impact_report(path_obj, [])
        return ProjectAnalysis(
            project_scan=project_scan,
            source_analysis=source_analysis,
            dependency_usage=dep_usages,
            compatibility_reports=[],
            project_impact=empty_impact,
            risk_assessments=[],
        )

    # Index API usages by normalized package name
    api_usages_by_pkg: dict[str, list[APIUsage]] = {}
    for usage in source_analysis.api_usages:
        norm_name = normalize_package_name(usage.package_name)
        if norm_name not in api_usages_by_pkg:
            api_usages_by_pkg[norm_name] = []
        api_usages_by_pkg[norm_name].append(usage)

    compatibility_reports: list[CompatibilityReport] = []
    impact_entries: list[DependencyImpactEntry] = []

    # Stages 4-7 — Process each dependency in original order
    for dep_result in project_scan.results:
        pkg_name = dep_result.name

        # Handle parse error or PyPI lookup error on dependency
        if dep_result.has_error:
            err_msg = dep_result.error_message or f"Analysis error for package '{pkg_name}'"
            impact_entries.append(create_error_entry(pkg_name, err_msg))
            continue

        curr_ver_str = dep_result.current_version
        targ_ver_str = dep_result.latest_version

        if not curr_ver_str or not targ_ver_str:
            err_msg = f"Missing current or target version for package '{pkg_name}'"
            impact_entries.append(create_error_entry(pkg_name, err_msg))
            continue

        # Fetch PyPI JSON metadata (reuses network cache if already fetched during scan)
        pypi_data, pypi_err = fetch_pypi_json(pkg_name)

        # Get change info and discover change sources using metadata
        change_info = get_change_info(pkg_name, curr_ver_str, targ_ver_str)
        sources = discover_change_sources(pkg_name, pypi_data=pypi_data)

        # Extract documented changes from fetched change documents
        documented_changes: list[DocumentedChange] = []
        if fetch_docs and sources:
            # Fetch candidate sources (prioritizing top sources)
            for source in sources[:2]:
                doc = fetch_change_document(source)
                if doc.is_success:
                    extracted = extract_documented_changes(doc)
                    documented_changes.extend(extracted)

        # Retrieve relevant API usages for this package
        norm_pkg = normalize_package_name(pkg_name)
        pkg_api_usages = api_usages_by_pkg.get(norm_pkg, [])

        # Analyze compatibility
        comp_report = analyze_compatibility(
            package_name=pkg_name,
            current_version=curr_ver_str,
            target_version=targ_ver_str,
            api_usages=pkg_api_usages,
            documented_changes=documented_changes,
            update_type=dep_result.update_type,
        )

        compatibility_reports.append(comp_report)
        impact_entries.append(
            DependencyImpactEntry(
                package_name=pkg_name,
                status=AnalysisStatus.ANALYZED,
                report=comp_report,
            )
        )

    # Stage 8 — Project Impact Aggregation
    project_impact = build_project_impact_report(path_obj, impact_entries)

    # Stage 9 — Risk Assessment
    risk_report = assess_project_risk(path_obj, compatibility_reports)
    risk_assessments = risk_report.assessments

    return ProjectAnalysis(
        project_scan=project_scan,
        source_analysis=source_analysis,
        dependency_usage=dep_usages,
        compatibility_reports=compatibility_reports,
        project_impact=project_impact,
        risk_assessments=risk_assessments,
    )
