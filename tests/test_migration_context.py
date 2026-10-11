"""
test_migration_context.py — Unit tests for PyGuardian Migration Evidence & Context Builder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from packaging.version import Version
import pytest

from pyguardian.change_analysis import APIChangeMatch, DocumentedChange
from pyguardian.change_sources import ChangeDocument, ChangeSource
from pyguardian.compatibility import CompatibilityReport, ImpactAssessment
from pyguardian.dependencies import Dependency
from pyguardian.migration import MigrationCandidate, MigrationPlan
from pyguardian.migration_context import (
    DEFAULT_CONTEXT_LINES,
    MigrationContext,
    SourceContext,
    build_migration_context,
    build_migration_contexts,
    build_source_context,
    extract_relevant_changes,
    extract_relevant_documents,
    extract_relevant_imports,
    extract_relevant_usages,
)
from pyguardian.packages import PackageInfo, UpdateType
from pyguardian.pipeline import ProjectAnalysis
from pyguardian.project import DependencyResult, ProjectScan
from pyguardian.risk import RiskAssessment, RiskLevel, assess_impact
from pyguardian.source import APIUsage, ImportUsage, SourceAnalysis


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_sample_file(path: Path, num_lines: int = 50) -> Path:
    """Create a sample python source file with numbered lines."""
    lines = [f"line_{i} = {i}  # Code line {i}" for i in range(1, num_lines + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _create_test_candidate(
    package_name: str = "pandas",
    current_version: str = "1.5.3",
    target_version: str = "2.0.0",
    affected_api: str | None = "pandas.read_csv",
    file_path: Path | None = Path("app/loader.py"),
    line_number: int | None = 20,
    change_type: str = "deprecated",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    confidence: str = "high",
    description: str = "pandas.read_csv is deprecated in favor of read_parquet",
    source_url: str | None = "https://pandas.pydata.org/docs/release/2.0.0.html",
    include_evidence: bool = True,
) -> MigrationCandidate:
    """Construct a synthetic MigrationCandidate with complete evidence provenance."""
    curr_v = Version(current_version) if current_version else None
    targ_v = Version(target_version) if target_version else None

    evidence: RiskAssessment | None = None
    if include_evidence:
        usage = APIUsage(
            package_name=package_name,
            api_path=affected_api or f"{package_name}.module",
            file_path=file_path or Path("app/main.py"),
            line_number=line_number or 1,
            usage_type="call",
        )
        doc_change = DocumentedChange(
            package_name=package_name,
            version=targ_v,
            api_path=affected_api,
            change_type=change_type,
            description=description,
            source_url=source_url or "",
            confidence=confidence,
        )
        match = APIChangeMatch(
            api_usage=usage,
            documented_change=doc_change,
            match_type="exact" if affected_api else "package",
            confidence=confidence,
        )
        impact = ImpactAssessment(
            package_name=package_name,
            current_version=curr_v,
            target_version=targ_v,
            update_type=UpdateType.MAJOR,
            affected_api=affected_api,
            file_path=file_path,
            line_number=line_number,
            change_type=change_type,
            description=description,
            confidence=confidence,
            evidence=match,
        )
        evidence = assess_impact(impact)

    return MigrationCandidate(
        package_name=package_name,
        affected_api=affected_api,
        file_path=file_path,
        line_number=line_number,
        current_version=curr_v,
        target_version=targ_v,
        change_type=change_type,
        risk_level=risk_level,
        confidence=confidence,
        description=description,
        source_url=source_url,
        evidence=evidence,
    )


def _create_mock_project_analysis(
    project_path: Path,
    pkg_name: str = "pandas",
    api_usages: list[APIUsage] | None = None,
    import_usages: list[ImportUsage] | None = None,
    documented_changes: list[DocumentedChange] | None = None,
) -> ProjectAnalysis:
    """Construct a mock ProjectAnalysis containing AST results and reports."""
    req_file = project_path / "requirements.txt"
    results = [
        DependencyResult(
            dependency=Dependency(
                name=pkg_name,
                raw_line=f"{pkg_name}==1.5.3",
                specifier="==1.5.3",
                source_file=req_file,
                line_number=1,
            ),
            package_info=PackageInfo(name=pkg_name, latest_version=Version("2.0.0")),
            update_type=UpdateType.MAJOR,
        )
    ]
    scan = ProjectScan(
        project_path=project_path,
        project_name="test_proj",
        requirements_files=[req_file],
        results=results,
    )

    source_analysis = SourceAnalysis(
        project_path=project_path,
        usages=import_usages or [],
        api_usages=api_usages or [],
        files_scanned=1,
    )

    comp_reports: list[CompatibilityReport] = []
    if documented_changes and api_usages:
        assessments: list[ImpactAssessment] = []
        for chg in documented_changes:
            for usg in api_usages:
                if usg.package_name == chg.package_name:
                    m = APIChangeMatch(
                        api_usage=usg,
                        documented_change=chg,
                        match_type="exact" if chg.api_path == usg.api_path else "package",
                        confidence=chg.confidence,
                    )
                    assessments.append(
                        ImpactAssessment(
                            package_name=pkg_name,
                            current_version=Version("1.5.3"),
                            target_version=Version("2.0.0"),
                            update_type=UpdateType.MAJOR,
                            affected_api=usg.api_path if m.match_type == "exact" else None,
                            file_path=usg.file_path,
                            line_number=usg.line_number,
                            change_type=chg.change_type,
                            description=chg.description,
                            confidence=chg.confidence,
                            evidence=m,
                        )
                    )
        comp_reports.append(
            CompatibilityReport(
                package_name=pkg_name,
                current_version=Version("1.5.3"),
                target_version=Version("2.0.0"),
                update_type=UpdateType.MAJOR,
                assessments=assessments,
            )
        )

    return ProjectAnalysis(
        project_scan=scan,
        source_analysis=source_analysis,
        compatibility_reports=comp_reports,
    )


# ---------------------------------------------------------------------------
# Requirement 1-20 Unit Tests
# ---------------------------------------------------------------------------

def test_1_exact_api_candidate_gets_source_context(tmp_path: Path):
    """Test 1 — Exact API candidate with valid file and line gets a non-null SourceContext."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=30)
    candidate = _create_test_candidate(file_path=src_file, line_number=15)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.has_source_context
    assert ctx.source_context is not None
    assert ctx.source_context.content is not None
    assert ctx.source_context.file_path == src_file


def test_2_source_context_includes_affected_line(tmp_path: Path):
    """Test 2 — Source context content includes the exact affected line."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=30)
    candidate = _create_test_candidate(file_path=src_file, line_number=15)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is not None
    assert "line_15 = 15  # Code line 15" in (ctx.source_context.content or "")


def test_3_source_context_includes_bounded_surrounding_lines(tmp_path: Path):
    """Test 3 — Source context includes bounded lines before and after line 20 with default context."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=50)
    candidate = _create_test_candidate(file_path=src_file, line_number=20)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is not None
    # 20 - 8 = 12, 20 + 8 = 28
    assert ctx.source_context.start_line == 12
    assert ctx.source_context.end_line == 28
    assert ctx.source_context.line_count == 17
    assert "line_12 = 12" in ctx.source_context.content
    assert "line_20 = 20" in ctx.source_context.content
    assert "line_28 = 28" in ctx.source_context.content
    assert "line_11 = 11" not in ctx.source_context.content
    assert "line_29 = 29" not in ctx.source_context.content


def test_4_context_window_at_beginning_of_file(tmp_path: Path):
    """Test 4 — Context window at beginning of file clamps start_line to 1."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=30)
    candidate = _create_test_candidate(file_path=src_file, line_number=3)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis, context_lines=8)

    assert ctx.source_context is not None
    assert ctx.source_context.start_line == 1
    assert ctx.source_context.end_line == 11
    assert "line_1 = 1" in ctx.source_context.content
    assert "line_11 = 11" in ctx.source_context.content


def test_5_context_window_at_end_of_file(tmp_path: Path):
    """Test 5 — Context window at end of file clamps end_line to total lines."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=25)
    candidate = _create_test_candidate(file_path=src_file, line_number=23)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis, context_lines=8)

    assert ctx.source_context is not None
    assert ctx.source_context.start_line == 15
    assert ctx.source_context.end_line == 25
    assert "line_15 = 15" in ctx.source_context.content
    assert "line_25 = 25" in ctx.source_context.content


def test_6_oversized_line_number_handled_safely(tmp_path: Path):
    """Test 6 — Line number beyond file length returns None without raising exceptions."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=10)
    candidate = _create_test_candidate(file_path=src_file, line_number=100)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is None
    assert not ctx.has_source_context


def test_7_missing_source_file_does_not_crash(tmp_path: Path):
    """Test 7 — Non-existent source file does not crash and returns source_context = None."""
    missing_file = tmp_path / "non_existent.py"
    candidate = _create_test_candidate(file_path=missing_file, line_number=5)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is None
    assert not ctx.has_source_context
    assert ctx.candidate is candidate


def test_8_unreadable_source_file_does_not_crash(tmp_path: Path):
    """Test 8 — Directory path passed as file_path or unreadable file returns None gracefully."""
    dir_path = tmp_path / "somedir"
    dir_path.mkdir()
    candidate = _create_test_candidate(file_path=dir_path, line_number=1)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is None


def test_9_package_level_candidate_produces_no_fake_source_context(tmp_path: Path):
    """Test 9 — Package-level candidate with None file/line produces source_context = None."""
    candidate = _create_test_candidate(
        package_name="pandas",
        affected_api=None,
        file_path=None,
        line_number=None,
    )
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is None
    assert ctx.candidate.is_package_level
    assert not ctx.has_source_context


def test_10_relevant_api_usages_included(tmp_path: Path):
    """Test 10 — Relevant APIUsage entries are extracted and present in MigrationContext."""
    src_file = tmp_path / "app" / "loader.py"
    usage1 = APIUsage(
        package_name="pandas",
        api_path="pandas.read_csv",
        file_path=src_file,
        line_number=10,
        usage_type="call",
    )
    usage2 = APIUsage(
        package_name="requests",
        api_path="requests.get",
        file_path=src_file,
        line_number=20,
        usage_type="call",
    )
    analysis = _create_mock_project_analysis(tmp_path, api_usages=[usage1, usage2])
    candidate = _create_test_candidate(
        package_name="pandas",
        affected_api="pandas.read_csv",
        file_path=src_file,
        line_number=10,
    )

    ctx = build_migration_context(candidate, analysis)

    assert len(ctx.relevant_usages) >= 1
    assert any(u.api_path == "pandas.read_csv" for u in ctx.relevant_usages)
    assert not any(u.package_name == "requests" for u in ctx.relevant_usages)


def test_11_exact_api_candidate_prefers_exact_api_usages(tmp_path: Path):
    """Test 11 — Exact API candidates include exact API path usages and exclude other APIs of same package."""
    src_file = tmp_path / "app" / "loader.py"
    u_exact = APIUsage(
        package_name="pandas",
        api_path="pandas.read_csv",
        file_path=src_file,
        line_number=10,
        usage_type="call",
    )
    u_other = APIUsage(
        package_name="pandas",
        api_path="pandas.DataFrame",
        file_path=src_file,
        line_number=12,
        usage_type="class_reference",
    )
    analysis = _create_mock_project_analysis(tmp_path, api_usages=[u_exact, u_other])
    candidate = _create_test_candidate(
        package_name="pandas",
        affected_api="pandas.read_csv",
        file_path=src_file,
        line_number=10,
    )

    ctx = build_migration_context(candidate, analysis)

    assert u_exact in ctx.relevant_usages
    assert u_other not in ctx.relevant_usages


def test_12_package_level_candidates_include_package_usages_without_claiming_exact_match(tmp_path: Path):
    """Test 12 — Package-level candidates gather all package usages without claiming exact matches."""
    src_file = tmp_path / "app" / "loader.py"
    u1 = APIUsage("pandas", "pandas.read_csv", src_file, 10, "call")
    u2 = APIUsage("pandas", "pandas.DataFrame", src_file, 12, "class_reference")
    analysis = _create_mock_project_analysis(tmp_path, api_usages=[u1, u2])

    candidate = _create_test_candidate(
        package_name="pandas",
        affected_api=None,
        file_path=None,
        line_number=None,
    )

    ctx = build_migration_context(candidate, analysis)

    assert ctx.candidate.is_package_level
    assert u1 in ctx.relevant_usages
    assert u2 in ctx.relevant_usages


def test_13_relevant_documented_changes_preserved(tmp_path: Path):
    """Test 13 — Relevant documented changes are preserved in the migration context."""
    doc_chg = DocumentedChange(
        package_name="pandas",
        version=Version("2.0.0"),
        api_path="pandas.read_csv",
        change_type="deprecated",
        description="pandas.read_csv is deprecated",
        source_url="https://pandas.pydata.org/docs/release/2.0.0.html",
        confidence="high",
    )
    analysis = _create_mock_project_analysis(
        tmp_path,
        api_usages=[APIUsage("pandas", "pandas.read_csv", tmp_path / "app.py", 1, "call")],
        documented_changes=[doc_chg],
    )
    candidate = _create_test_candidate(
        package_name="pandas",
        affected_api="pandas.read_csv",
    )

    ctx = build_migration_context(candidate, analysis)

    assert len(ctx.relevant_changes) >= 1
    assert any(c.description == "pandas.read_csv is deprecated" for c in ctx.relevant_changes)


def test_14_relevant_change_documents_preserved(tmp_path: Path):
    """Test 14 — Authoritative change documents associated with package are preserved."""
    src = ChangeSource(
        package_name="pandas",
        url="https://pandas.pydata.org/docs/release/2.0.0.html",
        source_type="release_notes",
        is_official=True,
    )
    doc = ChangeDocument(
        package_name="pandas",
        source=src,
        title="Pandas 2.0.0 Release Notes",
        content="# Pandas 2.0.0\nWhat's new...",
        fetched_at=datetime.now(timezone.utc),
    )
    candidate = _create_test_candidate(package_name="pandas")
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis, documents=[doc])

    assert len(ctx.relevant_documents) == 1
    assert ctx.relevant_documents[0].title == "Pandas 2.0.0 Release Notes"
    assert ctx.relevant_documents[0].package_name == "pandas"


def test_15_source_urls_and_provenance_preserved(tmp_path: Path):
    """Test 15 — Source URLs and evidence links are preserved on candidate and context."""
    url = "https://pandas.pydata.org/docs/release/2.0.0.html"
    candidate = _create_test_candidate(source_url=url)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.candidate.source_url == url
    assert ctx.candidate.evidence is not None
    assert ctx.candidate.evidence.package_name == "pandas"


def test_16_current_and_target_versions_preserved(tmp_path: Path):
    """Test 16 — Current and target versions are preserved accurately on context properties."""
    candidate = _create_test_candidate(current_version="1.5.3", target_version="2.1.0")
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.current_version == Version("1.5.3")
    assert ctx.target_version == Version("2.1.0")
    assert ctx.package_name == "pandas"


def test_17_project_path_preserved(tmp_path: Path):
    """Test 17 — Project root path is preserved on MigrationContext."""
    candidate = _create_test_candidate()
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.project_path == tmp_path


def test_18_multiple_candidates_produce_deterministic_contexts_in_order(tmp_path: Path):
    """Test 18 — build_migration_contexts preserves exact candidate list ordering."""
    c1 = _create_test_candidate(affected_api="pandas.read_csv", line_number=10)
    c2 = _create_test_candidate(affected_api="pandas.DataFrame", line_number=20)
    c3 = _create_test_candidate(affected_api="pandas.Series", line_number=30)

    plan = MigrationPlan(project_path=tmp_path, candidates=[c1, c2, c3])
    analysis = _create_mock_project_analysis(tmp_path)

    contexts = build_migration_contexts(plan, analysis)

    assert len(contexts) == 3
    assert [ctx.candidate.affected_api for ctx in contexts] == [
        "pandas.read_csv",
        "pandas.DataFrame",
        "pandas.Series",
    ]


def test_19_missing_optional_evidence_does_not_crash(tmp_path: Path):
    """Test 19 — Candidate with minimal fields and no evidence object builds context cleanly."""
    candidate = MigrationCandidate(
        package_name="pandas",
        affected_api=None,
        file_path=None,
        line_number=None,
        current_version=None,
        target_version=None,
        change_type=None,
        risk_level=RiskLevel.LOW,
        confidence="low",
        description="Unknown package change",
        source_url=None,
        evidence=None,
    )
    analysis = _create_mock_project_analysis(tmp_path)

    ctx = build_migration_context(candidate, analysis)

    assert ctx.source_context is None
    assert ctx.relevant_usages == ()
    assert ctx.relevant_imports == ()
    assert ctx.relevant_changes == ()
    assert ctx.relevant_documents == ()
    assert ctx.package_name == "pandas"


def test_20_input_project_analysis_not_mutated(tmp_path: Path):
    """Test 20 — Input ProjectAnalysis is not mutated by context builder."""
    u = APIUsage("pandas", "pandas.read_csv", tmp_path / "app.py", 10, "call")
    analysis = _create_mock_project_analysis(tmp_path, api_usages=[u])
    original_api_usages = list(analysis.source_analysis.api_usages)
    original_results = list(analysis.project_scan.results)

    candidate = _create_test_candidate(file_path=tmp_path / "app.py", line_number=10)
    _ = build_migration_context(candidate, analysis)

    assert analysis.source_analysis.api_usages == original_api_usages
    assert analysis.project_scan.results == original_results


# ---------------------------------------------------------------------------
# Additional Targeted Tests (Security, Imports, Custom Windows)
# ---------------------------------------------------------------------------

def test_security_refuses_reading_env_and_private_keys(tmp_path: Path):
    """Test security — build_source_context refuses to read .env or .pem files."""
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_KEY=12345\nDATABASE_URL=postgres://...", encoding="utf-8")

    key_file = tmp_path / "server.key"
    key_file.write_text("-----BEGIN PRIVATE KEY-----\n...", encoding="utf-8")

    ctx_env = build_source_context(env_file, 1, project_path=tmp_path)
    ctx_key = build_source_context(key_file, 1, project_path=tmp_path)

    assert ctx_env is None
    assert ctx_key is None


def test_relevant_imports_extracted_for_same_file(tmp_path: Path):
    """Test imports — relevant import statements in the same file are linked to context."""
    src_file = tmp_path / "app" / "loader.py"
    other_file = tmp_path / "app" / "other.py"

    imp1 = ImportUsage("pandas", None, "pd", src_file, 1, "import")
    imp2 = ImportUsage("numpy", None, "np", src_file, 2, "import")
    imp3 = ImportUsage("pandas", None, "pd", other_file, 1, "import")

    analysis = _create_mock_project_analysis(
        tmp_path,
        import_usages=[imp1, imp2, imp3],
    )
    candidate = _create_test_candidate(file_path=src_file, line_number=10)

    ctx = build_migration_context(candidate, analysis)

    assert imp1 in ctx.relevant_imports
    assert imp2 not in ctx.relevant_imports  # Different package
    assert imp3 not in ctx.relevant_imports  # Different file


def test_custom_context_window_size(tmp_path: Path):
    """Test custom context_lines parameter."""
    src_file = _create_sample_file(tmp_path / "app" / "loader.py", num_lines=50)
    candidate = _create_test_candidate(file_path=src_file, line_number=20)
    analysis = _create_mock_project_analysis(tmp_path)

    ctx_3 = build_migration_context(candidate, analysis, context_lines=3)
    assert ctx_3.source_context is not None
    assert ctx_3.source_context.start_line == 17
    assert ctx_3.source_context.end_line == 23
    assert ctx_3.source_context.line_count == 7
