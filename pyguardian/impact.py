"""
impact.py — Dependency ↔ Source Usage Mapping layer.

Responsibilities:
- Map declared dependencies (from dependencies.py) to actual AST source imports
  (from source.py).
- Normalize distribution names and Python package import names for matching.
- Retain full source location intelligence (file path, line number, import details).
- Provide a clean, deterministic, in-memory mapping model.

Design Principles:
- Pure mapping logic: NO filesystem I/O, NO network calls, NO AST parsing inside mapper.
- Fault-tolerant: operates on structured inputs (Sequence[Dependency], SourceAnalysis).
- Unused dependencies are preserved with zero usages (is_used=False).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from pyguardian.dependencies import Dependency
from pyguardian.source import ImportUsage, SourceAnalysis


def normalize_package_name(name: str) -> str:
    """
    Normalize package names and import names for comparison.

    Normalizes names by:
    - Converting to lowercase
    - Replacing hyphens ('-') with underscores ('_')

    Examples:
        'my-package' -> 'my_package'
        'My_Package' -> 'my_package'
        'pandas'     -> 'pandas'
        ''           -> ''
    """
    if not name:
        return ""
    return name.lower().replace("-", "_")


@dataclass(frozen=True)
class DependencyUsage:
    """
    Represents the usage mapping of a single declared dependency across source code.

    Attributes:
        dependency: The declared Dependency object.
        usages:     List of ImportUsage instances corresponding to this dependency.
    """

    dependency: Dependency
    usages: list[ImportUsage] = field(default_factory=list)

    @property
    def is_used(self) -> bool:
        """Returns True if the dependency is imported anywhere in the source code."""
        return len(self.usages) > 0

    @property
    def usage_count(self) -> int:
        """Returns the total number of import occurrences found across all files."""
        return len(self.usages)

    @property
    def files_used_in(self) -> list[Path]:
        """
        Returns a list of unique source file paths where this dependency is imported,
        preserving the order of first appearance.
        """
        seen: set[Path] = set()
        files: list[Path] = []
        for usage in self.usages:
            if usage.file_path not in seen:
                seen.add(usage.file_path)
                files.append(usage.file_path)
        return files


def map_dependencies_to_usage(
    dependencies: Sequence[Dependency],
    source_analysis: SourceAnalysis,
) -> list[DependencyUsage]:
    """
    Match declared dependencies against actual import usages discovered via AST analysis.

    Args:
        dependencies:    Already-parsed Dependency entries from requirements file(s).
        source_analysis: Aggregated AST source import analysis for the project.

    Returns:
        List of DependencyUsage objects, one for each input dependency in original order.

    Behavior:
    - Case-insensitive matching with hyphen/underscore normalization (e.g. 'my-pkg' ↔ 'my_pkg').
    - Retains all exact import locations (file, line number, symbol, alias).
    - Preserves declared dependencies that have no matching imports (is_used=False).
    - Skips mapping for invalid dependencies (parse_error set on Dependency).
    - Pure function: performs zero I/O and zero AST parsing.
    """
    if not dependencies:
        return []

    # Index usages by normalized package name
    usage_map: dict[str, list[ImportUsage]] = defaultdict(list)
    for usage in source_analysis.usages:
        norm_pkg = normalize_package_name(usage.package_name)
        if norm_pkg:
            usage_map[norm_pkg].append(usage)

    mapped_results: list[DependencyUsage] = []
    for dep in dependencies:
        if not dep.is_valid:
            # Dependency had a parse error line; cannot match
            mapped_results.append(DependencyUsage(dependency=dep, usages=[]))
            continue

        norm_dep_name = normalize_package_name(dep.name)
        usages_for_dep = usage_map.get(norm_dep_name, [])
        mapped_results.append(
            DependencyUsage(
                dependency=dep,
                usages=list(usages_for_dep),
            )
        )

    return mapped_results
