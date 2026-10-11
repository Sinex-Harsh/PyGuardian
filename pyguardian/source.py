"""
source.py — AST-based Python source code import & API symbol usage analysis.

This module provides static analysis of Python source files to extract both
package import declarations (`ImportUsage`) and detailed symbol/API usage locations
(`APIUsage`) using Python's standard-library `ast` module.

Design Principles:
- Pure static analysis: does NOT execute or import target project code.
- No network I/O: does NOT query PyPI or inspect installed packages.
- Isolated module: independent of packages.py, dependencies.py, and cli.py.
- Fault-tolerant: syntax errors in individual files do not crash the scan.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Directory names ignored by default during source tree discovery
DEFAULT_IGNORED_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    "build",
    "dist",
}

# Standard library module names to exclude from external dependency API usage tracking
STDLIB_MODULES: set[str] = getattr(
    sys,
    "stdlib_module_names",
    {
        "os",
        "sys",
        "json",
        "pathlib",
        "math",
        "typing",
        "re",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "unittest",
        "dataclasses",
        "logging",
        "subprocess",
        "urllib",
        "copy",
        "io",
        "time",
        "random",
        "ast",
    },
)


@dataclass(frozen=True)
class ImportUsage:
    """
    Represents a single import of a package in a source file.

    Attributes:
        package_name:  Top-level package name (e.g. 'pandas' for 'import pandas.core').
        imported_name: Specific symbol imported, if 'from X import Y' (e.g. 'DataFrame').
        alias:         Local alias defined with 'as' keyword (e.g. 'pd' or 'DF'), if any.
        file_path:     Path to the source file where this import was found.
        line_number:   1-based line number of the import statement.
        import_type:   Kind of import: 'import' or 'from_import'.
    """

    package_name: str
    imported_name: Optional[str]
    alias: Optional[str]
    file_path: Path
    line_number: int
    import_type: str


@dataclass(frozen=True)
class APIUsage:
    """
    Represents a single usage of a dependency API or symbol in a source file.

    Attributes:
        package_name: Top-level package name (e.g. 'pandas').
        api_path:     Fully qualified API path (e.g. 'pandas.DataFrame' or 'pandas.read_csv').
        file_path:    Path to the source file where this API usage was found.
        line_number:  1-based line number of the usage.
        usage_type:   Kind of usage ('call', 'class_reference', 'function_reference', 'attribute').
    """

    package_name: str
    api_path: str
    file_path: Path
    line_number: int
    usage_type: str


@dataclass(frozen=True)
class FileParseError:
    """
    Represents a syntax or file reading error encountered for a source file.

    Attributes:
        file_path:     Path to the source file that failed to parse.
        error_message: Human-readable error description.
    """

    file_path: Path
    error_message: str


@dataclass(frozen=True)
class SingleFileResult:
    """
    The result of statically analyzing a single Python source file.

    Attributes:
        file_path:   Path to the analyzed file.
        usages:      List of detected import usages.
        api_usages:  List of detected API symbol usages.
        parse_error: Set if the file could not be parsed due to syntax or I/O error.
    """

    file_path: Path
    usages: list[ImportUsage] = field(default_factory=list)
    api_usages: list[APIUsage] = field(default_factory=list)
    parse_error: Optional[FileParseError] = None

    @property
    def is_successful(self) -> bool:
        return self.parse_error is None


@dataclass(frozen=True)
class SourceAnalysis:
    """
    Aggregated AST import and API analysis results for an entire project source tree.

    Attributes:
        project_path:  Root directory of the project analyzed.
        usages:        List of all detected import usages across scanned files.
        api_usages:    List of all detected API symbol usages across scanned files.
        parse_errors:  List of parse failures encountered during scanning.
        files_scanned: Total number of Python files discovered and analyzed.
    """

    project_path: Path
    usages: list[ImportUsage] = field(default_factory=list)
    api_usages: list[APIUsage] = field(default_factory=list)
    parse_errors: list[FileParseError] = field(default_factory=list)
    files_scanned: int = 0


def extract_top_level_package(module_name: str) -> str:
    """
    Extract the top-level package name from a dot-separated module string.

    Examples:
        'pandas'            -> 'pandas'
        'pandas.core.frame' -> 'pandas'
    """
    if not module_name:
        return ""
    return module_name.split(".")[0]


@dataclass
class SymbolBinding:
    """Maps a locally accessible symbol name to its package and resolved prefix."""

    local_name: str
    package_name: str
    resolved_prefix: str
    import_type: str


class _SymbolCollectorVisitor(ast.NodeVisitor):
    """
    AST Visitor that collects import statements and populates a symbol resolution map.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.usages: list[ImportUsage] = []
        self.symbol_map: dict[str, SymbolBinding] = {}

    def visit_Import(self, node: ast.Import) -> None:
        line_no = getattr(node, "lineno", 1)
        for alias_node in node.names:
            raw_name = alias_node.name
            top_pkg = extract_top_level_package(raw_name)
            if not top_pkg:
                continue

            usage = ImportUsage(
                package_name=top_pkg,
                imported_name=None,
                alias=alias_node.asname,
                file_path=self.file_path,
                line_number=line_no,
                import_type="import",
            )
            self.usages.append(usage)

            local_name = alias_node.asname if alias_node.asname else raw_name
            if not alias_node.asname and "." in raw_name:
                local_name = top_pkg

            resolved_prefix = raw_name if alias_node.asname else top_pkg
            self.symbol_map[local_name] = SymbolBinding(
                local_name=local_name,
                package_name=top_pkg,
                resolved_prefix=resolved_prefix,
                import_type="import",
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Skip relative imports (e.g. `from .utils import helper`)
        if node.level > 0 or not node.module:
            return

        line_no = getattr(node, "lineno", 1)
        top_pkg = extract_top_level_package(node.module)
        if not top_pkg:
            return

        for alias_node in node.names:
            usage = ImportUsage(
                package_name=top_pkg,
                imported_name=alias_node.name,
                alias=alias_node.asname,
                file_path=self.file_path,
                line_number=line_no,
                import_type="from_import",
            )
            self.usages.append(usage)

            imported_symbol = alias_node.name
            if imported_symbol == "*":
                continue

            local_name = alias_node.asname if alias_node.asname else imported_symbol
            resolved_prefix = f"{node.module}.{imported_symbol}"
            self.symbol_map[local_name] = SymbolBinding(
                local_name=local_name,
                package_name=top_pkg,
                resolved_prefix=resolved_prefix,
                import_type="from_import",
            )
        self.generic_visit(node)


def _unwrap_attribute(node: ast.AST) -> Optional[list[str]]:
    """Unwrap a chain of ast.Attribute and ast.Name nodes into a list of string identifiers."""
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Attribute):
        parent_parts = _unwrap_attribute(node.value)
        if parent_parts is None:
            return None
        return parent_parts + [node.attr]
    return None


def _resolve_api_path(binding: SymbolBinding, parts: list[str]) -> str:
    """Resolve the full API path from a SymbolBinding and expression parts."""
    if len(parts) == 1:
        return binding.resolved_prefix
    return f"{binding.resolved_prefix}.{'.'.join(parts[1:])}"


class _APIUsageVisitor(ast.NodeVisitor):
    """
    AST Visitor that detects API and symbol usages using local symbol bindings.
    """

    def __init__(
        self,
        file_path: Path,
        symbol_map: dict[str, SymbolBinding],
        declared_packages: Optional[set[str]] = None,
    ) -> None:
        self.file_path = file_path
        self.symbol_map = symbol_map
        self.declared_packages = declared_packages
        self.api_usages: list[APIUsage] = []
        self.processed_nodes: set[int] = set()

    def _is_target_package(self, package_name: str) -> bool:
        if self.declared_packages is not None:
            return package_name in self.declared_packages
        return package_name not in STDLIB_MODULES

    def _mark_processed(self, node: ast.AST) -> None:
        self.processed_nodes.add(id(node))
        if isinstance(node, ast.Attribute):
            self._mark_processed(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if id(node) in self.processed_nodes:
            self.generic_visit(node)
            return

        func_parts = _unwrap_attribute(node.func)
        if func_parts and func_parts[0] in self.symbol_map:
            binding = self.symbol_map[func_parts[0]]
            if self._is_target_package(binding.package_name):
                api_path = _resolve_api_path(binding, func_parts)
                line_no = getattr(node, "lineno", getattr(node.func, "lineno", 1))
                self.api_usages.append(
                    APIUsage(
                        package_name=binding.package_name,
                        api_path=api_path,
                        file_path=self.file_path,
                        line_number=line_no,
                        usage_type="call",
                    )
                )
                self._mark_processed(node.func)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if id(node) in self.processed_nodes:
            self.generic_visit(node)
            return

        parts = _unwrap_attribute(node)
        if parts and parts[0] in self.symbol_map:
            binding = self.symbol_map[parts[0]]
            if self._is_target_package(binding.package_name):
                api_path = _resolve_api_path(binding, parts)
                line_no = getattr(node, "lineno", 1)
                last_part = parts[-1]
                if last_part[0].isupper():
                    utype = "class_reference"
                else:
                    utype = "function_reference"
                self.api_usages.append(
                    APIUsage(
                        package_name=binding.package_name,
                        api_path=api_path,
                        file_path=self.file_path,
                        line_number=line_no,
                        usage_type=utype,
                    )
                )
                self._mark_processed(node)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if id(node) in self.processed_nodes:
            return

        if not isinstance(node.ctx, ast.Load):
            return

        if node.id in self.symbol_map:
            binding = self.symbol_map[node.id]
            if binding.import_type == "from_import" and self._is_target_package(binding.package_name):
                line_no = getattr(node, "lineno", 1)
                utype = "class_reference" if node.id[0].isupper() else "function_reference"
                self.api_usages.append(
                    APIUsage(
                        package_name=binding.package_name,
                        api_path=binding.resolved_prefix,
                        file_path=self.file_path,
                        line_number=line_no,
                        usage_type=utype,
                    )
                )
                self.processed_nodes.add(id(node))


def analyze_file(
    file_path: Path,
    declared_packages: Optional[set[str]] = None,
) -> SingleFileResult:
    """
    Statically parse and analyze a single Python source file for import and API usages.

    Does not execute code or import the file.

    Args:
        file_path: Path to the Python file to analyze.
        declared_packages: Optional set of package names to restrict API analysis to.

    Returns:
        SingleFileResult containing discovered import usages, API usages, or a parse_error object.
    """
    path_obj = Path(file_path)
    try:
        source_code = path_obj.read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        return SingleFileResult(
            file_path=path_obj,
            parse_error=FileParseError(
                file_path=path_obj,
                error_message=f"Encoding error: {err}",
            ),
        )
    except OSError as err:
        return SingleFileResult(
            file_path=path_obj,
            parse_error=FileParseError(
                file_path=path_obj,
                error_message=f"File read error: {err}",
            ),
        )

    try:
        tree = ast.parse(source_code, filename=str(path_obj))
    except SyntaxError as err:
        return SingleFileResult(
            file_path=path_obj,
            parse_error=FileParseError(
                file_path=path_obj,
                error_message=f"Syntax error at line {err.lineno}: {err.msg}",
            ),
        )

    # Pass 1: Collect import statements and build symbol map
    symbol_visitor = _SymbolCollectorVisitor(file_path=path_obj)
    symbol_visitor.visit(tree)

    # Pass 2: Analyze API symbol usages using local symbol bindings
    api_visitor = _APIUsageVisitor(
        file_path=path_obj,
        symbol_map=symbol_visitor.symbol_map,
        declared_packages=declared_packages,
    )
    api_visitor.visit(tree)

    return SingleFileResult(
        file_path=path_obj,
        usages=symbol_visitor.usages,
        api_usages=api_visitor.api_usages,
    )


def analyze_api_usage(
    file_path: Path,
    declared_packages: Optional[set[str]] = None,
) -> list[APIUsage]:
    """
    Statically parse a single Python file and return discovered external API symbol usages.

    Args:
        file_path: Path to the Python file to analyze.
        declared_packages: Optional set of package names to restrict API analysis to.

    Returns:
        List of APIUsage objects detected in the file.
    """
    result = analyze_file(file_path, declared_packages=declared_packages)
    return result.api_usages


def _discover_python_files(root_dir: Path, skip_dirs: set[str]) -> list[Path]:
    """
    Find all .py files under root_dir recursively, skipping any directory in skip_dirs.
    """
    python_files: list[Path] = []
    try:
        entries = sorted(root_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return python_files

    for entry in entries:
        if entry.is_dir():
            if entry.name in skip_dirs:
                continue
            python_files.extend(_discover_python_files(entry, skip_dirs))
        elif entry.is_file() and entry.suffix == ".py":
            python_files.append(entry)

    return python_files


def analyze_source_tree(
    project_path: Path,
    ignored_dirs: Optional[Sequence[str]] = None,
    declared_packages: Optional[set[str]] = None,
) -> SourceAnalysis:
    """
    Discover and analyze all Python source files within a project directory.

    Args:
        project_path: Path to the project root directory.
        ignored_dirs: Optional custom sequence of directory names to skip.
        declared_packages: Optional set of package names to restrict API analysis to.

    Returns:
        SourceAnalysis aggregating import usages, API usages, and parse errors across the project.
    """
    path_obj = Path(project_path)
    if not path_obj.exists() or not path_obj.is_dir():
        return SourceAnalysis(
            project_path=path_obj,
            parse_errors=[
                FileParseError(
                    file_path=path_obj,
                    error_message=f"Directory does not exist or is not a directory: {path_obj}",
                )
            ],
            files_scanned=0,
        )

    skip_dirs = set(ignored_dirs) if ignored_dirs is not None else DEFAULT_IGNORED_DIRS

    all_usages: list[ImportUsage] = []
    all_api_usages: list[APIUsage] = []
    all_parse_errors: list[FileParseError] = []
    files_scanned = 0

    python_files = _discover_python_files(path_obj, skip_dirs)
    for py_file in python_files:
        files_scanned += 1
        result = analyze_file(py_file, declared_packages=declared_packages)
        if result.parse_error:
            all_parse_errors.append(result.parse_error)
        else:
            all_usages.extend(result.usages)
            all_api_usages.extend(result.api_usages)

    return SourceAnalysis(
        project_path=path_obj,
        usages=all_usages,
        api_usages=all_api_usages,
        parse_errors=all_parse_errors,
        files_scanned=files_scanned,
    )
