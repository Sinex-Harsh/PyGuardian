"""
dependencies.py — Dependency file detection and parsing.

Responsibilities:
- Locate dependency files within a project directory.
- Parse requirements.txt into structured Dependency objects.
- Handle malformed lines gracefully.

V0.1 supports: requirements.txt only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.specifiers import SpecifierSet

# We use packaging.requirements for correct, standards-compliant parsing.
# This handles all valid requirement formats:
#   numpy==1.26.4
#   requests>=2.28,<3
#   pandas[excel]~=2.0
#   -r other-requirements.txt   (we skip these for now)
from packaging.requirements import Requirement, InvalidRequirement


@dataclass
class Dependency:
    """
    Represents a single dependency entry from a requirements file.

    Attributes:
        name:        Normalized package name (e.g. "numpy").
        raw_line:    The original text line from the file.
        specifier:   The version constraint string (e.g. "==1.26.4", ">=2.0").
        extras:      Optional extras (e.g. ["excel"] for pandas[excel]).
        source_file: Path to the file this dependency came from.
        line_number: Line number in the source file (1-indexed).
        parse_error: If the line could not be parsed, the error message.
    """

    name: str
    raw_line: str
    specifier: str = ""
    extras: list[str] = field(default_factory=list)
    source_file: Optional[Path] = None
    line_number: int = 0
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Returns True if the dependency was parsed without errors."""
        return self.parse_error is None

    @property
    def declared_version(self) -> str:
        """
        Extract a bare version string from the specifier, if possible.

        For pinned requirements (e.g. ``numpy==1.26.4``), returns ``"1.26.4"``.
        For range specifiers (e.g. ``requests>=2.28,<3``), returns the version
        from the first specifier clause (``"2.28"`` in this case).
        For bare package names with no specifier, returns ``""``.

        This property is the bridge between Dependency and classify_update(),
        which expects a plain version string, not a constraint expression.
        project.py uses this to determine "what version does this project
        currently declare?" without needing to know about SpecifierSet internals.

        Examples::

            Dependency(name="numpy", specifier="==1.26.4").declared_version
            # → "1.26.4"

            Dependency(name="requests", specifier=">=2.28,<3").declared_version
            # → "2.28" (lower bound; not perfect but better than nothing)

            Dependency(name="flask", specifier="").declared_version
            # → ""
        """
        if not self.specifier:
            return ""
        try:
            spec_set = SpecifierSet(self.specifier)
            specs = list(spec_set)
            if not specs:
                return ""
            # Prefer an exact pin (==) if present; fall back to the first clause.
            for spec in specs:
                if spec.operator == "==":
                    return spec.version
            return specs[0].version
        except Exception:
            return ""

    def __str__(self) -> str:
        if self.specifier:
            return f"{self.name}{self.specifier}"
        return self.name


def _is_ignorable_line(line: str) -> bool:
    """
    Return True for lines that should be skipped during parsing.

    We skip:
    - Empty lines
    - Comment lines (starting with #)
    - Option flags (starting with - e.g., -r, --index-url)
    - URL-based requirements (starting with http/https/git+)
      These are valid pip syntax but not pip-installable package names
      that we can look up on PyPI yet.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("-"):
        return True
    if re.match(r"^(https?://|git\+)", stripped, re.IGNORECASE):
        return True
    return False


def parse_requirements_file(filepath: Path) -> list[Dependency]:
    """
    Parse a requirements.txt file and return a list of Dependency objects.

    Each valid line is parsed using packaging.requirements.Requirement,
    which correctly handles all standard pip requirement formats.

    Lines that cannot be parsed are returned as Dependency objects with
    parse_error set — they are never silently dropped.

    Args:
        filepath: Absolute or relative path to the requirements.txt file.

    Returns:
        A list of Dependency objects (valid or with parse errors).

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError:   If the file cannot be read.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Requirements file not found: {filepath}\n"
            f"Make sure the path is correct and the file exists."
        )

    dependencies: list[Dependency] = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, raw_line in enumerate(lines, start=1):
        # Strip inline comments and surrounding whitespace
        line = raw_line.split("#")[0].strip()

        if _is_ignorable_line(line):
            continue

        try:
            req = Requirement(line)
            dep = Dependency(
                name=req.name,
                raw_line=raw_line.rstrip("\n"),
                specifier=str(req.specifier),
                extras=sorted(req.extras),
                source_file=filepath,
                line_number=line_number,
            )
        except InvalidRequirement as exc:
            # Don't silently discard invalid lines — surface them.
            dep = Dependency(
                name=line,           # Use raw line as the name placeholder
                raw_line=raw_line.rstrip("\n"),
                source_file=filepath,
                line_number=line_number,
                parse_error=str(exc),
            )

        dependencies.append(dep)

    return dependencies


def find_requirements_files(project_path: Path) -> list[Path]:
    """
    Search a project directory for requirements.txt files.

    Currently checks:
    - requirements.txt (root level)

    Future versions will also check:
    - requirements/*.txt
    - pyproject.toml
    - Pipfile
    - poetry.lock

    Args:
        project_path: Root directory of the Python project.

    Returns:
        A list of found requirements file paths (may be empty).
    """
    candidates = [
        project_path / "requirements.txt",
    ]

    found = [p for p in candidates if p.exists() and p.is_file()]
    return found
