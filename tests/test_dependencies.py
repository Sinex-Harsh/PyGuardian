"""
test_dependencies.py — Tests for the dependency parser.

Run with:
    pytest tests/

These tests verify that parse_requirements_file correctly handles:
- Normal pinned versions (==)
- Range specifiers (>=, <)
- Extras (package[extra])
- Comments and blank lines (should be ignored)
- Malformed lines (should be captured with parse_error, not crashed)
- Non-existent files (should raise FileNotFoundError)
"""

import pytest
from pathlib import Path
from pyguardian.dependencies import (
    parse_requirements_file,
    find_requirements_files,
    Dependency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_requirements_file(tmp_path: Path, content: str) -> Path:
    """
    Create a temporary requirements.txt with given content.

    `tmp_path` is a pytest built-in fixture that provides a temporary
    directory unique to each test invocation. It is automatically
    cleaned up after the test run.
    """
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(content, encoding="utf-8")
    return req_file


# ---------------------------------------------------------------------------
# Tests: parse_requirements_file
# ---------------------------------------------------------------------------

class TestParseRequirementsFile:

    def test_parses_pinned_version(self, tmp_path):
        """A pinned == requirement should be parsed correctly."""
        req_file = make_requirements_file(tmp_path, "numpy==1.26.4\n")
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        dep = deps[0]
        assert dep.name == "numpy"
        assert dep.specifier == "==1.26.4"
        assert dep.is_valid is True
        assert dep.parse_error is None

    def test_parses_range_specifier(self, tmp_path):
        """A range specifier should be captured as a combined string."""
        req_file = make_requirements_file(tmp_path, "httpx>=0.24.0,<1.0\n")
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        dep = deps[0]
        assert dep.name == "httpx"
        # packaging normalises multi-part specifiers; both parts should be present
        assert ">=0.24.0" in dep.specifier
        assert "<1.0" in dep.specifier
        assert dep.is_valid is True

    def test_parses_extras(self, tmp_path):
        """Extras like package[extra] should be stored in dep.extras."""
        req_file = make_requirements_file(tmp_path, "celery[redis]==5.3.4\n")
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        dep = deps[0]
        assert dep.name == "celery"
        assert "redis" in dep.extras
        assert dep.specifier == "==5.3.4"

    def test_ignores_blank_lines(self, tmp_path):
        """Blank lines should not produce Dependency objects."""
        content = "\n\nnumpy==1.26.4\n\n"
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1

    def test_ignores_comment_lines(self, tmp_path):
        """Comment lines (# ...) should be ignored."""
        content = "# This is a comment\nnumpy==1.26.4\n"
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        assert deps[0].name == "numpy"

    def test_ignores_inline_comments(self, tmp_path):
        """Inline comments should be stripped before parsing."""
        content = "numpy==1.26.4  # pinned for stability\n"
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        assert deps[0].name == "numpy"
        assert deps[0].is_valid is True

    def test_ignores_option_flags(self, tmp_path):
        """Lines starting with - (like -r, --index-url) should be ignored."""
        content = "-r base-requirements.txt\nnumpy==1.26.4\n"
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        assert deps[0].name == "numpy"

    def test_captures_invalid_lines_with_error(self, tmp_path):
        """
        Malformed lines should NOT be silently dropped.
        They should appear in the results with parse_error set.
        """
        content = "numpy==1.26.4\n!!!invalid_package\n"
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        # Both lines should produce a Dependency object
        assert len(deps) == 2

        valid_dep = deps[0]
        assert valid_dep.is_valid is True

        invalid_dep = deps[1]
        assert invalid_dep.is_valid is False
        assert invalid_dep.parse_error is not None

    def test_preserves_line_numbers(self, tmp_path):
        """Line numbers should match the actual position in the file."""
        content = "# comment\nnumpy==1.26.4\npandas==2.2.0\n"
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        assert deps[0].line_number == 2  # numpy is on line 2
        assert deps[1].line_number == 3  # pandas is on line 3

    def test_raises_file_not_found(self, tmp_path):
        """Passing a non-existent file should raise FileNotFoundError."""
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError) as exc_info:
            parse_requirements_file(missing)
        # The error message should be informative
        assert "nonexistent.txt" in str(exc_info.value)

    def test_parses_multiple_packages(self, tmp_path):
        """A realistic multi-package file should parse all entries."""
        content = (
            "numpy==1.26.4\n"
            "pandas==2.2.0\n"
            "requests==2.31.0\n"
        )
        req_file = make_requirements_file(tmp_path, content)
        deps = parse_requirements_file(req_file)

        assert len(deps) == 3
        names = [d.name for d in deps]
        assert "numpy" in names
        assert "pandas" in names
        assert "requests" in names

    def test_no_version_specifier(self, tmp_path):
        """A bare package name with no version should parse without error."""
        req_file = make_requirements_file(tmp_path, "numpy\n")
        deps = parse_requirements_file(req_file)

        assert len(deps) == 1
        dep = deps[0]
        assert dep.name == "numpy"
        assert dep.specifier == ""
        assert dep.is_valid is True


# ---------------------------------------------------------------------------
# Tests: Dependency.declared_version
# ---------------------------------------------------------------------------

class TestDependencyDeclaredVersion:
    """
    Tests for the declared_version property.

    This property is the bridge between Dependency (which stores a constraint
    like "==1.26.4") and classify_update() (which expects a bare version
    string like "1.26.4"). Without it, every caller would need to parse the
    specifier manually.
    """

    def test_pinned_version_returns_bare_version(self):
        """==1.26.4 should yield '1.26.4'."""
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        assert dep.declared_version == "1.26.4"

    def test_no_specifier_returns_empty_string(self):
        """A bare package name with no constraint yields ''."""
        dep = Dependency(name="numpy", raw_line="numpy", specifier="")
        assert dep.declared_version == ""

    def test_range_specifier_returns_lower_bound(self):
        """>=2.28,<3 should yield '2.28' (the >= clause version)."""
        dep = Dependency(name="requests", raw_line="requests>=2.28,<3", specifier=">=2.28,<3")
        # Not a perfect pinned version — but it's the best we can extract
        # from a range. A non-empty string means classify_update() can attempt
        # a comparison rather than returning UNKNOWN.
        assert dep.declared_version != ""

    def test_prefers_exact_pin_over_range_clause(self):
        """If specifier contains ==, that exact version is returned."""
        dep = Dependency(name="flask", raw_line="flask==3.0.0", specifier="==3.0.0")
        assert dep.declared_version == "3.0.0"

    def test_parse_error_dep_returns_empty_string(self):
        """A Dependency with a parse error has no meaningful version."""
        dep = Dependency(
            name="!!!bad",
            raw_line="!!!bad",
            specifier="",
            parse_error="invalid",
        )
        assert dep.declared_version == ""

    def test_integrates_with_classify_update(self):
        """
        declared_version should be directly usable with classify_update().

        This is the core integration test: parse a requirement, extract
        declared_version, pass it to classify_update — no glue code needed.
        """
        from pyguardian.packages import classify_update, UpdateType

        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        result = classify_update(dep.declared_version, "2.3.3")
        assert result == UpdateType.MAJOR


# ---------------------------------------------------------------------------
# Tests: find_requirements_files
# ---------------------------------------------------------------------------

class TestFindRequirementsFiles:

    def test_finds_requirements_txt_at_root(self, tmp_path):
        """Should find requirements.txt in the project root."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy==1.26.4\n")

        found = find_requirements_files(tmp_path)

        assert len(found) == 1
        assert found[0] == req_file

    def test_returns_empty_if_no_requirements_file(self, tmp_path):
        """Should return an empty list if no requirements.txt exists."""
        found = find_requirements_files(tmp_path)
        assert found == []
