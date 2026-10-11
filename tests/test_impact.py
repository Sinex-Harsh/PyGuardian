"""
test_impact.py — Unit tests for Dependency ↔ Source Usage Mapping (pyguardian/impact.py).
"""

from pathlib import Path
import pytest

from pyguardian.dependencies import Dependency
from pyguardian.impact import (
    DependencyUsage,
    map_dependencies_to_usage,
    normalize_package_name,
)
from pyguardian.source import (
    FileParseError,
    ImportUsage,
    SourceAnalysis,
)


# ---------------------------------------------------------------------------
# Unit tests: normalize_package_name
# ---------------------------------------------------------------------------

class TestNormalizePackageName:

    def test_lowercase_conversion(self):
        assert normalize_package_name("Pandas") == "pandas"
        assert normalize_package_name("REQUESTS") == "requests"

    def test_hyphen_to_underscore(self):
        assert normalize_package_name("my-package") == "my_package"
        assert normalize_package_name("scikit-learn") == "scikit_learn"

    def test_combined_case_and_hyphen(self):
        assert normalize_package_name("My-Package") == "my_package"
        assert normalize_package_name("FOO-BAR_BAZ") == "foo_bar_baz"

    def test_already_normalized(self):
        assert normalize_package_name("pandas") == "pandas"
        assert normalize_package_name("my_package") == "my_package"

    def test_empty_string(self):
        assert normalize_package_name("") == ""


# ---------------------------------------------------------------------------
# Unit tests: DependencyUsage dataclass and properties
# ---------------------------------------------------------------------------

class TestDependencyUsageModel:

    def test_is_used_property(self):
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        usage = ImportUsage(
            package_name="pandas",
            imported_name=None,
            alias=None,
            file_path=Path("app.py"),
            line_number=1,
            import_type="import",
        )

        used_dep = DependencyUsage(dependency=dep, usages=[usage])
        unused_dep = DependencyUsage(dependency=dep, usages=[])

        assert used_dep.is_used is True
        assert unused_dep.is_used is False

    def test_usage_count_property(self):
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        u1 = ImportUsage("pandas", None, None, Path("app.py"), 1, "import")
        u2 = ImportUsage("pandas", "DataFrame", None, Path("analysis.py"), 12, "from_import")

        dep_usage = DependencyUsage(dependency=dep, usages=[u1, u2])
        assert dep_usage.usage_count == 2

    def test_files_used_in_property(self):
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        p1 = Path("app.py")
        p2 = Path("analysis.py")

        u1 = ImportUsage("pandas", None, "pd", p1, 1, "import")
        u2 = ImportUsage("pandas", "DataFrame", None, p1, 10, "from_import")  # Same file
        u3 = ImportUsage("pandas", "Series", None, p2, 5, "from_import")     # Different file

        dep_usage = DependencyUsage(dependency=dep, usages=[u1, u2, u3])
        # Preserves unique files in order of appearance
        assert dep_usage.files_used_in == [p1, p2]

    def test_files_used_in_empty_when_unused(self):
        dep = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        dep_usage = DependencyUsage(dependency=dep, usages=[])
        assert dep_usage.files_used_in == []


# ---------------------------------------------------------------------------
# Integration / Unit tests: map_dependencies_to_usage
# ---------------------------------------------------------------------------

class TestMapDependenciesToUsage:

    def test_direct_match(self):
        dep_pandas = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        usage = ImportUsage("pandas", None, None, Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[usage],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep_pandas], source_analysis)

        assert len(results) == 1
        res = results[0]
        assert res.dependency == dep_pandas
        assert res.is_used is True
        assert res.usage_count == 1
        assert res.usages == [usage]

    def test_multiple_usages_preserved(self):
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")

        u1 = ImportUsage("pandas", None, "pd", Path("app.py"), 1, "import")
        u2 = ImportUsage("pandas", "DataFrame", None, Path("analysis.py"), 12, "from_import")
        u3 = ImportUsage("pandas", "Series", None, Path("utils.py"), 5, "from_import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u1, u2, u3],
            files_scanned=3,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        res = results[0]
        assert res.is_used is True
        assert res.usage_count == 3
        assert res.usages == [u1, u2, u3]
        assert res.files_used_in == [Path("app.py"), Path("analysis.py"), Path("utils.py")]

    def test_unused_dependency(self):
        dep_numpy = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")
        u_pandas = ImportUsage("pandas", None, "pd", Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u_pandas],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep_numpy], source_analysis)

        assert len(results) == 1
        res = results[0]
        assert res.dependency == dep_numpy
        assert res.is_used is False
        assert res.usage_count == 0
        assert res.usages == []
        assert res.files_used_in == []

    def test_multiple_dependencies(self):
        dep_pandas = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        dep_requests = Dependency(name="requests", raw_line="requests==2.31.0", specifier="==2.31.0")
        dep_numpy = Dependency(name="numpy", raw_line="numpy==1.26.4", specifier="==1.26.4")

        u_pandas1 = ImportUsage("pandas", None, "pd", Path("app.py"), 1, "import")
        u_pandas2 = ImportUsage("pandas", "DataFrame", None, Path("analysis.py"), 12, "from_import")
        u_requests = ImportUsage("requests", "get", None, Path("api.py"), 5, "from_import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u_pandas1, u_pandas2, u_requests],
            files_scanned=3,
        )

        results = map_dependencies_to_usage(
            [dep_pandas, dep_requests, dep_numpy],
            source_analysis,
        )

        assert len(results) == 3

        res_pandas = results[0]
        assert res_pandas.dependency.name == "pandas"
        assert res_pandas.is_used is True
        assert res_pandas.usage_count == 2
        assert res_pandas.files_used_in == [Path("app.py"), Path("analysis.py")]

        res_requests = results[1]
        assert res_requests.dependency.name == "requests"
        assert res_requests.is_used is True
        assert res_requests.usage_count == 1
        assert res_requests.files_used_in == [Path("api.py")]

        res_numpy = results[2]
        assert res_numpy.dependency.name == "numpy"
        assert res_numpy.is_used is False
        assert res_numpy.usage_count == 0
        assert res_numpy.files_used_in == []

    def test_aliased_imports(self):
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        u_aliased = ImportUsage("pandas", None, "pd", Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u_aliased],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        assert results[0].is_used is True
        assert results[0].usages[0].alias == "pd"

    def test_from_imports(self):
        dep = Dependency(name="requests", raw_line="requests==2.31.0", specifier="==2.31.0")
        u_from = ImportUsage("requests", "get", None, Path("app.py"), 1, "from_import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u_from],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        assert results[0].is_used is True
        assert results[0].usages[0].imported_name == "get"

    def test_nested_imports(self):
        """
        source.py extracts top-level package_name as 'pandas' for 'import pandas.core.frame'.
        The mapper should match this to the 'pandas' dependency.
        """
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        u_nested = ImportUsage("pandas", None, None, Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u_nested],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        assert results[0].is_used is True

    def test_distribution_name_normalization(self):
        """
        Distribution name: 'my-package'
        Source import: 'my_package'
        Must match cleanly via normalization.
        """
        dep = Dependency(name="my-package", raw_line="my-package==1.0.0", specifier="==1.0.0")
        u = ImportUsage("my_package", None, None, Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        res = results[0]
        assert res.is_used is True
        assert res.usage_count == 1
        assert res.usages[0] == u

    def test_case_insensitive_normalization(self):
        """
        Distribution name: 'PyYAML'
        Source import: 'yaml' (wait, yaml is a name mismatch, PyYAML vs yaml, PyYAML remains unmatched)
        Distribution name: 'My-Package'
        Source import: 'MY_PACKAGE'
        """
        dep = Dependency(name="My-Package", raw_line="My-Package==1.0.0", specifier="==1.0.0")
        u = ImportUsage("MY_PACKAGE", None, None, Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        assert results[0].is_used is True

    def test_unmatched_dependency_distribution_mismatch(self):
        """
        When distribution name != import name and cannot be normalized (e.g. scikit-learn vs sklearn),
        the mapper should classify it as unmatched (is_used=False) rather than guessing.
        """
        dep = Dependency(name="scikit-learn", raw_line="scikit-learn==1.4.0", specifier="==1.4.0")
        u = ImportUsage("sklearn", None, None, Path("model.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        res = results[0]
        assert res.is_used is False
        assert res.usage_count == 0
        assert res.usages == []

    def test_empty_project(self):
        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[],
            files_scanned=0,
        )
        results = map_dependencies_to_usage([], source_analysis)
        assert results == []

    def test_source_parse_errors_with_valid_usages(self):
        """
        If SourceAnalysis contains parse errors for some files, valid usages from other
        files must still be mapped properly.
        """
        dep = Dependency(name="pandas", raw_line="pandas==2.2.0", specifier="==2.2.0")
        u_valid = ImportUsage("pandas", None, "pd", Path("good.py"), 1, "import")
        parse_err = FileParseError(Path("bad.py"), "SyntaxError at line 5")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u_valid],
            parse_errors=[parse_err],
            files_scanned=2,
        )

        results = map_dependencies_to_usage([dep], source_analysis)

        assert len(results) == 1
        res = results[0]
        assert res.is_used is True
        assert res.usage_count == 1
        assert res.usages == [u_valid]

    def test_invalid_dependency_in_list(self):
        """
        If a Dependency object has a parse error, it should be mapped as unused with usages=[].
        """
        dep_invalid = Dependency(
            name="!!!bad",
            raw_line="!!!bad",
            parse_error="Invalid requirement string",
        )
        u = ImportUsage("bad", None, None, Path("app.py"), 1, "import")

        source_analysis = SourceAnalysis(
            project_path=Path("/project"),
            usages=[u],
            files_scanned=1,
        )

        results = map_dependencies_to_usage([dep_invalid], source_analysis)

        assert len(results) == 1
        assert results[0].is_used is False
        assert results[0].usages == []
