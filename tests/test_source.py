"""
test_source.py — Unit tests for PyGuardian's AST source code import and API usage intelligence (source.py).
"""

from pathlib import Path
import pytest

from pyguardian.source import (
    APIUsage,
    FileParseError,
    ImportUsage,
    SingleFileResult,
    SourceAnalysis,
    analyze_api_usage,
    analyze_file,
    analyze_source_tree,
    extract_top_level_package,
)


# ---------------------------------------------------------------------------
# Unit tests: extract_top_level_package
# ---------------------------------------------------------------------------

class TestExtractTopLevelPackage:

    def test_single_name(self):
        assert extract_top_level_package("pandas") == "pandas"

    def test_nested_module(self):
        assert extract_top_level_package("pandas.core.frame") == "pandas"

    def test_deeply_nested_module(self):
        assert extract_top_level_package("a.b.c.d.e") == "a"

    def test_empty_string(self):
        assert extract_top_level_package("") == ""


# ---------------------------------------------------------------------------
# Unit tests: analyze_file (Single File AST Import Analysis)
# ---------------------------------------------------------------------------

class TestAnalyzeFile:

    def test_basic_import(self, tmp_path):
        py_file = tmp_path / "app.py"
        py_file.write_text("import pandas\n", encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 1
        usage = res.usages[0]
        assert usage.package_name == "pandas"
        assert usage.imported_name is None
        assert usage.alias is None
        assert usage.import_type == "import"
        assert usage.line_number == 1
        assert usage.file_path == py_file

    def test_aliased_import(self, tmp_path):
        py_file = tmp_path / "app.py"
        py_file.write_text("import pandas as pd\n", encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 1
        usage = res.usages[0]
        assert usage.package_name == "pandas"
        assert usage.imported_name is None
        assert usage.alias == "pd"
        assert usage.import_type == "import"

    def test_from_import(self, tmp_path):
        py_file = tmp_path / "app.py"
        py_file.write_text("from pandas import DataFrame\n", encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 1
        usage = res.usages[0]
        assert usage.package_name == "pandas"
        assert usage.imported_name == "DataFrame"
        assert usage.alias is None
        assert usage.import_type == "from_import"

    def test_aliased_from_import(self, tmp_path):
        py_file = tmp_path / "app.py"
        py_file.write_text("from pandas import DataFrame as DF\n", encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 1
        usage = res.usages[0]
        assert usage.package_name == "pandas"
        assert usage.imported_name == "DataFrame"
        assert usage.alias == "DF"
        assert usage.import_type == "from_import"

    def test_multiple_imports_in_one_file(self, tmp_path):
        code = (
            "import pandas\n"
            "import requests\n"
            "from flask import Flask\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 3
        pkgs = [u.package_name for u in res.usages]
        assert pkgs == ["pandas", "requests", "flask"]

    def test_multiple_names_on_single_import_line(self, tmp_path):
        code = "import os, sys, math\n"
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 3
        pkgs = [u.package_name for u in res.usages]
        assert pkgs == ["os", "sys", "math"]
        assert all(u.line_number == 1 for u in res.usages)

    def test_nested_package_import(self, tmp_path):
        code = "import pandas.core.frame\n"
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 1
        assert res.usages[0].package_name == "pandas"

    def test_nested_package_from_import(self, tmp_path):
        code = "from pandas.core.frame import DataFrame\n"
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 1
        assert res.usages[0].package_name == "pandas"
        assert res.usages[0].imported_name == "DataFrame"

    def test_relative_imports_ignored(self, tmp_path):
        code = (
            "from .utils import helper\n"
            "from ..config import settings\n"
            "from . import internal_mod\n"
            "import requests\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        # Only requests should be extracted
        assert len(res.usages) == 1
        assert res.usages[0].package_name == "requests"

    def test_correct_line_numbers(self, tmp_path):
        code = (
            "# Comment line 1\n"
            "x = 10\n"
            "\n"
            "import numpy\n"  # Line 4
            "\n"
            "from flask import Flask\n"  # Line 6
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert res.is_successful
        assert len(res.usages) == 2
        assert res.usages[0].package_name == "numpy"
        assert res.usages[0].line_number == 4
        assert res.usages[1].package_name == "flask"
        assert res.usages[1].line_number == 6

    def test_syntax_error_handling(self, tmp_path):
        code = "def bad_function(\n    import pandas\n"  # Invalid syntax
        py_file = tmp_path / "broken.py"
        py_file.write_text(code, encoding="utf-8")

        res = analyze_file(py_file)

        assert not res.is_successful
        assert res.parse_error is not None
        assert res.parse_error.file_path == py_file
        assert "Syntax error" in res.parse_error.error_message
        assert len(res.usages) == 0

    def test_nonexistent_file_handling(self, tmp_path):
        missing_file = tmp_path / "does_not_exist.py"

        res = analyze_file(missing_file)

        assert not res.is_successful
        assert res.parse_error is not None
        assert res.parse_error.file_path == missing_file


# ---------------------------------------------------------------------------
# Unit tests: API & Symbol Usage Analysis (V0.2 Step 5)
# ---------------------------------------------------------------------------

class TestAPIUsageAnalysis:

    def test_package_qualified_call(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        assert len(api_usages) == 1
        usage = api_usages[0]
        assert usage.package_name == "pandas"
        assert usage.api_path == "pandas.read_csv"
        assert usage.line_number == 2
        assert usage.usage_type == "call"

    def test_package_qualified_class(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame(data)\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        assert len(api_usages) == 1
        usage = api_usages[0]
        assert usage.package_name == "pandas"
        assert usage.api_path == "pandas.DataFrame"
        assert usage.usage_type == "call"

    def test_direct_from_import_usage(self, tmp_path):
        code = (
            "from pandas import DataFrame\n"
            "df = DataFrame(data)\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        assert len(api_usages) == 1
        usage = api_usages[0]
        assert usage.package_name == "pandas"
        assert usage.api_path == "pandas.DataFrame"
        assert usage.usage_type == "call"

    def test_aliased_from_import_usage(self, tmp_path):
        code = (
            "from pandas import DataFrame as DF\n"
            "df = DF(data)\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        assert len(api_usages) == 1
        usage = api_usages[0]
        assert usage.package_name == "pandas"
        assert usage.api_path == "pandas.DataFrame"
        assert usage.usage_type == "call"

    def test_multiple_apis_retained(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv('x.csv')\n"
            "df2 = pd.DataFrame(df)\n"
            "res = pd.concat([df, df2])\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        paths = [u.api_path for u in api_usages]
        assert paths == ["pandas.read_csv", "pandas.DataFrame", "pandas.concat"]

    def test_nested_attribute_path(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "res = pd.io.json.read_json('data.json')\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        assert len(api_usages) == 1
        usage = api_usages[0]
        assert usage.package_name == "pandas"
        assert usage.api_path == "pandas.io.json.read_json"
        assert usage.usage_type == "call"

    def test_multiple_dependencies(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "import requests as req\n"
            "df = pd.read_csv('data.csv')\n"
            "resp = req.get('https://example.org')\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        pkgs = set(u.package_name for u in api_usages)
        assert pkgs == {"pandas", "requests"}
        paths = [u.api_path for u in api_usages]
        assert "pandas.read_csv" in paths
        assert "requests.get" in paths

    def test_function_reference(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "reader = pd.read_csv\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        assert len(api_usages) == 1
        usage = api_usages[0]
        assert usage.package_name == "pandas"
        assert usage.api_path == "pandas.read_csv"
        assert usage.usage_type == "function_reference"

    def test_method_limitation_does_not_guess_object_types(self, tmp_path):
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame(data)\n"
            "df.to_csv('out.csv')\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        # pd.DataFrame is detected
        # df.to_csv is NOT detected as pandas.DataFrame.to_csv (as per limitation specification)
        paths = [u.api_path for u in api_usages]
        assert paths == ["pandas.DataFrame"]

    def test_relative_imports_do_not_create_external_api_usages(self, tmp_path):
        code = (
            "from .utils import helper\n"
            "from ..config import settings\n"
            "res = helper()\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)
        assert api_usages == []

    def test_stdlib_imports_excluded_by_default(self, tmp_path):
        code = (
            "import os\n"
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "p = Path('file.txt')\n"
            "data = json.loads('{}')\n"
            "os.path.join('a', 'b')\n"
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)

        # Stdlib APIs should be excluded by default
        assert api_usages == []

    def test_line_number_accuracy(self, tmp_path):
        code = (
            "# Line 1\n"
            "# Line 2\n"
            "import pandas as pd\n"  # Line 3
            "\n"
            "x = 100\n"  # Line 5
            "df = pd.read_csv('x.csv')\n"  # Line 6
        )
        py_file = tmp_path / "app.py"
        py_file.write_text(code, encoding="utf-8")

        api_usages = analyze_api_usage(py_file)
        assert len(api_usages) == 1
        assert api_usages[0].line_number == 6


# ---------------------------------------------------------------------------
# Unit tests: analyze_source_tree (Directory & Tree Analysis)
# ---------------------------------------------------------------------------

class TestAnalyzeSourceTree:

    def test_multi_file_directory_aggregation(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        f1 = src_dir / "main.py"
        f1.write_text("import numpy as np\nfrom pandas import DataFrame\ndf = DataFrame()\n", encoding="utf-8")

        f2 = src_dir / "utils.py"
        f2.write_text("import requests\nresp = requests.get('https://e.org')\n", encoding="utf-8")

        analysis = analyze_source_tree(tmp_path)

        assert analysis.files_scanned == 2
        assert len(analysis.parse_errors) == 0
        pkgs = set(u.package_name for u in analysis.usages)
        assert pkgs == {"numpy", "pandas", "requests"}

        api_paths = set(u.api_path for u in analysis.api_usages)
        assert api_paths == {"pandas.DataFrame", "requests.get"}

    def test_ignored_directories(self, tmp_path):
        # Create standard source file
        app_file = tmp_path / "app.py"
        app_file.write_text("import flask\napp = flask.Flask(__name__)\n", encoding="utf-8")

        # Create files inside ignored directories
        venv_file = tmp_path / ".venv" / "lib" / "installed.py"
        venv_file.parent.mkdir(parents=True)
        venv_file.write_text("import numpy\nnp = numpy.array([1])\n", encoding="utf-8")

        git_file = tmp_path / ".git" / "hook.py"
        git_file.parent.mkdir(parents=True)
        git_file.write_text("import gitpkg\ngitpkg.run()\n", encoding="utf-8")

        cache_file = tmp_path / "__pycache__" / "cached.py"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("import cachepkg\ncachepkg.load()\n", encoding="utf-8")

        analysis = analyze_source_tree(tmp_path)

        assert analysis.files_scanned == 1
        assert len(analysis.usages) == 1
        assert analysis.usages[0].package_name == "flask"

        assert len(analysis.api_usages) == 1
        assert analysis.api_usages[0].api_path == "flask.Flask"

    def test_syntax_error_file_aggregated_without_crashing(self, tmp_path):
        good_file = tmp_path / "good.py"
        good_file.write_text("import requests\nrequests.get('x')\n", encoding="utf-8")

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken_syntax(\n", encoding="utf-8")

        analysis = analyze_source_tree(tmp_path)

        assert analysis.files_scanned == 2
        assert len(analysis.usages) == 1
        assert analysis.usages[0].package_name == "requests"
        assert len(analysis.api_usages) == 1
        assert analysis.api_usages[0].api_path == "requests.get"
        assert len(analysis.parse_errors) == 1
        assert analysis.parse_errors[0].file_path == bad_file

    def test_nonexistent_directory(self, tmp_path):
        missing_dir = tmp_path / "no_such_dir"

        analysis = analyze_source_tree(missing_dir)

        assert analysis.files_scanned == 0
        assert len(analysis.parse_errors) == 1
        assert "Directory does not exist" in analysis.parse_errors[0].error_message
