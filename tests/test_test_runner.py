"""
test_test_runner.py — Unit tests for Deterministic Test Runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyguardian.test_runner import TestResult, TestStatus, run_project_tests


def test_non_existent_project_returns_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "does_not_exist"
    res = run_project_tests(bad_path)

    assert isinstance(res, TestResult)
    assert res.status == TestStatus.ERROR
    assert res.return_code == -1
    assert "does not exist" in res.stderr


def test_passing_test_project_returns_pass_status(tmp_path: Path) -> None:
    # Create a minimal passing test project
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    res = run_project_tests(tmp_path)
    assert res.status == TestStatus.PASS
    assert res.is_passed is True
    assert res.return_code == 0
    assert "test_sample.py" in res.stdout or "1 passed" in res.stdout


def test_failing_test_project_returns_fail_status(tmp_path: Path) -> None:
    # Create a minimal failing test project
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_fail():\n    assert 1 == 2\n", encoding="utf-8")

    res = run_project_tests(tmp_path)
    assert res.status == TestStatus.FAIL
    assert res.is_passed is False
    assert res.return_code != 0
    assert "AssertionError" in res.stdout or "FAIL" in res.stdout or "FAILED" in res.stdout


def test_command_string_is_captured(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok(): pass\n", encoding="utf-8")

    res = run_project_tests(tmp_path, extra_args=["-k", "test_ok"])
    assert "pytest" in res.command
    assert "-k test_ok" in res.command


def test_project_files_remain_unmodified(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    content = "def test_ok(): assert True\n"
    test_file.write_text(content, encoding="utf-8")

    run_project_tests(tmp_path)

    assert test_file.read_text(encoding="utf-8") == content
