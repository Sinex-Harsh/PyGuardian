"""
test_runner.py — Deterministic Project Test Execution Layer for PyGuardian.

Architecture:
project_path
      ↓
run_project_tests(project_path, extra_args)
      ↓
TestResult (status: PASS / FAIL / NOT_AVAILABLE / ERROR, return_code, stdout, stderr, command)

Responsibilities:
- Execute project tests deterministically using standard pytest suite.
- Use sys.executable -m pytest via subprocess.run(shell=False).
- Capture stdout, stderr, returncode, and command string safely.
- Handle missing pytest or project directory errors gracefully.
- ZERO AI failure diagnosis or autonomous code repair.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status Enum
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    """Execution status classification for test runner."""
    __test__ = False
    PASS = "pass"
    FAIL = "fail"
    NOT_AVAILABLE = "not_available"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestResult:
    """
    Result of executing project tests.

    Attributes:
        status:      Evaluated TestStatus classification.
        return_code: Process return code (0 for pass, >0 for fail/error).
        stdout:      Captured standard output text.
        stderr:      Captured standard error text.
        command:     String representation of command executed.
    """

    __test__ = False
    status: TestStatus
    return_code: int
    stdout: str
    stderr: str
    command: str

    @property
    def is_passed(self) -> bool:
        """True if tests passed with exit code 0."""
        return self.status == TestStatus.PASS


# ---------------------------------------------------------------------------
# Test Runner API
# ---------------------------------------------------------------------------

def run_project_tests(
    project_path: Path,
    extra_args: Optional[Sequence[str]] = None,
    timeout_seconds: int = 60,
) -> TestResult:
    """
    Deterministically execute pytest for a target project directory.

    Args:
        project_path:    Directory path of the Python project to test.
        extra_args:      Optional extra command-line flags to pass to pytest.
        timeout_seconds: Maximum time limit in seconds before aborting test run.

    Returns:
        TestResult containing execution status, return code, stdout, and stderr.
    """
    path_obj = Path(project_path)
    if not path_obj.exists() or not path_obj.is_dir():
        return TestResult(
            status=TestStatus.ERROR,
            return_code=-1,
            stdout="",
            stderr=f"Project directory does not exist or is not a directory: '{project_path}'",
            command="",
        )

    cmd = [sys.executable, "-m", "pytest"]
    if extra_args:
        cmd.extend(extra_args)

    cmd_str = " ".join(cmd)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(path_obj),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
        status = TestStatus.PASS if proc.returncode == 0 else TestStatus.FAIL
        return TestResult(
            status=status,
            return_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            command=cmd_str,
        )
    except subprocess.TimeoutExpired as exc:
        return TestResult(
            status=TestStatus.FAIL,
            return_code=-1,
            stdout=exc.stdout or "",
            stderr=f"Test execution timed out after {timeout_seconds} seconds.",
            command=cmd_str,
        )
    except Exception as exc:
        return TestResult(
            status=TestStatus.NOT_AVAILABLE,
            return_code=-1,
            stdout="",
            stderr=f"Failed to execute test runner: {exc}",
            command=cmd_str,
        )
