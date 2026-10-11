# 🛡️ PyGuardian

**Python Dependency Intelligence & Auto-Migration Assistant**

PyGuardian analyzes Python project dependencies, detects compatibility risks from package updates against your source code, generates targeted migration patches, and applies them safely with interactive diff review and automated test execution.

---

## Features

- **Dependency Scanner**: Parses `requirements.txt` dependencies and queries PyPI for the latest releases (categorized by MAJOR, MINOR, and PATCH updates).
- **AST Source Code Analysis**: Traverses Python project source trees to detect package imports, function calls, and API usages.
- **Compatibility & Risk Scoring**: Evaluates upgrade risks based on package changelogs and detected call sites.
- **AI-Assisted Migration Reasoning**: Produces code migration recommendations and structured source edits using configurable LLM providers (e.g., local Ollama or custom providers).
- **Diff Review & Human Approval**:
  - Validates patch syntax with Python's AST parser before presentation.
  - Guards against modifications to security-sensitive files (such as `.env` and configuration secrets).
  - Displays colorized diffs in the terminal and requires explicit confirmation before writing any changes to disk.
- **Atomic Patch Application & Test Validation**:
  - Applies patches with automated rollback if changes fail.
  - Automatically executes project tests (`pytest`) following migration to verify code behavior.

---

## Installation

### Prerequisites
- Python 3.10+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/pyguardian.git
cd pyguardian

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS

# Install in editable mode with test dependencies
pip install -e .[test]
```

---

## Usage

### 1. Dependency Scan
Scan dependencies in the current directory or a specified path:

```bash
# Scan current directory
pyguardian scan

# Scan specific project directory
pyguardian scan path/to/project
```

### 2. Migration Workflow
Run the end-to-end migration assistant to inspect risks, propose patches, review diffs, and run tests:

```bash
# Interactive migration (prompts for confirmation before applying changes)
pyguardian migrate

# Auto-approve patch application (-y or --yes)
pyguardian migrate -y
# or
pyguardian migrate --yes

# Apply patch without running post-migration tests
pyguardian migrate --no-test
```

---

## Safety & Approval Model

PyGuardian is built with strict safety boundaries:
1. **In-Memory Generation**: Patches are generated in-memory as unified diffs and validated with AST parsing before any user prompt.
2. **Security File Guard**: Files containing credentials or sensitive configuration (e.g., `.env`) are blocked from automated modification.
3. **Explicit Human Approval**: Disks are never modified without user consent unless `-y` / `--yes` is explicitly passed.
4. **Atomic Rollback**: If an error occurs during file application, original files are automatically restored.
5. **Deterministic Verification**: Project test suites are executed via pytest to confirm migration success.

---

## Development & Testing

Run the test suite using pytest:

```bash
pytest tests/
```

All 449 tests verify the dependency scanner, AST analyzer, risk engine, migration context, patch generation, diff reviewer, and end-to-end workflows.
