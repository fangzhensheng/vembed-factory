# Code Quality Tools Reference

## Black (Code Formatter)

**Version**: 24.4.2
**Config**: line-length=100

Ensures consistent code formatting across the project.

```bash
# Check formatting
black --check --line-length=100 .

# Auto-format
black --line-length=100 .
```

**What it checks**:
- Line length (max 100 characters)
- Spacing and indentation
- String formatting consistency
- Import organization

---

## Isort (Import Sorter)

**Version**: 5.13.2
**Config**: profile=black, line-length=100

Organizes imports into three groups: future, third-party, first-party.

```bash
# Check import ordering
isort --check-only --profile black --line-length=100 .

# Auto-sort
isort --profile black --line-length=100 .
```

**What it checks**:
- Import group ordering (future → stdlib → third-party → first-party)
- Alphabetical sorting within groups
- One import per line consistency

---

## Ruff (Linter)

**Version**: 0.4.4
**Config**: PEP 8 + security checks

Fast Python linter combining many tools (flake8, pylint, etc.).

```bash
# Run all checks
ruff check .

# Show source code for errors
ruff check --show-source <file>

# Auto-fix what can be fixed
ruff check --fix .
```

**What it checks**:
- PEP 8 compliance
- Unused imports and variables
- Security vulnerabilities
- Code style issues

**Common error codes**:
- `E501` - Line too long (but Black handles this)
- `F841` - Local variable assigned but never used
- `F401` - Imported but unused
- `E402` - Module level import not at top

---

## Pytest (Test Runner)

**Version**: Latest (optional)

Runs unit tests and generates coverage reports.

```bash
# Run tests
pytest -v tests/

# Run with coverage
pytest -v --cov=vembed --cov-report=term-missing tests/

# Run specific test file
pytest -v tests/unit/test_trainer.py

# Run specific test function
pytest -v tests/unit/test_trainer.py::test_function_name
```

**What it does**:
- Executes unit tests
- Reports test results (passed/failed)
- Generates coverage metrics
- Shows which lines are untested

---

## Python Compilation Check

Basic syntax validation using Python's built-in compiler.

```bash
python -m py_compile vembed/**/*.py
```

**What it checks**:
- Python syntax errors
- Module import errors
- Compilation issues

---

## Configuration Files

### .flake8
Ruff configuration (extends flake8 config)

### pyproject.toml
- Black configuration (line-length=100)
- Isort configuration (profile=black)
- Pytest configuration

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | One or more checks failed |
| 2 | Invalid arguments |

---

## Performance Tips

1. **Run checks in order**: Black → Isort → Ruff (failures cascade)
2. **Use --check flags first**: Verify before auto-fixing
3. **For large codebases**: Run on specific files/directories first
4. **Parallel runs**: Can't run Black and Ruff simultaneously (file conflicts)

---

## Upgrading Tools

To update a single tool:
```bash
source .venv/bin/activate
pip install --upgrade black isort ruff
```

Check installed versions:
```bash
black --version
isort --version
ruff --version
pytest --version
```
