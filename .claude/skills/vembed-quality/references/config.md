# VEmbed Quality Check Configuration

## Project Configuration Files

### .flake8
Located in project root. Configured for Ruff compatibility.

```
[flake8]
max-line-length = 100
```

### pyproject.toml
Main configuration file for build, dependencies, and tools.

**Black section**:
```toml
[tool.black]
line-length = 100
target-version = ['py310']
```

**Isort section**:
```toml
[tool.isort]
profile = "black"
line_length = 100
multi_line_mode = 3
```

**Ruff section**:
```toml
[tool.ruff]
line-length = 100
target-version = "py310"
```

**Pytest section**:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

## CI/CD Pipeline

### GitHub Actions (.github/workflows/ci.yml)

**Quality Stage**:
- Runs on: ubuntu-latest, Python 3.10
- Tools: black, isort, ruff, mypy
- Continues on error: mypy check only (gradually enforced)

**Test Stage**:
- Runs on: ubuntu-latest
- Versions: Python 3.10, 3.11, 3.12
- Dependencies: pytest, pytest-cov
- Coverage: Uploaded to codecov if main branch

## Local Environment

### Requirements

```
torch==2.7.*
torchvision
transformers>=4.37.0
accelerate>=0.27.0
datasets>=2.16.0
# ... other dependencies in requirements.txt
```

### Development Tools

These should be in your venv:

```
black==24.4.2
isort==5.13.2
ruff==0.4.4
mypy
pytest
pytest-cov
```

Install dev dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install black==24.4.2 isort==5.13.2 ruff==0.4.4 pytest pytest-cov
```

## Pre-commit Hooks (Optional)

You can set up git hooks to run checks automatically before commits.

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
source .venv/bin/activate
black --check --line-length=100 . || exit 1
isort --check-only --profile black --line-length=100 . || exit 1
ruff check . || exit 1
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Line Length Standard

**Standard**: 100 characters
**Reasoning**: Balances readability with modern screen sizes
**Enforcement**: Black, Isort, Ruff

Why not 80?
- Original PEP 8 standard from era of smaller monitors
- Modern development environments support 100+ chars
- Improves readability by reducing line wrapping

## Python Version

**Target**: Python 3.10+
**Tested**: Python 3.10, 3.11, 3.12

Minimum version enforced by:
- pyproject.toml `target-version`
- GitHub Actions matrix
- Type hints using modern syntax (e.g., `list[T]` instead of `List[T]`)

## Rules by Tool

### Black Rules
- Line length: 100
- String quotes: prefer double quotes
- Trailing commas: on multiline collections
- Parentheses: minimal and necessary

### Isort Rules
- Profile: Black-compatible
- Line length: 100
- Import groups: future, stdlib, third-party, first-party, local
- Multi-line mode: Vertical hanging indent

### Ruff Rules
- PEP 8 compliance (E/W codes)
- Security checks (S codes)
- Unused variable detection (F codes)
- Imports organization (I codes)

### Pytest Rules
- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Minimum coverage target: (configured per project, check ci.yml)

## Coverage Requirements

Current setup captures coverage but doesn't enforce minimums.

Track in `--cov-report=term-missing` output:
- Which lines are untested (marked MISSING)
- File-by-file coverage percentages
- Total project coverage

Gradual improvement approach preferred over strict minimums.

## Troubleshooting Configuration

### "Different versions locally vs CI"
Check `.github/workflows/ci.yml` for exact versions being used, match locally.

### "Black wants to format, Ruff says it's wrong"
Both tools should be compatible with `profile=black`. If conflict:
1. Run `black --line-length=100 <file>`
2. Run `ruff check --fix <file>`
3. Recheck both tools

### "Import ordering differs between local and CI"
Ensure both using `profile=black` and `line-length=100`.

### "Line length issues"
Check all three (Black, Isort, Ruff) are set to `line-length=100`:
```bash
grep -r "line-length\|line_length" pyproject.toml .flake8
```

## Default Behavior

When running `/vembed-quality` without flags:
1. Loads `pyproject.toml` and `.flake8` automatically
2. Uses configured line-length (100)
3. Uses configured profiles (black for isort)
4. Checks (does not fix) by default
5. Reports all issues found

To auto-fix, use `--format` flag.
