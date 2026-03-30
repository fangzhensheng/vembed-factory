---
name: vembed-quality
description: |
  Run comprehensive code quality checks for vembed-factory project. Execute this whenever the user mentions code review, CI/CD validation, testing, or preparing code for commit/merge. Use this skill to verify code formatting (Black), import ordering (Isort), linting (Ruff), and optionally run unit tests with coverage reports. Supports quick checks, full testing, and auto-formatting. Use this skill proactively whenever the user finishes coding changes and needs validation before committing.
compatibility:
  - venv: required (python 3.10+)
  - tools: black, isort, ruff, pytest (optional)
  - os: macOS, Linux
---

# VEmbed Quality Check Skill

Run comprehensive code quality checks for the vembed-factory project using the project's CI/CD pipeline tools.

## When to Use This Skill

Use this skill whenever you need to:
- **Validate code quality** before committing changes
- **Check formatting compliance** (Black line-length=100)
- **Verify import ordering** (Isort with Black profile)
- **Run linting checks** (Ruff for PEP 8 and security)
- **Execute unit tests** (with optional coverage reports)
- **Auto-fix formatting issues**
- **Prepare for code review** or pull requests

## Quick Start

The skill runs a series of code quality checks in sequence:

1. **Black** - Code formatting validation (line-length=100)
2. **Isort** - Import ordering verification (Black-compatible)
3. **Ruff** - Linting for PEP 8 and security issues
4. **Compilation** - Python syntax validation

Then optionally:
5. **Pytest** - Unit tests with coverage (if --full flag used)

## Usage Modes

### Quick Check (Recommended for Development)
```bash
/vembed-quality
```
Runs: Black + Isort + Ruff + Compilation checks
Perfect for: Before committing code, validating changes locally

### Full Check with Tests
```bash
/vembed-quality --full
```
Runs: All basic checks + Unit tests (pytest)
Perfect for: Pre-release validation, feature completion

### Full Check with Coverage Reports
```bash
/vembed-quality --full --coverage
```
Runs: All checks + Tests + Coverage metrics
Perfect for: Release preparation, coverage goals

### Auto-Format Code
```bash
/vembed-quality --format
```
Runs: Auto-format with Black + Isort, then validate
Perfect for: Fixing formatting issues before commit

## What Each Check Does

| Tool | Check | Config | Purpose |
|------|-------|--------|---------|
| **Black** | Code formatting | line-length=100 | Ensures consistent code style |
| **Isort** | Import ordering | profile=black | Organizes imports correctly |
| **Ruff** | Linting | PEP 8 + security | Detects code issues and vulnerabilities |
| **Pytest** | Unit tests | (optional) | Verifies functionality with tests |

## Requirements

- **Python**: 3.10+
- **Virtual Environment**: Must be activated (`source .venv/bin/activate`)
- **Tools**: Pre-installed in venv (black, isort, ruff)
- **Optional**: pytest for --full mode

## Expected Output

### Success
```
✨ All checks passed!
  ✓ Black formatting
  ✓ Isort import ordering
  ✓ Ruff linting
  ✓ Python compilation
```

### Failure
```
❌ Black formatting issues detected
  Run: black --line-length=100 <files>

  Error output:
    [specific formatting issues]
```

## Common Workflows

### Before Committing Code
```bash
# 1. Make your changes
# 2. Activate environment
source .venv/bin/activate

# 3. Run quality checks
/vembed-quality

# 4. If passed, commit
git add .
git commit -m "your message"

# 5. If failed, fix issues and rerun
```

### Fixing Formatting Issues
```bash
# 1. Auto-format
/vembed-quality --format

# 2. Verify it worked
/vembed-quality

# 3. Commit the fixes
git add .
git commit -m "style: auto-format code"
```

### Release Checklist
```bash
# 1. Full validation with tests
/vembed-quality --full --coverage

# 2. Review coverage report
# (check that coverage meets project standards)

# 3. Fix any remaining issues
# 4. All green? Ready to merge!
```

## Troubleshooting

### "Virtual environment not activated"
```bash
source .venv/bin/activate
/vembed-quality
```

### "Command not found: black/isort/ruff"
Reinstall tools:
```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### "Black formatting issues"
Let the skill auto-format:
```bash
/vembed-quality --format
```

### "Ruff linting errors"
Fix high-priority errors first:
1. Read the error output carefully
2. Some errors are auto-fixable: `ruff check --fix`
3. Others require manual fixes

### Test failures (--full mode)
```bash
# Run tests directly to debug
source .venv/bin/activate
pytest -v tests/

# Run with coverage
pytest -v --cov=vembed --cov-report=html tests/
```

## Integration with CI/CD

This skill mirrors the GitHub Actions pipeline (`.github/workflows/ci.yml`), so passing locally guarantees passing in CI. Use before pushing to avoid failed CI checks.

## Related Documentation

- **Python Coding Standards**: `.claude/skills/python-coding-standards.md`
- **CI/CD Configuration**: `.github/workflows/ci.yml`
- **CI Quality Report**: `.claude/CI_QUALITY_REPORT.md`
- **Usage Guide**: `.claude/USAGE_GUIDE.md`

## See Also

- Linting error details: Run `ruff check --show-source` for line-by-line explanations
- Black formatting guide: `black --help` for formatting options
- Test execution: `pytest -v` for verbose test output
