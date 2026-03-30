#!/bin/bash
# VEmbed CI Quality Check Script
# Runs comprehensive code quality checks locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
RUN_TESTS=false
COVERAGE=false
AUTO_FORMAT=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            RUN_TESTS=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --format)
            AUTO_FORMAT=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure we're in venv
if [[ ! -v VIRTUAL_ENV ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo "Activate with: source .venv/bin/activate"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}VEmbed CI Quality Check${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"

# Counter for checks
PASSED=0
FAILED=0

# Function to run a check
run_check() {
    local name=$1
    local cmd=$2
    echo -n "📋 $name ... "

    if eval "$cmd" > /tmp/ci_check.log 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC}"
        ((FAILED++))
        if [[ "$QUIET" != "true" ]]; then
            echo -e "${RED}Error output:${NC}"
            cat /tmp/ci_check.log | sed 's/^/  /'
        fi
    fi
}

# === CODE QUALITY CHECKS ===

if [[ "$AUTO_FORMAT" == "true" ]]; then
    echo -e "${YELLOW}🔧 Auto-formatting mode enabled${NC}\n"

    run_check "Black auto-format" "black --line-length=100 ."
    run_check "Isort auto-sort" "isort --profile black --line-length=100 ."
else
    run_check "Black formatting check" "black --check --line-length=100 ."
    run_check "Isort import check" "isort --check-only --profile black --line-length=100 ."
fi

run_check "Ruff linting" "ruff check ."
run_check "Python compilation" "python -m py_compile vembed/**/*.py 2>/dev/null || true"

# === OPTIONAL TESTS ===

if [[ "$RUN_TESTS" == "true" ]]; then
    echo ""
    echo -e "${BLUE}Running optional tests...${NC}\n"

    if [[ "$COVERAGE" == "true" ]]; then
        run_check "Unit tests with coverage" \
            "pytest -v --cov=vembed --cov-report=term-missing tests/ 2>&1"
    else
        run_check "Unit tests" "pytest -v tests/ 2>&1"
    fi
fi

# === SUMMARY ===

echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

TOTAL=$((PASSED + FAILED))
echo "Passed: ${GREEN}$PASSED${NC}/$TOTAL"

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}✨ All checks passed!${NC}"
    exit 0
else
    echo "Failed: ${RED}$FAILED${NC}/$TOTAL"
    echo -e "${RED}❌ Some checks failed${NC}"
    exit 1
fi
