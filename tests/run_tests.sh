#!/bin/bash
# Comprehensive test runner for vembed-factory

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         vembed-factory Unit Test Suite                  ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
RUN_ALL=false
RUN_LOSSES=false
RUN_OPTIMIZER=false
RUN_BIDIRECTIONAL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --losses)
            RUN_LOSSES=true
            shift
            ;;
        --optimizer)
            RUN_OPTIMIZER=true
            shift
            ;;
        --bidirectional)
            RUN_BIDIRECTIONAL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_tests.sh [--all|--losses|--optimizer|--bidirectional] [-v|--verbose]"
            exit 1
            ;;
    esac
done

# Default to all if nothing specified
if [ "$RUN_ALL" = false ] && [ "$RUN_LOSSES" = false ] && [ "$RUN_OPTIMIZER" = false ] && [ "$RUN_BIDIRECTIONAL" = false ]; then
    RUN_ALL=true
fi

PYTEST_ARGS="-v"
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --tb=long"
else
    PYTEST_ARGS="$PYTEST_ARGS --tb=short"
fi

# Run tests
if [ "$RUN_ALL" = true ]; then
    echo -e "${YELLOW}Running all unit tests...${NC}"
    python -m pytest unit $PYTEST_ARGS
    EXIT_CODE=$?
else
    EXIT_CODE=0

    if [ "$RUN_LOSSES" = true ]; then
        echo -e "${YELLOW}Running loss tests...${NC}"
        python -m pytest unit/test_losses.py unit/test_bidirectional_loss.py $PYTEST_ARGS
        EXIT_CODE=$((EXIT_CODE + $?))
    fi

    if [ "$RUN_OPTIMIZER" = true ]; then
        echo -e "${YELLOW}Running optimizer tests...${NC}"
        python -m pytest unit/test_optimizer.py $PYTEST_ARGS
        EXIT_CODE=$((EXIT_CODE + $?))
    fi

    if [ "$RUN_BIDIRECTIONAL" = true ]; then
        echo -e "${YELLOW}Running bidirectional loss tests...${NC}"
        python -m pytest unit/test_bidirectional_loss.py $PYTEST_ARGS
        EXIT_CODE=$((EXIT_CODE + $?))
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "║${GREEN} ✓ All tests passed!                              ${NC}║"
else
    echo -e "║${RED} ✗ Some tests failed (exit code: $EXIT_CODE)              ${NC}║"
fi
echo "╚══════════════════════════════════════════════════════════╝"

exit $EXIT_CODE
