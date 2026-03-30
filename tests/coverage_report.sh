#!/bin/bash
# 覆盖率报告生成脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 Running tests with coverage..."
pytest \
    --cov=vembed \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-config=.coveragerc \
    tests/

COVERAGE_PERCENTAGE=$(coverage report | grep "^TOTAL" | awk '{print $NF}' | sed 's/%//')

echo ""
echo "📊 Coverage Report Summary:"
echo "=========================="
coverage report --skip-covered
echo ""
echo "Overall Coverage: ${COVERAGE_PERCENTAGE}%"
echo ""

if (( $(echo "$COVERAGE_PERCENTAGE < 75" | bc -l) )); then
    echo "⚠️  Coverage below 75% target (${COVERAGE_PERCENTAGE}%)"
    echo "📁 Open htmlcov/index.html for detailed report"
    exit 1
else
    echo "✅ Coverage meets 75% target (${COVERAGE_PERCENTAGE}%)"
fi

echo ""
echo "🌐 HTML report generated at: htmlcov/index.html"
echo ""
echo "💡 To view the report locally:"
echo "   cd $PROJECT_ROOT && python -m http.server 8000 --directory htmlcov"
echo "   Then open http://localhost:8000"
