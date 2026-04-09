#!/bin/bash
# Test Suite Quick Reference & Helper Commands

# ============================================================================
# Quick Start
# ============================================================================

# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific component
pytest -v tests/test_anomaly_detector_comprehensive.py

# ============================================================================
# Test Filtering
# ============================================================================

# Run determinism tests only
pytest -v -k "determinism" tests/

# Run all tests except slow ones
pytest -v -m "not slow" tests/

# Run forward pass tests
pytest -v -k "forward" tests/

# Run edge case tests
pytest -v -k "edge_case" tests/

# ============================================================================
# Coverage Reports
# ============================================================================

# Generate coverage report
pytest --cov=src --cov-report=term-missing tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# ============================================================================
# Device-Specific Testing
# ============================================================================

# Run only CPU tests
pytest -v -k "cpu" tests/

# Run all (GPU tests skipped if unavailable)
pytest -v tests/

# Force CPU-only (disable CUDA)
CUDA_VISIBLE_DEVICES="" pytest -v tests/

# ============================================================================
# Debugging
# ============================================================================

# Run with print statements visible
pytest -vv -s tests/test_anomaly_detector_comprehensive.py

# Drop into debugger on first failure
pytest --pdb tests/

# Show local variables on failure
pytest -l tests/

# ============================================================================
# Parallel Execution
# ============================================================================

# Run 4 tests in parallel (requires pytest-xdist)
pytest -n 4 tests/

# ============================================================================
# Component-Specific Tests
# ============================================================================

# Anomaly Detector Tests
pytest -v tests/test_anomaly_detector_comprehensive.py

# Loss Function Tests
pytest -v tests/test_loss_comprehensive.py

# Dataset Tests
pytest -v tests/test_dataset_comprehensive.py

# Feature Extractor Tests
pytest -v tests/test_feature_extractors_comprehensive.py

# Integration Tests
pytest -v tests/test_integration_comprehensive.py

# ============================================================================
# Test Categories
# ============================================================================

# Initialization tests
pytest -v -k "initialization" tests/

# Forward pass tests
pytest -v -k "forward" tests/

# Backward/gradient tests
pytest -v -k "backward or gradient" tests/

# Determinism tests (critical for reproducibility)
pytest -v -k "determinism" tests/

# Edge case tests
pytest -v -k "edge or zero or single" tests/

# Device tests
pytest -v -k "device or cuda or cpu" tests/

# ============================================================================
# Output Control
# ============================================================================

# Quiet output (only show failures)
pytest -q tests/

# Verbose with variable names
pytest -vv tests/

# Show print statements
pytest -s tests/

# Don't capture output
pytest --capture=no tests/

# ============================================================================
# Failure Analysis
# ============================================================================

# Show last N lines of traceback
pytest --tb=short tests/

# Show full traceback
pytest --tb=long tests/

# Show local variables in traceback
pytest -l tests/

# Show full diff on assertion failures
pytest --tb=short -vv tests/

# ============================================================================
# Test Statistics
# ============================================================================

# Run and show test count
pytest --collect-only -q tests/ | tail -1

# Show test tree structure
pytest --collect-only -q tests/

# Count tests by file
pytest --collect-only -q tests/ | grep "test_" | wc -l

# ============================================================================
# Practical Examples
# ============================================================================

# Example 1: Quick sanity check
echo "Running quick sanity check..."
pytest -x tests/  # Stop on first failure

# Example 2: Check determinism only
echo "Checking determinism..."
pytest -v -k "determinism" tests/

# Example 3: Check all model tests with coverage
echo "Testing anomaly detector with coverage..."
pytest -v --cov=src.models.anomaly_detector tests/test_anomaly_detector_comprehensive.py

# Example 4: Development workflow
echo "Running subset for development..."
pytest -v -k "forward or backward" tests/test_anomaly_detector_comprehensive.py --tb=short -s

# Example 5: Pre-commit check
echo "Running pre-commit tests..."
pytest tests/ --tb=short -q

# Example 6: Full validation
echo "Running full test suite with coverage..."
pytest -v --cov=src tests/ --cov-report=html

# ============================================================================
# Installation Check
# ============================================================================

# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Verify pytest installation
pytest --version

# List available plugins
pytest --version

# ============================================================================
# Continuous Integration
# ============================================================================

# Run tests with CI settings
pytest -v --tb=short --co tests/ | grep "passed\|failed\|error"

# Generate JUnit XML for CI
pytest --junitxml=junit.xml tests/

# Generate JSON report for CI
pytest --json-report --json-report-file=report.json tests/
