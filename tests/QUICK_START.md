# Quick Start Guide - Running the Test Suite

## Prerequisites

```bash
# Activate virtual environment
source /home/mohammed/ml_env/bin/activate

# Install test dependencies
pip install pytest pytest-cov pytest-xdist
```

## Test Structure

```
tests/
├── conftest.py                              # Pytest fixtures & config
├── test_anomaly_detector_comprehensive.py   # 50+ tests
├── test_loss_comprehensive.py               # 40+ tests
├── test_dataset_comprehensive.py            # 50+ tests
├── test_feature_extractors_comprehensive.py # 40+ tests
├── test_integration_comprehensive.py        # 30+ tests
├── pytest.ini                               # Pytest config
├── run_tests.sh                             # Helper scripts
├── TEST_SUITE_README.md                     # Full documentation
├── IMPLEMENTATION_SUMMARY.md                # Implementation details
└── QUICK_START.md                           # This file
```

## Running Tests

### 1. Quick Sanity Check (30 seconds)

```bash
cd /home/mohammed/dev/Graduation\ Project/Real_world_Anomaly_Detection_in_Surveillance_Videos

# Run a few quick tests
pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorInitialization -v
```

### 2. Test Single Component (2-5 minutes)

```bash
# Test anomaly detector
pytest tests/test_anomaly_detector_comprehensive.py -v

# Test loss function
pytest tests/test_loss_comprehensive.py -v

# Test dataset
pytest tests/test_dataset_comprehensive.py -v

# Test feature extractors
pytest tests/test_feature_extractors_comprehensive.py -v

# Test integration
pytest tests/test_integration_comprehensive.py -v
```

### 3. Run All Tests (5-10 minutes)

```bash
# All tests with summary
pytest tests/ -v

# All tests, no output unless failure
pytest tests/ -q
```

### 4. Test Specific Aspect

```bash
# All determinism tests (critical for reproducibility)
pytest tests/ -k "determinism" -v

# All gradient/backward tests
pytest tests/ -k "gradient or backward" -v

# All edge case tests
pytest tests/ -k "edge or zero or single" -v

# All forward pass tests
pytest tests/ -k "forward" -v

# All device tests
pytest tests/ -k "device" -v
```

## Coverage Reports

### Generate Coverage Report

```bash
# Terminal output
pytest tests/ --cov=src --cov-report=term-missing

# HTML report
pytest tests/ --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html  # On macOS
# or
xdg-open htmlcov/index.html  # On Linux
```

## Debugging Tests

### Verbose Output

```bash
# Show print statements
pytest tests/test_anomaly_detector_comprehensive.py -v -s

# Very verbose with local variables
pytest tests/test_anomaly_detector_comprehensive.py -vv -l

# Drop into debugger on failure
pytest tests/test_anomaly_detector_comprehensive.py --pdb
```

## Common Issues & Solutions

### Issue: ImportError for src modules

```bash
# Solution: Make sure you're in the right directory
cd /home/mohammed/dev/Graduation\ Project/Real_world_Anomaly_Detection_in_Surveillance_Videos
pytest tests/
```

### Issue: CUDA tests are skipped

```bash
# This is expected if no GPU available
# To force CPU-only testing:
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### Issue: pytorchvideo not found

```bash
# Some I3D tests will be skipped, which is fine
# If you want to run I3D tests:
pip install pytorchvideo
```

### Issue: Tests hang or are slow

```bash
# Run only fast tests, exclude slow tests
pytest tests/ -m "not slow"

# Or run subset
pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward -v
```

## Test Categories & What They Check

### 1. **Anomaly Detector Tests** (50+)

✅ Model initialization  
✅ Forward pass correctness  
✅ Output shapes and validity  
✅ Gradient flow  
✅ Determinism in eval mode  
✅ Train vs eval behavior  
✅ Device handling (CPU/CUDA)  
✅ Edge cases (zero inputs, single segment, etc.)

```bash
pytest tests/test_anomaly_detector_comprehensive.py -v
```

### 2. **Loss Function Tests** (40+)

✅ Loss computation  
✅ Gradient flow  
✅ Component breakdown (ranking, smoothness, sparsity)  
✅ Edge cases (all-normal, all-anomalous batches)  
✅ Numerical stability  
✅ Lambda parameter sensitivity

```bash
pytest tests/test_loss_comprehensive.py -v
```

### 3. **Dataset Tests** (50+)

✅ Feature loading  
✅ Shape validation  
✅ Collate function for variable-length sequences  
✅ DataLoader integration  
✅ Edge cases (single sample, long sequences)  
✅ Batch independence  

```bash
pytest tests/test_dataset_comprehensive.py -v
```

### 4. **Feature Extractor Tests** (40+)

✅ I3D, R3D, Lightweight extractors  
✅ Forward pass correctness  
✅ Feature dimensions  
✅ Determinism in eval mode  
✅ Device handling  
✅ Edge cases

```bash
pytest tests/test_feature_extractors_comprehensive.py -v
```

### 5. **Integration Tests** (30+)

✅ Model + Loss pipeline  
✅ Model + DataLoader  
✅ Complete training loop  
✅ Device transfers  
✅ Checkpoint management  
✅ Inference pipeline

```bash
pytest tests/test_integration_comprehensive.py -v
```

## Real-World Test Workflows

### Workflow 1: Before Committing Code

```bash
# Run quick tests on modified component
pytest tests/test_anomaly_detector_comprehensive.py -v --tb=short -q
```

### Workflow 2: Full Validation Before Push

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --tb=short
```

### Workflow 3: Debugging a Failing Test

```bash
# Run with detailed output and drop into debugger
pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward::test_forward_pass_basic -vv -s --tb=long
```

### Workflow 4: Performance Check

```bash
# Time how long tests take
pytest tests/ -v --durations=10
```

### Workflow 5: CI/CD-like Testing

```bash
# Quiet mode, stop on first failure, generate report
pytest tests/ -x -q --junit-xml=report.xml
```

## Key Test Assertions Explained

### Determinism (Critical for ML)

```python
# Same input + same model state → identical output in eval mode
assert torch.allclose(output1, output2)
```

**Why**: Ensures reproducible inference for production systems

### Validity (Data Quality)

```python
# No NaN or Inf in outputs
assert not torch.isnan(output).any()
assert not torch.isinf(output).any()
```

**Why**: Catches numerical instability early

### Gradient Flow (Training)

```python
# Gradients must be computed and finite
assert param.grad is not None
assert not torch.isnan(param.grad).any()
```

**Why**: Ensures model can be trained without gradient issues

### Shapes (Integration)

```python
# Output shapes must match expected dimensions
assert output.shape == expected_shape
```

**Why**: Catches silent reshaping bugs and dimension mismatches

## Advanced Options

### Parallel Testing (faster)

```bash
# Requires: pip install pytest-xdist
pytest tests/ -n 4  # Run 4 tests in parallel
```

### Specific Test Selection

```bash
# Run only tests matching pattern
pytest tests/ -k "determinism and anomaly" -v

# Run all except specific pattern
pytest tests/ -k "not slow" -v
```

### Different Report Formats

```bash
# JUnit XML (for CI/CD)
pytest tests/ --junit-xml=report.xml

# JSON report
pytest tests/ --json-report --json-report-file=report.json

# HTML report
pytest tests/ --html=report.html
```

## Summary Table

| Task | Command | Time |
|------|---------|------|
| Quick check | `pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorInitialization -v` | 5s |
| Single component | `pytest tests/test_anomaly_detector_comprehensive.py -v` | 1m |
| All tests | `pytest tests/ -v` | 5-10m |
| With coverage | `pytest tests/ --cov=src --cov-report=html` | 10-15m |
| Parallel (4x faster) | `pytest tests/ -n 4` | 2-3m |
| Debug test | `pytest tests/test_anomaly_detector_comprehensive.py::TestClass::test_name -vvs --pdb` | varies |

## Next Steps

1. ✅ Run quick sanity check
2. ✅ Run full test suite
3. ✅ Review coverage report
4. ✅ Check for any failures
5. ✅ Commit code with confidence

## Getting Help

- **Full Documentation**: See `TEST_SUITE_README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Test Helper Commands**: See `run_tests.sh`
- **Pytest Docs**: <https://docs.pytest.org/>

---

**Happy Testing! 🚀**
