# Test Suite Documentation

## Overview

Comprehensive pytest test suite for the Real-world Anomaly Detection in Surveillance Videos project.

The test suite includes over **200+ test cases** covering:

- **Anomaly Detector Model** (50+ tests)
- **MIL Ranking Loss** (40+ tests)
- **VideoFeatureDataset** (50+ tests)
- **Feature Extractors** (40+ tests)
- **End-to-End Integration** (30+ tests)

## Test Files

### Core Component Tests

1. **`test_anomaly_detector_comprehensive.py`**
   - Model initialization and architecture
   - Forward pass with various input shapes
   - Eval mode determinism (critical for reproducibility)
   - Train vs eval behavior differences
   - Device consistency (CPU/CUDA)
   - Gradient flow and backward pass
   - Edge cases and numerical stability
   - Output validity (no NaN/Inf)

2. **`test_loss_comprehensive.py`**
   - MIL Ranking Loss initialization
   - Forward pass computation
   - Loss component breakdown (ranking, smoothness, sparsity)
   - Gradient flow
   - Edge cases (all-normal, all-anomalous, empty batches)
   - Lambda parameter sensitivity
   - Numerical stability

3. **`test_dataset_comprehensive.py`**
   - Dataset initialization and loading
   - Feature loading and shape validation
   - Label extraction
   - Collate function for variable-length sequences
   - DataLoader integration
   - Edge cases (single sample, zero features, long sequences)
   - Batch independence
   - Reproducibility

4. **`test_feature_extractors_comprehensive.py`**
   - I3D Feature Extractor
   - R3D Feature Extractor
   - Lightweight Feature Extractor
   - Eval mode determinism
   - Device consistency
   - Gradient computation
   - Edge cases (small spatial dims, zero input, etc.)
   - Output validity

5. **`test_integration_comprehensive.py`**
   - Model + Loss integration
   - Model + DataLoader integration
   - Complete training pipeline
   - Dataset + Model integration
   - Device transfer (CPU/CUDA)
   - Batch composition tests
   - Checkpoint management
   - Inference pipeline

### Fixtures

**`conftest.py`** provides reusable fixtures:

- **Device fixtures**: `device`, `cpu_device`, `device_param`
- **Data fixtures**: `synthetic_video_features`, `synthetic_labels`, `synthetic_3d_video`, etc.
- **Configuration fixtures**: `model_config`, `loss_config`
- **Utility fixtures**: `temp_dir`

## Testing Strategy

### Key Testing Dimensions

#### 🔴 1. Eval Mode Determinism

```python
model.eval()
output1 = model(input)
output2 = model(input)
assert torch.allclose(output1, output2)  # Must be identical
```

**Why**: Ensures reproducible inference and proper evaluation mode behavior.

#### 🟠 2. Train vs Eval Behavior

```python
model.train()
output_train = model(input)
model.eval()
output_eval = model(input)
# Outputs should be different (dropout, batchnorm, etc.)
```

**Why**: Catches incorrect mode switching and dropout behavior.

#### 🟡 3. Device Consistency

```python
model = model.to(device)
input = input.to(device)
output = model(input)
assert output.device == input.device
```

**Why**: Prevents hidden CPU tensors and device mismatch errors.

#### 🟡 4. Backward Stability

```python
loss = model(input).sum()
loss.backward()
assert not torch.isnan(grads)
assert not torch.isinf(grads)
```

**Why**: Ensures gradient flow is stable (no explosion/vanishing).

#### 🟡 5. Edge Cases

```python
# Test with:
# - All-zero inputs
# - Single sample
# - Single segment
# - Very long sequences
# - Unbalanced batches
```

**Why**: Catches bugs in corner cases that don't appear in normal usage.

#### 🟡 6. Component Consistency

```python
# Verify output shape matches input
# Ensure components chain correctly
# Check tensor dimensions throughout pipeline
```

**Why**: Catches silent reshaping bugs and dimension mismatches.

#### 🟡 7. Batch Independence

```python
# Process individually vs in batch
# Ensure outputs are independent
# Check for hidden state leakage
```

**Why**: Catches issues with shared buffers (e.g., RNN hidden states).

#### 🟡 8. Reproducibility

```python
torch.manual_seed(42)
output1 = model(input)
torch.manual_seed(42)
output2 = model(input)
assert torch.allclose(output1, output2)
```

**Why**: Ensures experiments are reproducible.

#### 🟡 9. Collate Edge Cases

```python
# Variable sequence lengths
# Batch size = 1
# Unbalanced labels
```

**Why**: Catches issues in data loading pipelines.

#### 🟡 10. Error Handling

```python
with pytest.raises(Exception):
    # Invalid operation
```

**Why**: Ensures proper error handling for invalid inputs.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_anomaly_detector_comprehensive.py

# Run specific test class
pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward

# Run specific test
pytest tests/test_anomaly_detector_comprehensive.py::TestAnomalyDetectorForward::test_forward_pass_basic
```

### Running with Verbosity

```bash
# Verbose output
pytest -v tests/

# Very verbose with print statements
pytest -vv -s tests/

# Show local variables on failure
pytest -l tests/
```

### Running Subsets

```bash
# Run tests matching pattern
pytest -k "determinism" tests/

# Run tests excluding pattern
pytest -k "not slow" tests/

# Run only CPU tests (exclude GPU tests)
pytest -m "not gpu" tests/

# Run only fast tests
pytest -m "not slow" tests/
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# View coverage
open htmlcov/index.html
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest -n 4 tests/
```

### Device-Specific Testing

```bash
# Run only CPU tests
pytest -k "cpu" tests/

# Run CUDA tests (skipped if GPU unavailable)
pytest tests/

# Force CPU-only
CUDA_VISIBLE_DEVICES="" pytest tests/
```

## Test Statistics

| Component | Tests | Coverage |
|-----------|-------|----------|
| Anomaly Detector | 50+ | Model, forward, backward, determinism |
| Loss Function | 40+ | Components, gradients, edge cases |
| Dataset | 50+ | Loading, collation, batching |
| Feature Extractors | 40+ | Forward, backward, determinism |
| Integration | 30+ | Pipeline, device transfer, inference |
| **Total** | **210+** | **Comprehensive coverage** |

## Key Assertions

The test suite uses critical assertions:

### Determinism (Must Pass)

```python
# In eval mode with same seed, outputs must be identical
assert torch.allclose(output1, output2)
```

### Validity (Must Pass)

```python
# No NaN or Inf in outputs
assert not torch.isnan(output).any()
assert not torch.isinf(output).any()
```

### Gradient Flow (Must Pass)

```python
# Gradients must be computed and finite
assert param.grad is not None
assert not torch.isnan(param.grad).any()
```

### Shapes (Must Pass)

```python
# Output shapes must match expected dimensions
assert output.shape == expected_shape
```

## Troubleshooting

### Import Errors

```bash
# Ensure src is importable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Missing Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# For feature extractors
pip install pytorchvideo ultralytics
```

### CUDA Tests Skipped

```bash
# Tests requiring CUDA are skipped if GPU unavailable
# This is expected behavior
pytest -v tests/ | grep "SKIPPED"
```

### Memory Issues

```bash
# Run tests with smaller batches (modify fixtures)
# Or skip memory-intensive tests
pytest -k "not large" tests/
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest --cov=src tests/
```

## Adding New Tests

### Template for New Test Class

```python
class TestNewComponent:
    """Test new component."""
    
    def test_initialization(self):
        """Test initialization."""
        obj = NewComponent()
        assert obj is not None
    
    def test_forward_pass(self, synthetic_features):
        """Test forward pass."""
        obj = NewComponent()
        output = obj(synthetic_features)
        assert output is not None
    
    def test_determinism(self, synthetic_features):
        """Test determinism in eval mode."""
        obj = NewComponent()
        obj.eval()
        
        with torch.no_grad():
            output1 = obj(synthetic_features)
            output2 = obj(synthetic_features)
        
        assert torch.allclose(output1, output2)
```

### Using Fixtures

```python
def test_with_fixtures(self, synthetic_video_features, model_config, device):
    """Test using fixtures."""
    model = AnomalyDetector(**model_config)
    model = model.to(device)
    
    features = synthetic_video_features.to(device)
    output = model(features)
    
    assert output.device == device
```

## Test Maintenance

### Regular Tasks

1. **Update fixtures** when data format changes
2. **Add tests** for new features
3. **Refactor** duplicated test code
4. **Review coverage** quarterly
5. **Update CI/CD** configs as needed

### Debugging Failed Tests

```bash
# Run with detailed output
pytest -vv -s test_file.py::TestClass::test_method

# Drop into debugger on failure
pytest --pdb tests/

# Show local variables on failure
pytest -l tests/

# Keep output even if test passes
pytest -s tests/
```

## References

- [pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
- [Testing Best Practices](https://docs.pytest.org/en/6.2.x/goodpractices.html)
