"""
tests/test_model.py
-------------------

Unit tests for the anomaly detection model.

Goal:
Verify model forward pass, output shapes, and numerical stability.

Test Plan
---------

1️⃣ Model initialization
    - verify model can be instantiated
    - verify all layers exist

2️⃣ Forward pass
    - ensure forward() executes without errors
    - verify output tensor shape

3️⃣ Variable input sizes
    - test different batch sizes
    - test different sequence lengths

4️⃣ Output values
    - ensure outputs contain no NaN or Inf
    - verify logits range

5️⃣ Determinism
    - same input → same output

6️⃣ Different inputs
    - different inputs → different outputs

7️⃣ Device compatibility
    - CPU
    - GPU (optional)

8️⃣ Gradient flow
    - ensure gradients propagate through network
"""