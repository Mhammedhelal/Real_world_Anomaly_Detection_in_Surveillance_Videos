"""
tests/test_loss.py
------------------

Unit tests for anomaly detection loss functions.

Goal:
Verify loss computation correctness and gradient behavior.

Test Plan
---------

1️⃣ Loss initialization
    - verify loss object creation

2️⃣ Forward computation
    - ensure loss returns scalar tensor
    - verify dtype

3️⃣ Loss with normal samples
    - ensure valid value returned

4️⃣ Loss with anomalous samples
    - verify behavior

5️⃣ Edge cases
    - empty batch
    - all-normal batch
    - all-anomaly batch

6️⃣ Gradient flow
    - ensure backward() works

7️⃣ Numerical stability
    - ensure no NaN or Inf values
"""