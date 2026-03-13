"""
tests/test_dataset.py
---------------------

Unit tests for the anomaly detection dataset.

Goal:
Verify dataset loading, preprocessing, and temporal sampling logic.

Test Plan
---------

1️⃣ Dataset initialization
    - verify dataset can be constructed
    - verify root paths are correct
    - verify annotation files are parsed

2️⃣ Dataset length
    - ensure __len__ returns correct number of samples

3️⃣ Sample loading
    - ensure __getitem__ returns expected structure
    - verify tensor shapes
    - verify dtype

4️⃣ Frame loading
    - ensure frames are loaded from disk
    - verify missing frame fallback behavior

5️⃣ Temporal window
    - ensure correct number of frames per sample
    - verify odd/even window constraints

6️⃣ Transform pipeline
    - ensure transforms are applied correctly
    - verify output tensor shape

7️⃣ Edge cases
    - empty dataset
    - corrupted frame
    - missing annotation
"""