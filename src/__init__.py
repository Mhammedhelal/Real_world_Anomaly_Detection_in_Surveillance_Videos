"""
Real-world Anomaly Detection in Surveillance Videos
"""

# Sub-packages are imported lazily to avoid hard-failing when optional
# dependencies (torchvision, pytorchvideo, ultralytics, cv2) are absent.
# Import the sub-package you need explicitly in your own code.

__all__ = ['models', 'data', 'engine', 'utils']


def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f"src.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'src' has no attribute {name!r}")
