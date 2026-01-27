"""
Prediction module for Segger.

Contains prediction scripts and utilities for the Segger model.
"""

__all__ = ["load_model", "segment", "predict_batch"]

try:
    from .predict_parquet import load_model, segment, predict_batch
except ImportError:
    # Handle missing dependencies (like cupy)
    def load_model(*args, **kwargs):
        raise ImportError("load_model requires cupy, which is not installed.")

    def segment(*args, **kwargs):
        raise ImportError("segment requires cupy, which is not installed.")

    def predict_batch(*args, **kwargs):
        raise ImportError("predict_batch requires cupy, which is not installed.")
