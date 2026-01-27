import importlib.util
import dask

# FIX: Disable dask-expr query planning to support spatialdata/squidpy
# Must be set before importing any module that uses dask.dataframe
dask.config.set({"dataframe.query-planning": False})

__version__ = "0.1.0"

cupy_available = importlib.util.find_spec("cupy") is not None

from stereosegger.data import *
from stereosegger.models import *
from stereosegger.training import *

# from stereosegger.validation import *

# segger.prediction requires cupy, which is not available in macOS
if cupy_available:
    from stereosegger.prediction import *

__all__ = ["data", "models", "prediction", "training"]
