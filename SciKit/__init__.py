"""
SciKit - Scientific toolkits for analysis, plotting, and so on.

Copyright (c) 2025 Shanlong Li
Licensed under the MIT License
"""

__version__ = "0.1.0"
__author__ = "Shanlong Li"
__license__ = "MIT"

# Import submodules here as they are developed
# Example:
# from . import analysis
# from . import plotting
# from . import utils

# Import CLI module
try:
    from . import cal
except ImportError:
    cal = None

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "cal",
]
