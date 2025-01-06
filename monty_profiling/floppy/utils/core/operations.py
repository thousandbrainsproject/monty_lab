# flop_analysis/core/operations.py

from typing import Set


class OperationRegistry:
    """Registry of FLOP operations for different libraries."""

    NUMPY_OPERATIONS: Set[str] = {
        # Basic arithmetic
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        # Linear algebra
        "dot",
        "matmul",
        "inner",
        "outer",
        # Statistical
        "mean",
        "std",
        "var",
        # Comparisons
        "greater",
        "less",
        "greater_equal",
        "less_equal",
        "equal",
        "not_equal",
        # Exponential and logarithmic
        "exp",
        "log",
        "log10",
        "log2",
        # Trigonometric
        "sin",
        "cos",
        "tan",
        # Random generation
        "random",
        "randn",
        "normal",
        "uniform",
        # Reduction operations
        "sum",
        "prod",
        "argmin",
        "argmax",
        "min",
        "max",
        # Matrix operations
        "inv",
        "pinv",
        "solve",
        "svd",
        "eig",
        "qr",
    }

    SKLEARN_OPERATIONS: Dict[str, Dict] = {
        "KDTree": {
            "module": "sklearn.neighbors",
            "methods": {"query", "kneighbors", "kneighbors_graph"},
        },
        "kneighbors_graph": {"module": "sklearn.neighbors", "methods": set()},
    }

    SCIPY_OPERATIONS: Dict[str, Dict] = {
        "KDTree": {"module": "scipy.spatial", "methods": {"query", "query_pairs"}},
        "convolve": {"module": "scipy.signal", "methods": set()},
    }


OPERATION_REGISTRY = OperationRegistry()
