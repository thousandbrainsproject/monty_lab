# flop_counting/operations.py
from typing import Dict, Any
import ast


class OperationRegistry:
    """Registry of operations that require explicit tracking"""

    # Instance method calls
    SKLEARN_OPERATIONS: Dict[str, Dict] = {
        "KDTree": {
            "module": "sklearn.neighbors",
            "methods": {"query", "kneighbors", "kneighbors_graph"},
        },
        "kneighbors_graph": {
            "module": "sklearn.neighbors",
            "methods": set(),  # Empty set indicates this is a function, not a class
        },
    }

    SCIPY_OPERATIONS: Dict[str, Dict] = {
        "KDTree": {"module": "scipy.spatial", "methods": {"query", "query_pairs"}},
        "Rotation": {
            "module": "scipy.spatial.transform",
            "methods": {
                "from_euler",
                "from_matrix",
                "from_quat",
                "as_euler",
                "as_matrix",
                "as_quat",
                "inv",
                "apply",
            },
        },
        "convolve": {
            "module": "scipy.signal",
            "methods": set(),  # Empty set indicates this is a function, not a class
        },
    }

    # Track basic arithmetic operations in Python code
    ARITHMETIC_OPERATIONS: Dict[type, Dict[str, Any]] = {
        ast.Add: {"name": "addition", "symbol": "+"},
        ast.Sub: {"name": "subtraction", "symbol": "-"},
        ast.Mult: {"name": "multiplication", "symbol": "*"},
        ast.Div: {"name": "division", "symbol": "/"},
        ast.Pow: {"name": "power", "symbol": "**"},
        ast.MatMult: {"name": "matrix_multiplication", "symbol": "@"},
        ast.FloorDiv: {"name": "floor_division", "symbol": "//"},
        ast.Mod: {"name": "modulo", "symbol": "%"},
        ast.LShift: {"name": "left_shift", "symbol": "<<"},
        ast.RShift: {"name": "right_shift", "symbol": ">>"},
        ast.BitOr: {"name": "bitwise_or", "symbol": "|"},
        ast.BitXor: {"name": "bitwise_xor", "symbol": "^"},
        ast.BitAnd: {"name": "bitwise_and", "symbol": "&"},
        ast.Invert: {"name": "bitwise_invert", "symbol": "~"},
        ast.UAdd: {"name": "unary_plus", "symbol": "+"},
        ast.USub: {"name": "unary_minus", "symbol": "-"},
        # Comparisons
        ast.Eq: {"name": "equals", "symbol": "=="},
        ast.NotEq: {"name": "not_equals", "symbol": "!="},
        ast.Lt: {"name": "less_than", "symbol": "<"},
        ast.LtE: {"name": "less_than_or_equal", "symbol": "<="},
        ast.Gt: {"name": "greater_than", "symbol": ">"},
        ast.GtE: {"name": "greater_than_or_equal", "symbol": ">="},
        # Boolean operations
        ast.And: {"name": "boolean_and", "symbol": "and"},
        ast.Or: {"name": "boolean_or", "symbol": "or"},
        ast.Not: {"name": "boolean_not", "symbol": "not"},
    }

OPERATION_REGISTRY = OperationRegistry()