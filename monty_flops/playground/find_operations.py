import numpy as np
import sklearn
from functools import wraps
import inspect
import warnings
from typing import Set, Dict, Any, Optional
import ast
from pathlib import Path
from inspect import getmodule


class FlopAnalyzer:
    """Integrated FLOP analysis combining runtime tracking, AST parsing, and inspect-based analysis."""

    NUMPY_FLOP_OPERATIONS = {
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

    def __init__(self):
        self.runtime_stats = {}
        self.static_analysis = {}
        self.call_analysis = {}
        self._original_funcs = {}

    def _is_numpy_call(self, func) -> bool:
        """Check if a function is from numpy."""
        module = getmodule(func)
        return module and module.__name__.startswith("numpy")

    def _wrap_function(self, func, op_name: str):
        """Wrap a function to track its usage."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.runtime_stats[op_name] = self.runtime_stats.get(op_name, 0) + 1
            # Track if it's a numpy call
            if self._is_numpy_call(func):
                self.call_analysis[op_name] = self.call_analysis.get(op_name, 0) + 1
            return func(*args, **kwargs)

        return wrapper

    def _patch_numpy(self):
        """Patch numpy functions for runtime tracking."""
        for op in self.NUMPY_FLOP_OPERATIONS:
            if hasattr(np, op):
                func = getattr(np, op)
                if callable(func):
                    self._original_funcs[op] = func
                    setattr(np, op, self._wrap_function(func, op))

    def _unpatch_numpy(self):
        """Restore original numpy functions."""
        for op, func in self._original_funcs.items():
            if hasattr(np, op):
                setattr(np, op, func)
        self._original_funcs.clear()

    def analyze_file(self, filename: str) -> Dict[str, Any]:
        """Perform static analysis on a Python file."""
        with open(filename, "r") as f:
            tree = ast.parse(f.read())

        analyzer = FlopOperatorFinder()
        analyzer.visit(tree)

        # Store results in static_analysis
        self.static_analysis[filename] = {
            "operators": analyzer.operations,
            "numpy_calls": analyzer.numpy_calls,
            "implicit_ops": analyzer.implicit_ops,
        }

        return self.static_analysis[filename]

    def __enter__(self):
        """Start runtime tracking."""
        self._patch_numpy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop runtime tracking."""
        self._unpatch_numpy()
        return False

    def get_comprehensive_stats(self) -> Dict[str, Dict]:
        """Get combined statistics from all analysis methods."""
        return {
            "runtime_tracking": self.runtime_stats,
            "static_analysis": self.static_analysis,
            "call_inspection": self.call_analysis,
        }


class FlopOperatorFinder(ast.NodeVisitor):
    """AST visitor to find FLOP operations in code."""

    def __init__(self):
        self.operations = []  # Explicit operators like +, -, *, /
        self.numpy_calls = []  # Direct numpy function calls
        self.implicit_ops = []  # Hidden operations like list comprehensions

    def visit_BinOp(self, node):
        """Track binary operations."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
        }
        op_type = type(node.op)
        if op_type in op_map:
            self.operations.append(
                {"type": op_map[op_type], "line": node.lineno, "col": node.col_offset}
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        """Track function calls."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in {"np", "numpy"}:
                    self.numpy_calls.append(
                        {
                            "func": node.func.attr,
                            "line": node.lineno,
                            "col": node.col_offset,
                        }
                    )
        self.generic_visit(node)

    def visit_ListComp(self, node):
        """Track list comprehensions which might involve floating point ops."""
        self.implicit_ops.append(
            {"type": "list_comprehension", "line": node.lineno, "col": node.col_offset}
        )
        self.generic_visit(node)


def analyze_codebase(directory: str) -> Dict[str, Any]:
    """Analyze an entire codebase for FLOP operations."""
    analyzer = FlopAnalyzer()
    results = {
        "files": {},
        "total_stats": {
            "explicit_ops": 0,
            "numpy_calls": 0,
            "implicit_ops": 0,
            "runtime_ops": 0,
        },
    }

    # Walk through all Python files
    for py_file in Path(directory).rglob("*.py"):
        with analyzer:  # This will track runtime operations if the file is imported
            try:
                # Perform static analysis
                file_results = analyzer.analyze_file(str(py_file))
                results["files"][str(py_file)] = file_results

                # Update totals
                results["total_stats"]["explicit_ops"] += len(file_results["operators"])
                results["total_stats"]["numpy_calls"] += len(
                    file_results["numpy_calls"]
                )
                results["total_stats"]["implicit_ops"] += len(
                    file_results["implicit_ops"]
                )

            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")

    # Add runtime statistics
    comprehensive_stats = analyzer.get_comprehensive_stats()
    results["runtime_analysis"] = comprehensive_stats["runtime_tracking"]
    results["call_inspection"] = comprehensive_stats["call_inspection"]

    return results


# Example usage:
if __name__ == "__main__":
    # 1. Analyze a single file with runtime tracking
    analyzer = FlopAnalyzer()
    with analyzer:
        # Your code here
        x = np.array([[1, 2], [3, 4]])
        y = np.dot(x, x)

    print("Single file analysis:", analyzer.get_comprehensive_stats())

    # 2. Analyze entire codebase
    results = analyze_codebase("/Users/hlee/tbp/tbp.monty/src/tbp/monty/frameworks")
    print("\nCodebase analysis:", results)
