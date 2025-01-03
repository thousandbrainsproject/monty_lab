from contextlib import ContextDecorator
from typing import Dict, Any
import numpy as np
from .wrappers import FunctionWrapper, UfuncWrapper
from .operations import (
    MatmulOperation,
    Addition,
    Subtraction,
    Multiplication,
    Division,
    ClipOperation,
    WhereOperation,
    MinOperation,
    MaxOperation,
    RoundOperation,
    IsnanOperation,
    SineOperation,
    CosineOperation,
    CrossOperation,
)


class FlopCounter(ContextDecorator):
    """Count FLOPs in NumPy operations automatically."""

    def __init__(self):
        self.flops = 0
        self._original_funcs: Dict[str, Any] = {}
        self._operations = {
            "matmul": MatmulOperation(),
            "add": Addition(),
            "subtract": Subtraction(),
            "multiply": Multiplication(),
            "divide": Division(),
            "clip": ClipOperation(),
            "where": WhereOperation(),
            "min": MinOperation(),
            "max": MaxOperation(),
            "round": RoundOperation(),
            "isnan": IsnanOperation(),
            "sin": SineOperation(),
            "cos": CosineOperation(),
            "cross": CrossOperation(),
        }

    def __enter__(self):
        # Define functions to wrap
        funcs_to_wrap = [
            ("add", np.add),
            ("subtract", np.subtract),
            ("multiply", np.multiply),
            ("divide", np.divide),
            ("matmul", np.matmul),
            ("dot", np.dot),
            ("clip", np.clip),
            ("where", np.where),
            ("min", np.min),
            ("max", np.max),
            ("round", np.round),
            ("isnan", np.isnan),
            ("sin", np.sin),
            ("cos", np.cos),
            ("cross", np.cross),
        ]

        # Store original functions and wrap them
        for func_name, func in funcs_to_wrap:
            self._original_funcs[func_name] = func
            wrapper = (
                UfuncWrapper(func, self, func_name)
                if isinstance(func, np.ufunc)
                else FunctionWrapper(func, self, func_name)
            )
            setattr(np, func_name, wrapper)

        return self

    def __exit__(self, *exc):
        # Restore original functions
        for func_name, orig_func in self._original_funcs.items():
            setattr(np, func_name, orig_func)
        return False

    def add_flops(self, count: int):
        """Add to the FLOP count."""
        self.flops += count
