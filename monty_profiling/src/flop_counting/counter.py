from contextlib import ContextDecorator
from typing import Dict, Any
import numpy as np
from .wrappers import FunctionWrapper, UfuncWrapper
from .operations import MatmulOperation, Addition, Subtraction, Multiplication, Division


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
        }

    def __enter__(self):
        funcs_to_wrap = [
            ("add", np.add),
            ("subtract", np.subtract),
            ("multiply", np.multiply),
            ("divide", np.divide),
            ("matmul", np.matmul),
            ("dot", np.dot),
        ]

        for func_name, func in funcs_to_wrap:
            self._wrap_function(func_name, func)

        return self

    def __exit__(self, *exc):
        for func_name, orig_func in self._original_funcs.items():
            setattr(np, func_name, orig_func)
        return False

    def _wrap_function(self, func_name: str, func: Any):
        self._original_funcs[func_name] = func
        wrapper = (
            UfuncWrapper(func, self, func_name)
            if isinstance(func, np.ufunc)
            else FunctionWrapper(func, self, func_name)
        )
        setattr(np, func_name, wrapper)

    def add_flops(self, count: int):
        self.flops += count
