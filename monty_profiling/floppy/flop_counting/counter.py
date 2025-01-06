from contextlib import ContextDecorator
from typing import Dict, Any
import numpy as np
from .wrappers import FunctionWrapper, UfuncWrapper
from .operations import *


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
            "arccos": ArccosOperation(),
            "tan": TangentOperation(),
            "arctan": ArcTangentOperation(),
            "arcsin": ArcSineOperation(),
            "linalg.norm": NormOperation(),
            "linalg.cond": CondOperation(),
            "linalg.inv": InvOperation(),
            "linalg.eig": EigOperation(),
            "log": LogOperation(),
            "mean": MeanOperation(),
            "std": StdOperation(),
            "var": VarOperation(),
            "average": AverageOperation(),
            "trace": TraceOperation(),
            "argmin": ArgminOperation(),
            "argmax": ArgmaxOperation(),
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
            ("arccos", np.arccos),
            ("tan", np.tan),
            ("arctan", np.arctan),
            ("arcsin", np.arcsin),
            ("linalg.norm", np.linalg.norm),
            ("linalg.cond", np.linalg.cond),
            ("linalg.inv", np.linalg.inv),
            ("linalg.eig", np.linalg.eig),
            ("log", np.log),
            ("mean", np.mean),
            ("std", np.std),
            ("var", np.var),
            ("average", np.average),
            ("trace", np.trace),
            ("argmin", np.argmin),
            ("argmax", np.argmax),
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
