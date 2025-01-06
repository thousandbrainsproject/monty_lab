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
            "floor_divide": FloorDivideOperation(),
            "mod": ModuloOperation(),
            "bitwise_and": BitwiseAndOperation(),
            "bitwise_or": BitwiseOrOperation(),
            "power": PowerOperation(),
        }

    def __enter__(self):
        # Define ufuncs to wrap
        ufuncs_to_wrap = [
            ("add", np.add),
            ("subtract", np.subtract),
            ("multiply", np.multiply),
            ("divide", np.divide),
            ("matmul", np.matmul),
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
            ("log", np.log),
            ("mean", np.mean),
            ("std", np.std),
            ("var", np.var),
            ("average", np.average),
            ("trace", np.trace),
            ("argmin", np.argmin),
            ("argmax", np.argmax),
        ]

        # Define regular functions to wrap
        funcs_to_wrap = [
            ("dot", np.dot),  # dot is a special case, handled by FunctionWrapper
            ("linalg.norm", np.linalg.norm),
            ("linalg.cond", np.linalg.cond),
            ("linalg.inv", np.linalg.inv),
            ("linalg.eig", np.linalg.eig),
        ]

        # Store original functions and wrap them
        for func_name, func in ufuncs_to_wrap:
            self._original_funcs[func_name] = func
            wrapper = UfuncWrapper(func, self, func_name)
            if "." not in func_name:  # Only set direct numpy attributes
                setattr(np, func_name, wrapper)

        # Wrap regular functions
        for func_name, func in funcs_to_wrap:
            self._original_funcs[func_name] = func
            wrapper = FunctionWrapper(func, self, func_name)
            if "." in func_name:
                # Handle nested attributes like np.linalg.norm
                module_path = func_name.split(".")
                obj = np
                for part in module_path[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, module_path[-1], wrapper)
            else:
                setattr(np, func_name, wrapper)

        return self

    def __exit__(self, *exc):
        # Restore original functions only
        for func_name, orig_func in self._original_funcs.items():
            setattr(np, func_name, orig_func)
        return False

    def add_flops(self, count: int):
        """Add to the FLOP count."""
        self.flops += count
