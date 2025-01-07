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
        self._is_active = False  # Add flag to control when to count FLOPs

    def __enter__(self):
        print("Debug - Entering FlopCounter context")
        # Wrap all functions first
        self._wrap_functions()
        # Only start counting after all wrapping is done
        self._is_active = True
        return self

    def add_flops(self, count: int):
        """Add to the FLOP count only if counter is active."""
        if self._is_active:
            self.flops += count

    def _wrap_functions(self):
        """Separate function to handle all the wrapping."""
        ufuncs_to_wrap = [
            ("add", np.add),
            ("subtract", np.subtract),
            ("multiply", np.multiply),
            ("divide", np.divide),
            ("matmul", np.matmul),
            ("clip", np.clip),
            ("where", np.where),
            ("min", np.amin),
            ("max", np.amax),
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

        for func_name, func in ufuncs_to_wrap:
            self._original_funcs[func_name] = func
            wrapper = UfuncWrapper(func, self, func_name)
            if "." not in func_name:
                setattr(np, func_name, wrapper)
                if func.__name__ != func_name:
                    actual_name = func.__name__
                    self._original_funcs[actual_name] = func
                    setattr(np, actual_name, wrapper)

    def __exit__(self, *exc):
        self._is_active = False
        print("Debug - Exiting FlopCounter context")  # Debug line
        # Restore original functions
        for func_name, orig_func in self._original_funcs.items():
            if "." in func_name:
                module_path = func_name.split(".")
                obj = np
                for part in module_path[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, module_path[-1], orig_func)
            else:
                setattr(np, func_name, orig_func)
        return False

    def __call__(self, func, *args, **kwargs):
        """Count FLOPs for the given function."""

        operation = self._operations.get(func)
        if operation is None:
            return None

        flops = operation.count_flops(*args, result=None)

        if flops is not None:
            self.flops += flops
        return flops
