# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

from .operations import (
    AbsoluteOperation,
    Addition,
    ArccosOperation,
    ArcSineOperation,
    ArcTangent2Operation,
    ArcTangentOperation,
    AverageOperation,
    CondOperation,
    ConvolveOperation,
    CosineOperation,
    CrossOperation,
    DegreesOperation,
    DiffOperation,
    Division,
    EigOperation,
    EinsumOperation,
    ExponentialOperation,
    InnerOperation,
    InvOperation,
    LinspaceOperation,
    LogOperation,
    MatmulOperation,
    MeanOperation,
    MedianOperation,
    ModuloOperation,
    Multiplication,
    NormOperation,
    OuterOperation,
    PowerOperation,
    RadiansOperation,
    SineOperation,
    SolveOperation,
    StdOperation,
    Subtraction,
    SumOperation,
    TangentOperation,
    TraceOperation,
    VarOperation,
)


class OperationRegistry:
    """Registry for managing FLOP counting operations and their method mappings."""

    def __init__(self):
        self._operations: Dict[str, Any] = {}
        self._method_mappings: Dict[str, str] = {}
        self._module_locations: Dict[str, Tuple[Any, str]] = {}

    def register(
        self,
        ufunc_name: str,
        operation_class: Type,
        method_name: Optional[str] = None,
        module_path: Optional[str] = None,
    ) -> None:
        """Register an operation with its ufunc and optional method name.

        Args:
            ufunc_name: Name of the NumPy ufunc (e.g., 'add', 'multiply')
            operation_class: Class that implements the FLOP counting for this operation
            method_name: Optional alternative name for when the method is called directly
                       on an array (e.g., 'mod' for 'remainder')
            module_path: Optional module path for nested functions (e.g., 'linalg')
        """
        self._operations[ufunc_name] = operation_class()
        self._method_mappings[method_name or ufunc_name] = ufunc_name

        # Determine the module and attribute for the function
        if module_path == "linalg":
            self._module_locations[ufunc_name] = (np.linalg, ufunc_name.split(".")[-1])
        elif module_path == "ma":
            self._module_locations[ufunc_name] = (np.ma, ufunc_name.split(".")[-1])
        else:
            self._module_locations[ufunc_name] = (np, ufunc_name)

    def get_operation(self, name: str) -> Optional[Any]:
        """Get the operation instance for a given ufunc name."""
        return self._operations.get(name)

    def get_ufunc_name(self, method_name: str) -> Optional[str]:
        """Get the ufunc name for a given method name."""
        return self._method_mappings.get(method_name)

    def get_module_location(self, name: str) -> Optional[Tuple[Any, str]]:
        """Get the module and attribute name for a given operation.

        Returns:
            Tuple of (module, attribute_name) or None if not found
        """
        return self._module_locations.get(name)

    def get_all_operations(self) -> Dict[str, Any]:
        """Get all registered operations."""
        return self._operations

    @classmethod
    def create_default_registry(cls) -> "OperationRegistry":
        """Create and return a registry with all default operations registered."""
        registry = cls()

        # Register arithmetic operations
        registry.register("absolute", AbsoluteOperation, method_name="abs")
        registry.register("add", Addition)
        registry.register("subtract", Subtraction)
        registry.register("multiply", Multiplication)
        registry.register("divide", Division)
        registry.register("power", PowerOperation)
        registry.register("square", PowerOperation)
        registry.register("sqrt", PowerOperation)
        registry.register("cbrt", PowerOperation)
        registry.register("reciprocal", PowerOperation)
        registry.register("remainder", ModuloOperation, method_name="mod")

        # Register mathematical functions
        registry.register("sin", SineOperation)
        registry.register("cos", CosineOperation)
        registry.register("tan", TangentOperation)
        registry.register("arcsin", ArcSineOperation)
        registry.register("arccos", ArccosOperation)
        registry.register("arctan", ArcTangentOperation)
        registry.register("arctan2", ArcTangent2Operation)
        registry.register("degrees", DegreesOperation)
        registry.register("radians", RadiansOperation)
        registry.register("log", LogOperation)
        registry.register("exp", ExponentialOperation)

        # Register array operations
        registry.register("convolve", ConvolveOperation)
        registry.register("diff", DiffOperation)
        registry.register("einsum", EinsumOperation)
        registry.register("inner", InnerOperation)
        registry.register("linspace", LinspaceOperation)
        registry.register("matmul", MatmulOperation)
        registry.register("outer", OuterOperation)
        registry.register("dot", MatmulOperation, method_name="dot")
        registry.register("sum", SumOperation)
        registry.register("nansum", SumOperation)
        registry.register("ma.sum", SumOperation, module_path="ma")
        registry.register("ma.average", AverageOperation, module_path="ma")
        registry.register("mean", MeanOperation)
        registry.register("median", MedianOperation)
        registry.register("std", StdOperation)
        registry.register("var", VarOperation)
        registry.register("average", AverageOperation)
        registry.register("trace", TraceOperation, method_name="trace")

        # Register linear algebra operations
        registry.register("linalg.norm", NormOperation, module_path="linalg")
        registry.register("linalg.cond", CondOperation, module_path="linalg")
        registry.register("linalg.inv", InvOperation, module_path="linalg")
        registry.register("linalg.eig", EigOperation, module_path="linalg")
        registry.register("linalg.solve", SolveOperation, module_path="linalg")
        registry.register("cross", CrossOperation)

        return registry
