"""Registry for managing FLOP counting operations."""

from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

from .operations import (
    Addition,
    ArccosOperation,
    ArcSineOperation,
    ArcTangentOperation,
    ArgmaxOperation,
    ArgminOperation,
    AverageOperation,
    BitwiseAndOperation,
    BitwiseOrOperation,
    ClipOperation,
    CondOperation,
    CosineOperation,
    CrossOperation,
    Division,
    EigOperation,
    FloorDivideOperation,
    InvOperation,
    IsnanOperation,
    LogOperation,
    MatmulOperation,
    MaxOperation,
    MeanOperation,
    MinOperation,
    ModuloOperation,
    Multiplication,
    NormOperation,
    PowerOperation,
    RoundOperation,
    SineOperation,
    StdOperation,
    Subtraction,
    SumOperation,
    TangentOperation,
    TraceOperation,
    VarOperation,
    WhereOperation,
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
        registry.register("add", Addition)
        registry.register("subtract", Subtraction)
        registry.register("multiply", Multiplication)
        registry.register("divide", Division)
        registry.register("power", PowerOperation)
        registry.register("square", PowerOperation)
        registry.register("sqrt", PowerOperation)
        registry.register("cbrt", PowerOperation)
        registry.register("reciprocal", PowerOperation)
        registry.register("floor_divide", FloorDivideOperation)
        registry.register("remainder", ModuloOperation, method_name="mod")

        # Register bitwise operations
        registry.register("bitwise_and", BitwiseAndOperation)
        registry.register("bitwise_or", BitwiseOrOperation)

        # Register mathematical functions
        registry.register("sin", SineOperation)
        registry.register("cos", CosineOperation)
        registry.register("tan", TangentOperation)
        registry.register("arcsin", ArcSineOperation)
        registry.register("arccos", ArccosOperation)
        registry.register("arctan", ArcTangentOperation)
        registry.register("log", LogOperation)

        # Register array operations
        registry.register("clip", ClipOperation)
        registry.register("matmul", MatmulOperation)
        registry.register("sum", SumOperation)
        registry.register("mean", MeanOperation)
        registry.register("std", StdOperation)
        registry.register("var", VarOperation)
        registry.register("average", AverageOperation)
        registry.register("trace", TraceOperation)
        registry.register("min", MinOperation)
        registry.register("max", MaxOperation)
        registry.register("argmin", ArgminOperation)
        registry.register("argmax", ArgmaxOperation)
        registry.register("round", RoundOperation)
        registry.register("where", WhereOperation)
        registry.register("isnan", IsnanOperation)

        # Register linear algebra operations
        registry.register("linalg.norm", NormOperation, module_path="linalg")
        registry.register("linalg.cond", CondOperation, module_path="linalg")
        registry.register("linalg.inv", InvOperation, module_path="linalg")
        registry.register("linalg.eig", EigOperation, module_path="linalg")
        registry.register("cross", CrossOperation)

        return registry
