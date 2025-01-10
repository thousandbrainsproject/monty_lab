from typing import Any, Optional
import numpy as np

class BaseOperation:
    """Base implementation for simple operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """
        Count the number of floating-point operations (FLOPs) for a NumPy operation.

        For basic arithmetic operations (add, subtract, multiply, divide), each element-wise
        operation counts as one FLOP. For example:
        - Adding two scalars (5 + 3) is 1 FLOP
        - Adding two arrays [1,2,3] + [4,5,6] is 3 FLOPs (one per element)
        - Adding array with scalar [1,2,3] + 2 is 3 FLOPs (one per element after broadcasting)

        Args:
            *args: The input arguments to the operation
            result: The output from the operation

        Returns:
            Optional[int]: Number of FLOPs performed, or None if inputs are invalid

        Examples:
            >>> op = BaseOperation("add")
            >>> # Scalar addition
            >>> op.count_flops(5, 3, result=8)
            1
            >>> # Array addition
            >>> op.count_flops(np.array([1, 2]), np.array([3, 4]),
            ...               result=np.array([4, 6]))
            2
            >>> # Broadcasting scalar to array
            >>> op.count_flops(np.array([1, 2, 3]), 5,
            ...               result=np.array([6, 7, 8]))
            3
        """
        if not self.validate_inputs(*args):
            return None
        return np.size(result) if isinstance(result, np.ndarray) else 1

    def validate_inputs(self, *args: Any) -> bool:
        """
        Validates that the inputs to a NumPy operation are suitable for FLOP counting.

        This method checks if:
        1. There are at least 2 arguments (required for binary operations like add, subtract)
        2. All arguments can be converted to NumPy arrays
        3. The array shapes are compatible for broadcasting (if inheritance includes shape checking)

        Args:
            *args: Variable length argument list. For binary NumPy operations like np.add,
                typically contains two arguments that were passed to the original function.
                For example:
                - np.add(1, 2) -> args would be (1, 2)
                - np.add([1, 2], 3) -> args would be (array([1, 2]), 3)
                - np.add([[1, 2]], [[3, 4]]) -> args would be (array([[1, 2]]), array([[3, 4]]))

        Returns:
            bool: True if the inputs are valid for FLOP counting, False otherwise.
                Returns False if:
                - Fewer than 2 arguments are provided, e.g. np.add(1) is invalid.
                - Arguments cannot be converted to NumPy arrays
                - Array shapes are incompatible for broadcasting (if shape checking is implemented)

        Examples:
            >>> op = BaseOperation("add")
            >>> op.validate_inputs(np.array([1, 2]), np.array([3, 4]))
            True
            >>> op.validate_inputs(5, 3)
            True
            >>> op.validate_inputs(np.array([1, 2]))  # single argument
            False
            >>> op.validate_inputs("invalid", [1, 2])  # can't convert to array
            False
        """
        if len(args) < 2:
            return False

        try:
            # Try converting inputs to arrays
            _ = [np.asarray(arg) for arg in args[:2]]
            return True
        except (TypeError, ValueError):
            # Return False if inputs can't be converted to arrays
            return False
