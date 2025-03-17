# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Optional, Protocol


class FlopOperation(Protocol):
    """Protocol defining the interface for FLOP counting operations.

    All operation classes should implement this protocol to ensure consistent
    interface across the codebase. This includes operations for arithmetic,
    linear algebra, trigonometry, and other mathematical operations.

    Example:
        class MyOperation(FlopOperation):
            def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
                # Implementation specific to the operation
                return flop_count
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Count the number of floating point operations (FLOPs) for this operation.

        Args:
            *args: Variable length argument list containing the input operands.
                Can be one or more arguments where each can be a scalar
                or numpy array.
            result: The result of the operation, used to determine the
                final shape after any transformations (e.g., broadcasting).
            **kwargs: Additional keyword arguments that match the underlying
                numpy operation parameters.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                Returns None if the operation is not supported or if the
                FLOP count cannot be determined.
        """
        ...
