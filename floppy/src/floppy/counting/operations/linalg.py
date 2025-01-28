from typing import Any, Optional, Tuple, Union

import numpy as np

__all__ = [
    "CrossOperation",
    "MatmulOperation",
    "TraceOperation",
    "NormOperation",
    "CondOperation",
    "InvOperation",
    "EigOperation",
]

class CrossOperation:
    """FLOP counter for vector cross product operations."""

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """Count FLOPs for cross product operation.

        Note: Cross product is only defined for 3D vectors (and 7D, though rarely used).
        For 3D vectors, cross product requires:
        - 6 multiplications
        - 3 subtractions
        Total: 9 FLOPs per cross product
        """
        # Get number of cross products being computed
        num_operations = max(
            1, result.shape[0] if isinstance(result, np.ndarray) else 1
        )
        return 9 * num_operations


class MatmulOperation:
    """FLOP counter for matrix multiplication operations."""

    def _compute_broadcast_batch_shape(
        self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Compute the broadcasted shape of batch dimensions.

        Args:
            shape1: Shape of first array
            shape2: Shape of second array

        Returns:
            Tuple[int, ...]: Broadcasted batch shape
        """
        batch1 = shape1[:-2] if len(shape1) > 2 else ()
        batch2 = shape2[:-2] if len(shape2) > 2 else ()

        if not batch1 and not batch2:
            return ()

        return np.broadcast_shapes(batch1, batch2)

    def count_flops(self, *args: Any, result: Any) -> Optional[int]:
        """
        Count FLOPs for matrix multiplication with broadcasting support.

        For matrix multiplication C = A @ B:
        - Each element in the result requires one multiplication and one addition per
          inner dimension element
        - Total FLOPs = 2 * M * N * P * batch_size
        where:
            M = rows in result
            N = inner dimension (columns of A, rows of B)
            P = columns in result
            batch_size = product of broadcasted batch dimensions

        Args:
            *args: Input arrays
            result: Result of the operation

        Returns:
            Optional[int]: Number of FLOPs or None if invalid
        """
        try:
            a, b = np.asarray(args[0]), np.asarray(args[1])

            # Handle 1D vector cases
            if a.ndim == 1 and b.ndim == 1:
                # Vector dot product: 2N-1 FLOPs (N multiplications, N-1 additions)
                return 2 * a.shape[0] - 1

            if a.ndim == 1:  # 1D × 2D
                M, N = 1, a.shape[0]
                P = b.shape[1]
            elif b.ndim == 1:  # 2D × 1D
                M, N = a.shape
                P = 1
            else:  # ND × ND
                # Use negative indexing to handle arbitrary batch dimensions
                # shape = (*batch_dims, M, N) for first array
                # shape = (*batch_dims, N, P) for second array
                M = a.shape[-2]
                N = a.shape[-1]  # same as b.shape[-2]
                P = b.shape[-1]

            # Compute broadcasted batch shape
            batch_shape = self._compute_broadcast_batch_shape(a.shape, b.shape)
            batch_size = np.prod(batch_shape) if batch_shape else 1

            # Each element requires N multiplications and N-1 additions
            # For each M×P elements in the result
            return batch_size * M * P * (2 * N - 1)

        except Exception as e:
            raise ValueError(f"Error counting matmul FLOPs: {str(e)}")


class TraceOperation:
    """FLOP count for trace operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for trace operation.

        Trace is the sum of diagonal elements, requiring (n-1) additions
        where n is the number of diagonal elements.
        """
        n = min(args[0].shape)  # Get the minimum dimension (for non-square matrices)
        return n - 1


class NormOperation:
    """FLOP count for np.linalg.norm operation with different norms.

    This class implements FLOP counting for both vector and matrix norms.
    For vectors, it supports p-norms including L1, L2, and L∞.
    For matrices, it supports common matrix norms including Frobenius,
    spectral (2-norm), nuclear, and induced norms (L1 and L∞).
    """

    def count_flops(
        self,
        *args: Any,
        ord: Optional[Union[int, float, str]] = None,
        axis: Optional[Union[int, tuple]] = None,
        keepdims: bool = False,
        result: Any = None,
    ) -> int:
        """Count FLOPs for norm calculation.

        Args:
            args: Variable length argument list. First argument is the input array.
            ord: Order of the norm.
            axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms.
                 If axis is a 2-tuple, it specifies the axes that hold 2-D matrices for matrix norm computation.
                 If axis is None, either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.
            keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with size one.
            result: Not used, kept for API consistency.

        Returns:
            Number of floating point operations.

        Note:
            Vector norm FLOP counts:
            - L2 norm: 2n FLOPs (n multiplies, n-1 adds,  10 sqrt)
            - L1 norm: 2n-1 FLOPs (n absolute values, n-1 additions)
            - L∞ norm: 2n-1 FLOPs (n absolute values, n-1 comparisons)
            - General p-norm: uses PowerOperation for p-th powers
                * n absolute values
                * n power operations (via PowerOperation)
                * (n-1) additions
                * 1 final power operation

            Matrix norm FLOP counts:
            - Frobenius: mn*2 FLOPs (mn multiplies, mn-1 adds, 1 sqrt)
            - L1 (max col sum): mn+m-1 FLOPs (mn abs, m(n-1) adds, m-1 comparisons)
            - L∞ (max row sum): mn+n-1 FLOPs (mn abs, n(m-1) adds, n-1 comparisons)
            - L2 (spectral): ~14n³ FLOPs (SVD based on Trefethen and Bau)
            - Nuclear: ~14n³ + n FLOPs (SVD + sum of singular values)
        """
        x = args[0]

        if axis is None:
            # If no axis specified, compute norm over entire array
            if x.ndim <= 1:
                return self._count_vector_norm_flops(x, ord)
            elif x.ndim == 2:
                return self._count_matrix_norm_flops(x, ord)
            else:
                # For higher dimensions, treat as vector norm over flattened array
                return self._count_vector_norm_flops(x.reshape(-1), ord)

        elif isinstance(axis, tuple) and len(axis) == 2:
            # Matrix norm along specified axes
            # Count FLOPs for each matrix in the remaining dimensions
            matrices_count = np.prod(
                [x.shape[i] for i in range(x.ndim) if i not in axis]
            )
            single_matrix_flops = self._count_matrix_norm_flops(
                x.transpose((*axis, *[i for i in range(x.ndim) if i not in axis]))[
                    0, 0
                ],
                ord,
            )
            return matrices_count * single_matrix_flops

        elif isinstance(axis, (int, tuple)):
            # Vector norm along specified axis/axes
            # Count FLOPs for each vector
            vectors_count = np.prod(
                [
                    x.shape[i]
                    for i in range(x.ndim)
                    if i not in (axis if isinstance(axis, tuple) else (axis,))
                ]
            )
            vector_size = np.prod(
                [x.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))]
            )
            return vectors_count * self._count_vector_norm_flops(
                np.ones(vector_size), ord
            )

        else:
            raise ValueError(f"Invalid axis parameter: {axis}")

    def _count_vector_norm_flops(
        self, x: np.ndarray, ord: Optional[Union[int, float, str]]
    ) -> int:
        """Count FLOPs for vector norm calculation."""
        n = np.size(x)

        if ord is None or ord == 2:
            return 2 * n + 10  # n multiplies, n-1 adds, 10 sqrt
        elif ord == 1:
            return 2 * n - 1  # n absolute values, n-1 additions
        elif ord in (np.inf, float("inf"), -np.inf, float("-inf")):
            return 2 * n - 1  # n absolute values, n-1 comparisons
        else:
            # For general p-norm, assume worst case ~40 FLOPs for power operation (see PowerOperation)
            # 1. n absolute values
            # 2. n power operations
            # 3. (n-1) additions
            # 4. 1 final power (1/p)

            power_flops = n + 40 * n + (n - 1) + 40
            return power_flops

    def _count_matrix_norm_flops(
        self, x: np.ndarray, ord: Optional[Union[int, float, str]]
    ) -> int:
        """Count FLOPs for matrix norm calculation."""
        m, n = x.shape

        if ord is None or ord == "fro":
            return m * n * 2  # mn multiplies, mn-1 adds, 1 sqrt
        elif ord == 1:
            return m * n + m - 1  # mn abs, m(n-1) adds, m-1 comparisons
        elif ord in (np.inf, float("inf")):
            return m * n + n - 1  # mn abs, n(m-1) adds, n-1 comparisons
        elif ord == 2:
            # Spectral norm (largest singular value)
            # Using estimate from Trefethen and Bau (see CondOperation)
            k = min(m, n)
            return 14 * k**3  # SVD complexity
        elif ord == "nuc":
            # Nuclear norm (sum of singular values)
            # SVD + sum of singular values
            k = min(m, n)
            return 14 * k**3 + k  # SVD + k-1 additions
        else:
            raise ValueError(
                f"FLOP count for norm order '{ord}' for matrix norm not implemented"
            )


class CondOperation:
    """FLOP count for np.linalg.cond operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for condition number calculation.

        For a square matrix (m=n), the total FLOP count is:
        - SVD decomposition (~14n^3)
        An estimate for complexity of SVD is ~2mn^2 + 11n^3 per equation 11.22
        in "Numerical Linear Algebra" by Trefethen and Bau.

        - Division of largest by smallest singular value (1)

        Total: ~14n^3 + 1 FLOPs
        """
        n = args[0].shape[0]
        return 14 * n**3 + 1


class InvOperation:
    """FLOP count for np.linalg.inv operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for matrix inversion.

        Matrix inversion using LU decomposition:
        - LU decomposition (~2/3 n³)
        - Forward and backward substitution (~2n²)
        An estimate for complexity of LU decomposition is ~2/3 n³ FLOPs per equation 20.8
        in "Numerical Linear Algebra" by Trefethen and Bau.
        Total: ~2/3 n³ + 2n² FLOPs
        """
        n = args[0].shape[0]
        return (2 * n**3) // 3 + 2 * n**2


class EigOperation:
    """FLOP count for np.linalg.eig operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for eigenvalue decomposition.

        Eigenvalue decomposition using QR algorithm:
        - Reduction to Hessenberg form (~10/3 n³)
        An estimate for complexity of reduction to Hessenberg form is ~10/3 n³ FLOPs per equation 26.1
        in "Numerical Linear Algebra" by Trefethen and Bau.
        - QR iterations using Householder reflections: ~4/3 n³ FLOPs per iteration (~20 iterations)
        An estimate for complexity of QR iterations is ~4/3 n³ FLOPs per equation 10.9
        in "Numerical Linear Algebra" by Trefethen and Bau.

        Total: 10/3 n³ + 4/3 n³ * 20 = 10/3 n³ + 80/3 n³ = 90/3 n³ = 30 n³ FLOPs
        """
        n = args[0].shape[0]
        return 30 * n**3
