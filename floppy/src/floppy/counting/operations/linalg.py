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
    "OuterOperation",
    "InnerOperation",
    "EinsumOperation",
    "SolveOperation",
]

class CrossOperation:
    """Counts floating point operations (FLOPs) for vector cross product operations.

    Handles both single vector pairs and batched computations of 3D cross products.
    Each 3D cross product requires 9 FLOPs (6 multiplications, 3 subtractions).

    Example shapes:
        Single: (3,) x (3,) -> (3,)
        Batched: (N, 3) x (N, 3) -> (N, 3)
    """

    # Constants for the cross product operation
    VECTOR_DIM = 3  # Standard 3D vector dimension
    MULTS_PER_CROSS = 6  # Number of multiplications per cross product
    SUBS_PER_CROSS = 3  # Number of subtractions per cross product
    FLOPS_PER_CROSS = MULTS_PER_CROSS + SUBS_PER_CROSS

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing vector cross products.

        Args:
            *args: Tuple[np.ndarray, ...], Input arrays where each array contains vectors
                  to compute cross product. Typically two 3D vectors.
            result: np.ndarray, The resulting array from the cross product operation.
                   Used to determine the number of cross products computed.
            **kwargs: Additional numpy.cross parameters (e.g., axis, out).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Cross product computation:
            - Only defined for 3D vectors (and 7D vectors, though rarely used)
            - Each 3D cross product requires 9 FLOPs:
                * 6 multiplications (2 per component)
                * 3 subtractions (1 per component)
            - For batched inputs, total FLOPs = 9 * number_of_cross_products
        """
        # Validate input
        if not isinstance(result, np.ndarray):
            return None

        # Validate vector dimension
        if result.shape[-1] != self.VECTOR_DIM:
            return None  # Not a 3D vector cross product

        # Calculate number of cross products from result shape
        # For single vector: shape = (3,)
        # For batch: shape = (N, 3) where N is batch size
        batch_size = result.shape[0] if result.ndim > 1 else 1

        return self.FLOPS_PER_CROSS * batch_size


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
            batch_size = 1
            if batch_shape:
                for dim in batch_shape:
                    batch_size *= dim

            # Each element requires N multiplications and N-1 additions
            # For each M×P elements in the result
            return batch_size * M * P * (2 * N - 1)

        except Exception as e:
            raise ValueError(f"Error counting matmul FLOPs: {str(e)}")


class TraceOperation:
    """FLOP count for trace operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for trace operation.

        Args:
            *args: Input arrays (first argument is the array to compute trace)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.trace parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Trace is the sum of diagonal elements, requiring (n-1) additions
            where n is the number of diagonal elements.
        """
        arr = args[0]
        if len(arr.shape) < 2:
            raise ValueError("Input array must be at least 2D")

        # Get last two dimensions which define the matrices
        matrix_shape = arr.shape[-2:]
        if 0 in matrix_shape:
            return 0

        # Calculate FLOPs for one matrix
        n = min(matrix_shape)
        flops_per_matrix = max(0, n - 1)

        # If more than 2D, multiply by number of matrices
        if len(arr.shape) > 2:
            num_matrices = np.prod(arr.shape[:-2])
            return flops_per_matrix * num_matrices

        return flops_per_matrix


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
        reduction: bool = False,
        reduction_axis: Optional[Union[int, tuple]] = None,
        **kwargs: Any,
    ) -> int:
        """Count FLOPs for norm calculation.

        Args:
            args: Variable length argument list. First argument is the input array.
            ord: Order of the norm.
            axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms.
                 If axis is a 2-tuple, it specifies the axes that hold 2-D matrices for matrix norm computation.
                 If axis is None, either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.
            keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with size one.
            result: Result of the operation.
            reduction: Whether this is being called as part of a reduction operation.
            reduction_axis: The axis along which reduction is being performed.
            **kwargs: Additional keyword arguments.

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
        # If this is a reduction operation, return 0 since the FLOPs are already counted
        if reduction:
            return 0

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

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for condition number calculation.

        Args:
            *args: Input arrays (first argument is the matrix to compute condition number)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.linalg.cond parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            For a square matrix (m=n), the total FLOP count is:
            - SVD decomposition (~14n^3)
            - Division of largest by smallest singular value (1)
            Total: ~14n^3 + 1 FLOPs

            An estimate for complexity of SVD is ~2mn^2 + 11n^3 per equation 11.22
            in "Numerical Linear Algebra" by Trefethen and Bau.
        """
        n = args[0].shape[0]
        return 14 * n**3 + 1


class InvOperation:
    """FLOP count for np.linalg.inv operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for matrix inversion.

        Matrix inversion using LU decomposition requires:
        - LU decomposition (~2/3 n³)
        - Forward and backward substitution (~2n²)

        Args:
            *args: Input arrays (first argument is the matrix to invert)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.linalg.inv parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations for matrix inversion

        Note:
            An estimate for complexity of LU decomposition is ~2/3 n³ FLOPs per equation 20.8
            in "Numerical Linear Algebra" by Trefethen and Bau.
            Total: ~2/3 n³ + 2n² FLOPs
        """
        n = args[0].shape[0]
        if n == 1:  # Special case for 1x1 matrices
            return 1  # Just one division
        return (2 * n**3) // 3 + 2 * n**2


class EigOperation:
    """FLOP count for np.linalg.eig operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for eigenvalue decomposition.

        Args:
            *args: Input arrays (first argument is the matrix to compute eigenvalues)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.linalg.eig parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Eigenvalue decomposition using QR algorithm requires:
            - Reduction to Hessenberg form (~10/3 n³)
            - QR iterations using Householder reflections: ~4/3 n³ FLOPs per iteration (~20 iterations)
            Total: ~30n³ FLOPs

            Estimates from "Numerical Linear Algebra" by Trefethen and Bau:
            - Hessenberg form: ~10/3 n³ FLOPs (equation 26.1)
            - QR iterations: ~4/3 n³ FLOPs per iteration (equation 10.9)
        """
        n = args[0].shape[0]
        return 30 * n**3


class OuterOperation:
    """FLOP count for outer product operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for outer product operation.

        For vectors a (M,) and b (N,), outer product creates an (M,N) matrix
        where each element is a multiplication: result[i,j] = a[i] * b[j]

        Args:
            *args: Input arrays (two vectors)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.outer parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of multiplication operations (M * N)
        """
        return np.size(
            result
        )  # Size of result is M * N, one multiplication per element


class InnerOperation:
    """FLOP count for inner product operation."""

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for inner product operation.

        For vectors a (M,) and b (M,), inner product performs:
        - M multiplications (one per element pair)
        - M-1 additions to sum up the products

        For arrays with more dimensions, this is done over the last axis.

        Args:
            *args: Input arrays
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.inner parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations
        """
        a, b = args[0], args[1]
        inner_dim = a.shape[
            -1
        ]  # Size of the last axis over which inner product is computed
        result_size = np.size(result)  # Number of inner products computed

        # Each inner product requires inner_dim multiplications and (inner_dim - 1) additions
        flops_per_inner = 2 * inner_dim - 1
        return flops_per_inner * result_size


class EinsumOperation:
    """FLOP count for einsum operation.

    This class implements FLOP counting for numpy's einsum operation.
    The FLOP count depends on the einsum equation and input array shapes.
    """

    def _parse_subscripts(self, subscripts: str) -> tuple[str, list[str], str]:
        """Parse einsum subscripts into input and output specifications.

        Args:
            subscripts: The einsum equation string (e.g., "ij,jk->ik")

        Returns:
            Tuple of (full_subscript, input_specs, output_spec)
        """
        # Split into input and output
        full = subscripts.replace(" ", "")
        input_output = full.split("->")

        if len(input_output) == 1:
            input_spec = input_output[0]
            output_spec = ""  # Implicit output
        else:
            input_spec, output_spec = input_output

        # Split input specs
        input_specs = input_spec.split(",")

        return full, input_specs, output_spec

    def _compute_intermediate_size(
        self, spec_chars: str, shapes: list[tuple[int, ...]]
    ) -> int:
        """Compute size of intermediate result for a set of dimensions.

        Args:
            spec_chars: String of dimension characters
            shapes: List of input array shapes

        Returns:
            Product of unique dimension sizes
        """
        # Map each dimension character to its size
        dim_sizes = {}
        for shape, spec in zip(shapes, spec_chars):
            for size, dim in zip(shape, spec):
                if dim in dim_sizes:
                    assert dim_sizes[dim] == size, (
                        f"Inconsistent sizes for dimension {dim}"
                    )
                else:
                    dim_sizes[dim] = size

        # Compute product of all unique dimensions
        size = 1
        for dim in set(spec_chars):
            size *= dim_sizes[dim]
        return size

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for einsum operation.

        Args:
            *args: First arg is subscripts string, followed by input arrays
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.einsum parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            For each element in the output:
            - One multiplication per input array (except the first)
            - One addition per iteration (except the first)

            Example: "ij,jk->ik" with shapes (M,N) and (N,K)
            - Output shape: (M,K)
            - For each M,K element: N multiplications and N-1 additions
            Total: M*K*N*2 - M*K FLOPs
        """
        if len(args) < 2:
            return 0

        subscripts = args[0]
        arrays = args[1:]
        shapes = [np.shape(arr) for arr in arrays]

        # Parse the einsum specification
        full, input_specs, output_spec = self._parse_subscripts(subscripts)

        # Special case for trace-like operations (e.g., "ii->")
        if len(input_specs) == 1 and len(set(input_specs[0])) < len(input_specs[0]):
            # Just need to sum along diagonal - similar to trace
            n = shapes[0][0]  # Size of the diagonal
            return n - 1  # n-1 additions

        # For matrix multiplication style operations:
        # Count multiplications and additions for the contracted dimensions
        contracted_dims = set()
        for spec in input_specs:
            for dim in spec:
                if sum(1 for s in input_specs if dim in s) > 1:
                    contracted_dims.add(dim)

        if not contracted_dims:
            # Element-wise operation
            return (len(arrays) - 1) * 2 * np.size(result)

        # Compute size of the intermediate result (product of all dimensions)
        intermediate_size = self._compute_intermediate_size(
            "".join(input_specs), shapes
        )

        # For each element in intermediate result:
        # - One multiplication per input array (except first)
        # - One addition per iteration (except first)
        mults_per_element = len(arrays) - 1
        adds_per_element = len(arrays) - 2

        return intermediate_size * (mults_per_element + adds_per_element)


class SolveOperation:
    """FLOP count for np.linalg.solve operation.

    This class implements FLOP counting for solving a system of linear equations Ax = b.
    The implementation uses LU decomposition followed by forward and backward substitution.
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> int:
        """Count FLOPs for solving linear system Ax = b.

        Args:
            *args: Input arrays (A matrix and b vector/matrix)
            result: The result array
            **kwargs: Additional keyword arguments that match numpy.linalg.solve parameters.
                     These currently don't affect the FLOP count.

        Returns:
            int: Number of floating point operations

        Note:
            Solving Ax = b using LU decomposition requires:
            1. LU decomposition of A: ~2/3 n³ FLOPs
            2. Forward substitution (Ly = b): n² FLOPs
            3. Backward substitution (Ux = y): n² FLOPs

            For multiple right-hand sides (b is n×k matrix):
            - Forward/backward substitution cost multiplied by k

            Total FLOPs: 2/3 n³ + 2kn² where:
            - n is the dimension of the system
            - k is the number of right-hand sides

            Reference: "Numerical Linear Algebra" by Trefethen and Bau
        """
        if len(args) < 2:
            return 0

        A, b = args[0], args[1]
        n = A.shape[0]  # System dimension

        # Handle multiple right-hand sides
        k = 1 if b.ndim == 1 else b.shape[1]

        if n == 1:  # Special case for 1x1 systems
            return 1  # Just one division

        # LU decomposition cost + forward/backward substitution cost
        lu_flops = (2 * n**3) // 3  # ~2/3 n³ for LU
        solve_flops = 2 * k * n**2  # 2n² per right-hand side

        return lu_flops + solve_flops
