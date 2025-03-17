# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ..base.protocols import FlopOperation

__all__ = [
    "CondOperation",
    "CrossOperation",
    "EigOperation",
    "EinsumOperation",
    "InnerOperation",
    "InvOperation",
    "MatmulOperation",
    "NormOperation",
    "OuterOperation",
    "SolveOperation",
    "TraceOperation",
]


class CrossOperation(FlopOperation):
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
            *args: Input arrays where each array contains vectors
                  to compute cross product. Typically two 3D vectors.
            result: The resulting array from the cross product operation.
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
    """Counts floating point operations (FLOPs) for matrix multiplication operations.

    Handles various matrix multiplication cases including:
    - Vector dot product: (N,) x (N,) -> scalar
    - Matrix-vector multiplication: (M,N) x (N,) -> (M,)
    - Vector-matrix multiplication: (N,) x (N,M) -> (M,)
    - Matrix-matrix multiplication: (M,N) x (N,P) -> (M,P)
    - Batched matrix multiplication: (*,M,N) x (*,N,P) -> (*,M,P)

    Each element in the result requires N multiplications and (N-1) additions,
    where N is the inner dimension (columns of first matrix, rows of second matrix).
    Total FLOPs = batch_size * M * P * (2N - 1) where:
    - M = rows in result
    - N = inner dimension
    - P = columns in result
    - batch_size = product of broadcasted batch dimensions
    """

    def _compute_broadcast_batch_shape(
        self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Compute the broadcasted shape of batch dimensions.

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

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Returns the number of floating point operations for matrix multiplication.

        Args:
            *args: Input arrays where each array contains matrices/vectors
                  to compute matrix multiplication. Typically two arrays.
            result: The resulting array from the matrix multiplication operation.
                   Used to determine the number of operations computed.
            **kwargs: Additional numpy.matmul parameters.
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Matrix multiplication computation:
            - For vector dot product (N,) x (N,): 2N-1 FLOPs
              * N multiplications
              * (N-1) additions
            - For matrix multiplication (M,N) x (N,P): M*P*(2N-1) FLOPs
              * M*P*N multiplications
              * M*P*(N-1) additions
            - For batched inputs, total FLOPs = batch_size * M * P * (2N-1)
              where batch_size is the product of broadcasted batch dimensions
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
            raise ValueError(f"Error counting matmul FLOPs: {e!s}")


class TraceOperation:
    """Counts floating point operations (FLOPs) for matrix trace operations.

    Handles various trace computation cases including:
    - Single matrix: (M,N) -> scalar
    - Batched matrices: (*,M,N) -> (*,)

    The trace operation sums the diagonal elements of a matrix,
    requiring (n-1) additions where n is the number of diagonal elements.
    For batched inputs, the operation is performed independently on each matrix.

    Example shapes:
        Single: (M,N) -> scalar
        Batched: (B,M,N) -> (B,)
    """

    def count_flops(self, *args: Any, result: Any, **kwargs: Any) -> Optional[int]:
        """Returns the number of floating point operations for computing matrix trace.

        Args:
            *args: Input arrays where the first array contains matrices
                  to compute trace. Must be at least 2D.
            result: The resulting array from the trace operation.
                   Used to determine the number of traces computed.
            **kwargs: Additional numpy.trace parameters (e.g., offset, axis1, axis2).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Trace computation:
            - Only defined for matrices (at least 2D arrays)
            - Each trace requires (n-1) additions where n is the number of diagonal elements
            - For batched inputs, total FLOPs = (n-1) * number_of_matrices
            - The number of diagonal elements is min(M,N) where M,N are matrix dimensions
        """
        # Validate input
        if not isinstance(args[0], np.ndarray):
            return None

        arr = args[0]
        if len(arr.shape) < 2:
            return None  # Not a matrix

        # Get last two dimensions which define the matrices
        matrix_shape = arr.shape[-2:]
        if 0 in matrix_shape:
            return 0  # Empty matrix

        # Calculate FLOPs for one matrix
        n = min(matrix_shape)
        flops_per_matrix = max(0, n - 1)

        # If more than 2D, multiply by number of matrices
        if len(arr.shape) > 2:
            num_matrices = np.prod(arr.shape[:-2])
            return flops_per_matrix * num_matrices

        return flops_per_matrix


class NormOperation:
    """Counts floating point operations (FLOPs) for vector and matrix norm operations.

    Handles various norm computations including:
    - Vector norms (L1, L2, L∞, and general p-norms)
    - Matrix norms (Frobenius, spectral, nuclear, and induced norms)
    - Batched computations along specified axes

    Example shapes:
        Vector L2: (N,) -> scalar
        Matrix Frobenius: (M,N) -> scalar
        Batched vector: (B,N) -> (B,)
        Batched matrix: (B,M,N) -> (B,)
    """

    # Constants for vector norm operations
    L2_SQRT_COST = 20  # Cost of square root operation (see PowerOperation)
    POWER_COST = 40  # Cost of power operation (see PowerOperation)

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing norms.

        Args:
            *args: Input arrays where the first array contains vectors/matrices
                  to compute norm. Must be at least 1D.
            result: The resulting array from the norm operation.
                   Used to determine the number of norms computed.
            **kwargs: Additional numpy.linalg.norm parameters (e.g., ord, axis).
                     These affect the FLOP count based on the norm type.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Vector norm computation:
            - L2 norm: 2n + 20 FLOPs (n multiplies, n-1 adds, 20 sqrt)
            - L1 norm: 2n-1 FLOPs (n absolute values, n-1 additions)
            - L∞ norm: 2n-1 FLOPs (n absolute values, n-1 comparisons)
            - General p-norm: ~42n FLOPs
                * n absolute values
                * n power operations (40 FLOPs each)
                * (n-1) additions
                * 1 final power (1/p)

            Matrix norm computation:
            - Frobenius: 2mn FLOPs (mn multiplies, mn-1 adds, 1 sqrt)
            - L1 (max col sum): mn+m-1 FLOPs (mn abs, m(n-1) adds, m-1 comparisons)
            - L∞ (max row sum): mn+n-1 FLOPs (mn abs, n(m-1) adds, n-1 comparisons)
            - L2 (spectral): ~14k³ FLOPs (SVD based on Trefethen and Bau)
            - Nuclear: ~14k³ + k FLOPs (SVD + sum of singular values)
        """
        x = args[0]
        ord = kwargs.get("ord")
        axis = kwargs.get("axis")

        if axis is None:
            # If no axis specified, compute norm over entire array
            if x.ndim <= 1:
                return self._count_vector_norm_flops(x, ord)
            if x.ndim == 2:
                return self._count_matrix_norm_flops(x, ord)
            # For higher dimensions, treat as vector norm over flattened array
            return self._count_vector_norm_flops(x.reshape(-1), ord)

        if isinstance(axis, tuple) and len(axis) == 2:
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

        if isinstance(axis, (int, tuple)):
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

        raise ValueError(f"Invalid axis parameter: {axis}")

    def _count_vector_norm_flops(
        self, x: np.ndarray, ord: Optional[Union[int, float, str]]
    ) -> int:
        """Count FLOPs for vector norm calculation."""
        n = np.size(x)
        if n == 0:
            return 0

        if ord is None or ord == 2:
            return 2 * n + self.L2_SQRT_COST  # n mults, n adds, sqrt cost
        if ord == 1:
            return 2 * n - 1  # n absolute values, n-1 additions
        if ord in (np.inf, float("inf"), -np.inf, float("-inf")):
            return 2 * n - 1  # n absolute values, n-1 comparisons
        # For general p-norm
        # 1. n absolute values
        # 2. n power operations
        # 3. (n-1) additions
        # 4. 1 final power (1/p)
        return n + self.POWER_COST * n + (n - 1) + self.POWER_COST

    def _count_matrix_norm_flops(
        self, x: np.ndarray, ord: Optional[Union[int, float, str]]
    ) -> int:
        """Count FLOPs for matrix norm calculation."""
        m, n = x.shape
        if m == 0 or n == 0:
            return 0

        if ord is None or ord == "fro":
            return m * n * 2 - 1 + self.L2_SQRT_COST  # mn mults, mn-1 adds, sqrt cost
        if ord == 1:
            return m * n + m - 1  # mn abs, m(n-1) adds, m-1 comparisons
        if ord in (np.inf, float("inf")):
            return m * n + n - 1  # mn abs, n(m-1) adds, n-1 comparisons
        if ord == 2:
            # Spectral norm (largest singular value)
            # Using estimate from Trefethen and Bau (see CondOperation)
            k = min(m, n)
            return 14 * k**3  # SVD complexity
        if ord == "nuc":
            # Nuclear norm (sum of singular values)
            # SVD + sum of singular values
            k = min(m, n)
            return 14 * k**3 + k  # SVD + k-1 additions
        raise ValueError(
            f"FLOP count for norm order '{ord}' for matrix norm not implemented"
        )


class CondOperation:
    """Counts floating point operations (FLOPs) for matrix condition number operations.

    Handles condition number computation for square matrices using singular value decomposition (SVD).
    The condition number is the ratio of the largest to smallest singular value.

    Example shapes:
        Single: (N,N) -> scalar
        Batched: (*,N,N) -> (*,)

    The condition number computation requires:
    - SVD decomposition (~14n³ FLOPs)
    - Division of largest by smallest singular value (1 FLOP)
    Total: ~14n³ + 1 FLOPs per matrix

    Reference: "Numerical Linear Algebra" by Trefethen and Bau equation 11.22
    """

    # Constants for condition number operation
    SVD_COST = 14  # Cost of SVD decomposition per n³
    DIV_COST = 1  # Cost of division operation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing matrix condition number.

        Args:
            *args: Input arrays where the first array contains matrices
                  to compute condition number. Must be square matrices.
            result: The resulting array from the condition number operation.
                   Used to determine the number of matrices processed.
            **kwargs: Additional numpy.linalg.cond parameters (e.g., p).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Condition number computation:
            - Only defined for square matrices
            - Each matrix requires:
                * SVD decomposition (~14n³ FLOPs)
                * Division of largest by smallest singular value (1 FLOP)
            - For batched inputs, total FLOPs = (14n³ + 1) * number_of_matrices
        """
        # Validate input
        if not isinstance(args[0], np.ndarray):
            return None

        arr = args[0]
        if arr.ndim < 2:
            return None  # Not a matrix

        # Get matrix dimensions
        matrix_shape = arr.shape[-2:]
        if matrix_shape[0] != matrix_shape[1]:
            return None  # Not a square matrix

        # Calculate FLOPs for one matrix
        n = matrix_shape[0]
        if n == 0:
            return 0  # Empty matrix

        flops_per_matrix = self.SVD_COST * n**3 + self.DIV_COST

        # If more than 2D, multiply by number of matrices
        if arr.ndim > 2:
            num_matrices = np.prod(arr.shape[:-2])
            return flops_per_matrix * num_matrices

        return flops_per_matrix


class InvOperation:
    """Counts floating point operations (FLOPs) for matrix inversion operations.

    Handles matrix inversion computation for square matrices using LU decomposition
    followed by forward and backward substitution.

    Example shapes:
        Single: (N,N) -> (N,N)
        Batched: (*,N,N) -> (*,N,N)

    The matrix inversion computation requires:
    - LU decomposition (~2/3 n³ FLOPs)
    - Forward substitution (n² FLOPs)
    - Backward substitution (n² FLOPs)
    Total: ~2/3 n³ + 2n² FLOPs per matrix

    Reference: "Numerical Linear Algebra" by Trefethen and Bau equation 20.8
    """

    # Constants for matrix inversion operation
    LU_COST = 2 / 3  # Cost of LU decomposition per n³
    SUBST_COST = 1  # Cost of forward/backward substitution per n²

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing matrix inverse.

        Args:
            *args: Input arrays where the first array contains matrices
                  to compute inverse. Must be square matrices.
            result: The resulting array from the matrix inversion operation.
                   Used to determine the number of matrices processed.
            **kwargs: Additional numpy.linalg.inv parameters.
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Matrix inversion computation:
            - Only defined for square matrices
            - Each matrix requires:
                * LU decomposition (~2/3 n³ FLOPs)
                * Forward substitution (n² FLOPs)
                * Backward substitution (n² FLOPs)
            - For batched inputs, total FLOPs = (2/3 n³ + 2n²) * number_of_matrices
            - Special case for 1x1 matrices: 1 FLOP (single division)
        """
        # Validate input
        if not isinstance(args[0], np.ndarray):
            return None

        arr = args[0]
        if arr.ndim < 2:
            return None  # Not a matrix

        # Get matrix dimensions
        matrix_shape = arr.shape[-2:]
        if matrix_shape[0] != matrix_shape[1]:
            return None  # Not a square matrix

        # Calculate FLOPs for one matrix
        n = matrix_shape[0]
        if n == 0:
            return 0  # Empty matrix

        if n == 1:
            return 1  # Special case for 1x1 matrices

        # LU decomposition + forward/backward substitution
        lu_flops = int(self.LU_COST * n**3)  # Cast to int since LU_COST is float
        subst_flops = 2 * self.SUBST_COST * n**2
        flops_per_matrix = lu_flops + subst_flops

        # If more than 2D, multiply by number of matrices
        if arr.ndim > 2:
            num_matrices = np.prod(arr.shape[:-2])
            return flops_per_matrix * num_matrices

        return flops_per_matrix


class EigOperation:
    """Counts floating point operations (FLOPs) for eigenvalue decomposition operations.

    Handles eigenvalue decomposition computation for square matrices using the QR algorithm.
    The computation includes reduction to Hessenberg form followed by QR iterations.

    Example shapes:
        Single: (N,N) -> (N,) for eigenvalues, (N,N) for eigenvectors
        Batched: (*,N,N) -> (*,N) for eigenvalues, (*,N,N) for eigenvectors

    The eigenvalue decomposition computation requires:
    - Reduction to Hessenberg form (~10/3 n³ FLOPs)
    - QR iterations using Householder reflections (~4/3 n³ FLOPs per iteration)
    - Approximately 20 iterations for convergence
    Total: ~30n³ FLOPs per matrix

    Reference: "Numerical Linear Algebra" by Trefethen and Bau
    - Hessenberg form: equation 26.1
    - QR iterations: equation 10.9
    """

    # Constants for eigenvalue decomposition operation
    HESSENBERG_COST = 10.0 / 3  # Cost of reduction to Hessenberg form per n³
    QR_ITER_COST = 4.0 / 3  # Cost of one QR iteration per n³
    NUM_ITERATIONS = 20  # Typical number of QR iterations for convergence

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing eigenvalues.

        Args:
            *args: Input arrays where the first array contains matrices
                  to compute eigenvalues. Must be square matrices.
            result: The resulting array from the eigenvalue decomposition operation.
                   Used to determine the number of matrices processed.
            **kwargs: Additional numpy.linalg.eig parameters.
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Eigenvalue decomposition computation:
            - Only defined for square matrices
            - Each matrix requires:
                * Reduction to Hessenberg form (~10/3 n³ FLOPs)
                * QR iterations (~4/3 n³ FLOPs per iteration)
                * Approximately 20 iterations for convergence
            - For batched inputs, total FLOPs = 30n³ * number_of_matrices
            - The actual number of iterations may vary based on matrix properties
        """
        # Validate input
        if not isinstance(args[0], np.ndarray):
            return None

        arr = args[0]
        if arr.ndim < 2:
            return None  # Not a matrix

        # Get matrix dimensions
        matrix_shape = arr.shape[-2:]
        if matrix_shape[0] != matrix_shape[1]:
            return None  # Not a square matrix

        # Calculate FLOPs for one matrix
        n = matrix_shape[0]
        if n == 0:
            return 0  # Empty matrix

        # Calculate total cost including both Hessenberg reduction and QR iterations
        total_cost = (
            self.HESSENBERG_COST + self.QR_ITER_COST * self.NUM_ITERATIONS
        ) * n**3
        flops_per_matrix = int(round(total_cost))  # Round to nearest integer

        # If more than 2D, multiply by number of matrices
        if arr.ndim > 2:
            num_matrices = np.prod(arr.shape[:-2])
            return flops_per_matrix * num_matrices

        return flops_per_matrix


class OuterOperation:
    """Counts floating point operations (FLOPs) for vector outer product operations.

    Handles both single vector pairs and batched computations of outer products.
    Each element in the result matrix requires one multiplication operation.

    Example shapes:
        Single: (M,) x (N,) -> (M,N)
        Batched: (B,M) x (B,N) -> (B,M,N)

    The outer product computation requires:
    - M*N multiplications for single vectors
    - B*M*N multiplications for batched vectors
    where:
    - M is the length of the first vector
    - N is the length of the second vector
    - B is the batch size (if applicable)
    """

    # Constants for the outer product operation
    MULTS_PER_ELEMENT = 1  # Number of multiplications per element in result

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing vector outer products.

        Args:
            *args: Input arrays where each array contains vectors
                  to compute outer product. Typically two vectors.
            result: The resulting array from the outer product operation.
                   Used to determine the number of operations computed.
            **kwargs: Additional numpy.outer parameters.
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Outer product computation:
            - Each element in the result matrix requires one multiplication
            - Total FLOPs = M * N where:
                * M is the length of the first vector
                * N is the length of the second vector
            - For batched inputs, total FLOPs = B * M * N where B is batch size
        """
        # Validate input
        if not isinstance(result, np.ndarray):
            return None

        # Each element in the result requires one multiplication
        return self.MULTS_PER_ELEMENT * np.size(result)


class InnerOperation:
    """Counts floating point operations (FLOPs) for vector inner product operations.

    Handles both single vector pairs and batched computations of inner products.
    Each inner product requires N multiplications and (N-1) additions, where N is
    the size of the last axis over which the inner product is computed.

    Example shapes:
        Single: (N,) x (N,) -> scalar
        Batched: (B,N) x (B,N) -> (B,)
        Multi-dimensional: (...,N) x (...,N) -> (...)

    The inner product computation requires:
    - N multiplications (one per element pair)
    - (N-1) additions to sum the products
    where N is the size of the last axis.
    For batched inputs, the operation is performed independently on each pair.
    """

    # Constants for the inner product operation
    MULTS_PER_ELEMENT = 1  # Number of multiplications per element pair
    ADDS_PER_SUM = 1  # Number of additions per sum operation

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for inner product operations.

        Args:
            *args: Input arrays where each array contains vectors
                  to compute inner product. Must have same shape.
            result: The resulting array from the inner product operation.
                   Used to determine the number of operations computed.
            **kwargs: Additional numpy.inner parameters.
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Inner product computation:
            - Each element pair requires one multiplication
            - Each sum requires (N-1) additions where N is the number of elements
            - For batched inputs, the operation is performed independently on each pair
            - For empty arrays, returns 0 as no operations are performed
        """
        try:
            a, b = np.asarray(args[0]), np.asarray(args[1])

            # Handle empty arrays
            if a.size == 0 or b.size == 0:
                return 0

            # Get the last axis size (N) which is the dimension over which inner product is computed
            N = a.shape[-1]
            if b.shape[-1] != N:
                return None  # Shapes must match on last axis

            # Calculate number of inner products from result shape
            # For single vectors: shape = ()
            # For batched vectors: shape = (batch_size,)
            # For multi-dimensional: shape = (...,)
            batch_size = np.prod(result.shape) if result.shape else 1

            # Each inner product requires:
            # - N multiplications (one per element pair)
            # - (N-1) additions to sum the products
            flops_per_inner = N + (N - 1)

            return flops_per_inner * batch_size

        except Exception as e:
            raise ValueError(f"Error counting inner product FLOPs: {e!s}")


class EinsumOperation:
    """Counts floating point operations (FLOPs) for einsum operations.

    Handles various einsum computation cases including:
    - Matrix multiplication: "ij,jk->ik" with shapes (M,N) and (N,K) -> (M,K)
    - Vector dot product: "i,i->" with shapes (N,) and (N,) -> scalar
    - Trace operation: "ii->" with shape (N,N) -> scalar
    - Element-wise operations: "i,i->i" with shapes (N,) and (N,) -> (N,)
    - Batched operations: "bij,bjk->bik" with shapes (B,M,N) and (B,N,K) -> (B,M,K)

    The FLOP count depends on the einsum equation and input array shapes:
    - For matrix multiplication: M*K*(2N-1) FLOPs
        * M*K*N multiplications
        * M*K*(N-1) additions
    - For trace operations: (N-1) additions
    - For element-wise operations: (num_inputs-1)*2*output_size FLOPs
        * (num_inputs-1)*output_size multiplications
        * (num_inputs-1)*output_size additions

    Example shapes:
        Matrix multiplication: (M,N) x (N,K) -> (M,K)
        Vector dot product: (N,) x (N,) -> scalar
        Trace: (N,N) -> scalar
        Batched matrix multiplication: (B,M,N) x (B,N,K) -> (B,M,K)
    """

    # Constants for einsum operation
    MULTS_PER_ELEMENT = 1  # Number of multiplications per element
    ADDS_PER_ELEMENT = 1  # Number of additions per element
    TRACE_ADDS_PER_DIAG = 1  # Number of additions per diagonal element in trace

    def _parse_subscripts(self, subscripts: str) -> Tuple[str, List[str], str]:
        """Parse einsum subscripts into input and output specifications.

        Args:
            subscripts: The einsum equation string (e.g., "ij,jk->ik")

        Returns:
            Tuple[str, List[str], str]: A tuple containing:
                - The full subscript string with spaces removed
                - List of input subscript specifications
                - Output subscript specification (empty string if implicit)
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
        self, spec_chars: str, shapes: List[Tuple[int, ...]]
    ) -> int:
        """Compute size of intermediate result for a set of dimensions.

        Args:
            spec_chars: String of dimension characters from input specifications
            shapes: List of input array shapes

        Returns:
            int: Product of unique dimension sizes

        Raises:
            AssertionError: If inconsistent sizes are found for the same dimension
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

        # Compute product of all unique dimensions that appear in input specs
        size = 1
        for dim in set(spec_chars):
            if dim in dim_sizes:  # Only consider dimensions that appear in input specs
                size *= dim_sizes[dim]
        return size

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for computing einsum operations.

        Args:
            *args: First argument is the einsum equation string, followed by input arrays
                  to compute the einsum operation. Must have matching shapes according
                  to the equation.
            result: The resulting array from the einsum operation.
                   Used to determine the number of operations computed.
            **kwargs: Additional numpy.einsum parameters (e.g., optimize, order).
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Einsum computation FLOPs:
            - For matrix multiplication (e.g., "ij,jk->ik"):
                * M*K*N multiplications
                * M*K*(N-1) additions
            - For trace operations (e.g., "ii->"):
                * (N-1) additions where N is matrix dimension
            - For element-wise operations (e.g., "i,i->i"):
                * (num_inputs-1)*output_size multiplications
                * (num_inputs-1)*output_size additions
            - For batched operations, multiply by batch size
            - For diagonal operations (e.g., "ii->i"):
                * 0 FLOPs as it's just indexing
        """
        if len(args) < 2:
            return None

        subscripts = args[0]
        arrays = args[1:]
        shapes = [np.shape(arr) for arr in arrays]

        # Parse the einsum specification
        full, input_specs, output_spec = self._parse_subscripts(subscripts)

        # Special case for diagonal operations (e.g., "ii->i")
        if len(input_specs) == 1 and len(set(input_specs[0])) < len(input_specs[0]):
            if len(output_spec) == 1 and output_spec[0] in input_specs[0]:
                return 0  # Just indexing, no operations needed
            # Just need to sum along diagonal - similar to trace
            n = shapes[0][0]  # Size of the diagonal
            return (n - 1) * self.TRACE_ADDS_PER_DIAG

        # For matrix multiplication style operations:
        # Count multiplications and additions for the contracted dimensions
        contracted_dims = set()
        for spec in input_specs:
            for dim in spec:
                if sum(1 for s in input_specs if dim in s) > 1:
                    contracted_dims.add(dim)

        if not contracted_dims:
            # Element-wise operation or full array sum
            if not output_spec:  # Full array sum case (e.g., "ij->")
                total_elements = np.size(arrays[0])
                return total_elements - 1  # Need (n-1) additions to sum n elements

            # Check if this is an outer product operation
            # For outer product, output should contain all input dimensions
            if len(output_spec) == len("".join(input_specs)):
                return np.size(result)  # Just count multiplications for outer product

            return (len(arrays) - 1) * 2 * np.size(result)

        # For matrix multiplication, we need to count:
        # 1. Multiplications: M*K*N where:
        #    - M is the size of the first dimension of first matrix
        #    - K is the size of the second dimension of second matrix
        #    - N is the size of the contracted dimension
        # 2. Additions: M*K*(N-1) for summing the products

        # Get the dimensions from the shapes
        dim_sizes = {}
        for shape, spec in zip(shapes, input_specs):
            for size, dim in zip(shape, spec):
                if dim in dim_sizes:
                    # Only check size consistency for contracted dimensions
                    if dim in contracted_dims and dim_sizes[dim] != size:
                        return None  # Return None for incompatible shapes
                else:
                    dim_sizes[dim] = size

        # For matrix multiplication, we need to count all dimensions
        total_size = 1
        for dim in set("".join(input_specs)):
            if dim in dim_sizes:  # Only consider dimensions that appear in input specs
                total_size *= dim_sizes[dim]

        # Count multiplications and additions
        mults = total_size  # One multiplication per element in intermediate result
        adds = total_size - np.size(
            result
        )  # One less addition than multiplication per output element

        return mults + adds


class SolveOperation:
    """Counts floating point operations (FLOPs) for solving linear systems.

    Handles solving systems of linear equations Ax = b using LU decomposition
    followed by forward and backward substitution. Supports both single systems
    and multiple right-hand sides.

    Example shapes:
        Single system: (N,N) x (N,) -> (N,)
        Multiple right-hand sides: (N,N) x (N,K) -> (N,K)
        Batched systems: (*,N,N) x (*,N) -> (*,N)
        Batched multiple right-hand sides: (*,N,N) x (*,N,K) -> (*,N,K)

    The computation requires:
    - LU decomposition: ~2/3 n³ FLOPs
    - Forward substitution (Ly = b): n² FLOPs per right-hand side
    - Backward substitution (Ux = y): n² FLOPs per right-hand side
    Total: ~2/3 n³ + 2kn² FLOPs where:
    - n is the system dimension
    - k is the number of right-hand sides
    - For batched inputs, multiply by batch size

    Reference: "Numerical Linear Algebra" by Trefethen and Bau
    - LU decomposition: equation 20.8
    - Forward/backward substitution: equations 17.1, 17.2
    """

    # Constants for solve operation
    LU_COST = 2 / 3  # Cost of LU decomposition per n³
    SUBST_COST = 1  # Cost of forward/backward substitution per n²
    SINGLE_DIV_COST = 1  # Cost of single division for 1x1 systems

    def count_flops(
        self, *args: Any, result: np.ndarray, **kwargs: Any
    ) -> Optional[int]:
        """Returns the number of floating point operations for solving linear systems.

        Args:
            *args: Input arrays where the first array contains the coefficient matrix A
                  and the second array contains the right-hand side(s) b.
                  Must be compatible shapes for solving Ax = b.
            result: The resulting array from the solve operation.
                   Used to determine the number of systems solved.
            **kwargs: Additional numpy.linalg.solve parameters.
                     These do not affect the FLOP count.

        Returns:
            Optional[int]: Number of floating point operations (FLOPs).
                          Returns None if operation cannot be performed.

        Note:
            Linear system solving computation:
            - Only defined for square coefficient matrices
            - Each system requires:
                * LU decomposition (~2/3 n³ FLOPs)
                * Forward substitution (n² FLOPs per right-hand side)
                * Backward substitution (n² FLOPs per right-hand side)
            - For multiple right-hand sides, multiply substitution costs by k
            - For batched inputs, multiply by batch size
            - Special case for 1x1 systems: 1 FLOP (single division)
        """
        if len(args) < 2:
            return None

        A, b = args[0], args[1]
        if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
            return None

        # Get system dimensions
        n = A.shape[0]  # System dimension
        if n == 0:
            return 0  # Empty system

        if n == 1:
            return self.SINGLE_DIV_COST  # Special case for 1x1 systems

        # Handle multiple right-hand sides
        k = 1 if b.ndim == 1 else b.shape[1]

        # LU decomposition cost + forward/backward substitution cost
        lu_flops = int(self.LU_COST * n**3)  # Cast to int since LU_COST is float
        solve_flops = 2 * k * self.SUBST_COST * n**2  # 2n² per right-hand side

        return lu_flops + solve_flops
