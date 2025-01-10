import numpy as np
from typing import Any, Optional

class NormOperation:
    """FLOP count for np.linalg.norm operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for norm calculation.

        For Frobenius/L2 norm:
        - n multiplications for squares
        - (n-1) additions for sum
        - 1 square root
        Total: 2n FLOPs
        """
        return 2 * np.size(args[0])


class CondOperation:
    """FLOP count for np.linalg.cond operation."""

    def count_flops(self, *args: Any, result: Any) -> int:
        """Count FLOPs for condition number calculation.

        Condition number requires:
        - SVD decomposition (~14n³ for an nxn matrix)
        - Division of largest by smallest singular value
        Total: ~14n³ + 1 FLOPs
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
        - QR iterations (~9n³ per iteration, typically ~2 iterations)
        Total: ~22n³ FLOPs (approximate)
        """
        n = args[0].shape[0]
        return 22 * n**3
