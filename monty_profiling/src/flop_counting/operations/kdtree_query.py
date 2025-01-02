import numpy as np
from typing import Any, Optional, Tuple, List
from .base import BaseOperation
import warnings


class KDTreeQueryOperation(BaseOperation):
    """
    Operation class for k-d tree query operations.
    """

    def __init__(self):
        super().__init__("kdtree_query")
        self.points_examined = 0

    def validate_inputs(self, *args: Any) -> bool:
        """
        Validate inputs for KDTree query operation.

        Expected input format:
        - args[0]: query points (n_queries, n_dimensions)
        - args[1]: reference points from KDTree (n_points, n_dimensions)
        - kwargs may include:
            - k: number of nearest neighbors (default=1)
            - return_distance: bool
        """
        if len(args) < 2:
            return False

        try:
            query_points = np.asarray(args[0])
            ref_points = np.asarray(args[1])

            # Check dimensions match
            if query_points.ndim != 2 or ref_points.ndim != 2:
                return False
            if query_points.shape[1] != ref_points.shape[1]:
                return False

            return True
        except Exception:
            return False

    def _count_distance_flops(self, n_dimensions: int) -> int:
        """
        Count FLOPs for a single Euclidean distance calculation.

        For each dimension:
        - 1 subtraction (coordinate difference)
        - 1 multiplication (square)
        Plus:
        - (n_dimensions - 1) additions to sum squares
        - 1 square root

        Args:
            n_dimensions: Number of dimensions in the space

        Returns:
            int: Total FLOPs for one distance calculation
        """
        subtractions = n_dimensions  # One per dimension
        multiplications = n_dimensions  # Square each difference
        additions = n_dimensions - 1  # Sum the squares
        sqrt = 1  # Final square root

        return subtractions + multiplications + additions + sqrt

    def _count_box_distance_flops(self, n_dimensions: int) -> int:
        """
        Count FLOPs for computing distance to a bounding box.

        Per dimension:
        - 2 comparisons
        - Up to 1 subtraction (worst case)
        - 1 multiplication
        Plus:
        - (n_dimensions - 1) additions
        - 1 square root

        Args:
            n_dimensions: Number of dimensions in the space

        Returns:
            int: Total FLOPs for bounding box distance calculation
        """
        comparisons = 2 * n_dimensions  # Min/max comparisons per dimension
        subtractions = n_dimensions  # Worst case: one subtraction per dimension
        multiplications = n_dimensions  # Square each component
        additions = n_dimensions - 1  # Sum squares
        sqrt = 1  # Final square root

        return comparisons + subtractions + multiplications + additions + sqrt

    def count_flops(self, *args: Any, result: Any, **kwargs) -> Optional[int]:
        """
        Count total FLOPs for KDTree query operation.

        Args:
            *args: Should contain:
                - Query points (n_queries, n_dimensions)
                - Reference points (n_points, n_dimensions)
            result: Query result (ignored)
            **kwargs: May include:
                - k: number of nearest neighbors (default=1)
                - points_examined: number of points actually examined
                - nodes_visited: number of nodes visited in tree

        Returns:
            Optional[int]: Total number of FLOPs or None if invalid
        """
        if not self.validate_inputs(*args):
            return None

        try:
            query_points = np.asarray(args[0])
            ref_points = np.asarray(args[1])
            n_queries = query_points.shape[0]
            n_dimensions = query_points.shape[1]

            # Get actual points and nodes examined from kwargs
            points_examined = kwargs.get("points_examined", ref_points.shape[0])
            nodes_visited = kwargs.get("nodes_visited", points_examined)
            k = kwargs.get("k", 1)

            # Per query point:
            flops_per_query = (
                # Distance calculations for examined points
                points_examined * self._count_distance_flops(n_dimensions)
                +
                # Bounding box calculations for visited nodes
                nodes_visited * self._count_box_distance_flops(n_dimensions)
                +
                # Priority queue comparisons (k comparisons per examined point)
                points_examined * k
            )

            total_flops = n_queries * flops_per_query

            # Store points examined for debugging
            self.points_examined = points_examined

            return total_flops

        except Exception as e:
            warnings.warn(f"Error counting KDTree query FLOPs: {str(e)}")
            return None

    def get_operation_details(self) -> dict:
        """Return detailed information about the last operation."""
        return {
            "points_examined": self.points_examined,
            "flops_per_distance": self._count_distance_flops(3),  # Example for 3D
            "flops_per_box": self._count_box_distance_flops(3),  # Example for 3D
        }
