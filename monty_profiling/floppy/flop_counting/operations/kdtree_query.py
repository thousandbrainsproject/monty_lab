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

    def validate_inputs(self, query_points, tree_points, **kwargs) -> bool:
        """Validate inputs for KDTree query."""
        try:
            query_points = np.asarray(query_points)
            tree_points = np.asarray(tree_points)

            if query_points.ndim != 2 or tree_points.ndim != 2:
                return False
            if query_points.shape[1] != tree_points.shape[1]:
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

    def count_flops(
        self,
        query_points,
        tree_points,
        k=1,
        points_examined=None,
        nodes_visited=None,
        result=None,
        **kwargs,
    ) -> Optional[int]:
        """
        Count total FLOPs for KDTree query operation.

        Args:
            query_points: Query points array (n_queries, n_dimensions)
            tree_points: Points in the KDTree (n_points, n_dimensions)
            k: Number of nearest neighbors
            points_examined: Number of points examined during search
            nodes_visited: Number of nodes visited during search
            result: Query result (unused, kept for consistency)
        """
        if not self.validate_inputs(query_points, tree_points):
            return None

        try:
            n_queries = query_points.shape[0]
            n_dimensions = query_points.shape[1]

            # Use provided counts or defaults
            points_examined = (
                points_examined if points_examined is not None else tree_points.shape[0]
            )
            nodes_visited = (
                nodes_visited if nodes_visited is not None else points_examined
            )

            # Calculate FLOPs per operation
            dist_flops = self._count_distance_flops(n_dimensions)
            box_flops = self._count_box_distance_flops(n_dimensions)

            # Total FLOPs per query
            flops_per_query = (
                points_examined * dist_flops  # Point distance calculations
                + nodes_visited * box_flops  # Bounding box calculations
                + points_examined * k  # Priority queue operations
            )

            # Store for stats
            self.points_examined = points_examined

            return n_queries * flops_per_query

        except Exception as e:
            warnings.warn(f"Error counting KDTree query FLOPs: {str(e)}")
            return None
