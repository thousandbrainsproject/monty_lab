# flop_counting/wrappers/kdtree.py
import numpy as np
from typing import Any, Union, Tuple
from contextlib import contextmanager
import warnings
from .base import OperationWrapper


class KDTreeWrapper(OperationWrapper):
    """Wrapper for sklearn.neighbors.KDTree with FLOP counting."""

    def __init__(self, kdtree: Any, flop_counter: Any):
        super().__init__(kdtree, flop_counter, "kdtree_query")
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Initialize monitoring attributes."""
        self.points_examined = 0
        self.nodes_visited = 0
        self._monitoring = False

    @contextmanager
    def _monitor_traversal(self):
        """Track tree traversal statistics."""
        self._monitoring = True
        self.points_examined = 0
        self.nodes_visited = 0

        original_min_distance = getattr(self.operation.tree, "min_distance", None)
        original_query_radius = getattr(self.operation.tree, "_query_radius", None)

        def wrapped_min_distance(self_tree, *args, **kwargs):
            if self._monitoring:
                self.points_examined += 1
            return original_min_distance(self_tree, *args, **kwargs)

        def wrapped_query_radius(self_tree, *args, **kwargs):
            if self._monitoring:
                self.nodes_visited += 1
            return original_query_radius(self_tree, *args, **kwargs)

        try:
            setattr(self.operation.tree, "min_distance", wrapped_min_distance)
            setattr(self.operation.tree, "_query_radius", wrapped_query_radius)
            yield
        finally:
            if original_min_distance is not None:
                setattr(self.operation.tree, "min_distance", original_min_distance)
            if original_query_radius is not None:
                setattr(self.operation.tree, "_query_radius", original_query_radius)
            self._monitoring = False

    def query(
        self,
        X: np.ndarray,
        k: int = 1,
        return_distance: bool = True,
        dualtree: bool = False,
        breadth_first: bool = False,
        sort_results: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Query the KDTree while counting FLOPs."""
        X = np.asarray(X)

        with self._monitor_traversal():
            result = self.operation.query(
                X,
                k=k,
                return_distance=return_distance,
                dualtree=dualtree,
                breadth_first=breadth_first,
                sort_results=sort_results,
            )

        try:
            flops = self.flop_counter._operations["kdtree_query"].count_flops(
                X,
                self.operation.data,
                result=result,
                k=k,
                points_examined=self.points_examined,
                nodes_visited=self.nodes_visited,
            )
            if flops is not None:
                self.flop_counter.add_flops(flops)
        except Exception as e:
            warnings.warn(f"Error counting KDTree query FLOPs: {str(e)}")

        return result
