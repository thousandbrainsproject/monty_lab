# flop_counting/wrappers/kdtree.py
from typing import Any, Union, Tuple
from contextlib import contextmanager
import numpy as np
import warnings


class KDTreeWrapper:
    """Wrapper for both scipy and sklearn KDTree with FLOP counting."""

    def __init__(self, kdtree: Any, flop_counter: Any, is_scipy: bool = False):
        """
        Initialize wrapper.

        Args:
            kdtree: Original KDTree instance
            flop_counter: FlopCounter instance
            is_scipy: Whether this is a scipy KDTree (vs sklearn)
        """
        self._tree = kdtree
        self.flop_counter = flop_counter
        self.is_scipy = is_scipy
        self.points_examined = 0
        self.nodes_visited = 0
        self._monitoring = False

    @contextmanager
    def _monitor_traversal(self):
        """Track tree traversal statistics."""
        self._monitoring = True
        self.points_examined = 0
        self.nodes_visited = 0
        try:
            yield
        finally:
            self._monitoring = False

    def query(
        self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1, **kwargs
    ):
        """
        Query the KDTree while counting FLOPs.

        Handles both scipy and sklearn query signatures:
        scipy: query(x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1)
        sklearn: query(X, k=1, return_distance=True, dualtree=False,
                      breadth_first=False, sort_results=True)
        """
        print(
            f"KDTreeWrapper.query called ({'scipy' if self.is_scipy else 'sklearn'} version)"
        )
        x = np.asarray(x)

        # Execute the query
        with self._monitor_traversal():
            if self.is_scipy:
                result = self._tree.query(
                    x,
                    k=k,
                    eps=eps,
                    p=p,
                    distance_upper_bound=distance_upper_bound,
                    workers=workers,
                )
            else:
                # sklearn version
                result = self._tree.query(x, k=k, p=p, workers=workers, **kwargs)

            # Estimate points examined
            if hasattr(self._tree, "data"):
                n_points = len(self._tree.data)
            else:
                # scipy stores points in 'data' or '_data'
                n_points = len(getattr(self._tree, "_data", []))

            self.points_examined = int(k * np.log2(n_points))
            self.nodes_visited = self.points_examined

        # Count FLOPs
        if hasattr(self.flop_counter, "_operations"):
            operation = self.flop_counter._operations.get("kdtree_query")
            if operation is not None:
                try:
                    flops = operation.count_flops(
                        query_points=x,
                        tree_points=self.data,
                        k=k,
                        points_examined=self.points_examined,
                        nodes_visited=self.nodes_visited,
                        result=result,
                        p=p,
                    )
                    if flops is not None:
                        self.flop_counter.add_flops(flops)
                except Exception as e:
                    warnings.warn(f"Error counting KDTree query FLOPs: {str(e)}")

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attributes to the wrapped KDTree."""
        return getattr(self._tree, name)

    @property
    def data(self):
        """Access to the underlying data, handling both scipy and sklearn."""
        if self.is_scipy:
            return getattr(self._tree, "_data", None)
        return getattr(self._tree, "data", None)
