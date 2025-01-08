import numpy as np
from contextlib import contextmanager
from tbp.monty.frameworks.models.object_model import ObjectModel

class FlopCountingObjectModel(ObjectModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flop_counter = None

    @contextmanager
    def _get_flop_counter(self):
        """Get the active FlopCounter from the context if available."""
        import inspect

        # Walk up the stack to find the FlopCounter context
        frame = inspect.currentframe()
        while frame:
            # Look for FlopCounter instance in locals
            if "self" in frame.f_locals:
                obj = frame.f_locals["self"]
                if hasattr(obj, "flops"):
                    self._flop_counter = obj
                    break
            frame = frame.f_back

        try:
            yield self._flop_counter
        finally:
            pass

    def find_nearest_neighbors(
        self,
        search_locations,
        num_neighbors,
        return_distance=False,
    ):
        """Find nearest neighbors in graph for list of search locations.

        FLOP counting for KDTree query:
        1. Distance calculations (per comparison):
            - 3 subtractions for coordinate differences (3 FLOPs)
            - 3 multiplications for squaring differences (3 FLOPs)
            - 2 additions for summing squared differences (2 FLOPs)
            - 1 square root operation (1 FLOP)
            Total per distance = 9 FLOPs

        2. Tree Traversal (per search point):
            - log(N) levels of tree to traverse
            - 1 comparison per level
            - Approximately log(N) * 1 FLOPs

        3. K-nearest neighbor maintenance (per distance calculation):
            - Heap insertion: log(k) comparisons
            - Each comparison involves 1 FLOP
            - Total per insertion = log(k) FLOPs
        """
        num_search_points = len(search_locations)
        num_tree_points = len(self._location_tree.data)

        # Calculate FLOPs
        flops_per_dist = 9
        traversal_flops = num_search_points * np.log2(num_tree_points)
        heap_ops_per_point = np.log2(num_tree_points) * np.log2(num_neighbors)
        total_heap_flops = num_search_points * heap_ops_per_point
        examined_points = np.log2(num_tree_points)
        distance_flops = num_search_points * examined_points * flops_per_dist
        total_flops = int(distance_flops + traversal_flops + total_heap_flops)

        # Add FLOPs to counter if available
        with self._get_flop_counter() as counter:
            if counter is not None:
                counter.add_flops(total_flops)

        # Perform KDTree query
        (distances, nearest_node_ids) = self._location_tree.query(
            search_locations,
            k=num_neighbors,
            p=2,
            workers=1,
        )

        if return_distance:
            return distances, total_flops
        else:
            return nearest_node_ids, total_flops