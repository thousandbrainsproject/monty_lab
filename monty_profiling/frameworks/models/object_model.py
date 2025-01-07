import numpy as np

from tbp.monty.frameworks.models.object_model import ObjectModel


class FlopCountingObjectModel(ObjectModel):
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

        The previous version was a significant underestimate as it:
        - Ignored tree traversal costs
        - Ignored k-nearest neighbor maintenance
        - Assumed all distances were computed (while KD-tree actually prunes many)

        Returns:
            If return_distance is True, return (distances, flop_count).
            Otherwise, return (indices of nearest neighbors, flop_count).
        """
        num_search_points = len(search_locations)
        num_tree_points = len(self._location_tree.data)

        # 1. Distance calculation FLOPs
        flops_per_dist = 9

        # 2. Tree traversal FLOPs
        # Each search point traverses approximately log(N) levels
        traversal_flops = num_search_points * np.log2(num_tree_points)

        # 3. K-nearest neighbor maintenance FLOPs
        # For each distance calculation that makes it to the heap
        # We need log(k) operations to maintain the heap
        # Assuming we check about log(N) points per search point
        heap_ops_per_point = np.log2(num_tree_points) * np.log2(num_neighbors)
        total_heap_flops = num_search_points * heap_ops_per_point

        # Average case: we don't compute all distances due to tree pruning
        # Typically examine O(log N) points instead of N points
        examined_points = np.log2(num_tree_points)
        distance_flops = num_search_points * examined_points * flops_per_dist

        total_flops = distance_flops + traversal_flops + total_heap_flops

        # Perform KDTree query
        (distances, nearest_node_ids) = self._location_tree.query(
            search_locations,
            k=num_neighbors,
            p=2,  # euclidean distance
            workers=1,  # using more than 1 worker slows down run on lambda
        )

        if return_distance:
            return distances, total_flops
        else:
            return nearest_node_ids, total_flops
