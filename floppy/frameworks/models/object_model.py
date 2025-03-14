import numpy as np
import torch
import torch_geometric
from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    get_correct_k_n,
    remove_close_points,
)
from torch_geometric.data import Data


class FlopCountingObjectModel(GraphObjectModel):
    """Object model that counts FLOPs for KDTree operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flop_counter = None

    def _build_adjacency_graph(
        self, locations, features, k_n, graph_delta_thresholds, old_graph_index
    ):
        locations_reduced, clean_ids = remove_close_points(
            np.array(locations), features, graph_delta_thresholds, old_graph_index
        )
        num_nodes = locations_reduced.shape[0]
        node_features = np.linspace(0, num_nodes - 1, num_nodes).reshape((num_nodes, 1))
        feature_mapping = dict()
        feature_mapping["node_ids"] = [0, 1]

        for feature_id in features.keys():
            feats = np.array([features[feature_id][i] for i in clean_ids])
            if len(feats.shape) == 1:
                feats = feats.reshape((feats.shape[0], 1))

            feature_mapping[feature_id] = [
                node_features.shape[1],
                node_features.shape[1] + feats.shape[1],
            ]
            node_features = np.column_stack((node_features, feats))

            if feature_id == "pose_vectors":
                norm = torch.tensor(feats[:, :3], dtype=torch.float)

        assert np.all(
            locations[:old_graph_index] == locations_reduced[:old_graph_index]
        ), "Old graph points shouldn't change"

        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(locations_reduced, dtype=torch.float)

        graph = Data(x=x, pos=pos, norm=norm, feature_mapping=feature_mapping)

        if k_n is not None:
            k_n = get_correct_k_n(k_n, num_nodes)

            scipy_graph = kneighbors_graph(
                locations_reduced, n_neighbors=k_n, include_self=False
            )
            # Count FLOPs for kneighbors_graph calculation
            n = len(locations_reduced)
            # Calculate distances from each point to all others (n points * (n-1) other points * 9 FLOPs per distance)
            distance_flops = n * (n - 1) * 9
            # Add approximate cost for partial sorting to find k nearest neighbors
            # Using O(n log k) complexity for each point's partial sort
            if k_n > 0:
                sorting_flops = n * (n - 1) * np.log2(k_n)
            if self.flop_counter is not None:
                self.flop_counter.add_flops(distance_flops + sorting_flops)

            scipygraph = torch_geometric.utils.from_scipy_sparse_matrix(scipy_graph)
            edge_index = scipygraph[0]

            displacements = []
            for e, edge_start in enumerate(edge_index[0]):
                edge_end = edge_index[1][e]
                displacements.append(
                    locations_reduced[edge_end] - locations_reduced[edge_start]
                )

            edge_attr = torch.tensor(np.array(displacements), dtype=torch.float)

            graph.edge_index = edge_index
            graph.edge_attr = edge_attr

        return graph
