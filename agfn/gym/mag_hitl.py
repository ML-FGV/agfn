import torch
import torch.nn as nn
import numpy as np

from agfn.commons import EdgeType, outer_product, elementwise_maximum
from agfn.gym.mag import MAG


class MAG_HITL(MAG):
    def __init__(
        self,
        num_nodes: int,
        batch_size: int,
        log_reward: nn.Module,
        max_degree: int | None = None,
    ):
        self.num_edge_types = len(EdgeType)

        super(MAG_HITL, self).__init__(
            num_nodes, batch_size, log_reward, max_degree, device="cpu"
        )

        self.edges = torch.triu_indices(num_nodes, num_nodes, offset=1).t()
        self.transition_step = 0

        # batch_index, node_index_1, node_index_2
        self.edge_list = torch.zeros((0, 3), dtype=torch.long, device=self.device)
        self.edge_is_bidirected = torch.zeros(
            (0,), dtype=torch.bool, device=self.device
        )

        # Remove unused attributes
        del self.indices, self.edge_degree

        # alpha indicates whether there is a directed path from x to y
        # beta indicates whether there is an almost directed path from x to y
        self.alpha = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )
        self.beta = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )

        node_indices = torch.arange(self.num_nodes, device=self.device)
        self.alpha[:, node_indices, node_indices] = 1.0

        self.left_causes_right_action_indices = (
            torch.arange(self.max_num_edges)
            + self.max_num_edges * EdgeType.LEFT_CAUSES_RIGHT.to_int()
        )
        self.right_causes_left_action_indices = (
            torch.arange(self.max_num_edges)
            + self.max_num_edges * EdgeType.RIGHT_CAUSES_LEFT.to_int()
        )
        self.latent_confounding_action_indices = (
            torch.arange(self.max_num_edges)
            + self.max_num_edges * EdgeType.LATENT_CONFOUNDING.to_int()
        )

        # A forward action corresponds to adding a edge of a given type
        self.forward_mask = torch.ones(
            (self.batch_size, self.max_num_edges * self.num_edge_types),
            device=self.device,
        )

        # A backward action corresponds to removing an edge
        self.backward_mask = torch.zeros_like(self.forward_mask)

        # A state has been modified if an edge has been added or removed
        # (We use numpy arrays for indexing since it works better with
        # Python non-native types, e.g., EdgeType)
        self.state_has_been_modified = np.zeros(
            (self.batch_size,), dtype=bool, device=self.device
        )

    def add_to_edge_list(
        self, indices: np.ndarray, edge_nodes: torch.Tensor, is_bidirected: bool
    ):
        indices = torch.from_numpy(indices)
        self.edge_list = torch.vstack(
            [
                self.edge_list,
                torch.hstack([indices.unsqueeze(1), edge_nodes[indices]]),
            ]
        )

        if is_bidirected:
            self.edge_list = torch.vstack(
                [
                    self.edge_list,
                    torch.hstack(
                        [indices.unsqueeze(1), edge_nodes[..., [1, 0]][indices]]
                    ),
                ]
            )

            self.edge_is_bidirected = torch.hstack(
                [
                    self.edge_is_bidirected,
                    torch.ones_like(indices, dtype=self.edge_is_bidirected.dtype),
                    torch.ones_like(indices, dtype=self.edge_is_bidirected.dtype),
                ]
            )
        else:
            self.edge_is_bidirected = torch.hstack(
                [
                    self.edge_is_bidirected,
                    torch.zeros_like(indices, dtype=self.edge_is_bidirected.dtype),
                ]
            )

    def update_directed_paths(self, indices: torch.Tensor, edge_nodes: torch.Tensor):
        left_edge_nodes = edge_nodes[indices, 0]
        right_edge_nodes = edge_nodes[indices, 1]

        # Marks x -> y as a directed path
        self.alpha[indices, left_edge_nodes, right_edge_nodes] = 1.0

        # For each (a, b), if there is a path a -> x and y -> b, then there is a path a -> x -> y -> b
        # when the edge x -> y is appended to the graph
        self.alpha[indices] = elementwise_maximum(
            self.alpha[indices],
            outer_product(
                self.alpha[indices, :, left_edge_nodes],
                self.alpha[indices, right_edge_nodes, :],
            ),
        )

    def update_almost_directed_paths(
        self,
        indices: torch.Tensor,
        edge_nodes: torch.Tensor,
        is_bidirected: bool = False,
    ):
        left_edge_nodes = edge_nodes[indices, 0]
        right_edge_nodes = edge_nodes[indices, 1]

        # If there is a directed path from a to x and from y to b, adding a bidirected
        # edge from x to y will create an almost directed path from a to b
        # Similarly, if there is a bidirected path from a to y and from x to b, adding a
        # bidirected edge from x to y will create an almost directed path from a to b
        if is_bidirected:
            self.beta[indices, left_edge_nodes, right_edge_nodes] = 1.0
            self.beta[indices, right_edge_nodes, left_edge_nodes] = 1.0

            self.beta[indices] = elementwise_maximum(
                self.beta[indices],
                outer_product(
                    self.alpha[indices, :, left_edge_nodes],
                    self.alpha[indices, right_edge_nodes, :],
                ),
            )
            self.beta[indices] = elementwise_maximum(
                self.beta[indices],
                outer_product(
                    self.alpha[indices, :, right_edge_nodes],
                    self.alpha[indices, left_edge_nodes, :],
                ),
            )
        else:
            # If there is an almost directed path from a to x and a directed path from y to b, adding a
            # directed edge from x to y will create an almost directed path from a to b

            self.beta[indices] = elementwise_maximum(
                self.beta[indices],
                outer_product(
                    self.beta[indices, :, left_edge_nodes],
                    self.alpha[indices, right_edge_nodes, :],
                ),
                outer_product(
                    self.alpha[indices, :, left_edge_nodes],
                    self.beta[indices, right_edge_nodes, :],
                ),
            )

    def mask_added_edges(
        self, batch_ids: torch.Tensor, edge_nodes_indices: torch.Tensor
    ):
        for edge_type in range(self.num_edge_types):
            self.forward_mask[
                batch_ids, self.max_num_edges * edge_type + edge_nodes_indices
            ] = 0.0

    def mask_actions_leading_to_cycles(self):
        left_edge_nodes = self.edges[:, 0]
        right_edge_nodes = self.edges[:, 1]

        has_directed_path_from_right_to_left = self.alpha[
            :, right_edge_nodes, left_edge_nodes
        ]
        has_almost_directed_path_from_right_to_left = self.beta[
            :, right_edge_nodes, left_edge_nodes
        ]

        has_directed_path_from_left_to_right = self.alpha[
            :, left_edge_nodes, right_edge_nodes
        ]
        has_almost_directed_path_from_left_to_right = self.beta[
            :, left_edge_nodes, right_edge_nodes
        ]

        self.forward_mask[:, self.left_causes_right_action_indices] *= (
            1 - has_directed_path_from_right_to_left
        ) * (1 - has_almost_directed_path_from_right_to_left)

        self.forward_mask[:, self.right_causes_left_action_indices] *= (
            1 - has_directed_path_from_left_to_right
        ) * (1 - has_almost_directed_path_from_left_to_right)

        self.forward_mask[:, self.latent_confounding_action_indices] *= (
            1 - has_directed_path_from_left_to_right
        ) * (1 - has_directed_path_from_right_to_left)

    def apply(self, indices: torch.Tensor):
        # Indices should be stored as numpy arrays
        can_add_edges = (self.forward_mask.sum(dim=1) > 0).numpy()

        self.is_initial[:] = 0
        self.stopped = torch.from_numpy(~can_add_edges)

        edge_nodes_indices = indices % self.max_num_edges
        edge_nodes = self.edges[edge_nodes_indices]

        # Check the edge type based on the index
        edge_types = indices.numpy() // self.max_num_edges
        edge_types = np.array([EdgeType.from_int(et) for et in edge_types])

        (left_causes_right_indices,) = np.where(
            (edge_types == EdgeType.LEFT_CAUSES_RIGHT) & can_add_edges
        )
        (right_causes_left_indices,) = np.where(
            (edge_types == EdgeType.RIGHT_CAUSES_LEFT) & can_add_edges
        )
        (latent_confounding_indices,) = np.where(
            (edge_types == EdgeType.LATENT_CONFOUNDING) & can_add_edges
        )

        self.add_to_edge_list(
            left_causes_right_indices, edge_nodes, is_bidirected=False
        )
        self.add_to_edge_list(
            right_causes_left_indices, edge_nodes[..., [1, 0]], is_bidirected=False
        )
        self.add_to_edge_list(
            latent_confounding_indices, edge_nodes, is_bidirected=True
        )

        # Recall: self.alpha[batch, x, y] indicates whether there is a directed path from x to y
        # Recall: self.beta[batch, x, y] indicates whether there is an almost directed path from x to y

        self.mask_added_edges(
            self.batch_ids[can_add_edges], edge_nodes_indices[can_add_edges]
        )

        # Update alpha
        self.update_directed_paths(left_causes_right_indices, edge_nodes)
        self.update_directed_paths(right_causes_left_indices, edge_nodes[..., [1, 0]])

        # Update beta
        self.update_almost_directed_paths(
            left_causes_right_indices, edge_nodes, is_bidirected=False
        )
        self.update_almost_directed_paths(
            right_causes_left_indices, edge_nodes[..., [1, 0]], is_bidirected=False
        )
        self.update_almost_directed_paths(
            latent_confounding_indices, edge_nodes, is_bidirected=True
        )

        # Mask out all edges that may lead to either directed or almost directed cycles
        self.mask_actions_leading_to_cycles()

        # Remove mask from all backward actions related to the added edges
        self.backward_mask[self.batch_ids[can_add_edges], indices[can_add_edges]] = 1.0

        self.state_has_been_modified = can_add_edges

        self.transition_step += 1

    def backward(self, indices):
        del indices
        raise NotImplementedError("Backward is not implemented for MAG_HITL")

    def f2b(self, actions):
        # forward_action = num_edges * edge_type + edge_index
        # backward_action = edge_index
        return actions % self.max_num_edges

    @property
    def adjacency_matrix(self):
        adj = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )
        adj[self.edge_list[:, 0], self.edge_list[:, 1], self.edge_list[:, 2]] = 1.0
        return adj

    @property
    def parents_and_spouses(self):
        parents = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )  # Directed edges
        spouses = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )  # Bidirected edges
        is_bidirected = self.edge_is_bidirected

        parents[
            self.edge_list[~is_bidirected, 0],
            self.edge_list[~is_bidirected, 1],
            self.edge_list[~is_bidirected, 2],
        ] = 1.0
        spouses[
            self.edge_list[is_bidirected, 0],
            self.edge_list[is_bidirected, 1],
            self.edge_list[is_bidirected, 2],
        ] = 1.0

        return parents, spouses
