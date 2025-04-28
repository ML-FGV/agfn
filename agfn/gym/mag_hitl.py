import torch
import torch.nn as nn
import numpy as np

from agfn.commons import EdgeType
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

        self.edges = torch.triu_indices(num_nodes, num_nodes, offset=1)

        assert len(self.actions) == self.num_edge_types * self.max_num_edges
        assert self.device == "cpu", "MAG_HITL must be instantiated on CPU"

        # batch_index, node_index_1, node_index_2
        self.edge_list = torch.zeros((0, 3), dtype=torch.long, device=self.device)

        self.forward_mask = torch.ones(
            (self.batch_size, self.actions.shape[0]), device=self.device
        )
        self.backward_mask = torch.zeros(
            (self.batch_size, self.max_num_edges), device=self.device
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
        self.beta[:, node_indices, node_indices] = 1.0

        self.left_causes_right_action_indices = (
            torch.arange(self.max_num_edges)
            + self.max_num_edges * EdgeType.LEFT_CAUSES_RIGHT
        )
        self.right_causes_left_action_indices = (
            torch.arange(self.max_num_edges)
            + self.max_num_edges * EdgeType.RIGHT_CAUSES_LEFT
        )
        self.latent_confounding_action_indices = (
            torch.arange(self.max_num_edges)
            + self.max_num_edges * EdgeType.LATENT_CONFOUNDING
        )

    def add_to_edge_list(self, indices, edge_nodes):
        self.edge_list = torch.vstack(
            [
                self.edge_list,
                torch.hstack([indices.unsqueeze(0), edge_nodes]),
            ]
        )

    def update_directed_paths(self, indices: torch.Tensor, edge_nodes: torch.Tensor):
        # Marks x -> y as a directed path
        self.alpha[indices, edge_nodes[indices, 0], edge_nodes[indices, 1]] = 1.0

        # For each (a, b), if there is a path a -> x and y -> b, then there is a path a -> x -> y -> b
        # when the edge x -> y is appended to the graph
        self.alpha[indices] = torch.maximum(
            self.alpha[indices],
            self.alpha[indices, edge_nodes[indices, 0], :]
            @ self.alpha[indices, :, edge_nodes[indices, 1]],
        )

    def update_almost_directed_paths(
        self, indices: torch.Tensor, edge_nodes: torch.Tensor, bidirected: bool = False
    ):
        # If there is a directed path from a to x and from y to b, adding a bidirected
        # edge from x to y will create an almost directed path from a to b
        # Similarly, if there is a bidirected path from a to y and from x to b, adding a
        # bidirected edge from x to y will create an almost directed path from a to b
        if bidirected:
            self.beta[indices] = torch.maximum(
                self.beta[indices],
                self.alpha[indices, edge_nodes[indices, 0], :]
                @ self.alpha[indices, :, edge_nodes[indices, 1]],
            )
            self.beta[indices] = torch.maximum(
                self.beta[indices],
                self.alpha[indices, edge_nodes[indices, 1], :]
                @ self.alpha[indices, :, edge_nodes[indices, 0]],
            )
            return

        # If there is an almost directed path from a to x and a directed path from y to b, adding a
        # directed edge from x to y will create an almost directed path from a to b
        self.beta[indices] = torch.maximum(
            self.beta[indices],
            self.beta[indices, edge_nodes[indices, 0], :]
            @ self.alpha[indices, :, edge_nodes[indices, 1]],
            self.alpha[indices, edge_nodes[indices, 0], :]
            @ self.beta[indices, :, edge_nodes[indices, 1]],
        )

    def mask_added_edges(self, edge_nodes_indices: torch.Tensor):
        for edge_type in range(self.num_edge_types):
            self.forward_mask[
                self.batch_ids, self.max_num_edges * edge_type + edge_nodes_indices
            ] = 0.0

    def mask_actions_leading_to_cycles(
        self,
        left_causes_right_indices: torch.Tensor,
        right_causes_left_indices: torch.Tensor,
        latent_confounding_indices: torch.Tensor,
    ):
        self.forward_mask[
            left_causes_right_indices, self.left_causes_right_action_indices
        ] = (
            self.forward_mask[
                left_causes_right_indices, self.left_causes_right_action_indices
            ]
            * (1 - self.alpha.transpose(1, 2))
            * (1 - self.beta.transpose(1, 2))
        )
        self.forward_mask[
            right_causes_left_indices, self.right_causes_left_action_indices
        ] = (
            self.forward_mask[
                right_causes_left_indices, self.right_causes_left_action_indices
            ]
            * (1 - self.alpha)
            * (1 - self.beta)
        )
        self.forward_mask[
            latent_confounding_indices, self.latent_confounding_action_indices
        ] = (
            self.forward_mask[
                latent_confounding_indices, self.latent_confounding_action_indices
            ]
            * (1 - self.alpha.transpose(1, 2))
            * (1 - self.alpha)
        )

    def remove_backward_mask_from_added_edges(
        self, indices: torch.Tensor, edge_nodes_indices: torch.Tensor
    ):
        self.backward_mask[indices, edge_nodes_indices[indices]] = 1.0

    def apply(self, indices: torch.Tensor):
        can_add_edges = self.forward_mask.sum(dim=1) > 0

        edge_nodes_indices = indices % self.max_num_edges
        edge_nodes = self.edges[edge_nodes_indices]

        # Check the edge type based on the index
        edge_types = indices.numpy() // self.max_num_edges
        edge_types = [EdgeType(et) for et in edge_types]

        left_causes_right_indices = np.argwhere(
            (edge_types == EdgeType.LEFT_CAUSES_RIGHT) & can_add_edges
        )
        right_causes_left_indices = np.argwhere(
            (edge_types == EdgeType.RIGHT_CAUSES_LEFT) & can_add_edges
        )

        latent_confounding_indices = np.argwhere(
            (edge_types == EdgeType.LATENT_CONFOUNDING) & can_add_edges
        )

        self.add_to_edge_list(left_causes_right_indices, edge_nodes)
        self.add_to_edge_list(right_causes_left_indices, edge_nodes[..., [1, 0]])
        self.add_to_edge_list(latent_confounding_indices, edge_nodes)
        self.add_to_edge_list(latent_confounding_indices, edge_nodes[..., [1, 0]])

        self.mask_added_edges(edge_nodes_indices[can_add_edges])

        # Recall: self.alpha[batch, x, y] indicates whether there is a directed path from x to y
        # Recall: self.beta[batch, x, y] indicates whether there is an almost directed path from x to y

        # Update alpha
        self.update_directed_paths(left_causes_right_indices, edge_nodes)
        self.update_directed_paths(right_causes_left_indices, edge_nodes[..., [1, 0]])

        # Update beta
        self.update_almost_directed_paths(left_causes_right_indices, edge_nodes)
        self.update_almost_directed_paths(
            right_causes_left_indices, edge_nodes[..., [1, 0]]
        )
        self.update_almost_directed_paths(
            latent_confounding_indices, edge_nodes, bidirected=True
        )

        # Mask out all edges that may lead to either directed or almost directed cycles
        self.mask_actions_leading_to_cycles(
            left_causes_right_indices,
            right_causes_left_indices,
            latent_confounding_indices,
        )

        # Remove mask from all backward actions related to the added edges
        self.remove_backward_mask_from_added_edges(
            left_causes_right_indices, edge_nodes_indices
        )
        self.remove_backward_mask_from_added_edges(
            right_causes_left_indices, edge_nodes_indices
        )
        self.remove_backward_mask_from_added_edges(
            latent_confounding_indices, edge_nodes_indices
        )

    def backward(self, indices):
        del indices
        raise NotImplementedError("Backward is not implemented for MAG_HITL")

    def f2b(self, actions):
        # forward_action = num_edges * edge_type + edge_index
        # backward_action = edge_index
        return actions % self.max_num_edges
