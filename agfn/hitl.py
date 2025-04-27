import torch
import numpy as np
from typing import Optional

from agfn.commons import EdgeType
from agfn.gym.gym import MAG, LogReward
from agfn.gflownet import GFlowNet


def posterior_update(
    responses: list,
    queries: list,
    num_edges: int,
    pi: float = 0.95,
):
    indices = torch.arange(4).unsqueeze(0).repeat(num_edges, 1)
    responses = torch.tensor(responses).unsqueeze(1)

    prior = torch.ones((num_edges, 4)) / 4
    posterior = prior.clone()

    posterior[queries] = prior[queries] * pi * (indices[queries] == responses) + (
        1 - pi
    ) * (indices[queries] != responses) * (1 / 3)

    posterior = posterior / posterior.sum(dim=1, keepdim=True)

    return posterior


def find_next_query(
    gfn: GFlowNet,
    log_reward: LogReward,
    queries: list,
    responses: list,
    mask_queried_edges: bool = True,
    pi: float = 0.95,
    initial_state: Optional[MAG] = None,
    num_iterations: int = 25,
):
    if initial_state is None:
        num_nodes = log_reward.dataset.shape[1]
        initial_state = MAG(num_nodes, batch_size=128, log_reward=log_reward)

    num_edges = (num_nodes * (num_nodes - 1)) // 2

    edge_indices = np.arange(num_edges)
    if mask_queried_edges:
        edge_indices = np.setdiff1d(edge_indices, queries)

    largest_entropy = -np.inf
    best_edge_to_query = None

    for edge_idx in edge_indices:
        queries_cp = queries.copy()
        responses_cp = responses.copy()

        expected_entropy = 0.0
        # First, compute the human-induced posterior distribution
        for response in EdgeType:
            queries_cp.append(edge_idx)
            responses_cp.append(response.value)

            posterior_dist = posterior_update(responses_cp, queries_cp, num_edges, pi)

            gfn.set_posterior(posterior_dist)

            # Compute the cross-entropy of the resulting distribution
            # with the target distribution
            expected_entropy += gfn.cross_entropy_with_reward(
                initial_state=initial_state, num_iterations=num_iterations
            )

        expected_entropy = expected_entropy / len(EdgeType)

        if expected_entropy > largest_entropy:
            largest_entropy = expected_entropy
            best_edge_to_query = edge_idx

    # Returns the index of the highest-uncertainty edge to query
    return best_edge_to_query


def get_response_from_oracle(query: int, adj: torch.Tensor) -> EdgeType:
    i, j = torch.triu_indices(adj.shape[0], adj.shape[1], offset=1)
    i, j = i[query].item(), j[query].item()

    if adj[i, j] == 1 and adj[j, i] == 1:
        return EdgeType.LATENT_CONFOUNDING
    elif adj[i, j] == 1:
        return EdgeType.LEFT_CAUSES_RIGHT
    elif adj[j, i] == 1:
        return EdgeType.RIGHT_CAUSES_LEFT
    else:
        return EdgeType.NO_RELATIONSHIP
