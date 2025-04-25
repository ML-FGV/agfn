import torch
import numpy as np

from typing import Callable

from agfn.gym.gym import MAG, LogReward
from agfn.gflownet import GFlowNet


class EdgeType:
    NO_RELATIONSHIP = 0
    LEFT_CAUSES_RIGHT = 1
    RIGHT_CAUSES_LEFT = 2
    LATENT_CONFOUNDING = 3


def posterior_update(
    responses: list,
    queries: list,
    num_edges: int,
    pi: float = 0.95,
):
    prior = torch.ones((num_edges, 4)) / 4
    posterior = prior.clone()
    posterior[queries, responses] = prior * pi
    posterior = posterior / posterior.sum(dim=1, keepdim=True)
    return posterior


def find_next_query(
    gfn: GFlowNet,
    log_reward: LogReward,
    queries: list,
    responses: list,
    mask_queried_edges: bool = True,
    oracle: Callable = lambda x: EdgeType.NO_RELATIONSHIP,
    pi: float = 0.95,
    initial_state: MAG = None,
    num_iterations: int = None,
):
    num_nodes = log_reward.dataset.shape[1]
    num_edges = (num_nodes * (num_nodes - 1)) // 2

    edge_indices = np.arange(num_edges)
    if mask_queried_edges:
        edge_indices = np.setdiff1d(edge_indices, queries)

    queries_cp = queries.copy()
    responses_cp = responses.copy()

    largest_entropy = -np.inf
    best_edge_to_query = None

    for edge_idx in edge_indices:
        # Compute the distributions' entropy when the edge is added

        # First, compute the human-induced posterior distribution
        response = oracle(edge_idx)
        responses_cp.append(response)

        posterior_dist = posterior_update(responses_cp, queries_cp, pi)

        gfn.set_posterior(posterior_dist)

        # Compute the cross-entropy of the resulting distribution
        # with the target distribution
        entropy = gfn.cross_entropy_with_reward(
            initial_state=initial_state,
            num_iterations=num_iterations,
        )

        if entropy > largest_entropy:
            largest_entropy = entropy
            best_edge_to_query = edge_idx

    # Returns the index of the highest-uncertainty edge to query
    return best_edge_to_query
