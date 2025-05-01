import torch
import numpy as np
import itertools

from typing import Optional

from agfn.commons import EdgeType
from agfn.gym import MAG_HITL, LogReward
from agfn.gflownet import GFlowNet


def posterior_from_responses(
    responses: list,
    queries: list,
    num_node_pairs: int,
    pi: float = 0.95,
):
    indices = torch.arange(4).unsqueeze(0).repeat(num_node_pairs, 1)
    responses = torch.tensor(responses).unsqueeze(1)

    prior = torch.ones((num_node_pairs, 4)) / 4
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
    num_nodes: int,
    mask_queried_edges: bool = True,
    pi: float = 0.95,
    initial_state: Optional[MAG_HITL] = None,
    num_iterations: int = 25,
):
    if initial_state is None:
        initial_state = MAG_HITL(num_nodes, batch_size=128, log_reward=log_reward)

    num_node_pairs = (num_nodes * (num_nodes - 1)) // 2

    edge_indices = np.arange(num_node_pairs)
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

            posterior_dist = posterior_from_responses(
                responses_cp, queries_cp, num_node_pairs, pi
            )

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


def log_score_from_responses(
    states: MAG_HITL, queries: list, responses: list
) -> torch.Tensor:
    # log_reward + log_posterior (mean-field approximation)
    posterior_dist = posterior_from_responses(
        queries=queries, responses=responses, num_node_pairs=states.max_num_edges
    )
    log_scores = states.log_reward()

    adj = states.adjacency_matrix

    # For each edge, find the corresponding posterior distribution
    for i, j in itertools.combinations(range(states.num_nodes), 2):
        # Find the edge type for each graph in the batch
        left_causes_right = (adj[:, i, j] == 1) & (adj[:, j, i] == 0)
        right_causes_left = (adj[:, i, j] == 0) & (adj[:, j, i] == 1)
        latent_confounding = (adj[:, i, j] == 1) & (adj[:, j, i] == 1)
        no_relation = (adj[:, i, j] == 0) & (adj[:, j, i] == 0)

        # Edges are lexicographically ordered. Find the index of (i, j).
        idx = i * (states.num_nodes - 2) + j - 1 - (i**2 - i) // 2

        # Update the score according to the (expert-induced) posterior distribution
        log_scores[left_causes_right] += torch.log(
            posterior_dist[idx, EdgeType.LEFT_CAUSES_RIGHT.value]
        )

        log_scores[right_causes_left] += torch.log(
            posterior_dist[idx, EdgeType.RIGHT_CAUSES_LEFT.value]
        )

        log_scores[latent_confounding] += torch.log(
            posterior_dist[idx, EdgeType.LATENT_CONFOUNDING.value]
        )

        log_scores[no_relation] += torch.log(
            posterior_dist[idx, EdgeType.NO_RELATIONSHIP.value]
        )

    return log_scores
