import torch
import numpy as np
import itertools

from typing import Optional
from enum import Enum

from agfn.commons import EdgeType
from agfn.gym import MAG_HITL, LogReward
from agfn.modules.gflownet import GFlowNet


class ElicitationStrategy(Enum):
    REWARD_CROSS_ENTROPY = "reward_cross_entropy"
    ENTROPY = "entropy"
    RANDOM = "random"


def posterior_from_responses(
    responses: list, queries: list, num_node_pairs: int, pis: list
):
    indices = (
        torch.arange(4).unsqueeze(0).repeat(num_node_pairs, 1)
    )  # (num_node_pairs, 4)
    queries = torch.tensor(queries, dtype=torch.long)  # (num_node_pairs,)
    responses = torch.tensor(responses).unsqueeze(1)  # (num_node_pairs, 1)
    pis = torch.tensor(pis, dtype=torch.float).unsqueeze(1)  # (num_node_pairs, 1)

    log_prior = torch.ones((num_node_pairs, 4)) / 4
    log_posterior = log_prior.clone()

    log_likelihood = torch.log(
        pis * (indices[queries] == responses)
        + (1 - pis) * (indices[queries] != responses) * (1 / 3)
    )

    log_posterior.index_add_(0, queries, log_likelihood)

    posterior = torch.exp(log_posterior)
    posterior = posterior / posterior.sum(dim=1, keepdim=True)

    return posterior


def find_next_query_from_reward_cross_entropy(
    gfn: GFlowNet,
    log_reward: LogReward,
    queries: list,
    responses: list,
    pis: list,
    num_nodes: int,
    mask_queried_edges: bool = True,
    initial_state: Optional[MAG_HITL] = None,
    num_iterations: int = 25,
    prior_pi: float = 0.95,
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
        pis_cp = pis.copy()

        expected_entropy = 0.0
        # First, compute the human-induced posterior distribution
        for response in EdgeType:
            queries_cp.append(edge_idx)
            responses_cp.append(response.value)
            pis_cp.append(prior_pi)

            posterior_dist = posterior_from_responses(
                responses_cp, queries_cp, num_node_pairs, pis_cp
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


def find_next_query_from_entropy(
    queries: list, responses: list, pis: list, num_nodes: int
) -> int:
    posterior = posterior_from_responses(
        responses=responses,
        queries=queries,
        num_node_pairs=num_nodes * (num_nodes - 1) // 2,
        pis=pis,
    )

    entropy = -(posterior * torch.log(posterior)).sum(dim=1)
    return entropy.argmax().item()


def find_next_query_randomly(num_nodes: int) -> int:
    num_edges = num_nodes * (num_nodes - 1) // 2
    return np.random.choice(num_edges)


def find_next_query(
    strategy: ElicitationStrategy,
    queries: list,
    responses: list,
    num_nodes: int,
    pis: list = None,
    gfn: GFlowNet = None,
    log_reward: LogReward = None,
    mask_queried_edges: bool = True,
    initial_state: Optional[MAG_HITL] = None,
    num_iterations: int = 25,
    prior_pi: float = 0.95,
):
    match strategy:
        case ElicitationStrategy.REWARD_CROSS_ENTROPY:
            return find_next_query_from_reward_cross_entropy(
                gfn,
                log_reward,
                queries,
                responses,
                pis,
                num_nodes,
                mask_queried_edges,
                initial_state,
                num_iterations,
                prior_pi,
            )
        case ElicitationStrategy.ENTROPY:
            return find_next_query_from_entropy(queries, responses, pis, num_nodes)
        case ElicitationStrategy.RANDOM:
            return find_next_query_randomly(num_nodes)


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
    states: MAG_HITL,
    queries: list,
    responses: list,
    log_reward_coef: float = 0.5,
    pis: list = [],
) -> torch.Tensor:
    # log_reward + log_posterior (mean-field approximation)
    log_posterior_dist = (1 - log_reward_coef) * posterior_from_responses(
        queries=queries,
        responses=responses,
        num_node_pairs=states.max_num_edges,
        pis=pis,
    ).log()
    log_scores = log_reward_coef * states.log_reward()

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
        log_scores[left_causes_right] += log_posterior_dist[
            idx, EdgeType.LEFT_CAUSES_RIGHT.value
        ]

        log_scores[right_causes_left] += log_posterior_dist[
            idx, EdgeType.RIGHT_CAUSES_LEFT.value
        ]

        log_scores[latent_confounding] += log_posterior_dist[
            idx, EdgeType.LATENT_CONFOUNDING.value
        ]

        log_scores[no_relation] += log_posterior_dist[
            idx, EdgeType.NO_RELATIONSHIP.value
        ]

    return log_scores


# import pandas as pd
# from agfn.commons import EdgeType


# def str_to_list(string):
#     s = string.split()
#     s[0] = s[0][1:]
#     if len(s) == 5:
#         s = s[:-1]
#     else:
#         s[len(s) - 1] = s[len(s) - 1][:-1]
#     l = []

#     for i in range(len(s)):
#         l.append(float(s[i]))
#     return l


# def get_response_from_LLM_oracle(
#     query: int, responses: pd.DataFrame
# ) -> tuple[EdgeType, float]:  # returns edge type and its uncertainty
#     edge_probs = np.array(str_to_list(responses.iloc[query, 2].strip()))
#     pi = edge_probs.max()

#     if pi == 1:
#         pi = 0.95

#     edge = edge_probs.argmax()

#     if edge == 0:
#         return EdgeType.NO_RELATIONSHIP, pi
#     if edge == 1:
#         return EdgeType.LEFT_CAUSES_RIGHT, pi
#     if edge == 2:
#         return EdgeType.RIGHT_CAUSES_LEFT, pi
#     if edge == 3:
#         return EdgeType.LATENT_CONFOUNDING, pi


# def posterior_from_responses_pi_list(
#     responses: list,
#     queries: list,
#     num_node_pairs: int,
#     pis: list,
# ):
#     indices = torch.arange(4).unsqueeze(0).repeat(num_node_pairs, 1)
#     queries = torch.tensor(queries, dtype=torch.long)
#     pis = torch.tensor(pis, dtype=torch.long).unsqueeze(0).T

#     responses = torch.tensor(responses).unsqueeze(1)

#     log_prior = torch.ones((num_node_pairs, 4)) / 4
#     log_posterior = log_prior.clone()

#     log_likelihood = torch.log(
#         pis * (indices[queries] == responses)
#         + (1 - pis) * (indices[queries] != responses) * (1 / 3),
#     )

#     log_posterior.index_add_(0, queries, log_likelihood)

#     posterior = torch.exp(log_posterior)
#     posterior = posterior / posterior.sum(dim=1, keepdim=True)

#     return posterior


# def find_next_query_from_entropy_pi_list(
#     queries: list,
#     responses: list,
#     num_nodes: int,
#     pis: list,
# ) -> int:
#     posterior = posterior_from_responses_pi_list(
#         responses=responses,
#         queries=queries,
#         num_node_pairs=num_nodes * (num_nodes - 1) // 2,
#         pis=pis,
#     )

#     entropy = -(posterior * torch.log(posterior)).sum(dim=1)
#     return entropy.argmax().item()
