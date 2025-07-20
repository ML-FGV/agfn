import torch
import numpy as np
import itertools
import tqdm

from typing import Optional
from enum import Enum

from agfn.commons import EdgeType
from agfn.gym import MAG_HITL, LogReward
from agfn.modules.gflownet import GFlowNet
from agfn.modules.expert import Expert


class ElicitationStrategy(Enum):
    REWARD_CROSS_ENTROPY = "reward_cross_entropy"
    ENTROPY = "entropy"
    RANDOM = "random"


class AGFN:
    def __init__(
        self,
        gflownet: GFlowNet,
        expert: Expert,
        strategy: ElicitationStrategy,
        log_reward: LogReward,
        prior_pi: float = 0.95,
    ):
        self.gflownet = gflownet
        self.expert = expert
        self.strategy = strategy
        self.log_reward = log_reward

        self.prior_pi = prior_pi

        self.queries = []
        self.responses = []
        self.pis = []

    def train(
        self,
        num_epochs: int = 125,
        batch_size: int = 256,
        lr: float = 1e-3,
        lr_logz: float = 1e-1,
        progress_bar: bool = True,
    ) -> list[float]:
        self.gflownet.train()
        opt = torch.optim.Adam(
            [
                {"params": self.gflownet.pf.parameters(), "lr": lr},
                {"params": self.gflownet.log_z, "lr": lr_logz},
            ]
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

        log_z_values = []

        pbar = tqdm.tqdm(range(num_epochs), disable=not progress_bar)

        for _ in pbar:
            initial_state = MAG_HITL(self.expert.num_nodes, batch_size, self.log_reward)
            opt.zero_grad()
            loss = self.gflownet(initial_state)
            loss.backward()
            opt.step()
            sch.step()

            log_z_values.append(self.gflownet.log_z.detach().cpu().item())

            pbar.set_postfix(loss=loss.item())

        return log_z_values

    def sample(
        self,
        num_samples: int = 100,
        batch_size: int = 256,
        progress_bar: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        graphs = []
        log_rewards = []

        for _ in tqdm.trange(num_samples, disable=not progress_bar):
            initial_state = MAG_HITL(self.expert.num_nodes, batch_size, self.log_reward)
            state = self.gflownet.sample(initial_state)
            graphs.append(state.adjacency_matrix.cpu().numpy())
            log_rewards.append(state.log_reward().cpu().numpy())

        graphs = np.concatenate(graphs, axis=0)
        log_rewards = np.concatenate(log_rewards, axis=0)

        return graphs, log_rewards

    def posterior_from_responses(self):
        indices = torch.arange(4).unsqueeze(0).repeat(self.expert.num_node_pairs, 1)
        queries = torch.tensor(self.queries, dtype=torch.long)
        responses = torch.tensor(self.responses).unsqueeze(1)
        pis = torch.tensor(self.pis, dtype=torch.float).unsqueeze(1)

        log_prior = torch.ones((self.expert.num_node_pairs, 4)) / 4
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
        self,
        mask_queried_edges: bool = True,
        initial_state: Optional[MAG_HITL] = None,
        num_iterations: int = 25,
    ) -> int:
        if initial_state is None:
            initial_state = MAG_HITL(
                self.expert.num_nodes,
                batch_size=128,
                log_reward=self.log_reward,
            )

        edge_indices = np.arange(self.expert.num_node_pairs)
        if mask_queried_edges:
            edge_indices = np.setdiff1d(edge_indices, self.queries)

        largest_entropy = -np.inf
        best_edge_to_query = None

        for edge_idx in edge_indices:
            queries_cp = self.queries.copy()
            responses_cp = self.responses.copy()
            pis_cp = self.pis.copy()

            expected_entropy = 0.0
            # First, compute the human-induced posterior distribution
            for response in EdgeType:
                queries_cp.append(edge_idx)
                responses_cp.append(response.value)
                pis_cp.append(self.prior_pi)

                posterior_dist = self.posterior_from_responses()

                self.gflownet.set_posterior(posterior_dist)

                # Compute the cross-entropy of the resulting distribution
                # with the target distribution
                expected_entropy += self.gflownet.cross_entropy_with_reward(
                    initial_state=initial_state, num_iterations=num_iterations
                )

            expected_entropy = expected_entropy / len(EdgeType)

            if expected_entropy > largest_entropy:
                largest_entropy = expected_entropy
                best_edge_to_query = edge_idx

        # Returns the index of the highest-uncertainty edge to query
        return best_edge_to_query

    def find_next_query_from_entropy(self) -> int:
        posterior = self.posterior_from_responses()

        entropy = -(posterior * torch.log(posterior)).sum(dim=1)
        return entropy.argmax().item()

    def find_next_query_randomly(self) -> int:
        num_edges = self.expert.num_nodes * (self.expert.num_nodes - 1) // 2
        return np.random.choice(num_edges)

    def find_next_query(
        self,
        mask_queried_edges: bool = True,
        initial_state: Optional[MAG_HITL] = None,
        num_iterations: int = 25,
        prior_pi: float = 0.95,
    ):
        match self.strategy:
            case ElicitationStrategy.REWARD_CROSS_ENTROPY:
                return self.find_next_query_from_reward_cross_entropy(
                    mask_queried_edges, initial_state, num_iterations, prior_pi
                )
            case ElicitationStrategy.ENTROPY:
                return self.find_next_query_from_entropy()
            case ElicitationStrategy.RANDOM:
                return self.find_next_query_randomly()

    def log_score_from_responses(
        self, states: MAG_HITL, log_reward_coef: float = 0.5
    ) -> torch.Tensor:
        # log_reward + log_posterior (mean-field approximation)
        log_posterior_dist = (
            1 - log_reward_coef
        ) * self.posterior_from_responses().log()
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
