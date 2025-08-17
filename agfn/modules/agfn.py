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
    LARGEST_CROSS_ENTROPY = "largest_cross_entropy"
    LARGEST_ENTROPY = "largest_entropy"
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

    def create_initial_state(self, batch_size: int = 256) -> MAG_HITL:
        return MAG_HITL(self.log_reward.num_nodes, batch_size, self.log_reward)

    def train(
        self,
        num_epochs: int = 125,
        batch_size: int = 256,
        lr: float = 1e-3,
        lr_logz: float = 1e-1,
        progress_bar: bool = True,
    ) -> list[float]:
        self.train_mode()
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
            initial_state = self.create_initial_state(batch_size)
            opt.zero_grad()
            loss = self.gflownet(initial_state)
            loss.backward()
            opt.step()
            sch.step()

            log_z_values.append(self.gflownet.log_z.detach().cpu().item())

            pbar.set_postfix(loss=loss.item())

        return log_z_values

    def eval_mode(self):
        self.gflownet.eval()

    def train_mode(self):
        self.gflownet.train()

    def sample(
        self,
        num_samples: int = 100,
        batch_size: int = 256,
        progress_bar: bool = True,
        log_reward_coef: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        graphs = []
        log_scores = []

        for _ in tqdm.trange(num_samples, disable=not progress_bar):
            initial_state = self.create_initial_state(batch_size)
            state = self.gflownet.sample(initial_state)
            log_score_samples = (
                self.log_score_from_responses(state, log_reward_coef).cpu().numpy()
            )

            graphs.append(state.adjacency_matrix.cpu().numpy())
            log_scores.append(log_score_samples)

        graphs = np.concatenate(graphs, axis=0)
        log_scores = np.concatenate(log_scores, axis=0)

        return graphs, log_scores

    def posterior_from_responses(self):
        num_node_pairs = (
            self.log_reward.num_nodes * (self.log_reward.num_nodes - 1)
        ) // 2

        indices = torch.arange(4).unsqueeze(0).repeat(num_node_pairs, 1)
        queries = torch.tensor(self.queries, dtype=torch.long)
        responses = torch.tensor(self.responses).unsqueeze(1)
        pis = torch.tensor(self.pis, dtype=torch.float).unsqueeze(1)

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
        self,
        mask_queried_edges: bool = True,
        initial_state: Optional[MAG_HITL] = None,
        num_iterations: int = 25,
    ) -> int:
        if initial_state is None:
            initial_state = self.create_initial_state(batch_size=128)

        num_node_pairs = (
            self.log_reward.num_nodes * (self.log_reward.num_nodes - 1) // 2
        )
        edge_indices = np.arange(num_node_pairs)
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
                responses_cp.append(response.to_int())
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

        entropy = -(posterior * torch.log(posterior + 1e-20)).sum(dim=1)
        return entropy.argmax().item()

    def find_next_query_randomly(self) -> int:
        num_edges = self.log_reward.num_nodes * (self.log_reward.num_nodes - 1) // 2
        return np.random.choice(num_edges)

    def find_next_query(
        self,
        mask_queried_edges: bool = True,
        initial_state: Optional[MAG_HITL] = None,
        num_iterations: int = 25,
        prior_pi: float = 0.95,
    ):
        match self.strategy:
            case ElicitationStrategy.LARGEST_CROSS_ENTROPY:
                return self.find_next_query_from_reward_cross_entropy(
                    mask_queried_edges, initial_state, num_iterations, prior_pi
                )
            case ElicitationStrategy.LARGEST_ENTROPY:
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
                idx, EdgeType.LEFT_CAUSES_RIGHT.to_int()
            ]

            log_scores[right_causes_left] += log_posterior_dist[
                idx, EdgeType.RIGHT_CAUSES_LEFT.to_int()
            ]

            log_scores[latent_confounding] += log_posterior_dist[
                idx, EdgeType.LATENT_CONFOUNDING.to_int()
            ]

            log_scores[no_relation] += log_posterior_dist[
                idx, EdgeType.NO_RELATIONSHIP.to_int()
            ]

        return log_scores

    def temperature_warmup(self, batch_size: int = 128, iterations: int = 8):
        self.log_reward.warmup(
            self.gflownet, self.create_initial_state(batch_size), iterations
        )

    def reset_queries(self):
        self.queries = []
        self.responses = []
        self.pis = []

    def feedback_loop(self, num_feedbacks: int = None, progress_bar: bool = True):
        initial_state = self.create_initial_state(batch_size=256)

        if num_feedbacks is None:
            num_feedbacks = initial_state.max_num_edges

        for _ in tqdm.trange(num_feedbacks, disable=not progress_bar):
            query = self.find_next_query(
                mask_queried_edges=True,
                initial_state=initial_state,
                num_iterations=25,
                prior_pi=1 - 1e-3,
            )

            response = self.expert(query)

            self.queries.append(query)
            self.responses.append(response.relationship.to_int())
            self.pis.append(
                response.confidence
                if response.confidence is not None
                else self.prior_pi
            )

            self.gflownet.set_posterior(self.posterior_from_responses())
