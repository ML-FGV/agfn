import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import itertools
import pandas as pd
import pathlib
import copy

from agfn.gym.meta import Environment
from agfn.ricf import reward_bic
from agfn.hitl import adj_to_edge_marks
from agfn.rpy2_utils import suppress_output, tensor3d_to_rmatrix

from rpy2.robjects.packages import STAP
from rpy2.robjects import numpy2ri, pandas2ri, NULL

numpy2ri.activate()
pandas2ri.activate()

HERE = pathlib.Path(__file__).parent


class LogRewardMeta(nn.Module):
    def __init__(self, device, temperature):
        super(LogRewardMeta, self).__init__()
        self.device = device
        self.temperature = temperature
        self.std = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=False)

        # self.temperature = 1.

    @torch.no_grad()
    def warmup(self, gflownet, initial_state, num_batches):
        if self.mu.data != 0.0:
            return
        log_rewards = list()
        for _ in range(num_batches):
            env = gflownet.sample(copy.deepcopy(initial_state))
            log_rewards.append(env.log_reward())
        log_rewards = torch.hstack(log_rewards) ** self.temperature
        is_nan = torch.isnan(log_rewards)
        self.mu.data = log_rewards[~is_nan].max()
        self.std.data = log_rewards[~is_nan].std()


class LogReward(LogRewardMeta):
    def __init__(
        self,
        dataset=None,
        gamma=0.5,
        num_samples=None,
        num_nodes=None,
        temperature=1.0,
        device="cpu",
    ):
        super(LogReward, self).__init__(device=device, temperature=temperature)
        if dataset is not None:
            self.dataset = nn.Parameter(dataset.to(device), requires_grad=False)
        else:
            self.dataset = nn.Parameter(
                torch.zeros((num_samples, num_nodes), device=device),
                requires_grad=False,
            )
        self.gamma = nn.Parameter(
            torch.tensor(gamma, device=device), requires_grad=False
        )
        # We assume unit variance for the (independent) error terms
        # The model is a structural Gaussian model:
        # X_{t} \sim \beta_{t}^{T} X_{-t} + \eps_{t},
        # with an appropriate sparsity for \beta_{t}
        pass

    @torch.no_grad()
    def forward(self, batch_state):
        return (reward_bic(batch_state, self.dataset, gamma=0.5) - self.mu) / (
            self.temperature * self.std
        )


class LogRewardSimple(nn.Module):
    def __init__(self, device):
        super(LogRewardSimple, self).__init__()
        self.device = device

    def forward(self, batch_state, *args):
        D, B = batch_state.directed_amat, batch_state.bidirected_amat
        return 0.3 * D.sum(dim=(1, 2)) + 0.15 * B.sum(dim=(1, 2))


class LogRewardSlightlyComplex(nn.Module):
    def __init__(self, num_nodes, seed=42, device="cpu"):
        super(LogRewardSlightlyComplex, self).__init__()
        self.num_nodes = num_nodes
        self.device = device
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self.d_w = torch.randn(
            (self.num_nodes, self.num_nodes), device=self.device, generator=g
        )
        g.manual_seed(2 * seed)
        self.b_w = torch.randn(
            (self.num_nodes, self.num_nodes), device=self.device, generator=g
        )

    def forward(self, batch_state, *args):
        D, B = batch_state.directed_amat, batch_state.bidirected_amat
        return (self.d_w * D).sum(dim=(1, 2)) + (self.b_w * B).sum(dim=(1, 2))


class LogRewardR(LogRewardMeta):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        temperature: float = 1.0,
        group_var_idx: int | None = None,
        device: str = "cpu",
        path: str = f"{HERE}/../R/computeBIC.R",
    ):
        super(LogRewardR, self).__init__(device=device, temperature=temperature)
        self.dataframe = dataframe
        self.rpath = path
        if group_var_idx is not None:
            self.group_var = self.dataframe.columns[group_var_idx]
        else:
            self.group_var = NULL

        with open(self.rpath) as f:
            self.compute_bic_module = STAP(f.read(), "compute_bic_module")

        self.total_evaluations = 0
        self.nan_evaluations = 0

    @torch.no_grad()
    def forward(self, batch_state, *args):
        del args
        with suppress_output():
            amatr = tensor3d_to_rmatrix(batch_state.adjacency_matrix)
            bics = self.compute_bic_module.computeBIC(
                amatr, self.dataframe, self.group_var
            )
        bics = torch.tensor(bics, device=self.device)
        self.total_evaluations += batch_state.batch_size
        self.nan_evaluations += torch.isnan(bics).sum().cpu().item()
        bics = torch.nan_to_num(bics, nan=-self.mu)
        return (-bics - self.mu) / (self.temperature * self.std)


class MAG(Environment):
    DIR_EDGES = 1
    BIDIR_EDGES = 2

    def __init__(
        self, num_nodes, batch_size, log_reward, max_degree=None, device="cpu"
    ):
        self.max_num_edges = (num_nodes * (num_nodes - 1)) // 2
        super(MAG, self).__init__(
            batch_size, self.max_num_edges + 1, log_reward, device
        )
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        # Directed edges
        self.actions = (
            torch.triu_indices(self.num_nodes, self.num_nodes, offset=1)
            .to(self.device)
            .t()
        )
        self.actions = torch.vstack(
            [
                self.actions,  # Directed
                self.actions[:, [1, 0]],  # Directed
                self.actions,
            ]
        )  # Bidirected
        self.backward_actions = (
            torch.triu_indices(self.num_nodes, self.num_nodes, offset=1)
            .to(self.device)
            .t()
        )
        self.forward_mask = torch.ones(
            (self.batch_size, self.actions.shape[0] + 1), device=self.device
        )
        self.backward_mask = torch.zeros(
            (self.batch_size, self.max_num_edges + 1), device=self.device
        )

        self.edge_list = torch.zeros((0, 4), dtype=torch.long, device=self.device)

        self.indices = (
            torch.zeros(
                (self.batch_size, self.num_nodes, self.num_nodes),
                dtype=torch.long,
                device=self.device,
            )
            - 1
        )
        self.num_edges = torch.zeros(
            (self.batch_size,), dtype=torch.long, device=self.device
        )

        # To generate sparse graphs
        self.degrees = torch.zeros(
            (self.batch_size, self.num_nodes), device=self.device
        )
        self.max_degree = self.num_nodes if max_degree is None else max_degree
        self.edge_degree = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )

        self.alpha = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes),
            device=self.device,
            dtype=bool,
        )
        self.beta = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes),
            device=self.device,
            dtype=bool,
        )

        self.alpha[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = 1.0

    @property
    @torch.no_grad()
    def directed_amat(self):
        amat = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )
        e = self.edge_list
        dir_e = e[:, 3] == self.DIR_EDGES
        amat[e[dir_e, 0], e[dir_e, 1], e[dir_e, 2]] = 1.0
        return amat

    @property
    @torch.no_grad()
    def bidirected_amat(self):
        amat = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes), device=self.device
        )
        e = self.edge_list
        bidir_e = e[:, 3] == self.BIDIR_EDGES
        amat[e[bidir_e, 0], e[bidir_e, 1], e[bidir_e, 2]] = 1.0
        return amat

    @torch.no_grad()
    def log_reward(self):
        return self._log_reward(self)

    @property
    def parents_and_spouses(self):
        return self.directed_amat, self.bidirected_amat

    @property
    @torch.no_grad()
    def adjacency_matrix(self):
        return self.directed_amat + self.bidirected_amat

    @torch.no_grad()
    def floyd_warshall_t(self, transpose=True):
        # The transitive closure of a DAG corresponds to its reachability matrix
        # D, B = self.directed_amat, self.bidirected_amat
        amat = self.directed_amat  # Compute directed paths
        dist = torch.where(amat == 1.0, 1.0, torch.inf)

        n_id = torch.arange(self.num_nodes).to(self.device)
        dist[:, n_id, n_id] = 0.0

        for k in range(self.num_nodes):
            pdist = dist[:, :, k, None] + dist[:, None, k, :]
            mask = dist > pdist
            dist = torch.where(mask, pdist, dist)

        # Return the reachability matrix and the number of bidirected edges within the path
        return ~torch.isinf(dist)

    @torch.no_grad()
    def node_feat(self):
        features = torch.eye(self.num_nodes).expand(
            self.batch_size, self.num_nodes, self.num_nodes
        )
        return features.flatten(end_dim=1).to(self.device)

    @torch.no_grad()
    def edge_index(self):
        edges = self.edge_list
        return (edges[:, 0].view(-1, 1) * self.num_nodes + edges[:, 1:3]).t()

    @torch.no_grad()
    def update_forward_mask(self):
        # Compute the sum S A A' A
        D, B = self.directed_amat, self.bidirected_amat
        D_t = torch.transpose(D, 1, 2)
        D_t[:, torch.arange(self.num_nodes), torch.arange(self.num_nodes)] = 1.0
        D_pow = torch.ones(
            (self.batch_size, self.num_nodes, self.num_nodes, self.num_nodes),
            device=self.device,
        )
        D_pow[:, 0] = torch.eye(
            self.num_nodes, self.num_nodes, device=self.device
        ).expand(self.batch_size, self.num_nodes, self.num_nodes)
        for i in range(1, self.num_nodes):
            D_pow[:, i] = D_pow[:, i - 1] @ D_t
        term = torch.zeros_like(D)
        for i in range(self.num_nodes):
            term += D_pow[:, i] @ B @ D_pow[:, self.num_nodes - i - 1]
        # Check if the path exists in the transpose graph
        path_bid = term > 0.0
        path_dir = D_pow[:, -1]

        # Directed actions
        path_bid_flat = (
            path_bid[:, self.dir_actions[:, 0], self.dir_actions[:, 1]] == 1.0
        )
        path_dir_flat = (
            path_dir[:, self.dir_actions[:, 0], self.dir_actions[:, 1]] == 1.0
        )
        D, B = self.directed_amat, self.bidirected_amat
        D_flat = D[:, self.dir_actions[:, 0], self.dir_actions[:, 1]] == 1.0
        D_t_flat = (
            torch.transpose(D, 1, 2)[:, self.dir_actions[:, 0], self.dir_actions[:, 1]]
            == 1.0
        )
        B_flat = B[:, self.dir_actions[:, 0], self.dir_actions[:, 1]] == 1.0

        self.forward_mask[:, : 2 * self.max_num_edges] = (
            (~path_bid_flat) & (~path_dir_flat) & (~D_flat) & (~D_t_flat) & (~B_flat)
        )

        # Bidirected actions
        path_dir_flat = (
            path_dir[:, self.bidir_actions[:, 0], self.bidir_actions[:, 1]] >= 1.0
        )
        path_dir_flat_t = (
            path_dir[:, self.bidir_actions[:, 1], self.bidir_actions[:, 0]] >= 1.0
        )
        D, B = self.directed_amat, self.bidirected_amat
        D_flat = D[:, self.bidir_actions[:, 0], self.bidir_actions[:, 1]] == 1.0
        D_t_flat = (
            torch.transpose(D, 1, 2)[
                :, self.bidir_actions[:, 0], self.bidir_actions[:, 1]
            ]
            == 1.0
        )
        B_flat = B[:, self.bidir_actions[:, 0], self.bidir_actions[:, 1]] == 1.0
        self.forward_mask[:, 2 * self.max_num_edges : -1] = (
            (~path_dir_flat) & (~path_dir_flat_t) & (~D_flat) & (~D_t_flat) & (~B_flat)
        )

        self.edge_degree = torch.maximum(
            self.degrees.unsqueeze(-1), self.degrees.unsqueeze(1)
        )

        self.forward_mask[:, : 2 * self.max_num_edges] = self.forward_mask[
            :, : 2 * self.max_num_edges
        ] * (
            self.edge_degree[:, self.dir_actions[:, 0], self.dir_actions[:, 1]]
            < self.max_degree
        )
        self.forward_mask[:, 2 * self.max_num_edges : -1] = self.forward_mask[
            :, 2 * self.max_num_edges : -1
        ] * (
            self.edge_degree[:, self.bidir_actions[:, 0], self.bidir_actions[:, 1]]
            < self.max_degree
        )

    @torch.no_grad()
    def update_edge_list(self, batchids, indices, is_bidirected, ord=[0, 1]):
        # Define the actions
        ones = torch.ones_like(self.actions[indices][:, 0]).view(-1, 1)
        actions = torch.hstack(
            [
                batchids.view(-1, 1),
                self.actions[indices][:, ord],
                ones * (self.BIDIR_EDGES if is_bidirected else self.DIR_EDGES),
            ]
        )

        # Update the edges' indices
        self.indices[
            batchids, self.actions[indices, ord[0]], self.actions[indices, ord[1]]
        ] = len(self.edge_list) + torch.arange(len(indices), device=self.device)

        # Update the list of edges
        self.edge_list = torch.vstack([self.edge_list, actions])

    @torch.no_grad()
    def apply(self, indices):
        is_stop_action = indices == self.actions.shape[0]

        indices_nonstop = indices[~is_stop_action]
        batchid_nonstop = self.batch_ids[~is_stop_action]

        is_directed_action = indices_nonstop < 2 * self.max_num_edges
        is_bidirected_action = (1 - is_directed_action.long()).bool()

        self.degrees[batchid_nonstop, self.actions[indices_nonstop, 0]] += 1.0
        self.degrees[batchid_nonstop, self.actions[indices_nonstop, 1]] += 1.0

        self.edge_degree = torch.maximum(
            self.degrees.unsqueeze(1), self.degrees.unsqueeze(-1)
        )

        # Append directed actions
        directed_indices = indices_nonstop[is_directed_action]
        directed_batchid = batchid_nonstop[is_directed_action]
        self.update_edge_list(
            batchids=directed_batchid, indices=directed_indices, is_bidirected=False
        )

        # Append bidirected actions
        bidirected_indices = indices_nonstop[is_bidirected_action]
        bidirected_batchid = batchid_nonstop[is_bidirected_action]
        for ord in [[0, 1], [1, 0]]:
            self.update_edge_list(
                bidirected_batchid, bidirected_indices, is_bidirected=True, ord=ord
            )

        # Update the edge index
        self.num_edges += (~is_stop_action).long()

        # Update the state's conditions
        self.stopped = is_stop_action.long()
        self.is_initial = self.num_edges == 0.0

        # Update indicator for (almost) directed paths when a directed edge is added
        D, B = self.directed_amat, self.bidirected_amat
        D, D_t, B = D == 1.0, torch.transpose(D == 1.0, 1, 2), B == 1.0

        # Mask directed actions (x -> y)
        y, x = self.actions[directed_indices, 0], self.actions[directed_indices, 1]
        cond = (
            self.alpha[directed_batchid, :, x].unsqueeze(-1).float()
            @ self.alpha[directed_batchid, y, :].unsqueeze(1).float()
        )
        self.alpha[directed_batchid] |= D_t[directed_batchid] | (cond > 0)

        cond1 = (
            self.beta[directed_batchid, :, x].unsqueeze(-1).float()
            @ self.alpha[directed_batchid, y, :].unsqueeze(1).float()
        )
        cond2 = (
            self.alpha[directed_batchid, :, x].unsqueeze(-1).float()
            @ self.beta[directed_batchid, y, :].unsqueeze(1).float()
        )
        self.beta[directed_batchid] |= B[directed_batchid] | (cond1 > 0) | (cond2 > 0)

        # Mask bidirected actions
        y, x = self.actions[bidirected_indices, 0], self.actions[bidirected_indices, 1]
        cond1 = (
            self.alpha[bidirected_batchid, :, x].unsqueeze(-1).float()
            @ self.alpha[bidirected_batchid, y, :].unsqueeze(1).float()
        )
        cond2 = (
            self.alpha[bidirected_batchid, :, y].unsqueeze(-1).float()
            @ self.alpha[bidirected_batchid, x, :].unsqueeze(1).float()
        )
        self.beta[bidirected_batchid] |= (
            B[bidirected_batchid] | (cond1 > 0) | (cond2 > 0)
        )

        umask = self.forward_mask.clone()
        # Transitive closure
        in_node_d, out_node_d = self.dir_actions[:, 0], self.dir_actions[:, 1]
        umask[:, : 2 * self.max_num_edges] = (
            (~self.beta[:, in_node_d, out_node_d])
            & (~self.alpha[:, in_node_d, out_node_d])
            & (~D[:, in_node_d, out_node_d])
            & (~B[:, in_node_d, out_node_d])
        )
        in_node_b, out_node_b = self.bidir_actions[:, 0], self.bidir_actions[:, 1]
        umask[:, 2 * self.max_num_edges : -1] = (
            (~self.alpha[:, in_node_b, out_node_b])
            & (~self.alpha[:, out_node_b, in_node_b])
            & (~D[:, in_node_b, out_node_b])
            & (~B[:, in_node_b, out_node_b])
            & (~D[:, out_node_b, in_node_b])
        )

        umask[:, : 2 * self.max_num_edges] *= (
            self.edge_degree[:, in_node_d, out_node_d] < self.max_degree
        )
        umask[:, 2 * self.max_num_edges : -1] *= (
            self.edge_degree[:, in_node_b, out_node_b] < self.max_degree
        )

        umask[self.batch_ids[is_stop_action], :-1] = 0.0
        self.forward_mask = umask.clone()

        # Update the backward mask
        in_node, out_node = self.bidir_actions[:, 0], self.bidir_actions[:, 1]
        umask = self.backward_mask.clone()
        umask = (
            (D[:, in_node, out_node])
            | (B[:, in_node, out_node])
            | (D[:, out_node, in_node])
        )
        umask[self.batch_ids[is_stop_action], :] = 0.0
        self.backward_mask = umask.clone()

    @property
    def dir_actions(self):
        return self.actions[: 2 * self.max_num_edges]

    @property
    def bidir_actions(self):
        return self.actions[2 * self.max_num_edges :]

    def f2b(self, indices):
        # forward to background
        return torch.where(
            indices == self.actions.shape[0],
            self.max_num_edges,
            torch.remainder(indices, self.max_num_edges),
        )

    @torch.no_grad()
    def backward(self, indices):
        # Similar to gym/graphs.py/Graph/backward
        indices = self.f2b(indices)
        is_non_initial = (self.is_initial != 1.0) & (self.stopped != 1.0)
        self.num_edges -= is_non_initial.long()

        indices_non_initial = indices[is_non_initial]
        batch_ids_non_initial = self.batch_ids[is_non_initial]

        actions = self.backward_actions[indices_non_initial]

        self.degrees[batch_ids_non_initial, actions[:, 0]] -= 1
        self.degrees[batch_ids_non_initial, actions[:, 1]] -= 1

        indices_to_remove = self.indices[
            batch_ids_non_initial, actions[:, 0], actions[:, 1]
        ]

        forward_direction = indices_to_remove != -1
        is_bidirected = self.edge_list[indices_to_remove, 3] == self.BIDIR_EDGES

        indices_to_remove = torch.hstack(
            [
                self.indices[
                    batch_ids_non_initial[forward_direction],
                    actions[forward_direction, 0],
                    actions[forward_direction, 1],
                ],
                self.indices[
                    batch_ids_non_initial[~forward_direction],
                    actions[~forward_direction, 1],
                    actions[~forward_direction, 0],
                ],
                self.indices[
                    batch_ids_non_initial[forward_direction & is_bidirected],
                    actions[forward_direction & is_bidirected, 1],
                    actions[forward_direction & is_bidirected, 0],
                ],
            ]
        )

        forward_actions = (
            torch.ones((self.batch_size,), dtype=indices.dtype, device=self.device)
            * self.actions.shape[0]
        )
        forward_actions[batch_ids_non_initial[forward_direction]] = indices_non_initial[
            forward_direction
        ]
        forward_actions[batch_ids_non_initial[~forward_direction]] = (
            indices_non_initial[~forward_direction] + self.max_num_edges
        )
        forward_actions[batch_ids_non_initial[forward_direction & is_bidirected]] = (
            indices_non_initial[forward_direction & is_bidirected]
            + 2 * self.max_num_edges
        )

        # assert (self.indices >= -1).all(), self.indices

        mask_to_remove = torch.ones(
            (len(self.edge_list),), dtype=bool, device=self.device
        )
        mask_to_remove[indices_to_remove] = False
        self.edge_list = self.edge_list[mask_to_remove]

        # Update indices
        self.indices[batch_ids_non_initial, actions[:, 0], actions[:, 1]] = -1
        self.indices[
            batch_ids_non_initial[is_bidirected],
            actions[is_bidirected, 1],
            actions[is_bidirected, 0],
        ] = -1
        placed_after_index = self.indices[
            batch_ids_non_initial, ..., None
        ] >= indices_to_remove.view(1, 1, -1)
        placed_after_index = placed_after_index.sum(dim=-1)
        self.indices[batch_ids_non_initial] -= placed_after_index

        # Update forward mask
        self.update_forward_mask()

        # Update backward mask
        D, B = self.directed_amat, self.bidirected_amat
        in_node, out_node = self.bidir_actions[:, 0], self.bidir_actions[:, 1]
        D_flat = D[:, in_node, out_node] == 1.0
        B_flat = B[:, in_node, out_node] == 1.0
        D_t_flat = D[:, out_node, in_node] == 1.0
        self.backward_mask = D_flat | D_t_flat | B_flat

        self.is_initial = self.num_edges == 0.0
        self.stopped[:] = 0.0
        # Return the corresponding forward actions (max_num_edges = stop_action_index)
        return forward_actions

    @torch.no_grad()
    def edge_counts_dir(self):
        amat = self.directed_amat
        actions = self.dir_actions
        amat = amat[:, actions[:, 0], actions[:, 1]]
        return amat, amat.sum(dim=0)

    @torch.no_grad()
    def edge_counts_bidir(self):
        amat = self.bidirected_amat
        amat = amat[:, self.bidir_actions[:, 0], self.bidir_actions[:, 1]]
        return amat, amat.sum(dim=0)

    @torch.no_grad()
    def path_counts(self):
        is_connected = self.floyd_warshall_t(transpose=False)
        is_connected = is_connected[:, self.dir_actions[:, 0], self.dir_actions[:, 1]]
        return is_connected, is_connected.sum(dim=0)

    @torch.no_grad()
    def merge(self, batch_state):
        new_edge_list = batch_state.edge_list.clone()
        new_edge_list[:, 0] += self.batch_size
        self.edge_list = torch.vstack([self.edge_list, new_edge_list])
        self.num_edges = torch.hstack([self.num_edges, batch_state.num_edges])
        self.degrees = torch.vstack([self.degrees, batch_state.degrees])
        super().merge(batch_state)

    def draw(self, idx, labels=None, amat=None):
        if amat is None:
            amat = self.adjacency_matrix[idx].cpu().numpy()
        graph = nx.from_numpy_array(amat, create_using=nx.MultiDiGraph())
        if labels is not None:
            nx.relabel_nodes(
                graph, mapping={i: labels[i] for i in range(len(labels))}, copy=False
            )
        nx.draw(graph, with_labels=True)

    @property
    def unique_input(self):
        return self.adjacency_matrix.flatten(start_dim=1)

    def is_mag(self):
        D, B = self.directed_amat, self.bidirected_amat
        D_exp = torch.matrix_exp(D)
        th = (
            torch.diagonal(D_exp, dim1=-1, dim2=-2).sum(dim=-1)
            - self.num_nodes
            + torch.sum(D_exp * B, dim=(-1, -2))
        )
        return torch.abs(th) < 1e-8

    # Static methods

    @staticmethod
    def sample_from_structured_gaussian(
        dag,
        num_samples,
        seed=42,
        beta_scale=0.5,
        beta_loc=2.0,
        omega_scale=0.1,
        omega_loc=0.1,
        device="cpu",
    ):
        # We randomly generate correlational and structural matrices
        # and sample `num_samples` variables from the ensuing distribution
        parents_matrix, spouses_matrix = (
            dag.directed_amat.squeeze(dim=0),
            dag.bidirected_amat.squeeze(dim=0),
        )
        num_nodes = parents_matrix.shape[0]
        nodes = torch.arange(num_nodes, device=device)

        g = torch.Generator(device=device)
        # Sample a beta
        g.manual_seed(seed)
        beta = (
            beta_scale
            * torch.rand(size=(num_nodes, num_nodes), device=device, generator=g)
            + beta_loc
        )
        beta = beta * parents_matrix

        # Sample `omega` from the correspondent Cholesky's decomposition
        g.manual_seed(2 * seed)
        omega = (
            omega_scale
            * torch.rand(size=(num_nodes, num_nodes), device=device, generator=g)
            + omega_loc
        )
        omega = torch.triu(omega).T @ torch.triu(omega) / num_nodes

        # Apply the structural masks
        g.manual_seed(3 * seed)
        omega = omega * spouses_matrix
        omega[nodes, nodes] = (
            omega_scale * torch.rand(size=(num_nodes,), device=device, generator=g)
            + omega_loc
        )

        # Identify the correspondent covariance matrix
        sigma = torch.linalg.inv(torch.eye(num_nodes, device=device) - beta)
        sigma = sigma.T @ omega @ sigma
        # Sample the variables from the ensuing normal distribution
        beta = beta
        omega = omega
        sigma = sigma
        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(num_nodes, device=device), covariance_matrix=sigma
        )
        samples = dist.sample(sample_shape=(num_samples,))
        return samples

    @staticmethod
    def generate_dag_with_bounded_degree(
        num_nodes,
        max_degree,
        num_samples,
        device="cpu",
        seed=42,
        beta_scale=0.5,
        beta_loc=2.0,
        omega_scale=0.1,
        omega_loc=0.1,
        nx_dag=None,
        return_mag=False,
    ):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        if nx_dag is None:
            in_deg_sequence = torch.randint(
                max_degree, size=(num_nodes,), device=device, generator=g
            )
            # randomly permute this sequence to create a list of out degrees
            rand_indices = torch.randperm(num_nodes)
            out_deg_sequence = in_deg_sequence[rand_indices]
            G = nx.directed_configuration_model(
                in_deg_sequence, out_deg_sequence, seed=seed
            )
            dag = nx.DiGraph()
            dag.add_nodes_from(G.nodes())
            dag.add_edges_from([(u, v) for (u, v) in G.edges() if u < v])
        else:
            dag = nx_dag

        # Sample from the resulting model
        mag = MAG(num_nodes, batch_size=1, log_reward=None, device=device)
        if len(dag.edges) > 0:
            mag.edge_list = torch.hstack(
                [
                    torch.zeros((len(dag.edges), 1), device=device),
                    torch.tensor(list(dag.edges), device=device),
                    torch.ones((len(dag.edges), 1), device=device),
                ]
            ).long()

        mag.stopped[:] = 1.0
        samples = MAG.sample_from_structured_gaussian(
            mag,
            num_samples,
            seed,
            beta_scale=beta_scale,
            beta_loc=beta_loc,
            omega_scale=omega_scale,
            omega_loc=omega_loc,
            device=device,
        )

        assert nx.is_directed_acyclic_graph(dag)
        if return_mag:
            return dag, samples, mag
        return dag, samples

    @staticmethod
    def mag_to_pcalg(amat):
        num_nodes = amat.shape[-1]
        pcalg = torch.zeros_like(amat)

        for edge in itertools.combinations(torch.arange(num_nodes), 2):
            directed = (amat[:, edge[0], edge[1]] == 1) & (
                amat[:, edge[1], edge[0]] != 1
            )
            pcalg[directed, edge[0], edge[1]] = 2
            pcalg[directed, edge[1], edge[0]] = 3

            directed = (amat[:, edge[1], edge[0]] == 1) & (
                amat[:, edge[0], edge[1]] != 1
            )
            pcalg[directed, edge[0], edge[1]] = 3
            pcalg[directed, edge[1], edge[0]] = 2

            bidirected = (amat[:, edge[1], edge[0]] == 1) & (
                amat[:, edge[0], edge[1]] == 1
            )
            pcalg[bidirected, edge[0], edge[1]] = 2
            pcalg[bidirected, edge[1], edge[0]] = 2
        return pcalg

    @staticmethod
    def pcalg_to_mag(pcalg):
        num_nodes = pcalg.shape[1]
        amat = torch.zeros_like(pcalg)

        for edge in itertools.combinations(torch.arange(num_nodes), 2):
            directed = (pcalg[:, edge[0], edge[1]] == 2) & (
                pcalg[:, edge[1], edge[0]] == 3
            )
            amat[directed, edge[0], edge[1]] = 1
            directed = (pcalg[:, edge[0], edge[1]] == 3) & (
                pcalg[:, edge[1], edge[0]] == 2
            )
            amat[directed, edge[1], edge[0]] = 1
            bidirected = (pcalg[:, edge[0], edge[1]] == 2) & (
                pcalg[:, edge[1], edge[0]] == 2
            )
            amat[bidirected, edge[0], edge[1]] = 1
            amat[bidirected, edge[1], edge[0]] = 1
        return amat

    @property
    def edge_marks(self):
        return adj_to_edge_marks(self.adjacency_matrix)
