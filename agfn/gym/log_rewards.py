import torch
import torch.nn as nn
import pandas as pd
import copy
import pathlib

from agfn.ricf import reward_bic
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
