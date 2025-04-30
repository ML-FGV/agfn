import torch
import torch.nn as nn

from copy import deepcopy

from agfn.gym import MAG
from agfn.policies.models import ForwardPolicy, BackwardPolicy


def huber_loss(diff, delta=9.0):
    diff_sq = diff * diff
    larger_than_delta = (diff_sq > delta**2).to(dtype=diff.dtype)
    return (
        delta * (diff_sq - 0.5 * delta) * larger_than_delta
        + 0.5 * diff_sq * (1 - larger_than_delta)
    ).mean()


class GFlowNet(nn.Module):
    def __init__(
        self, pf: ForwardPolicy, pb: BackwardPolicy, criterion="tb", device="cpu"
    ):
        super(GFlowNet, self).__init__()
        self.pf = pf
        self.pb = pb
        self.log_z = nn.Parameter(
            torch.randn((1,), dtype=torch.get_default_dtype(), device=device).squeeze(),
            requires_grad=True,
        )

        self.criterion = criterion
        self.device = device

    def forward(self, batch_state):
        match self.criterion:
            case "tb":
                loss = self._trajectory_balance(batch_state)
            case "cb":
                loss = self._contrastive_balance(batch_state)
            case "cbf":
                loss = self._contrastive_balance_full(batch_state)
            case "db":
                loss = self._detailed_balance(batch_state)
            case "subtb":
                loss = self._subtrajectory_balance(batch_state)
            case _:
                raise ValueError(
                    f"{self.criterion} should be either tb, cb, db, or dbc"
                )
        return loss

    def _sample_traj(self, batch_state):
        # dim 0: pf, dim 1: pb, dim 2: mask
        traj_stats = torch.zeros(
            (2, batch_state.batch_size, batch_state.max_trajectory_length),
            device=self.device,
        )
        F_traj = torch.zeros(
            (batch_state.batch_size, batch_state.max_trajectory_length + 1),
            device=self.device,
        )
        i = 0
        last_idx = torch.zeros(
            (batch_state.batch_size,), dtype=torch.long, device=self.device
        )

        is_stopped = torch.zeros(
            (batch_state.batch_size,), dtype=bool, device=self.device
        )
        while not is_stopped.all():
            # Sample the actions
            out = self.pf(batch_state)
            actions, pf, F = out[0], out[1], out[2]

            # Apply the actions
            batch_state.apply(actions)

            # Corresponding backward actions
            out = self.pb(batch_state, actions)
            pb = out[1]
            assert not torch.isinf(pb).any()

            # Save values
            traj_stats[0, ~is_stopped, i] = pf[~is_stopped]
            traj_stats[1, ~is_stopped, i] = pb[~is_stopped]

            F_traj[~is_stopped, i] = F[~is_stopped]
            # Check whether it already stopped
            is_stopped = batch_state.stopped
            i += 1
            last_idx += 1 - is_stopped.long()

        F_traj[batch_state.batch_ids, last_idx + 1] = batch_state.log_reward()
        return traj_stats, F_traj, last_idx + 1

    def _trajectory_balance(self, batch_state):
        traj_stats, F_traj, last_idx = self._sample_traj(batch_state)
        loss = (
            (traj_stats[0] - traj_stats[1]).sum(dim=1)
            - F_traj[batch_state.batch_ids, last_idx]
            + self.log_z
        )
        return (loss * loss).mean()

    def _detailed_balance(self, batch_state):
        traj_stats, F_traj, _ = self._sample_traj(batch_state)
        loss = traj_stats[0] + F_traj[:, :-1] - traj_stats[1] - F_traj[:, 1:]
        return (loss * loss).mean()

    def _subtrajectory_balance(self, batch_state):
        traj_stats, F_traj, _ = self._sample_traj(batch_state)
        max_traj_length = batch_state.max_trajectory_length
        i, j = torch.triu_indices(
            max_traj_length + 1, max_traj_length + 1, offset=1, device=self.device
        )

        traj_stats = torch.cumsum(traj_stats, dim=-1)
        traj_stats = torch.cat(
            [torch.zeros((*traj_stats.shape, 1), device=self.device), traj_stats],
            dim=-1,
        )
        pf = traj_stats[0, :, j] - traj_stats[0, :, i]
        pb = traj_stats[1, :, j] - traj_stats[1, :, i]
        loss = pf - pb + F_traj[:, i] - F_traj[:, j]
        lamb = 0.9 ** (j - i).view(1, -1)
        loss = ((loss * loss) * lamb).sum(dim=1) / lamb.sum()
        loss = self._remove_unlikely_states(batch_state, loss)
        return loss.mean()

    def _contrastive_balance_full(self, batch_state):
        traj_stats, F_traj, last_idx = self._sample_traj(batch_state)
        loss = (traj_stats[0] - traj_stats[1]).sum(dim=1) - F_traj[
            batch_state.batch_ids, last_idx
        ]
        loss = self._remove_unlikely_states(batch_state, loss)
        loss = (loss[:, None] - loss[None, :]) ** 2
        return loss.mean()

    def _contrastive_balance(self, batch_state):
        assert (batch_state.batch_size % 2) == 0, "batch size must be even"
        half_batch = batch_state.batch_size // 2
        traj_stats, F_traj, last_idx = self._sample_traj(batch_state)
        loss = (traj_stats[0] - traj_stats[1]).sum(dim=1) - F_traj[
            batch_state.batch_ids, last_idx
        ]
        loss = self._remove_unlikely_states(batch_state, loss)
        loss = loss[:half_batch] - loss[half_batch:]
        return (loss * loss).mean()

    @torch.no_grad()
    def sample(self, batch_state):
        while (batch_state.stopped < 1).any():
            out = self.pf(batch_state)
            actions = out[0]
            batch_state.apply(actions)
        return batch_state

    @torch.no_grad()
    def marginal_prob(self, batch_state, copy_env=False):
        # Use importance sampling to estimate the marginal probabilities
        if copy_env:
            batch_state = deepcopy(batch_state)
        forward_log_traj = torch.zeros(
            (batch_state.batch_size, batch_state.max_trajectory_length),
            device=self.device,
        )
        backward_log_traj = torch.zeros(
            (batch_state.batch_size, batch_state.max_trajectory_length),
            device=self.device,
        )

        idx = 0

        is_initial = torch.zeros(batch_state.batch_size, device=self.device, dtype=bool)
        while not is_initial.all():
            # Estimate the backward probabilities
            back_out = self.pb(batch_state)
            actions, backward_log_prob = back_out[0], back_out[1]

            forward_actions = batch_state.backward(actions)

            # Estimate the forward probabilities
            forward_out = self.pf(batch_state, actions=forward_actions)
            forward_log_prob = forward_out[1]

            forward_log_traj[~is_initial, idx] = forward_log_prob[~is_initial]
            backward_log_traj[~is_initial, idx] = backward_log_prob[~is_initial]

            is_initial = batch_state.is_initial.bool()
            idx += 1

        marginal_log = (forward_log_traj - backward_log_traj).sum(dim=1)
        return marginal_log

    def sample_many_backward(self, batch_states, num_trajectories):
        marginal_log = torch.zeros(
            (batch_states.batch_size, num_trajectories), device=self.device
        )
        for idx in range(num_trajectories):
            marginal_log[:, idx] = self.marginal_prob(batch_states, copy_env=True)
        return marginal_log

    class OffPolicyCtx:
        def __init__(self, gflownet):
            self.gflownet = gflownet

        def __enter__(self):
            self.curr_eps = self.gflownet.pf.eps
            self.gflownet.pf.eps = 1.0

        def __exit__(self, *unused_args):
            self.gflownet.pf.eps = self.curr_eps

    def off_policy(self):
        return self.OffPolicyCtx(self)

    def set_posterior(self, posterior):
        self.pf.set_posterior(posterior)

    def cross_entropy_with_reward(self, initial_state: MAG, num_iterations: int):
        entropy = 0.0
        for _ in range(num_iterations):
            initial_state_cp = deepcopy(initial_state)

            self.sample(initial_state_cp)

            entropy += -initial_state_cp.log_reward().sum()

        entropy = entropy / (initial_state.batch_size * num_iterations)
        return entropy
