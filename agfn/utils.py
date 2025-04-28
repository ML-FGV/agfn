import torch
import tqdm

from itertools import chain
from copy import deepcopy


def train(gflownet, initial_state, epochs, lr=1e-3, lr_logz=1e-1, use_scheduler=False):
    optimizer = torch.optim.AdamW(
        [
            {
                "params": chain(gflownet.pf.parameters(), gflownet.pb.parameters()),
                "lr": lr,
            },
            {"params": gflownet.log_z, "lr": lr_logz},
        ]
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs)

    losses = list()
    pbar = tqdm.tqdm(range(epochs), disable=False)
    for _ in pbar:
        optimizer.zero_grad()
        loss = gflownet(deepcopy(initial_state))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        # Update the exploration rate
        losses.append(loss.item())
        if use_scheduler:
            scheduler.step()
    return losses


@torch.no_grad()
def init_logz(gflownet, create_env, num_batches, batch_size):
    log_rewards = list()
    marginal_log_dist = list()
    for _ in tqdm.tqdm(range(num_batches)):
        env = create_env(batch_size)
        env = gflownet.sample(env)
        log_rewards.append(env.log_reward())
        marginal_log_prob = gflownet.sample_many_backward(env, num_trajectories=8)
        marginal_prob = torch.logsumexp(marginal_log_prob, dim=1)
        marginal_prob -= torch.log(torch.tensor(marginal_log_prob.shape[1]))
        marginal_log_dist.append(marginal_prob)
    log_rewards = torch.hstack(log_rewards)
    marginal_log_dist = torch.hstack(marginal_log_dist)
    gflownet.log_z.fill_(torch.logsumexp(log_rewards - marginal_log_dist, dim=0))
    gflownet.log_z.add_(-torch.log(torch.tensor(len(log_rewards))))

    assert gflownet.log_z.requires_grad


def unique(x, dim=-1):
    values, inverse, counts = torch.unique(
        x, return_inverse=True, return_counts=True, dim=dim
    )
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=values.device)
    # inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return (
        values,
        inverse,
        counts,
        inverse.new_empty(values.size(dim)).scatter_(dim, inverse, perm),
    )


def marginal_dist(env, marginal_log, log_rewards, dim=0, return_indices=False):
    if not torch.is_tensor(env):
        values, inverse, counts, indices = unique(env.unique_input, dim=dim)
    else:
        values, inverse, counts, indices = unique(env, dim=dim)
    # Compute learned distribution
    marginal_log = marginal_log - marginal_log.max()  # numerical stability
    marginal_log_batch = torch.zeros(
        (values.size(0), marginal_log.shape[1]), device=values.device
    )
    marginal_log_batch.scatter_add_(
        dim=0, index=inverse.view(-1, 1), src=marginal_log.exp()
    )
    marginal_log_batch = marginal_log_batch.sum(dim=-1)
    marginal_log_batch /= counts * marginal_log.shape[1]
    learned_dist = marginal_log_batch / marginal_log_batch.sum()
    # Compute the target distribution
    target_dist = (
        log_rewards[indices] - torch.logsumexp(log_rewards[indices], dim=0)
    ).exp()
    if return_indices:
        return learned_dist, target_dist, indices
    return learned_dist, target_dist


def marginal_dist_batch(gflownet, env, num_batches):
    learned_dist = list()
    target_dist = list()
    for _ in range(num_batches):
        terminal_states = gflownet.sample(deepcopy(env))
        marginal_log = gflownet.sample_many_backward(
            terminal_states, num_trajectories=8
        )
        learned_dist_batch, target_dist_batch = marginal_dist(
            terminal_states, marginal_log, terminal_states.log_reward()
        )

        learned_dist.append(learned_dist_batch)
        target_dist.append(target_dist_batch)

    return torch.hstack(learned_dist), torch.hstack(target_dist)
