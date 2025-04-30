import torch
import torch.nn as nn

from agfn.policies.meta import ForwardPolicyMeta
from agfn.gym import MAG_HITL


class ForwardPolicy(ForwardPolicyMeta):
    def __init__(self, num_nodes, hidden_dim, num_layers, device="cpu", eps=0.3):
        super(ForwardPolicy, self).__init__(eps)
        self.device = device
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = (
            2 * num_nodes * (num_nodes - 1)
        )  # n * (n - 1) / 2 edges 4 * actions for each edge

        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(self.num_nodes * self.num_nodes, hidden_dim))
        for _ in range(self.num_layers):
            self.mlp.append(nn.LeakyReLU())
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.LayerNorm((hidden_dim,)))

        self.mlp = self.mlp.to(self.device)
        self.get_logits = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.output_dim)
        ).to(self.device)

        self.posterior = None

    def get_latent_emb(self, batch_state: MAG_HITL):
        amat = batch_state.adjacency_matrix.flatten(start_dim=1)
        latent_emb = self.mlp(amat)
        return latent_emb

    def get_pol(self, latent_emb: torch.Tensor, mask: torch.Tensor):
        logits = self.get_logits(latent_emb)
        logits = mask * logits + (1 - mask) * self.masked_value
        # Currently, we avoid flow estimation for HITL (constraining, e.g.,
        # the training to trajectory-like balance objectives)
        flows = torch.zeros(
            (logits.shape[0],), device=logits.device, dtype=logits.dtype
        )
        pol = torch.softmax(logits, dim=1)

        if self.posterior is not None:
            # column-major flattening
            posterior_flattened = self.posterior.t().contiguous().view(-1)
            pol = pol * posterior_flattened
            pol = pol / pol.sum(dim=1, keepdims=True)

        return pol, flows

    def set_posterior(self, posterior: torch.Tensor):
        self.posterior = posterior


class BackwardPolicy(nn.Module):
    masked_value = -1e5

    def __init__(self, device="cpu"):
        super(BackwardPolicy, self).__init__()
        self.device = device

    def forward(self, batch_state: MAG_HITL, actions=None):
        mask = batch_state.backward_mask.clone()
        uniform_pol = torch.where(mask == 1.0, 1.0, self.masked_value)
        uniform_pol = torch.softmax(uniform_pol, dim=-1)

        if actions is not None:
            actions = batch_state.f2b(actions)
        else:
            actions = torch.multinomial(uniform_pol, num_samples=1, replacement=True)
            actions = actions.squeeze(dim=-1)

        pol = uniform_pol[batch_state.batch_ids, actions]
        print(actions[pol == 0.0], mask[pol == 0.0])

        return actions, torch.log(pol)
