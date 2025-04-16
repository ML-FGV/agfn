import torch
import torch.nn as nn
import torch_geometric as pyg

from agfn.utils import ForwardPolicyMeta
from copy import deepcopy


class ForwardPolicy(ForwardPolicyMeta):
    def __init__(self, num_nodes, hidden_dim, num_layers, device="cpu", eps=0.3):
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.output_dim = (3 * self.num_nodes * (self.num_nodes - 1)) // 2 + 1
        super(ForwardPolicy, self).__init__(eps)

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
        self.get_flows = nn.Linear(hidden_dim, 1).to(self.device)

    def get_latent_emb(self, batch_state):
        amat = batch_state.adjacency_matrix.flatten(start_dim=1)
        latent_emb = self.mlp(amat)
        return latent_emb

    def get_pol(self, latent_emb, mask):
        logits = self.get_logits(latent_emb)
        logits = mask * logits + (1 - mask) * self.masked_value
        gflows = self.get_flows(latent_emb).squeeze(dim=-1)
        pol = torch.softmax(logits, dim=-1)
        return pol, gflows

    def forward(self, batch_state, actions=None):
        latent_emb = self.get_latent_emb(batch_state)
        pol, gflows = self.get_pol(latent_emb, batch_state.forward_mask)
        if actions is None:
            actions = self.get_actions(pol, batch_state.forward_mask)
        return actions, torch.log(pol[batch_state.batch_ids, actions]), gflows


class KGForwardPolicy(ForwardPolicy):
    def forward(self, batch_state, actions=None, feedbacks=None):
        latent_emb = self.get_latent_emb(batch_state)
        mask = batch_state.forward_mask
        pol, gflows = self.get_pol(latent_emb, mask)
        if feedbacks is not None:
            feedbacks = feedbacks / feedbacks.max(dim=1, keepdims=True).values
            # the feedback does not affect the stopping probability
            pol[:, :-1] = pol[:, :-1] * torch.hstack(
                [
                    feedbacks[:, 1],  # directed i < j, i -> j
                    feedbacks[:, 2],  # directed i < j, j -> i
                    feedbacks[:, 3],  # undirected i < j, j <-> i
                ]
            ).view(1, -1)
            # if feedbacks[:, 0] == 0 for some edge, the process should continue
            e1, e2, e3 = torch.chunk(mask[:, :-1], dim=1, chunks=3)
            pol[:, -1] = pol[:, -1] * (
                (e1 * e2 * e3 * feedbacks[:, 0].view(1, -1) + (1 - e1 * e2 * e3) * 1e5)
                .min(dim=1)
                .values  # minimum only over valid edges
            )
            pol = pol / pol.sum(dim=1, keepdims=True)
        if actions is None:
            actions = self.get_actions(pol, mask)
        return actions, torch.log(pol[batch_state.batch_ids, actions]), gflows


class BackwardPolicy(nn.Module):
    masked_value = -1e5

    def __init__(self, device="cpu"):
        super(BackwardPolicy, self).__init__()
        self.device = device

    def forward(self, batch_state, actions=None):
        if actions is not None:
            actions = batch_state.f2b(actions)
        uniform_pol = torch.where(
            batch_state.backward_mask == 1.0, 1.0, self.masked_value
        )
        uniform_pol = torch.softmax(uniform_pol, dim=-1)

        if actions is None:
            actions = torch.multinomial(uniform_pol, num_samples=1, replacement=True)
            actions = actions.squeeze(dim=-1)
            actions = torch.where(
                batch_state.stopped == 1.0, batch_state.max_num_edges, actions
            )

        uniform_pol = torch.hstack(
            [
                uniform_pol * batch_state.backward_mask,
                (batch_state.backward_mask.sum(dim=1, keepdims=True) == 0).type(
                    uniform_pol.dtype
                ),
            ]
        )

        return actions, torch.log(uniform_pol[batch_state.batch_ids, actions])
