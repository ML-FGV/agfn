import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod

from agfn.gym import MAG, MAG_HITL
from typing import Union


class ForwardPolicyMeta(nn.Module, metaclass=ABCMeta):
    masked_value = -1e5

    def __init__(self, eps=0.3):
        super(ForwardPolicyMeta, self).__init__()
        self.eps = eps
        self.posterior = None

    @abstractmethod
    def get_latent_emb(self):
        pass

    @abstractmethod
    def get_pol(self):
        pass

    def set_posterior(self, posterior):
        self.posterior = posterior

    def get_actions(self, pol, mask=None):
        if mask is None:
            uniform_pol = torch.ones_like(pol)
        else:
            uniform_pol = torch.where(
                torch.isclose(mask, torch.ones_like(mask)), 1.0, self.masked_value
            )
        uniform_pol = torch.softmax(uniform_pol, dim=-1)
        eps = 0.0 if not self.training else self.eps
        exp_pol = pol * (1 - eps) + eps * uniform_pol
        actions = torch.multinomial(exp_pol, num_samples=1, replacement=True)
        actions = actions.squeeze(dim=-1)
        return actions

    def forward(self, batch_state: Union[MAG, MAG_HITL], actions=None):
        mask = batch_state.forward_mask.clone()
        latent_emb = self.get_latent_emb(batch_state)
        pol, gflows = self.get_pol(latent_emb, mask)
        if actions is None:
            actions = self.get_actions(pol, mask)
        return actions, torch.log(pol[batch_state.batch_ids, actions]), gflows
