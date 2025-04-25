import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod


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
            uniform_pol = torch.where(torch.isclose(mask, 1.0), 1.0, self.masked_value)
        uniform_pol = torch.softmax(uniform_pol, dim=-1)
        eps = 0.0 if not self.training else self.eps
        exp_pol = pol * (1 - eps) + eps * uniform_pol
        actions = torch.multinomial(exp_pol, num_samples=1, replacement=True)
        actions = actions.squeeze(dim=-1)
        return actions
