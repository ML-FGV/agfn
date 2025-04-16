import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    def __init__(
        self,
        batch_size: int,
        max_trajectory_length: int,
        log_reward: nn.Module,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.device = device
        self.max_trajectory_length = max_trajectory_length
        self.batch_ids = torch.arange(self.batch_size, device=self.device)
        self.traj_size = torch.ones((self.batch_size,), device=self.device)
        self.stopped = torch.zeros((self.batch_size), device=self.device)
        self.is_initial = torch.ones((self.batch_size,), device=self.device)
        self._log_reward = log_reward

    @abstractmethod
    def apply(self, actions: torch.Tensor):
        pass

    @abstractmethod
    def backward(self, actions: torch.Tensor):
        pass

    @torch.no_grad()
    def log_reward(self):
        return self._log_reward(self)

    @torch.no_grad()
    def merge(self, batch_state):
        self.batch_ids = torch.hstack([self.batch_ids, batch_state.batch_ids])
        self.batch_size += batch_state.batch_size
        self.stopped = torch.hstack([self.stopped, batch_state.stopped])
        self.is_initial = torch.hstack([self.is_initial, batch_state.is_initial])

    @property
    def unique_input(self):
        raise NotImplementedError
