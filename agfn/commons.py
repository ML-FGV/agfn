import torch
import numpy as np

from enum import Enum


class EdgeType(Enum):
    NO_RELATIONSHIP = 0
    LEFT_CAUSES_RIGHT = 1
    RIGHT_CAUSES_LEFT = 2
    LATENT_CONFOUNDING = 3


def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a.unsqueeze(2), b.unsqueeze(1))


def elementwise_maximum(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.stack(tensors, dim=0), dim=0).values


def is_mag(directed_adj, bidirected_adj, return_th=False):
    directed_adj_exp = torch.matrix_exp(directed_adj)
    th = (
        torch.diagonal(directed_adj_exp, dim1=-1, dim2=-2).sum(dim=-1)
        - directed_adj.shape[1]
        + torch.sum(directed_adj_exp * bidirected_adj, dim=(-1, -2))
    )
    if return_th:
        return torch.abs(th) < 1e-8, th
    return torch.abs(th) < 1e-8


def topk_numpy(array: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(array, -k)[-k:]
