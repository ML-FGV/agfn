import torch
import numpy as np

from enum import Enum


class EdgeType(Enum):
    NO_RELATIONSHIP = "NO_RELATIONSHIP"
    LEFT_CAUSES_RIGHT = "LEFT_CAUSES_RIGHT"
    RIGHT_CAUSES_LEFT = "RIGHT_CAUSES_LEFT"
    LATENT_CONFOUNDING = "LATENT_CONFOUNDING"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, EdgeType):
            return value

        if isinstance(value, str):
            for member in cls:
                if str(member.value) == value:
                    return member

        if isinstance(value, int):
            for member in cls:
                if member.value == value:
                    return member

        raise ValueError(f"Invalid edge type: {value}")

    def to_int(self) -> int:
        match self:
            case self.NO_RELATIONSHIP:
                return 0
            case self.LEFT_CAUSES_RIGHT:
                return 1
            case self.RIGHT_CAUSES_LEFT:
                return 2
            case self.LATENT_CONFOUNDING:
                return 3

    def unwrap_into_str(self, node_i: int, node_j: int) -> str:
        match self:
            case self.NO_RELATIONSHIP:
                return f"Node {node_i} and node {node_j} have no relationship."
            case self.LEFT_CAUSES_RIGHT:
                return f"Node {node_i} directly causes node {node_j}."
            case self.RIGHT_CAUSES_LEFT:
                return f"Node {node_j} directly causes node {node_i}."
            case self.LATENT_CONFOUNDING:
                return f"Node {node_i} and node {node_j} are indirectly associated, via latent confounding."

    @classmethod
    def from_int(cls, value: int) -> "EdgeType":
        match value:
            case 0:
                return cls.NO_RELATIONSHIP
            case 1:
                return cls.LEFT_CAUSES_RIGHT
            case 2:
                return cls.RIGHT_CAUSES_LEFT
            case 3:
                return cls.LATENT_CONFOUNDING
            case _:
                raise ValueError(f"Invalid edge type: {value}")


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
