import enum

import jax
import jax.numpy as jnp

from agfx.gym import AncestralGraph, damat
from agfx.special.ricf import batch_ricf


def _log_likelihood(
    data: jax.Array, beta: jax.Array, omega: jax.Array
) -> jax.Array:
    num_samples, num_nodes = data.shape
    Id = jnp.eye(num_nodes)

    # Sigma = (I - beta)^{-T} @ omega @ (I - beta)^{-1}
    inv_im_beta = jnp.linalg.inv(Id - beta)
    sigma = jnp.swapaxes(inv_im_beta, -2, -1) @ omega @ inv_im_beta

    XtX = data.T @ data
    trace_term = jnp.einsum("ij,bij->b", XtX, jnp.linalg.inv(sigma))
    _, logdet_sigma = jnp.linalg.slogdet(sigma)

    return (
        -(
            num_samples * num_nodes * jnp.log(2 * jnp.pi)
            + num_samples * logdet_sigma
            + trace_term
        )
        / 2
    )


def _bic(
    log_lik: jax.Array, num_edges: jax.Array, num_samples: int
) -> jax.Array:
    return -2 * log_lik + num_edges * jnp.log(num_samples)


class LogRewardUniform:
    def __call__(self, x: AncestralGraph):
        bs = len(x.batch_ids)
        return jnp.ones(bs)


class LogRewardSparse:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, x: AncestralGraph):
        dir_amat, bidir_amat = damat(x.amat)
        dir_edges = dir_amat.sum(axis=(1, 2))
        bidir_edges = bidir_amat.sum(axis=(1, 2)) / 2
        return -self.alpha * jnp.log(dir_edges + bidir_edges + 1)


class LogRewardBIC:
    def __init__(
        self,
        data: jax.Array,
        true_graph: AncestralGraph = None,
        temperature: float = 1.0,
    ):
        self.data = data
        self.temperature = temperature
        self.true_graph = true_graph

    def __call__(self, x: AncestralGraph):
        dir_amat, bidir_amat = damat(x.amat)
        beta, omega = batch_ricf(self.data, dir_amat, bidir_amat)

        num_samples = self.data.shape[0]
        num_edges = (
            dir_amat.sum(axis=(-1, -2)) + bidir_amat.sum(axis=(-1, -2)) / 2
        )

        log_lik = _log_likelihood(self.data, beta, omega)
        return -_bic(log_lik, num_edges, num_samples) / self.temperature


class LogRewardType(enum.Enum):
    UNIFORM = "uniform"
    SPARSE = "sparse"
    BIC = "bic"


def create_log_reward(
    logr_t: LogRewardType,
    data: jax.Array = None,
    graph: AncestralGraph = None,
    **kwargs,
):
    if logr_t == LogRewardType.UNIFORM:
        return LogRewardUniform()
    elif logr_t == LogRewardType.SPARSE:
        return LogRewardSparse(**kwargs)
    elif logr_t == LogRewardType.BIC:
        return LogRewardBIC(
            data=data, true_graph=graph, temperature=1.0, **kwargs
        )
