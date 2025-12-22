# Estimate the number of AGs with N nodes
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def sample_uniform_adj(n: int, key: jax.Array):
    i, j = jnp.triu_indices(n, k=1)
    num_edges = len(i)

    edges = jax.random.choice(key, 4, shape=(num_edges,), replace=True)

    # Check which ones are MAGs
    adj = jnp.zeros((n, n))
    adj = adj.at[i, j].set(jnp.where(edges == 1, 1, adj[i, j]))
    adj = adj.at[j, i].set(jnp.where(edges == 2, 1, adj[j, i]))
    adj = adj.at[i, j].set(jnp.where(edges == 3, 1, adj[i, j]))
    adj = adj.at[j, i].set(jnp.where(edges == 3, 1, adj[j, i]))

    _, key = jax.random.split(key, 2)

    return adj, key


@partial(jax.jit, static_argnames=("n",))
def is_mag(carry: tuple[jax.Array, jax.Array, jax.Array], _, *, n: int):
    count, adj, key = carry

    i_bi, j_bi = jnp.argwhere((adj == adj.T) & (adj == 1), size=n).T
    i_di, j_di = jnp.argwhere((adj != adj.T) & (adj == 1), size=n).T

    D = jnp.zeros_like(adj)
    B = jnp.zeros_like(adj)

    D = D.at[i_di, j_di].set(1)
    B = B.at[i_bi, j_bi].set(1)

    D_exp = jsp.linalg.expm(D)
    th = jnp.diagonal(D_exp, axis1=-1, axis2=-2).sum(axis=-1) - n + jnp.sum(D_exp * B, axis=(-1, -2))
    count += jnp.abs(th) < 1e-8

    adj, key = sample_uniform_adj(n, key)

    return (count, adj, key), _


def estimate_mags(n: int, num_samples: int):
    num_edges = (n * (n - 1)) / 2
    graph_count = 4**num_edges
    # Sample directed graphs uniformly

    # We first sample the state of each of the n * (n - 1) / 2 edges uniformly
    initial_key = jax.random.key(42)
    adj, key = sample_uniform_adj(n, initial_key)

    (count, _, _), _ = jax.lax.scan(
        f=partial(is_mag, n=n),
        init=(jnp.array([0]), adj, key),
        length=num_samples,
    )

    return count / num_samples * graph_count


if __name__ == "__main__":
    print(estimate_mags(5, 1e6))
