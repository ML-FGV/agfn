# Estimate the number of AGs with N nodes
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

# Local imports
from mpl_preamble import *


def sample_uniform_bigraph(n: int, key: jax.Array):
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


def sample_uniform_digraph(n: int, key: jax.Array):
    i, j = jnp.triu_indices(n, k=1)
    num_edges = len(i)

    edges = jax.random.choice(key, 3, shape=(num_edges,), replace=True)

    # Check which ones are MAGs
    adj = jnp.zeros((n, n))
    adj = adj.at[i, j].set(jnp.where(edges == 1, 1, adj[i, j]))
    adj = adj.at[j, i].set(jnp.where(edges == 2, 1, adj[j, i]))

    _, key = jax.random.split(key, 2)

    return adj, key


@partial(jax.jit, static_argnames=("n",))
def is_ag(carry: tuple[jax.Array, jax.Array, jax.Array], _, *, n: int):
    count, adj, key = carry

    D = jnp.logical_and(adj != adj.T, adj == 1)
    B = jnp.logical_and(adj == adj.T, adj == 1)

    D = jnp.where(D, 1, 0)
    B = jnp.where(B, 1, 0)

    D_exp = jsp.linalg.expm(D)

    th = jnp.trace(D_exp) - n + jnp.sum(D_exp * B)
    count += jnp.isclose(th, 0, atol=1e-8)

    adj, key = sample_uniform_bigraph(n, key)

    return (count, adj, key), _


@partial(jax.jit, static_argnames=("n",))
def is_dag(carry: tuple[jax.Array, jax.Array, jax.Array], _, *, n: int):
    count, adj, key = carry
    # Use NOTEARS's method; if the graph is a DAG, A^n = 0 (nilpotent),
    # tr(A^k) = 0 for all k \ge 1 (all eigenvalues are 0), and hence tr exp (A * A) = d
    # (since exp(A * A) includes the identity in the first component)
    D_sq = jnp.square(adj)
    D_exp = jsp.linalg.expm(D_sq)

    count += jnp.isclose(jnp.trace(D_exp) - n, 0, atol=1e-8)

    adj, key = sample_uniform_digraph(n, key)
    return (count, adj, key), _


def estimate_mags(n: int, num_samples: int):
    num_edges = (n * (n - 1)) // 2
    # Sample directed graphs uniformly

    # We first sample the state of each of the n * (n - 1) / 2 edges uniformly
    initial_key = jax.random.key(42)
    adj, key = sample_uniform_bigraph(n, initial_key)

    (count, _, _), _ = jax.lax.scan(
        f=partial(is_ag, n=n),
        init=(jnp.array([0]), adj, key),
        length=num_samples,
    )

    log_count = jnp.log(count) - jnp.log(num_samples) + num_edges * jnp.log(4)

    return jnp.exp(log_count).item()


def estimate_dags(n: int, num_samples: int):
    num_edges = (n * (n - 1)) / 2
    # Sample directed graphs uniformly

    # We first sample the state of each of the n * (n - 1) / 2 edges uniformly
    initial_key = jax.random.key(42)
    adj, key = sample_uniform_digraph(n, initial_key)

    (count, _, _), _ = jax.lax.scan(
        f=partial(is_dag, n=n),
        init=(jnp.array([0]), adj, key),
        length=num_samples,
    )

    log_count = jnp.log(count) - jnp.log(num_samples) + num_edges * jnp.log(3)

    return jnp.exp(log_count).item()


def make_plot(num_nodes_list: list[int], num_ags_list: list[int], num_dags_list: list[int]):
    plt.plot(num_nodes_list, num_ags_list, label="AGs")
    plt.plot(num_nodes_list, num_dags_list, label="DAGs")
    plt.ylabel("Number of Graphs")
    plt.xlabel("Number of Nodes")
    plt.yscale("log")
    # Define the forward and inverse functions
    # p = 4

    # def forward_sqrt(e):
    #     return e ** (1 / p)

    # def inverse_sq(e):
    #     return e**p

    # ax = plt.gca()
    # ax.set_yscale("function", functions=(forward_sqrt, inverse_sq))
    # ymax = max(num_ags_list)

    # ax.set_ylim(0, ymax * 1.2)

    plt.legend()


if __name__ == "__main__":
    num_nodes_list = [3, 4, 5, 6, 7]
    num_ags_list = []
    num_dags_list = []
    for num_nodes in num_nodes_list:
        num_ags = estimate_mags(num_nodes, 4e6)
        num_dags = estimate_dags(num_nodes, 4e6)

        num_ags_list.append(num_ags)
        num_dags_list.append(num_dags)

    plt.figure(figsize=(2.9, 2.9))
    make_plot(num_nodes_list, num_ags_list, num_dags_list)

    plt.savefig("estimate_mags.pdf", bbox_inches="tight")

    plt.show()
    # print(f"AGs [3]: {estimate_mags(3, 1e6):.2e}")
    # print(f"DAGs [3]: {estimate_dags(3, 1e6):.2e}")
    # print(f"AGs [4]: {estimate_mags(4, 1e6):.2e}")
    # print(f"DAGs [4]: {estimate_dags(4, 1e6):.2e}")
    # print(f"AGs [5]: {estimate_mags(5, 1e7):.2e}")
    # print(f"DAGs [5]: {estimate_dags(5, 1e7):.2e}")
    # print(f"AGs [6]: {estimate_mags(6, 1e7):.2e}")
    # print(f"DAGs [6]: {estimate_dags(6, 1e7):.2e}")
