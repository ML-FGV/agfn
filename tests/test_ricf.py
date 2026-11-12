import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agfx.special.ricf import _ricf, batch_ricf

jax.config.update("jax_enable_x64", True)


def _sample_sem(key, beta, omega, n_samples):
    """Generate data from X = X @ beta + eps, eps ~ N(0, omega)."""
    n = beta.shape[0]
    L = jnp.linalg.cholesky(omega)
    eps = jax.random.normal(key, (n_samples, n)) @ L.T
    # X (I - beta) = eps  =>  X = eps @ inv(I - beta)
    X = jnp.linalg.solve((jnp.eye(n) - beta).T, eps.T).T
    return X


def _parents_spouses(beta, omega):
    """Derive mask matrices from true parameters."""
    n = beta.shape[0]
    parents = (beta != 0).astype(jnp.float32)
    off_diag = omega - jnp.diag(jnp.diag(omega))
    spouses = (off_diag != 0).astype(jnp.float32)
    return parents, spouses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(key, beta_true, omega_true, n_samples=20_000, atol=0.05):
    parents, spouses = _parents_spouses(beta_true, omega_true)
    data = _sample_sem(key, beta_true, omega_true, n_samples)
    beta_hat, omega_hat = _ricf(data, parents, spouses)
    np.testing.assert_allclose(beta_hat, beta_true, atol=atol)
    np.testing.assert_allclose(omega_hat, omega_true, atol=atol)


# ---------------------------------------------------------------------------
# Trivial / structural tests
# ---------------------------------------------------------------------------

def test_no_edges_identity_noise():
    """Empty graph, identity noise: beta=0, omega=I should be recovered exactly."""
    n = 4
    beta = jnp.zeros((n, n))
    omega = jnp.eye(n)
    _run(jax.random.key(0), beta, omega, n_samples=10_000, atol=0.05)


def test_omega_symmetric(n=3):
    """Recovered omega must always be symmetric."""
    beta = jnp.array([[0, 0.5, 0], [0, 0, 0.4], [0, 0, 0]], dtype=jnp.float32)
    omega = jnp.array([[1.0, 0, 0.3], [0, 0.8, 0], [0.3, 0, 1.2]], dtype=jnp.float32)
    parents, spouses = _parents_spouses(beta, omega)
    data = _sample_sem(jax.random.key(1), beta, omega, 1_000)
    _, omega_hat = _ricf(data, parents, spouses)
    np.testing.assert_allclose(omega_hat, omega_hat.T, atol=1e-5)


def test_beta_zero_outside_parents():
    """Entries of beta outside the parent mask must remain zero."""
    beta = jnp.array([[0, 0.6, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.float32)
    omega = jnp.eye(3)
    parents, spouses = _parents_spouses(beta, omega)
    data = _sample_sem(jax.random.key(2), beta, omega, 5_000)
    beta_hat, _ = _ricf(data, parents, spouses)
    # Positions with no edge must be (near) zero.
    zero_mask = parents == 0
    np.testing.assert_allclose(beta_hat[zero_mask], 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Recovery tests — directed edges only
# ---------------------------------------------------------------------------

def test_chain_graph_recovery():
    """3-node chain 1->2->3, no bidirected edges."""
    beta = jnp.array([[0, 0.7, 0], [0, 0, -0.5], [0, 0, 0]], dtype=jnp.float64)
    omega = jnp.diag(jnp.array([1.0, 0.8, 1.2]))
    _run(jax.random.key(3), beta, omega)


def test_fork_graph_recovery():
    """3-node fork 1->2, 1->3."""
    beta = jnp.array([[0, 0.6, -0.4], [0, 0, 0], [0, 0, 0]], dtype=jnp.float64)
    omega = jnp.diag(jnp.array([0.9, 1.1, 0.7]))
    _run(jax.random.key(4), beta, omega)


def test_collider_graph_recovery():
    """3-node collider 1->3, 2->3."""
    beta = jnp.array([[0, 0, 0.5], [0, 0, -0.3], [0, 0, 0]], dtype=jnp.float64)
    omega = jnp.diag(jnp.array([1.0, 1.0, 0.6]))
    _run(jax.random.key(5), beta, omega)


# ---------------------------------------------------------------------------
# Recovery tests — bidirected edges
# ---------------------------------------------------------------------------

def test_bidir_only_recovery():
    """Two nodes with bidirected edge only (no directed edges)."""
    beta = jnp.zeros((2, 2), dtype=jnp.float64)
    omega = jnp.array([[1.0, 0.4], [0.4, 1.5]], dtype=jnp.float64)
    _run(jax.random.key(6), beta, omega)


def test_mixed_directed_and_bidirected():
    """3-node graph: 1->2, 1->3, 2->3, plus bidirected 1<->3."""
    beta = jnp.array([[0, 0.5, 0.3], [0, 0, -0.6], [0, 0, 0]], dtype=jnp.float64)
    omega = jnp.array([[1.0, 0, 0.35], [0, 0.9, 0], [0.35, 0, 1.1]], dtype=jnp.float64)
    _run(jax.random.key(7), beta, omega)


def test_bidirected_sibling_pair_in_larger_graph():
    """4-node graph with one bidirected pair (2<->4) and a chain."""
    beta = jnp.array(
        [[0, 0.4, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.5], [0, 0, 0, 0]],
        dtype=jnp.float64,
    )
    omega = jnp.array(
        [[1.0, 0, 0, 0], [0, 1.0, 0, 0.3], [0, 0, 0.8, 0], [0, 0.3, 0, 1.2]],
        dtype=jnp.float64,
    )
    _run(jax.random.key(8), beta, omega)


# ---------------------------------------------------------------------------
# Implied covariance recovery
# ---------------------------------------------------------------------------

def test_implied_covariance_recovery():
    """Check that Σ = (I-B)^{-T} Ω (I-B)^{-1} is recovered from the sample covariance."""
    beta_true = jnp.array([[0, 0.5, 0.3], [0, 0, -0.6], [0, 0, 0]], dtype=jnp.float64)
    omega_true = jnp.array([[1.0, 0, 0.35], [0, 0.9, 0], [0.35, 0, 1.1]], dtype=jnp.float64)
    parents, spouses = _parents_spouses(beta_true, omega_true)

    n = beta_true.shape[0]
    I = jnp.eye(n)
    inv_imb = jnp.linalg.inv(I - beta_true)
    sigma_true = inv_imb.T @ omega_true @ inv_imb

    data = _sample_sem(jax.random.key(9), beta_true, omega_true, 50_000)
    beta_hat, omega_hat = _ricf(data, parents, spouses)

    inv_imb_hat = jnp.linalg.inv(I - beta_hat)
    sigma_hat = inv_imb_hat.T @ omega_hat @ inv_imb_hat

    np.testing.assert_allclose(sigma_hat, sigma_true, atol=0.05)


# ---------------------------------------------------------------------------
# Batched API
# ---------------------------------------------------------------------------

def test_batch_ricf_shape():
    """batch_ricf must return (batch, n, n) tensors."""
    n, batch = 3, 5
    beta = jnp.zeros((n, n))
    omega = jnp.eye(n)
    parents = jnp.zeros((batch, n, n))
    spouses = jnp.zeros((batch, n, n))
    data = jax.random.normal(jax.random.key(10), (100, n))
    beta_hat, omega_hat = batch_ricf(data, parents, spouses)
    assert beta_hat.shape == (batch, n, n)
    assert omega_hat.shape == (batch, n, n)


def test_batch_ricf_matches_single():
    """Each element of batch_ricf must match the corresponding single call."""
    n = 3
    beta_true = jnp.array([[0, 0.5, 0], [0, 0, 0.4], [0, 0, 0]], dtype=jnp.float64)
    omega_true = jnp.diag(jnp.array([1.0, 0.8, 1.2]))
    parents, spouses = _parents_spouses(beta_true, omega_true)
    data = _sample_sem(jax.random.key(11), beta_true, omega_true, 10_000)

    b_single, o_single = _ricf(data, parents, spouses)
    b_batch, o_batch = batch_ricf(
        data,
        parents[None],   # (1, n, n)
        spouses[None],
    )
    np.testing.assert_allclose(b_batch[0], b_single, atol=1e-5)
    np.testing.assert_allclose(o_batch[0], o_single, atol=1e-5)
