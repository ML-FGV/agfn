import jax
import jax.numpy as jnp


def _masked_regression(
    covariates: jax.Array, targets: jax.Array, mask: jax.Array, lamb=1e-3
):
    X = covariates * mask
    A = X.T @ X + lamb * jnp.eye(covariates.shape[1])
    params = jnp.linalg.solve(A, X.T @ targets)
    residuals = targets - X @ params
    return params, (residuals * residuals).sum()


def _ricf_sweep(
    data: jax.Array,
    parents_matrix: jax.Array,
    spouses_matrix: jax.Array,
    omega: jax.Array,
    beta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    num_nodes = parents_matrix.shape[0]
    all_idx = jnp.arange(num_nodes)

    def _node_update(
        carry: tuple[jax.Array, jax.Array], i: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        omega, beta = carry

        # All node indices except i — static output shape (num_nodes-1,).
        idx = jnp.delete(all_idx, i, assume_unique_indices=True)

        omega_mimi = omega[idx[:, None], idx[None, :]]  # (n-1, n-1)

        # residuals: (num_nodes, num_samples)
        residuals = (data - data @ beta).T

        # pseudo_vars: (num_samples, n-1) — Ω_{-i,-i}^{-1} r_{-i}
        pseudo_vars = jnp.linalg.solve(omega_mimi, residuals[idx]).T

        covariates = jnp.concatenate(
            [data[:, idx], pseudo_vars], axis=-1
        )  # (num_samples, 2*(n-1))
        reg_mask = jnp.concatenate(
            [parents_matrix[idx, i], spouses_matrix[i, idx]]
        )

        params, reg_residuals = _masked_regression(
            covariates, data[:, i], reg_mask
        )

        beta_i = params[: num_nodes - 1]
        omega_i = params[num_nodes - 1 :]

        omega = omega.at[i, idx].set(omega_i)
        omega = omega.at[idx, i].set(omega_i)
        beta = beta.at[idx, i].set(beta_i)

        # Schur complement: ω_{ii} = σ²_residual + ω_{i,-i} Ω_{-i,-i}^{-1} ω_{-i,i}
        adjusted_corr = reg_residuals / data.shape[0]
        omega_mi_inv = jnp.linalg.solve(omega_mimi, omega[idx, i])
        omega = omega.at[i, i].set(
            adjusted_corr + omega[i, idx] @ omega_mi_inv
        )

        return (omega, beta), None

    (omega, beta), _ = jax.lax.scan(_node_update, (omega, beta), all_idx)
    return omega, beta


def _ricf(
    data: jax.Array,
    parents_matrix: jax.Array,
    spouses_matrix: jax.Array,
    max_iter: int = 1,
    tol: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    num_nodes = parents_matrix.shape[0]

    def cond_fun(state):
        _, _, iter_, delta = state
        return (delta >= tol) & (iter_ < max_iter)

    def body_fun(state):
        omega, beta, iter_, _ = state
        omega_new, beta_new = _ricf_sweep(
            data, parents_matrix, spouses_matrix, omega, beta
        )
        delta = jnp.max(jnp.abs(omega_new - omega) + jnp.abs(beta_new - beta))
        return omega_new, beta_new, iter_ + 1, delta

    omega, beta, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (jnp.eye(num_nodes), jnp.zeros((num_nodes, num_nodes)), 0, jnp.inf),
    )
    return beta, omega


# Batched versions — data is shared; graph structures are batched.
batch_ricf = jax.vmap(lambda d, pm, sm: _ricf(d, pm, sm), in_axes=(None, 0, 0))
