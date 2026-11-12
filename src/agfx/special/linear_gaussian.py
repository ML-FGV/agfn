import jax
import jax.numpy as jnp


def sample_from_graph(
    num_nodes: int,
    dir_amat: jax.Array,
    bidir_amat: jax.Array,
    num_samples: int,
    key: jax.Array,
    coef_range: tuple[float, float] = (0.1, 0.3),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    k_coef, k_sign, k_var, k_rho, k_data = jax.random.split(key, 5)

    # --- beta: nonzero only on directed edges, magnitude in coef_range ---
    coef_lo, coef_hi = coef_range
    magnitudes = jax.random.uniform(
        k_coef, (num_nodes, num_nodes), minval=coef_lo, maxval=coef_hi
    )
    signs = jax.random.choice(
        k_sign, jnp.array([-1.0, 1.0]), shape=(num_nodes, num_nodes)
    )
    beta = dir_amat * magnitudes * signs

    # --- omega: diagonal variances + off-diagonal covariances for bidirected edges ---
    variances = jax.random.uniform(k_var, (num_nodes,), minval=0.5, maxval=2.0)
    std = jnp.sqrt(variances)

    # covariance for bidirected pair (i,j): rho_ij * std_i * std_j, |rho| < 0.8
    rho = jax.random.uniform(
        k_rho, (num_nodes, num_nodes), minval=-0.8, maxval=0.8
    )
    omega = jnp.diag(variances) + bidir_amat * (
        rho * std[:, None] * std[None, :]
    )
    # symmetrize and nudge to positive definite
    omega = (omega + omega.T) / 2
    min_eig = jnp.linalg.eigvalsh(omega).min()
    omega = omega + jnp.maximum(0.0, -min_eig + 1e-6) * jnp.eye(num_nodes)

    # --- data: X = E @ inv(I - beta),  E rows ~ N(0, omega) ---
    L = jnp.linalg.cholesky(omega)  # omega = L L^T
    z = jax.random.normal(k_data, (num_samples, num_nodes))
    E = z @ L.T  # rows of E ~ N(0, omega)
    X = jnp.linalg.solve((jnp.eye(num_nodes) - beta).T, E.T).T

    return X, beta, omega


def sample_linear_gaussian(
    key: jax.Array,
    num_nodes: int,
    num_samples: int,
    edge_prob: float = 0.3,
    bidir_prob: float = 0.1,
    coef_range: tuple[float, float] = (0.1, 0.3),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key, k_dir, k_bidir = jax.random.split(key, 3)

    rows, cols = jnp.triu_indices(
        num_nodes, k=1
    )  # upper-triangle pairs (i < j)
    num_pairs = rows.shape[0]

    # --- directed edges (i -> j, i < j): always a DAG ---
    is_directed = jax.random.bernoulli(k_dir, edge_prob, (num_pairs,))
    dir_amat = (
        jnp.zeros((num_nodes, num_nodes)).at[rows, cols].set(is_directed)
    )

    # Transitive closure of directed edges: reach[i,j]=1 iff directed path i-->j.
    # Iterating n-1 times is sufficient for a DAG with n nodes.
    reach = dir_amat

    def closure(reach: jax.Array, _):
        return jnp.minimum(reach + dir_amat @ reach, 1.0), None

    reach, _ = jax.lax.scan(f=closure, init=reach, length=num_nodes - 1)

    # --- bidirected edges: forbidden when a directed path already connects the pair ---
    # (avoids almost-directed cycles; reach[j,i] is always 0 since edges go i<j)
    has_path = reach[rows, cols].astype(bool)
    is_bidir = (
        jax.random.bernoulli(k_bidir, bidir_prob, (num_pairs,))
        & ~has_path
        & ~is_directed
    )
    bidir_amat = (
        jnp.zeros((num_nodes, num_nodes))
        .at[rows, cols]
        .set(is_bidir)
        .at[cols, rows]
        .set(is_bidir)
    )
    X, beta, omega = sample_from_graph(
        num_nodes,
        dir_amat,
        bidir_amat,
        num_samples,
        key,
        coef_range=coef_range,
    )

    return X, dir_amat, bidir_amat, beta, omega
