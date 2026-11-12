import jax
import jax.numpy as jnp

from agfx.gym import AncestralGraph, from_amat, get_indicators
from agfx.policies import rollout_many


def eval_true_marginal(
    logr: callable,
    graphs: list[jax.Array],
    batch_size: int = 512,
):
    num_graphs = len(graphs)
    num_batches = num_graphs // batch_size

    graphs = jnp.array(graphs)
    graphs_to_scan = graphs[: num_batches * batch_size].reshape(
        num_batches, batch_size, *graphs.shape[1:]
    )
    other_graphs = graphs[num_batches * batch_size :]

    @jax.jit
    def scan_fn(_: jax.Array, graphs: jax.Array):
        states = from_amat(graphs)
        indicators = get_indicators(states)
        log_rewards = logr(states)
        return _, (indicators, log_rewards)

    _, (indicators_scan, logrs_scan) = jax.lax.scan(
        scan_fn, init=None, xs=graphs_to_scan
    )
    indicators_scan = indicators_scan.reshape(-1, *indicators_scan.shape[2:])
    logrs_scan = logrs_scan.reshape(-1, *logrs_scan.shape[2:])

    _, (indicators_other, logrs_other) = scan_fn(None, other_graphs)

    indicators = jnp.concatenate([indicators_scan, indicators_other])
    logrs = jnp.concatenate([logrs_scan, logrs_other])

    log_z = jax.nn.logsumexp(logrs, axis=0)

    indicators = jnp.log(indicators) + (logrs[..., None, None] - log_z)
    indicators = jax.nn.logsumexp(indicators, axis=0)

    return jnp.exp(indicators)


def eval_marginal(
    gfn: jax.Array,
    state: AncestralGraph,
    logr: callable,
    graphs: list[jax.Array],
    key: jax.Array,
    num_rollouts: int = 100,
    true_graph: AncestralGraph = None,
    should_eval_gt: bool = True,
):
    if should_eval_gt:
        true_marginals = eval_true_marginal(logr, graphs)
    else:
        true_marginals = None

    # Compute estimated marginals
    states = rollout_many(gfn, state, key, num_rollouts=num_rollouts)
    indicators = jax.vmap(get_indicators, in_axes=(0,), out_axes=0)(states)
    indicators = indicators.reshape(-1, *indicators.shape[2:])
    estimated_marginals = indicators.mean(axis=0)

    # Compute, if true_graph is not None, the probability of its edges under the model
    if true_graph is not None:
        true_indicators = get_indicators(true_graph)
        true_indicators = true_indicators.squeeze(axis=0)
        jax.debug.print(
            "true graph {}",
            (true_indicators * estimated_marginals).sum(axis=1),
        )

    # This should be carefully architected
    jax.debug.print("true {}\nestimated {}", true_marginals, estimated_marginals)

    return true_marginals, estimated_marginals


def shd(graph: AncestralGraph, true_graph: AncestralGraph):
    dist = jnp.abs(graph.amat - true_graph.amat)
    return dist.sum(axis=(1, 2))
