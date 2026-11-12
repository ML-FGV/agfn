import enum
import math

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp

from agfx.gym import AncestralGraph, _nil, get_indicators
from agfx.policies import GFlowNet, rollout_many


class QueryMethod(enum.Enum):
    RANDOM = "random"
    ENTROPY = "entropy"
    EDGES = "edges"


@struct.dataclass
class EIPolicyConfig:
    t: float = 1
    greedy: bool = False

    beta_rate: float = 0.9


class EIPolicy(nnx.Module):
    def __init__(
        self,
        num_nodes: int,
        pi: float = 0.9,
        maxsize: int = 200,
        config: EIPolicyConfig = EIPolicyConfig(),
    ):
        self.num_nodes = num_nodes
        self.num_pairs = math.comb(self.num_nodes, 2)
        self.num_relations = 4

        self.buffer_size = maxsize * self.num_pairs

        self.feedbacks = _nil * jnp.ones((self.buffer_size,), dtype=jnp.int32)
        self.relations = _nil * jnp.ones((self.buffer_size,), dtype=jnp.int32)

        self.pi = pi
        self.cursor = 0

        # Define the scheduling for the log-pooling
        self.beta = jnp.ones((self.num_pairs,))

        self.config = config

    def update(self, pair: jax.Array, feedback: jax.Array):
        # pair is the index of the pair, according to the lexicographic ordering
        self.feedbacks = self.feedbacks.at[self.cursor].set(feedback)
        self.relations = self.relations.at[self.cursor].set(pair)
        self.cursor = (self.cursor + 1) % (self.buffer_size)

        self.beta = self.beta.at[pair].set(
            min(0.5, self.beta[pair] * self.config.beta_rate)
        )

    def __call__(self, state: AncestralGraph, gfn: GFlowNet):
        # Compute the expert-induced posterior
        expert_posterior = get_posterior(self)  # (num_pairs, 4)
        expert_posterior = expert_posterior.flatten("F")  # (num_pairs * 4,)
        expert_posterior = expert_posterior[None, ...]

        # Compute the model's logits
        logits = gfn(state)
        logits = jnp.where(state.fmask == 1, logits, -jnp.inf)
        logits = nnx.log_softmax(logits, axis=1)  # (B, 4 * num_pairs)

        # Log-pool both policies
        beta = jnp.hstack([self.beta] * self.num_relations)[None]
        logits = beta * logits + (1 - beta) * expert_posterior
        logits = jnp.where(state.fmask == 1, logits, -jnp.inf)
        if self.config.greedy:
            logits = jnp.where(
                logits == jnp.max(logits, axis=1, keepdims=True), 1, -jnp.inf
            )
        else:
            logits = nnx.log_softmax(logits / self.config.t, axis=1)

        return logits

    def log_score(self, samples: AncestralGraph, logr: callable):
        logps, _ = mean_field_posterior(self, samples)
        logrs = logr(samples)
        return logps + logrs

    def set_config(self, config: EIPolicyConfig):
        self.config = config


# DAG, beta, neural network hyperparameters, evolution of some summary quantities, runtime


def get_posterior(pol: EIPolicy):
    # we first compute the sufficient statistics
    suff_stats = jnp.zeros((pol.num_pairs, pol.num_relations))
    suff_stats = suff_stats.at[pol.relations, pol.feedbacks].add(
        jnp.where(pol.feedbacks != _nil, 1, 0)
    )

    # the likelihood given j is pi ** suff_stats[j] * ((1 - pi) / 3) ** suff_stats[~j].sum()
    # the prior is uniform, not affecting the posterior
    log_posterior = suff_stats * jnp.log(pol.pi) + (
        suff_stats.sum(axis=1, keepdims=True) - suff_stats
    ) * (jnp.log(1 - pol.pi) - jnp.log(3))
    # Normalize the posterior distribution
    log_posterior = log_posterior - jax.nn.logsumexp(
        log_posterior, axis=1, keepdims=True
    )

    return log_posterior


def get_ei_distribution(
    pol: EIPolicy,
    gfn: GFlowNet,
    state: AncestralGraph,
    key: jax.Array,
    num_rollouts: int = 10,
):
    states = rollout_many(gfn, state, key, num_rollouts=num_rollouts, fpol=pol)
    indicators = jax.vmap(get_indicators, in_axes=(0,), out_axes=0)(states)
    empirical = indicators.mean(axis=(0, 1))  # (num_pairs, num_relations)
    return jnp.log(empirical + 1e-9)


def mean_field_posterior(pol: EIPolicy, state: AncestralGraph):
    log_posterior = get_posterior(pol)
    indicators = get_indicators(state)
    log_edges_posterior = jnp.log(indicators) + log_posterior[None, ...]
    log_graph_posterior = log_edges_posterior.max(axis=2)
    return log_graph_posterior, log_edges_posterior


def next_query(
    pol: EIPolicy,
    method: QueryMethod,
    key: jax.Array = None,
    gfn: GFlowNet = None,
    state: AncestralGraph = None,
    num_rollouts: int = 10,
):
    match method:
        case QueryMethod.RANDOM:
            key, subkey = jax.random.split(key, 2)
            return jax.random.randint(
                subkey, minval=0, maxval=pol.num_pairs, shape=(1,)
            ).squeeze(), key
        case QueryMethod.ENTROPY:
            expert_posterior = get_posterior(pol)
            # Compute the entropy
            entropy = -jnp.exp(expert_posterior) * expert_posterior
            entropy = entropy.sum(axis=1)  # (num_pairs,)
            return jnp.argmax(entropy), key
        case QueryMethod.EDGES:
            key, subkey = jax.random.split(key, 2)
            ei_distribution = get_ei_distribution(
                pol, gfn, state, subkey, num_rollouts
            )
            entropy = -jnp.exp(ei_distribution) * ei_distribution
            entropy = entropy.sum(axis=1)  # (num_pairs,)
            return jnp.argmax(entropy), key


def expert(true_graph: AncestralGraph, pi: float, rel: int, key: jax.Array):
    # This returns a sample from the queried variable
    true_indicators = get_indicators(true_graph)
    true_indicators = true_indicators.squeeze(axis=0)

    tir = true_indicators[rel]
    tinr = jnp.ones_like(tir) - tir
    p = pi * tir + (1 - pi) / 3 * tinr

    key, subkey = jax.random.split(key, 2)
    s = jax.random.categorical(subkey, logits=jnp.log(p))
    return s, key
