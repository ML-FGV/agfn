from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from mpl_preamble import *


def get_pi(pi_true: int, w: int):
    pi = jnp.ones((4,)) * (1 - pi_true) / 3
    pi = pi.at[w].set(pi_true)
    return pi


def sample(key: jax.Array, num_samples: int, *, pi: jax.Array):
    res = jax.random.categorical(key=key, logits=jnp.log(pi), shape=(num_samples,))
    _, key = jax.random.split(key)
    return key, res


def cumbincount(samples: jax.Array, size: int):
    onehot = jax.nn.one_hot(samples, num_classes=size)
    return jnp.cumsum(onehot, axis=0)


def _eval_posterior_pi_fixed(carry: None, counts: jax.Array, *, pi: jax.Array, w: int, num_samples: int):
    # Compute the posterior distribution
    pi_true = pi[w]
    pi_altn = (1 - pi_true) / 3

    log_posterior = counts * jnp.log(pi_true) + (num_samples - counts) * jnp.log(pi_altn)
    log_posterior = log_posterior - jax.nn.logsumexp(log_posterior, axis=0)
    return carry, log_posterior


def evaluate_posterior_pi_fixed(key: jax.Array, _, *, w: int, pi: jax.Array, num_samples: int):
    key, samples = sample(key, num_samples, pi=pi)

    counts = cumbincount(samples, len(pi))

    _, log_posterior = jax.lax.scan(
        f=partial(_eval_posterior_pi_fixed, pi=pi, w=w, num_samples=num_samples),
        init=None,
        xs=counts,
    )

    return key, log_posterior


def _eval_posterior_unknown_pi(carry: None, counts: jax.Array, *, alpha: float, beta: float, num_samples: int):
    alpha = alpha + counts
    beta = beta + counts.sum() - counts
    log_posterior = jsp.special.betaln(alpha, beta) + jnp.log(3) * (counts - num_samples)
    log_posterior = log_posterior - jax.nn.logsumexp(log_posterior, axis=0)
    return carry, log_posterior


def evaluate_posterior_pi_unknown(key: jax.Array, _, *, w: int, pi: jax.Array, num_samples: int):
    del w  # used for interface consistency

    key, samples = sample(key, num_samples=num_samples, pi=pi)

    counts = cumbincount(samples, len(pi))
    _, log_posterior = jax.lax.scan(
        f=partial(_eval_posterior_unknown_pi, alpha=1, beta=1, num_samples=num_samples),
        init=None,
        xs=counts,
    )

    return key, log_posterior


@partial(
    jax.jit,
    static_argnames=("func", "num_rep", "w", "num_samples"),
)
def eval_posterior_multiple_samples(
    carry: None,
    pi: jax.Array,
    *,
    func: callable,
    num_rep: int,
    w: int,
    num_samples: int,
):
    pi = get_pi(pi, w)
    key = jax.random.key(42)
    _, log_posterior = jax.lax.scan(
        f=partial(func, w=w, pi=pi, num_samples=num_samples),
        init=key,
        length=num_rep,
    )

    return carry, log_posterior


def eval_posterior_multiple_pi(
    func: callable,
    num_rep: int,
    w: int,
    pis: jax.Array,
    num_samples: int,
):
    _, log_posterior = jax.lax.scan(
        f=partial(eval_posterior_multiple_samples, func=func, num_rep=num_rep, w=w, num_samples=num_samples),
        init=None,
        xs=pis,
    )

    # We compute both the mean and standard deviations for each pi
    log_posterior_mu = log_posterior.mean(axis=1)
    log_posterior_std = log_posterior.std(axis=1)

    return (log_posterior_mu, log_posterior_std)


if __name__ == "__main__":
    w = 3

    num_samples = 10

    pi = 0.9

    _, log_posterior_pi_fixed = eval_posterior_multiple_samples(
        carry=None,
        pi=pi,
        func=evaluate_posterior_pi_fixed,
        num_rep=100,
        w=w,
        num_samples=num_samples,
    )

    _, log_posterior_pi_unknown = eval_posterior_multiple_samples(
        carry=None,
        pi=pi,
        func=evaluate_posterior_pi_unknown,
        num_rep=100,
        w=w,
        num_samples=num_samples,
    )

    fixed_mu = log_posterior_pi_fixed.mean(axis=0)
    unknown_mu = log_posterior_pi_unknown.mean(axis=0)

    fixed_std = log_posterior_pi_fixed.std(axis=0)
    unknown_std = log_posterior_pi_unknown.std(axis=0)

    plt.figure(figsize=(3, 2))

    xs = jnp.arange(num_samples)
    plt.plot(xs, fixed_mu[:, w], label="known $\pi$")
    plt.plot(xs, unknown_mu[:, w], label="unknown $\pi$")

    a = 0.2
    plt.fill_between(xs, fixed_mu[:, w] - a * fixed_std[:, w], fixed_mu[:, w] + a * fixed_std[:, w], alpha=0.2)
    plt.fill_between(xs, unknown_mu[:, w] - a * unknown_std[:, w], unknown_mu[:, w] + a * unknown_std[:, w], alpha=0.2)
    plt.xlabel("\# Feedbacks", size=12)
    plt.ylabel("$\log q_{r}(\omega^{\star} | \mathbf{f}_{r})$", size=12)

    plt.legend()

    plt.savefig("plot_posterior_for_pi.pdf", bbox_inches="tight")
