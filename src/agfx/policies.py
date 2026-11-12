import math
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from agfx.gym import AncestralGraph, fapply

_default_initializer = nnx.initializers.lecun_uniform()


def forward_fn(y: jax.Array, layer: nnx.Linear):
    y = nnx.leaky_relu(y)
    y = layer(y)
    return y, None


def forward(x: jax.Array, layers: list[nnx.Linear], nlayers: int):
    y, _ = jax.lax.scan(
        f=forward_fn,
        init=x,
        xs=layers,
        length=nlayers,
    )
    return y


class GFlowNet(nnx.Module):
    def __init__(
        self, num_nodes: int, dmid: int, nlayers: int = 2, *, seed: int = 43
    ):
        rngs = nnx.Rngs(seed)

        self.din = num_nodes * num_nodes
        self.dmid = dmid
        self.dout = 4 * math.comb(num_nodes, 2)
        self.nlayers = nlayers

        self.logz = nnx.Param(jnp.array(0.0))

        self.linear_in = nnx.Linear(
            self.din, dmid, rngs=rngs, kernel_init=_default_initializer
        )

        @nnx.split_rngs(splits=nlayers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: nnx.Rngs):
            return nnx.Linear(
                dmid, dmid, rngs=rngs, kernel_init=_default_initializer
            )

        self.layers = create_layer(rngs)
        self.linear_out = nnx.Linear(
            dmid, self.dout, rngs=rngs, kernel_init=_default_initializer
        )

    def __call__(self, x: AncestralGraph):
        y = x.value
        y = y.reshape(y.shape[0], -1)
        y = self.linear_in(y)
        y = forward(y, self.layers, self.nlayers)
        logits = self.linear_out(nnx.leaky_relu(y))  # (B, dout)
        logits = nnx.log_softmax(logits, axis=1)
        return logits


def isclose(eps: float, val: float, atol: float = 1e-6):
    return abs(eps - val) < atol


def eps_greedy(logits: jax.Array, eps: float, mask: jax.Array):
    ulogits = jnp.where(mask, 1, -jnp.inf)
    ulogits = nnx.log_softmax(ulogits, axis=-1)

    if isclose(eps, 0.0):
        return logits

    if isclose(eps, 1.0):
        return ulogits

    logits_wts = jnp.array([jnp.log(1 - eps), jnp.log(eps)])[:, None, None]
    logits_mix = jnp.stack([logits, ulogits], axis=0)
    logits = jax.nn.logsumexp(logits_mix + logits_wts, axis=0)

    return logits


def take(x: jax.Array, i: jax.Array):
    # fill_value=0 ensures this behaves gracefully when there is no bijection between
    # forward and backward actions; ideally, this would be encoded into `i`
    return jnp.take_along_axis(x, i[:, None], axis=1, fill_value=0).squeeze(1)


def sample_with_mask(
    gfn: GFlowNet,
    state: AncestralGraph,
    key: jax.Array,
    eps: float,
    mask: jax.Array,
    fpol: callable,
):
    logits = fpol(state, gfn)

    slogits = eps_greedy(logits, eps, mask=mask)

    key, subkey = jax.random.split(key, 2)

    actions = jax.random.categorical(key=subkey, logits=slogits, axis=-1)
    logits = take(logits, actions)
    return logits, actions, key


def bpol(state: AncestralGraph, _: GFlowNet = None):
    logits = jnp.where(state.bmask == 1, 1, -jnp.inf)
    logits = nnx.log_softmax(logits, axis=1)
    return logits


def fpol(state: AncestralGraph, gfn: GFlowNet):
    logits = gfn(state)
    logits = jnp.where(state.fmask == 1, logits, -jnp.inf)
    logits = nnx.log_softmax(logits, axis=-1)
    return logits


def fstep(
    carry: tuple[AncestralGraph, jax.Array],
    _,
    gfn: GFlowNet,
    fpol: callable,
    eps: float = 0.0,
):
    state, key = carry
    factive = ~state.fstopped

    # Sample actions
    flogits, actions, key = sample_with_mask(
        gfn, state, key, mask=state.fmask, eps=eps, fpol=fpol
    )
    bactions = actions % state.num_pairs

    # Move to the forward state
    fstate = fapply(state, actions)

    # Compute the backward (uniform) probabilities
    blogits = bpol(fstate)
    blogits = take(blogits, bactions)

    flogits = jnp.where(factive, flogits, 0)
    blogits = jnp.where(factive, blogits, 0)

    return (fstate, key), (flogits, blogits, actions)


def rollout(
    gfn: GFlowNet,
    state: AncestralGraph,
    key: jax.Array,
    eps: float = 0.0,
    fpol: callable = fpol,
):
    (new_state, new_key), (flogits, blogits, actions) = jax.lax.scan(
        f=partial(fstep, gfn=gfn, eps=eps, fpol=fpol),
        init=(state, key),
        length=state.L,
    )
    return new_state, new_key, flogits, blogits, actions


@partial(jax.jit, static_argnames=("eps", "num_rollouts"))
def rollout_many(
    gfn: GFlowNet,
    state: AncestralGraph,
    key: jax.Array,
    eps: float = 0.0,
    num_rollouts: int = 10,
    fpol: callable = fpol,
):
    def rollout_batch(key: jax.Array, _):
        new_state, new_key, *_ = rollout(gfn, state, key, eps, fpol=fpol)
        return new_key, new_state

    _, states = jax.lax.scan(rollout_batch, init=key, length=num_rollouts)
    return states
