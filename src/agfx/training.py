import enum
import time
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tqdm

from agfx.gym import AncestralGraph
from agfx.policies import GFlowNet, rollout


class LossType(enum.Enum):
    TB = "tb"
    CB = "cb"

    def from_diff(self, loss: jax.Array):
        match self:
            case LossType.TB:
                return jnp.mean(loss**2)
            case LossType.CB:
                return jnp.var(loss)


def create_opt(
    gfn: GFlowNet,
    iterations: int,
    lr: float = 1e-2,
    logz_lr: float = 1e-1,
    should_clip: bool = False,
):
    params = nnx.state(gfn, nnx.Param)

    def label_params(path, _):
        if "logz" in path:
            return "logz"
        return "default"

    param_labels = nnx.map_state(label_params, params)

    clip_default = (
        optax.clip_by_global_norm(1.0) if should_clip else optax.identity()
    )

    schedule = optax.schedules.linear_schedule(
        init_value=lr, end_value=lr * 1e-2, transition_steps=iterations
    )

    tx = optax.multi_transform(
        {
            "logz": optax.chain(
                optax.adam(learning_rate=logz_lr),
            ),
            "default": optax.chain(
                clip_default, optax.contrib.muon(learning_rate=schedule)
            ),
        },
        param_labels,
    )

    return nnx.Optimizer(gfn, tx=tx, wrt=nnx.Param)


@partial(nnx.jit, static_argnames=("logr", "loss_t"))
def train_step(
    gfn: GFlowNet,
    state: AncestralGraph,
    key: jax.Array,
    logr: callable,
    opt: nnx.Optimizer,
    loss_t: LossType,
):
    def loss_fn(gfn: GFlowNet):
        new_state, new_key, flogits, blogits, _ = rollout(
            gfn, state, key, eps=1e-1
        )
        log_rewards = logr(new_state)
        loss = (flogits - blogits).sum(axis=0) - log_rewards + gfn.logz
        loss = loss_t.from_diff(loss)
        return loss, new_key

    (loss, key), grads = nnx.value_and_grad(
        f=loss_fn, argnums=0, has_aux=True
    )(gfn)
    opt.update(gfn, grads)

    return loss, key


def train(
    gfn: GFlowNet,
    state: AncestralGraph,
    key: jax.Array,
    logr: callable,
    iterations: int,
    loss_t: LossType,
    return_elapsed: bool = False,
    lr: float = 1e-3,
):
    opt = create_opt(gfn, iterations, lr=lr)

    # Compile the function prior to execution
    train_step_compiled = train_step.lower(
        gfn, state, key, logr, opt, loss_t
    ).compile()

    losses = []

    s = time.time()

    for _ in (pbar := tqdm.trange(iterations)):
        loss, key = train_step_compiled(gfn, state, key, opt)

        pbar.set_postfix(loss=f"{loss:.2e}")

        losses.append(loss)

    if return_elapsed:
        return gfn, losses, time.time() - s

    return gfn, losses
