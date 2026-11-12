import flax.struct as struct
import jax
import typer

from agfx.evals import eval_marginal
from agfx.gym import (
    AncestralGraph,
    create_state,
    enumerate_ancestral_graphs,
    from_amat,
)
from agfx.policies import GFlowNet
from agfx.rewards import LogRewardType, create_log_reward
from agfx.special.linear_gaussian import sample_linear_gaussian
from agfx.training import LossType, train

app = typer.Typer()


@struct.dataclass
class RunOutput:
    s_o: AncestralGraph
    gfn: GFlowNet


@app.command()
def main(
    num_nodes: int,
    logr_t: LogRewardType = LogRewardType.UNIFORM,
    dmid: int = 64,
    bs: int = 64,
    seed: int = 42,
    iterations: int = 512,
    loss_t: LossType = LossType.TB,
    only_dags: bool = False,
):

    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    # Create the initial state
    state = create_state(bs, num_nodes, has_confounding=not only_dags)
    # Create the GFlowNet
    gfn = GFlowNet(num_nodes, dmid)

    # Simulate the data (which is needed for certain reward functions)
    data, dir_amat, bidir_amat, *_ = sample_linear_gaussian(
        k1, num_nodes, num_samples=600
    )
    true_graph = from_amat((dir_amat + bidir_amat)[None])

    # Create the reward function
    logr = create_log_reward(logr_t, data, true_graph)

    gfn, losses = train(gfn, state, k2, logr, iterations, loss_t)

    all_graphs = enumerate_ancestral_graphs(
        num_nodes, has_confounding=not only_dags
    )

    print(len(all_graphs))

    eval_marginal(gfn, state, logr, all_graphs, k3, true_graph=true_graph)

    return gfn


if __name__ == "__main__":
    app()
