from agfn.modules.gflownet import GFlowNet
from agfn.modules.expert import Expert, ExpertType, Oracle, LLMOracle
from agfn.modules.agfn import AGFN, ElicitationStrategy

from agfn.policies import ForwardPolicyHITL, BackwardPolicyHITL
from agfn.gym import LogReward


def create_gflownet(
    num_nodes: int, hidden_dim: int, num_layers: int, device: str = "cpu"
) -> GFlowNet:
    pf = ForwardPolicyHITL(num_nodes, hidden_dim, num_layers, device=device)
    pb = BackwardPolicyHITL(device=device)

    gfn = GFlowNet(pf, pb, criterion="cb", device=device)

    return gfn


def create_expert(expert_type: ExpertType, **kwargs) -> Expert:
    match expert_type:
        case ExpertType.ORACLE:
            return Oracle(**kwargs)
        case ExpertType.LLM_ORACLE:
            return LLMOracle(**kwargs)


def create_agfn(
    expert_type: ExpertType,
    strategy: ElicitationStrategy,
    log_reward: LogReward,
    prior_pi: float = 0.95,
    gflownet_kwargs: dict = {},
    expert_kwargs: dict = {},
) -> AGFN:
    expert = create_expert(expert_type, **expert_kwargs)
    gflownet = create_gflownet(expert.num_nodes, **gflownet_kwargs)

    return AGFN(expert, gflownet, strategy, log_reward, prior_pi)
