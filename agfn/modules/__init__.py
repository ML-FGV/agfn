from agfn.modules.gflownet import GFlowNet
from agfn.modules.expert import (
    Expert,
    ExpertType,
    Oracle,
    LLMOracle,
    LLMWikipedia,
    MultiStageLLM,
    EnsembleLLM,
)
from agfn.modules.agfn import AGFN, ElicitationStrategy

from agfn.policies import ForwardPolicyHITL, BackwardPolicyHITL
from agfn.gym import LogReward


def create_gflownet(
    num_nodes: int,
    hidden_dim: int,
    num_layers: int,
    criterion: str = "cb",
    device: str = "cpu",
) -> GFlowNet:
    pf = ForwardPolicyHITL(num_nodes, hidden_dim, num_layers, device=device)
    pb = BackwardPolicyHITL(device=device)

    gfn = GFlowNet(pf, pb, criterion=criterion, device=device)

    return gfn


def create_expert(
    expert_type: ExpertType, with_confidence: bool = False, **kwargs
) -> Expert:
    match expert_type:
        case ExpertType.ORACLE:
            expert = Oracle(**kwargs)
        case ExpertType.LLM_ORACLE:
            expert = LLMOracle(**kwargs)
        case ExpertType.LLM_WIKIPEDIA:
            expert = LLMWikipedia(**kwargs)
    expert.set_with_confidence(with_confidence)

    return expert


def create_multi_stage_llm(base_expert: Expert, max_depth: int = 2) -> MultiStageLLM:
    return MultiStageLLM(base_expert, max_depth=max_depth)


def create_expert_ensemble(
    expert_type: ExpertType,
    with_confidence: bool = False,
    ensemble_size: int = 10,
    max_depth: int = 1,
    **kwargs,
) -> EnsembleLLM:
    def expert_constructor():
        base_expert = create_expert(expert_type, with_confidence, **kwargs)
        if max_depth < 1:
            return base_expert
        return create_multi_stage_llm(base_expert, max_depth=max_depth)

    return EnsembleLLM(expert_constructor, ensemble_size=ensemble_size)


def create_agfn(
    expert_type: ExpertType,
    strategy: ElicitationStrategy,
    log_reward: LogReward,
    expert_with_confidence: bool = False,
    prior_pi: float = 0.95,
    gflownet_kwargs: dict = {},
    expert_kwargs: dict = {},
) -> AGFN:
    expert = create_expert_ensemble(
        expert_type, expert_with_confidence, **expert_kwargs
    )
    gflownet = create_gflownet(log_reward.num_nodes, **gflownet_kwargs)

    return AGFN(gflownet, expert, strategy, log_reward, prior_pi)
