import dspy
import torch

from abc import ABC, abstractmethod
from typing import NamedTuple
from enum import Enum

from agfn.commons import EdgeType


class ExpertType(Enum):
    ORACLE = "oracle"
    LLM_ORACLE = "llm_oracle"


class FeedbackWithoutConfidenceSignature(dspy.Signature):
    query: tuple[str, str] = dspy.InputField(
        description=(
            "Pair of nodes about which we want to know the relationship (defined by EdgeType). "
            "Each pair of nodes may be directly associated, via direct causation, or indirectly associated, via latent confounding. "
            "The nature of each node and of the underlying problem is described in the context field."
        )
    )

    context: str = dspy.InputField(
        description=(
            "The context of the problem, including the nature of each node (represented by a string) "
            "and the nature of the kind of relationships we are looking for."
        )
    )

    relationship: EdgeType = dspy.OutputField(
        description=(
            "This is the feedback. It describes the nature of the relationship between the input nodes. "
            "This information should be inferred from the provided context, background knowledge, and the nature of the problem."
        )
    )


class FeedbackWithConfidenceSignature(FeedbackWithoutConfidenceSignature):
    confidence: float = dspy.OutputField(
        ge=0.05,
        le=0.95,
        multiple_of=0.05,
        description="This is the confidence factor for the feedback. It is a value between 0.05 and 0.95.",
    )


class LLMOutputField(NamedTuple):
    relationship: EdgeType
    confidence: float


class Expert(ABC):
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.node_pairs = torch.triu_indices(
            num_nodes, num_nodes, offset=1
        ).t()  # (num_node_pairs, 2)

        self.num_node_pairs = len(self.node_pairs)

    @abstractmethod
    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        pass

    @abstractmethod
    def get_feedback(self, query) -> EdgeType:
        pass


class Oracle(Expert):
    def __init__(
        self, num_nodes: int, adj: torch.Tensor, confidence_factor: float = 0.95
    ):
        super().__init__(num_nodes)
        self.adj = adj
        self.confidence_factor = confidence_factor

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        return self.get_feedback(query), self.confidence_factor

    def get_feedback(self, query) -> EdgeType:
        i, j = self.node_pairs[query]

        if self.adj[i, j] == 1 and self.adj[j, i] == 1:
            return EdgeType.LATENT_CONFOUNDING
        elif self.adj[i, j] == 1:
            return EdgeType.LEFT_CAUSES_RIGHT
        elif self.adj[j, i] == 1:
            return EdgeType.RIGHT_CAUSES_LEFT

        return EdgeType.NO_RELATIONSHIP


class LLMOracle(Expert):
    def __init__(
        self,
        num_nodes: int,
        adj: torch.Tensor,
        llm_name: str = "ollama_chat/llama3.2",
        api_base: str = "http://localhost:11434",
        api_key: str = "",
    ):
        super().__init__(num_nodes)

        self.adj = adj

        # Set the default language model for dspy.
        self.lm = dspy.LM(llm_name, temperature=0.1, api_base=api_base, api_key=api_key)
        dspy.configure(lm=self.lm)

        # Create the LLM agent.
        self.llm = dspy.ChainOfThought(FeedbackWithConfidenceSignature)

    def _get_prompt_context(self, i: int, j: int) -> str:
        if self.adj[i, j] == 1 and self.adj[j, i] == 1:
            relationship = f"We know that node {i} and node {j} are indirectly associated, via latent confounding."
        elif self.adj[i, j] == 1:
            relationship = f"We know that node {i} causes node {j}."
        elif self.adj[j, i] == 1:
            relationship = f"We know that node {j} causes node {i}."
        else:
            relationship = f"We know that node {i} and node {j} are not associated."

        return (
            "You are given a pair of nodes and a context. \n"
            "You need to return the nature of the relationship between the input nodes. \n"
            f"{relationship}"
        )

    def _query_llm(self, query: tuple[str, str]) -> LLMOutputField:
        i, j = self.node_pairs[query]
        context = self._get_prompt_context(i, j)
        return self.llm(context=context, query=(i, j))

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        feedback = self._query_llm(query)
        return feedback.relationship, feedback.confidence

    def get_feedback(self, query) -> EdgeType:
        return self._query_llm(query).relationship
