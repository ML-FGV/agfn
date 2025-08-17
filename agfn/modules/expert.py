import dspy
import torch
import threading

from abc import abstractmethod
from typing import NamedTuple, Optional, Callable
from enum import Enum
from collections import Counter

from dspy.adapters.json_adapter import AdapterParseError

from agfn.commons import EdgeType


class ExpertType(Enum):
    ORACLE = "oracle"
    LLM_ORACLE = "llm_oracle"
    LLM_WIKIPEDIA = "llm_wikipedia"


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
            "This information should be inferred from the provided context, background knowledge, and the nature of the problem. \n"
            "EdgeType.LEFT_CAUSES_RIGHT means that node query.0 directly causes node query.1. \n"
            "EdgeType.RIGHT_CAUSES_LEFT means that node query.1 directly causes node query.0. \n"
            "EdgeType.LATENT_CONFOUNDING means that node query.0 and node query.1 are indirectly associated, via latent confounding. \n"
            "EdgeType.NO_RELATIONSHIP means that node query.0 and node query.1 have no relationship (no edge between them)."
        )
    )

    reasoning: str = dspy.OutputField(
        description="This is the reasoning process that led to the feedback. "
        "It is a string that describes the reasoning process that led to the feedback."
    )


class FeedbackWithConfidenceSignature(FeedbackWithoutConfidenceSignature):
    confidence: float = dspy.OutputField(
        ge=0.05,
        le=0.95,
        multiple_of=0.05,
        description="This is the confidence factor for the feedback. It is a value between 0.05 and 0.95.",
    )


class JudgementSignatureWithoutConfidence(FeedbackWithoutConfidenceSignature):
    tentative_relationship: str = dspy.InputField(
        description="This is the tentative relationship between the input nodes proposed by an LLM. "
        "It is a value that indicates the relationship between the input nodes. "
    )
    tentative_reasoning: str = dspy.InputField(
        description="This is the reasoning process that led to the tentative relationship proposed by an LLM. "
        "It is a string that describes the reasoning process that led to the tentative relationship."
    )

    judgement: bool = dspy.OutputField(
        description="This is the judgement of the relationship between the input nodes. "
        "It is a boolean value that indicates whether the relationship is correct or not."
    )


class JudgementSignatureWithConfidence(JudgementSignatureWithoutConfidence):
    tentative_confidence: float = dspy.InputField(
        description="This is the confidence factor for the tentative relationship. It is a value between 0.05 and 0.95.",
    )

    confidence: float = dspy.OutputField(
        ge=0.05,
        le=0.95,
        multiple_of=0.05,
        description="This is the confidence factor for the judgement. It is a value between 0.05 and 0.95.",
    )


class LLMOutputField(NamedTuple):
    relationship: EdgeType
    confidence: float
    reasoning: str


class ExpertResponse(NamedTuple):
    relationship: EdgeType
    confidence: Optional[float] = None


class Expert(dspy.Module):
    def __init__(self, num_nodes: int, with_confidence: bool = False):
        self.num_nodes = num_nodes
        self.node_pairs = torch.triu_indices(
            num_nodes, num_nodes, offset=1
        ).t()  # (num_node_pairs, 2)

        self.num_node_pairs = len(self.node_pairs)
        self.with_confidence = with_confidence

        self.context_cache = {}

    @abstractmethod
    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        pass

    @abstractmethod
    def get_feedback(self, query) -> EdgeType:
        pass

    def forward(self, query) -> ExpertResponse:
        if self.with_confidence:
            response = self.get_feedback_with_confidence(query)
            return ExpertResponse(relationship=response[0], confidence=response[1])
        else:
            return ExpertResponse(relationship=self.get_feedback(query))

    def __call__(self, *args, **kwargs) -> ExpertResponse:
        return super().__call__(*args, **kwargs)

    def set_with_confidence(self, with_confidence: bool):
        self.with_confidence = with_confidence

    def get_context(self, node_i: int, node_j: int) -> str:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def set_context(self, context: str, node_i: int, node_j: int):
        self.context_cache[(node_i, node_j)] = context

    def query_expert(self, query: int) -> LLMOutputField:
        raise NotImplementedError("This method should be implemented by the subclass.")


class Oracle(Expert):
    def __init__(
        self,
        num_nodes: int,
        adj: torch.Tensor,
        confidence_factor: float = 0.95,
        **kwargs,
    ):
        super().__init__(num_nodes)
        del kwargs
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
        **kwargs,
    ):
        super().__init__(num_nodes)

        del kwargs
        self.adj = adj

        # Set the default language model for dspy.
        self.lm = dspy.LM(llm_name, temperature=0.1, api_base=api_base, api_key=api_key)
        dspy.configure(lm=self.lm)

        # Create the LLM agent.
        self.expert = dspy.ChainOfThought(FeedbackWithConfidenceSignature)

    def get_context(self, i: int, j: int) -> str:
        if (i, j) in self.context_cache:
            return self.context_cache[(i, j)]

        if self.adj[i, j] == 1 and self.adj[j, i] == 1:
            relationship = f"We know that node {i} and node {j} are indirectly associated, via latent confounding."
        elif self.adj[i, j] == 1:
            relationship = f"We know that node {i} causes node {j}."
        elif self.adj[j, i] == 1:
            relationship = f"We know that node {j} causes node {i}."
        else:
            relationship = f"We know that node {i} and node {j} have no relationship."

        context = (
            "You are given a pair of nodes and a context. \n"
            "You need to return the nature of the relationship between the input nodes. \n"
            f"{relationship}"
        )
        self.set_context(context, i, j)
        return context

    def query_expert(self, query: int) -> LLMOutputField:
        i, j = self.node_pairs[query]
        context = self.get_context(i, j)
        return self.expert(context=context, query=(str(i), str(j)))

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        feedback = self.query_expert(query)
        return feedback.relationship, feedback.confidence

    def get_feedback(self, query) -> EdgeType:
        return self.query_expert(query).relationship


class LLMWikipedia(Expert):
    def __init__(
        self,
        num_nodes: int,
        url: str = "http://20.102.90.50:2017/wiki17_abstracts",
        **kwargs,
    ):
        super().__init__(num_nodes)
        del kwargs
        self.url = url

        self.ColBERTv2 = dspy.ColBERTv2(url=url)
        self.expert = dspy.ChainOfThought(FeedbackWithConfidenceSignature)

    def get_context(self, i: int, j: int) -> str:
        question = f"What is the relationship between {i} and {j}?"
        rag_response = self.ColBERTv2(question)
        return "\n".join(rag_response)

    def query_expert(self, query: int) -> LLMOutputField:
        i, j = self.node_pairs[query]
        context = self.get_context(i, j)
        return self.expert(context=context, query=(str(i), str(j)))

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        response = self.query_expert(query)
        return response.relationship, response.confidence

    def get_feedback(self, query) -> EdgeType:
        response = self.query_expert(query)
        return response.relationship


class MultiStageLLM(Expert):
    def __init__(self, base_expert: Expert, max_depth: int = 2, **kwargs):
        self.base_expert = base_expert
        self.judger = dspy.ChainOfThought(JudgementSignatureWithConfidence)

        # This is the depth of the recursive self-improvement process.
        self.max_depth = max_depth
        self.with_confidence = self.base_expert.with_confidence
        assert self.max_depth > 0, "The maximum depth must be greater than 0."

    def _get_judgement_context(self, node_i: int, node_j: int) -> str:
        context = self.base_expert.get_context(node_i, node_j)
        return (
            "You are given a pair of nodes and a reasoning process. \n"
            "The nature on the relationship between the input nodes has been proposed by an LLM. \n"
            "Your goal is to critically assess the proposal and provide a judgement. \n"
            "In case the judgement is False, you should provide a new proposal. \n"
            f"*Context*\n: {context}"
        )

    def _query_judger(
        self,
        query: int,
        depth: int = 0,
        tentative_relationship: EdgeType = None,
        tentative_confidence: float = None,
        tentative_reasoning: str = None,
    ) -> JudgementSignatureWithConfidence:
        node_i, node_j = self.base_expert.node_pairs[query]
        node_i, node_j = node_i.item(), node_j.item()
        if depth == 0:
            # Query the proposer.
            proposed_answer = self.base_expert.query_expert(query)
            return self._query_judger(
                query,
                depth + 1,
                proposed_answer.relationship,
                proposed_answer.confidence,
                proposed_answer.reasoning,
            )

        context = self._get_judgement_context(node_i, node_j)
        tentative_relationship_str = tentative_relationship.unwrap_into_str(
            node_i, node_j
        )

        try:
            response = self.judger(
                context=context,
                query=(str(node_i), str(node_j)),
                tentative_relationship=tentative_relationship_str,
                tentative_confidence=tentative_confidence,
                tentative_reasoning=tentative_reasoning,
            )
        except (AdapterParseError, ValueError, RuntimeError) as e:
            print(
                f"Error parsing the response from the LLM. Returning the tentative relationship. Error: {e}"
            )
            return JudgementSignatureWithConfidence(
                query=(str(node_i), str(node_j)),
                context=context,
                tentative_relationship=tentative_relationship_str,
                tentative_confidence=tentative_confidence,
                tentative_reasoning=tentative_reasoning,
                judgement=True,
                relationship=tentative_relationship,
                confidence=tentative_confidence,
                reasoning=tentative_reasoning,
            )

        # If the depth is the maximum depth, return the response.
        if depth == self.max_depth:
            return response

        # If the depth is not the maximum depth, query the judger again.
        return self._query_judger(
            query,
            depth + 1,
            response.relationship,
            response.confidence,
            response.reasoning,
        )

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        judgement = self._query_judger(query)
        return judgement.relationship, judgement.confidence

    def get_feedback(self, query) -> EdgeType:
        judgement = self._query_judger(query)
        return judgement.relationship


class EnsembleLLM(Expert):
    experts: list[Expert]

    def __init__(
        self,
        expert_constructor: Callable[[], Expert],
        ensemble_size: int = 10,
        **kwargs,
    ):
        del kwargs

        self.ensemble_size = ensemble_size
        self.experts = [expert_constructor() for _ in range(ensemble_size)]

        self.response_by_expert_id = {}
        self.with_confidence = all(expert.with_confidence for expert in self.experts)

    def majority_vote(
        self, responses: list[EdgeType], confidences: list[float]
    ) -> EdgeType:
        # Select the response with the highest count
        counts = Counter(responses)
        response_with_highest_count = max(counts, key=counts.get)
        # Compute the average confidence of the responses with the highest count

        if any(confidence is None for confidence in confidences):
            return response_with_highest_count, None

        confidences_with_highest_count = [
            confidence
            for response, confidence in zip(responses, confidences)
            if response == response_with_highest_count
        ]
        confidence = (
            sum(confidences_with_highest_count) / counts[response_with_highest_count]
        )

        return response_with_highest_count, confidence

    def _get_feedback_with_confidence(self, query, expert_id: int):
        self.response_by_expert_id[expert_id] = self.experts[
            expert_id
        ].get_feedback_with_confidence(query)

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        threads = []
        for i in range(self.ensemble_size):
            t = threading.Thread(
                target=self._get_feedback_with_confidence, args=(query, i)
            )
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        responses = [self.response_by_expert_id[i] for i in range(self.ensemble_size)]
        edge_types, confidences = zip(*responses)
        return self.majority_vote(edge_types, confidences)

    def _get_feedback(self, query, expert_id: int):
        self.response_by_expert_id[expert_id] = self.experts[expert_id].get_feedback(
            query
        )

    def get_feedback(self, query) -> EdgeType:
        threads = []
        for i in range(self.ensemble_size):
            t = threading.Thread(target=self._get_feedback, args=(query, i))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        edge_types = [self.response_by_expert_id[i] for i in range(self.ensemble_size)]
        edge_type_majority_vote, _ = self.majority_vote(edge_types)
        return edge_type_majority_vote
