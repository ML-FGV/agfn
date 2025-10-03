import dspy
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import NamedTuple
from enum import Enum

from agfn.commons import EdgeType


class ExpertType(Enum):
    ORACLE = "oracle"
    LLM_ORACLE = "llm_oracle"
    LLM_ORACLE_SACHS = "llm_oracle_sachs"


class FeedbackWithoutConfidenceSignature(dspy.Signature):
    query: tuple[str, str] = dspy.InputField(
        description=(
            "Pair of nodes about which we want to know the relationship (defined by EdgeType). "
            "Each pair of nodes may be directly associated, via direct causation, or indirectly associated, via latent confounding. "
            "The nature of each node and the underlying problem is described in the context field."
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

        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )

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
        error = True
        while error:
            try: 
                llm_response = self.llm(context=context, query=(i, j))
            except:
                print("LLM parsing error, querying again")
            else:
                error = False
        return llm_response

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        #feedback = self._query_llm(query)

        t = self.node_pairs
        
        i = int(t[query][0])
        j = int(t[query][1])

        #llm_edge = feedback.relationship.value
        is_llm_correct = 0
        real_edge = 0


        if true_sachs5[i][j] == true_sachs5[j][i] == 0:
            real_edge = 0
            #if llm_edge == 0:
                #is_llm_correct = 1
        elif true_sachs5[i][j] == 1 and true_sachs5[j][i] == 0:
            real_edge = 1
            #if llm_edge == 1:
                #is_llm_correct = 1
        elif true_sachs5[i][j] == 0 and true_sachs5[j][i] == 1:
            real_edge = 2
            #if llm_edge == 2:
                #is_llm_correct = 1
        elif true_sachs5[i][j] == true_sachs5[j][i] == 1:
            real_edge = 3
            #if llm_edge == 3:
                #is_llm_correct = 1
        #return feedback.relationship, feedback.confidence, is_llm_correct
        print(EdgeType(real_edge))
        return EdgeType(real_edge), 0.95, 1

    def get_feedback(self, query) -> EdgeType:
        return self._query_llm(query).relationship

with open("openai_key.txt", 'r') as file:
    key = file.read()
    
class LLMOracleSachs(Expert):
    def __init__(
        self,
        num_nodes: int,
        adj: torch.Tensor,
        llm_name: str = "openai/gpt-4o",
        #llm_name: str = "ollama_chat/llama3.2",
        #api_base: str = "http://localhost:11434",
        api_key: str = key,
    ):
        super().__init__(num_nodes)

        self.adj = adj

        # Set the default language model for dspy.
        #self.lm = dspy.LM(llm_name, temperature=0.1, api_base=api_base, api_key=api_key)
        self.lm = dspy.LM(llm_name, temperature=0.1, api_key=api_key)
        dspy.configure(lm=self.lm)

        # Create the LLM agent.
        #self.llm = dspy.ChainOfThought(FeedbackWithConfidenceSignature)
        self.llm = dspy.ChainOfThought(FeedbackWithoutConfidenceSignature)

        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )

    def _get_prompt_context(self, i: int, j: int) -> str:

        return (
            'You are an expert on the human immune system cell. You are investigating the cause-and-effect relationships between a specific set of  observed variables representing proteins and phospholipids: "Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk". Your task is to determine the causal relationship between the given variables. If there are both indirect and direct relationships between the variables, you should describe only the direct one.'
        )


    def _query_llm(self, query: tuple[str, str]) -> LLMOutputField:
        i, j = self.node_pairs[query]
        context = self._get_prompt_context(i, j)
        error = True
        while error:
            try: 
                llm_response = self.llm(context=context, query=(i, j))
            except:
                print("LLM parsing error, querying again")
            else:
                error = False
        #llm_response = self.llm(context=context, query=(i, j))
        #print(llm_response.relationship.value)
        return llm_response
        #return self.llm(context=context, query=(i, j))

    def get_feedback_with_confidence(self, query) -> tuple[EdgeType, float]:
        feedback = np.array([0,0,0,0])
        for i in range(10):
            r = self._query_llm(query).relationship.value
            feedback[r] += 1
        feedback = feedback/11
        #print(self._query_llm(query).relationship)
        
        t = self.node_pairs
        
        i = int(t[query][0])
        j = int(t[query][1])

        llm_edge = feedback.argmax()
        is_llm_correct = 0
        real_edge = 0

        
        #print(f'Query: {query}, LLM Edge: {feedback.argmax()}, Uncert.: {max(feedback)}, Real edge: {real_edge}')
        return EdgeType(feedback.argmax()), max(feedback)#, is_llm_correct

    def get_feedback(self, query) -> EdgeType:
        return self._query_llm(query).relationship
