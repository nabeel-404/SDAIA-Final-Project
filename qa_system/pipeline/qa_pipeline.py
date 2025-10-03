# The orchestrator of the pipeline is done here
from typing import Dict, List
from qa_system.retrieval import Retriever
from qa_system.reranker import Reranker
from qa_system.llm import LLM





class QAPipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker, llm: LLM) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def answer_question(self, question: str) -> Dict:
        reasoning_steps: List[str] = []

        #reasoning_steps.append("")
        candidates = self.retriever.retrieve(question, top_k=20)

        #reasoning_steps.append("")
        top_docs = self.reranker.rerank(question, candidates, top_k=5)

        contexts = [d.get("text", "") for d in top_docs]
        #reasoning_steps.append("")
        llm_out = self.llm.answer(question, contexts)

        return {
            "answer": llm_out.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "contexts": top_docs,
        }

