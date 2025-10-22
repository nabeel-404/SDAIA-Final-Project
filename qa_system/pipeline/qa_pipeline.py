# The orchestrator of the pipeline is done here
from typing import Dict, List

from qa_system.retrieval import Retriever
from qa_system.reranker import Reranker
from qa_system.llm import LLM
from qa_system.query_rewriter.rewriter import QueryRewriter
from qa_system.utils import Settings


class QAPipeline:
    def __init__(self, retriever: Retriever = None, reranker: Reranker = None, llm: LLM = None, query_rewriter: QueryRewriter = None) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.cfg = Settings()

    def answer_question(self, question: str) -> Dict:
        reasoning_steps: List[str] = []
        rewritten_queries = [question]
        top_docs = []
        
        # No Retriever configuration (Direct LLM only)
        if not self.retriever:
            reasoning_steps.append("Using direct LLM without retrieval...")
            llm_out = self.llm.answer(question, contexts=[])
            llm_reasoning = llm_out.get("reasoning_steps", [])
            reasoning_steps.extend(llm_reasoning)
            
            return {
                "answer": llm_out.get("answer", ""),
                "reasoning_steps": reasoning_steps,
                "contexts": [],
                "rewritten_queries": [question],
            }

        # Step 1: Query Rewriting (if available)
        if self.query_rewriter:
            reasoning_steps.append("Rewriting query for better retrieval...")
            rewritten_queries.extend(self.query_rewriter.rewrite_query(question))
            reasoning_steps.append(f"Generated {len(rewritten_queries)} query variations")     
        # Step 2: Retrieval with multiple queries
        reasoning_steps.append("Retrieving relevant documents...")
        candidates = self.retriever.retrieve_multiple(rewritten_queries, top_k=self.cfg.retrieval_top_k)

        # Step 3: Reranking (if available)
        if not self.reranker:
            reasoning_steps.append("Using retrieved documents without reranking...")
            top_docs = candidates[:self.cfg.rerank_top_k]  # Take top k from retrieval
        else:
            reasoning_steps.append("Reranking retrieved documents...")
            top_docs = self.reranker.rerank(question, candidates, top_k=self.cfg.rerank_top_k)

        # Step 4: LLM Answer Generation
        reasoning_steps.append("Generating answer with LLM...")
        contexts = [d.get("text", "") for d in top_docs]
        llm_out = self.llm.answer(question, contexts)
        
        
        llm_reasoning = llm_out.get("reasoning_steps", []) 
        reasoning_steps.append(llm_reasoning)

        return {
            "answer": llm_out.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "contexts": top_docs,
            "rewritten_queries": rewritten_queries if self.query_rewriter else [question],
        }
    
def build_pipeline(use_rewriter: bool = True) -> "QAPipeline":
    retriever = Retriever()
    reranker = Reranker()
    llm = LLM()
    qr = QueryRewriter() if use_rewriter else None
    return QAPipeline(retriever=retriever, reranker=reranker, llm=llm, query_rewriter=qr)



if __name__ == "__main__":
    import json
    import random
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Load the dataset and pick a random question
    with open('qa_system/data/hotpot_dev_distractor_v1.json', 'r') as f:
        data = json.load(f)
    
    # Pick a random question
    random_sample = random.choice(data)
    test_question = random_sample['question']
    ground_truth = random_sample['answer']
    
    print(f"Question: {test_question}")
    print(f"Ground Truth Answer: {ground_truth}")
    print("\n" + "="*80)
    
    
    retriever = Retriever()
    reranker = Reranker()
    llm = LLM()
    query_rewriter = QueryRewriter()
    pipeline = QAPipeline(retriever=retriever, reranker=reranker, llm=llm, query_rewriter=query_rewriter)
    
    result = pipeline.answer_question(test_question)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
   
    print("Our System Answer:", result["answer"])
    print("\nQuery Variations Used:")
    for i, query in enumerate(result["rewritten_queries"], 1):
        print(f"{i}. {query}")
    
    print("\nReasoning Steps:")
    for i, step in enumerate(result["reasoning_steps"], 1):
        print(f"{i}. {step}")
    
    print("\n" + "="*80)
    print("COMPARISON:")
    print(f"Ground Truth: {ground_truth}")
    print(f"Our Answer: {result['answer']}")
    