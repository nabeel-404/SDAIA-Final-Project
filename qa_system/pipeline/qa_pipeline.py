# The orchestrator of the pipeline is done here
from typing import Dict, List

from qa_system.retrieval import Retriever
from qa_system.reranker import Reranker
from qa_system.llm import LLM
from qa_system.query_rewriter.rewriter import QueryRewriter





class QAPipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker, llm: LLM, query_rewriter: QueryRewriter = None) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.query_rewriter = query_rewriter

    def answer_question(self, question: str) -> Dict:
        reasoning_steps: List[str] = []

        rewritten_queries = [question]

        # Step 1: Query Rewriting (if available)
        if self.query_rewriter:
            reasoning_steps.append("Rewriting query for better retrieval...")
            rewritten_queries.extend(self.query_rewriter.rewrite_query(question))
            reasoning_steps.append(f"Generated {len(rewritten_queries)} query variations")
            
        

        # Step 2: Retrieval with multiple queries
        reasoning_steps.append("Retrieving relevant documents...")
        if len(rewritten_queries) > 1:
            candidates = self.retriever.retrieve_multiple(rewritten_queries, top_k=50)
        else:
            candidates = self.retriever.retrieve(question, top_k=50)

        # Step 3: Reranking
        reasoning_steps.append("Reranking retrieved documents...")
        top_docs = self.reranker.rerank(question, candidates, top_k=10)

        # Step 4: LLM Answer Generation
        reasoning_steps.append("Generating answer with LLM...")
        contexts = [d.get("text", "") for d in top_docs]
        llm_out = self.llm.answer(question, contexts)
        
        # Add LLM reasoning steps to pipeline reasoning steps
        llm_reasoning = llm_out.get("reasoning_steps", []) 
        reasoning_steps.append(llm_reasoning)

        return {
            "answer": llm_out.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "contexts": top_docs,
            "rewritten_queries": rewritten_queries if self.query_rewriter else [question],
        }



if __name__ == "__main__":
    import json
    import random
    
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
    