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
        candidates = self.retriever.retrieve(question, top_k=100)

        #reasoning_steps.append("")
        top_docs = self.reranker.rerank(question, candidates, top_k=20)

        contexts = [d.get("text", "") for d in top_docs]
        #reasoning_steps.append("")
        llm_out = self.llm.answer(question, contexts)
        
        # Add LLM reasoning steps to pipeline reasoning steps
        llm_reasoning = llm_out.get("reasoning_steps", [])
        reasoning_steps.extend(llm_reasoning)

        return {
            "answer": llm_out.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "contexts": top_docs,
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
    pipeline = QAPipeline(retriever=retriever, reranker=reranker, llm=llm)
    
    result = pipeline.answer_question(test_question)
    
    print("Our System Answer:", result["answer"])
    print("\nReasoning Steps:")
    for i, step in enumerate(result["reasoning_steps"], 1):
        print(f"{i}. {step}")
    
    print("\n" + "="*80)
    print("COMPARISON:")
    print(f"Ground Truth: {ground_truth}")
    print(f"Our Answer: {result['answer']}")
    