"""
Command line interface for the QA pipeline. 

This can be used to test the system without the UI.

"""
import sys
from typing import List

from qa_system.pipeline import QAPipeline



def run_cli(args: List[str]) -> None:
    if not args:
        print("Usage: python main.py 'Your question here'")
        sys.exit(1)
    question = " ".join(args)

    retriever = None
    reranker = None
    llm = None
    pipeline = QAPipeline(retriever=retriever, reranker=reranker, llm=llm)

    result = pipeline.answer_question(question)

    print("Question:", question)
    print("Answer:", result["answer"]) 
    print("Reasoning:")
    for step in result["reasoning_steps"]:
        print("-", step)


if __name__ == "__main__":
    run_cli(sys.argv[1:])

