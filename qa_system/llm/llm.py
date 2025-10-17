
from typing import Dict, List
import ollama
import re

class LLM:
    def __init__(self) -> None:
        pass


    def answer(self, question: str, contexts: List[str]) -> Dict:
        # combine retrieved docs into a single context string
        context = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(contexts))

        prompt= f"""
        You are an advanced RAG-based question-answering agent.
        Always reason step by step, using the retrieved evidence to support every statement.

        Question:
        {question}

        Retrieved Context:
        {context}

        Instructions:
        1. PLAN:
            - Break the question into smaller reasoning steps or sub-questions.
        2. RETRIEVE:
            - Use only the provided context to find the most relevant facts for each step.
        3. REASON:
            - Combine these facts logically to answer the main question.
            - If the evidence is insufficient, clearly say “INSUFFICIENT EVIDENCE.”
        4. ANSWER FORMAT:
            - Just state the answer 
            
        """

        # Query the model
        response = ollama.chat(
            model='qwen3:0.6b',
            messages=[{'role': 'user', 'content': prompt}]
        )

        
        # Get the full response content
        full_content = response['message']['content'].strip()
        
        
        # Extract LLM reasoning from the <think> tags or the thinking field if it exists

        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, full_content, re.DOTALL)
        reasoning = ""
        if 'thinking' in response['message'] and response['message']['thinking']:
            reasoning = response['message']['thinking'].strip()
        elif think_matches:
            reasoning = '\n'.join(think_matches).strip()
        
        # Remove <think> tags from the answer
        answer = re.sub(think_pattern, '', full_content, flags=re.DOTALL).strip()
        
        


        return {"answer": answer, "reasoning_steps": reasoning}
   
if __name__ == "__main__":
    retrieved_docs = [
        "Saudi National Day commemorates the unification of the Kingdom of Saudi Arabia by King Abdulaziz in 1932.",
        "It is celebrated annually on September 23 across Saudi Arabia.",
        "The day features national festivities, fireworks, and cultural events to honor the country's heritage.",
        "Government offices, schools, and many businesses close for the public holiday."
    ]
    user_query = "When is the Saudi National Day?"

    llm = LLM()
    result = llm.answer(user_query, retrieved_docs)

    print("Answer:")
    print(result["answer"], "\n")
    print("Reasoning:")
    print(result["reasoning_steps"])





