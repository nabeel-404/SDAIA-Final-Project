# from typing import Dict, List


# class LLM:
#     def __init__(self) -> None:
#         pass

#     def answer(self, question: str, contexts: List[str]) -> Dict:
#         reasoning = [
#             "reasoning_step_1",
#             "reasoning_step_2",
#         ]
#         answer = (
#             "answer"
#         )
#         return {"answer": answer, "reasoning_steps": reasoning}


from typing import Dict, List
import ollama

class LLM:
    def __init__(self) -> None:
        pass

    #**tag** is used to highlight which parts of the context are relevant to the user’s question.
    #**Chain-of-thought** reasoning is used to derive the answer

    def answer(self, question: str, contexts: List[str]) -> Dict:
        # combine retrieved docs into a single context string
        context = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(contexts))

        prompt = f"""You are a reasoning QA agent. Use the following pieces (or multiple documents) of context to answer the question at the end. 
        If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.

        Passage:
        {context}

        First, **tag** up to K segments (by reference indices or labels) that are likely relevant to the question. (e.g. “Segment 3: …”, “Paragraph 5: …”)
        If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
        
        Then, use those tagged segments plus chain-of-thought reasoning to answer the question.

        Q: {question}
        A:
        """

        # Query the model
        response = ollama.chat(
            model='qwen3:0.6b',
            messages=[{'role': 'user', 'content': prompt}]
        )

        raw_answer = response['message']['content'].strip()
        
        # Extract reasoning from <think> tags and clean answer
        reasoning_steps = []
        if "<think>" in raw_answer and "</think>" in raw_answer:
            # Extract thinking content as reasoning steps
            think_start = raw_answer.find("<think>") + 7
            think_end = raw_answer.find("</think>")
            think_content = raw_answer[think_start:think_end].strip()
            
            # Split thinking into individual reasoning steps
            think_lines = [line.strip() for line in think_content.split('\n') if line.strip()]
            reasoning_steps.extend(think_lines)
            
            # Extract clean answer (part after </think>)
            answer_part = raw_answer[think_end + 8:].strip()
            clean_answer = answer_part.lstrip('\n')
        else:
            clean_answer = raw_answer

        return {"answer": clean_answer, "reasoning_steps": reasoning_steps}
   
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
    print(result["reasoning"])





