from typing import Dict, List


class LLM:
    def __init__(self) -> None:
        pass

    def answer(self, question: str, contexts: List[str]) -> Dict:
        reasoning = [
            "reasoning_step_1",
            "reasoning_step_2",
        ]
        answer = (
            "answer"
        )
        return {"answer": answer, "reasoning_steps": reasoning}

