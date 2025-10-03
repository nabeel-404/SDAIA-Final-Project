from typing import Dict, List


class Reranker:
    def __init__(self) -> None:
        pass

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        return docs[:top_k]

