# reranker.py
from typing import Dict, List, Tuple, Optional
from sentence_transformers import CrossEncoder
import torch


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
    Accepts a list of docs (dicts with at least 'text'/'chunk') and returns
    the same docs sorted by reranker_score (descending).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 16,
        fp16: bool = True,
        max_len: Optional[int] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"[Reranker] Loading {model_name} on {self.device}...")
        self.reranker = CrossEncoder(model_name, device=self.device, max_length=max_len)

        # Optional mixed precision for faster inference
        if fp16 and self.device.startswith("cuda"):
            self.reranker.model.half()
            print("[Reranker] Using half precision (fp16)")

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute relevance scores for (query, passage) pairs."""
        try:
            scores = self.reranker.predict(pairs, batch_size=self.batch_size)
            return [float(s) for s in scores]
        except RuntimeError as e:
            print(f"[Reranker] Runtime error: {e}")
            if "CUDA" in str(e):
                torch.cuda.empty_cache()
            raise

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        """Attach reranker scores and return docs sorted by them."""
        if not docs:
            print("[Reranker] Warning: received empty doc list.")
            return []

        # Extract valid text fields
        texts, idxs = [], []
        for i, d in enumerate(docs):
            t = d.get("text") or d.get("chunk") or d.get("content")
            if isinstance(t, str) and t.strip():
                texts.append(t)
                idxs.append(i)

        if not texts:
            print("[Reranker] Warning: no valid text fields found.")
            return docs[:top_k]

        # Build (query, doc_text) pairs
        pairs = [(query, t) for t in texts]
        scores = self._score_pairs(pairs)

        # Attach scores
        for i, s in zip(idxs, scores):
            docs[i]["reranker_score"] = s
            

        # Sort docs by score descending
        docs_sorted = sorted(
            docs,
            key=lambda d: d.get("reranker_score", d.get("retriever_score", 0.0)),
            reverse=True,
        )

        # Return top_k
        return docs_sorted[:top_k]
