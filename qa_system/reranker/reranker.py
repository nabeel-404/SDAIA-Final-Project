# reranker.py
from typing import Dict, List, Tuple, Optional
from sentence_transformers import CrossEncoder
import torch
from qa_system.utils import Settings


class Reranker:
    """
    Cross-encoder reranker using configurable model from Settings.
    Accepts a list of docs (dicts with at least 'text'/'chunk') and returns
    the same docs sorted by reranker_score (descending).
    """

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        batch_size: int = None,
        fp16: bool = None,
        max_len: Optional[int] = None,
    ) -> None:
        cfg = Settings()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size or cfg.reranker_batch_size
        model_name = model_name or cfg.reranker_model_name

        print(f"[Reranker] Loading {model_name} on {self.device}...")
        self.reranker = CrossEncoder(model_name, device=self.device, max_length=max_len)

        # Optional mixed precision for faster inference
        if (fp16 if fp16 is not None else cfg.reranker_fp16) and self.device.startswith("cuda"):
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

    def rerank(self, query: str, docs: List[Dict], top_k: int = None) -> List[Dict]:
        """Attach reranker scores and return docs sorted by them."""
        cfg = Settings()
        if top_k is None:
            top_k = cfg.rerank_top_k
            
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
