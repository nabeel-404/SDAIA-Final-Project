from typing import List, Dict
from pylate import models, indexes, retrieve
from qa_system.utils import Settings
import os
import json


class Retriever:
    def __init__(self) -> None:
        self.cfg = Settings()

        # Initialize PLAID index (use existing index)
        self.index = indexes.PLAID(
            index_folder=self.cfg.index_folder,
            index_name=self.cfg.index_name,
            override=False,
            device="cpu"

        )

        # Initialize the retriever
        self.retriever = retrieve.ColBERT(index=self.index)

        # Initialize ColBERT model
        self.model = self._init_model()

        # Load document_id -> text mapping from JSON
        self.document_ids_to_sentence = self._load_document_ids_to_sentence()

    def _init_model(self):
        """Initialize ColBERT model using Settings().model_name."""
        if not hasattr(self.cfg, "model_name") or not self.cfg.model_name:
            raise RuntimeError(
                "Settings().model_name is not set. Provide a valid ColBERT model path or name."
            )
        model = models.ColBERT(model_name_or_path=self.cfg.model_name)
        if model is None:
            raise RuntimeError(
                f"Failed to load ColBERT model from '{self.cfg.model_name}'."
            )
        return model

    def _load_document_ids_to_sentence(self) -> Dict[str, str]:
        """Load document ID -> text mapping from JSON file."""
        json_path = os.path.join(os.path.dirname(__file__), "document_ids_to_sentence.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Mapping file not found at {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve the top_k most relevant documents for a given query."""
        if top_k is None:
            top_k = self.cfg.retrieval_top_k
        return self.retrieve_multiple([query], top_k)
    
    def retrieve_multiple(self, queries: List[str], top_k: int = None) -> List[Dict]:
        """Retrieve documents for multiple queries and merge results."""
        if top_k is None:
            top_k = self.cfg.retrieval_top_k
        if self.model is None:
            raise RuntimeError("Model not initialized properly.")

        # Encode all queries
        query_emb = self.model.encode(
            queries,
            is_query=True,
            show_progress_bar=False
        )

        # Retrieve top-k results for each query
        all_results = self.retriever.retrieve(queries_embeddings=query_emb, k=top_k)

        
        merged_contexts = []
        
        for query_results in all_results:
            for result in query_results:
                doc_id = result["id"]
                text = self.document_ids_to_sentence.get(doc_id, "<text not found>")
                merged_contexts.append({
                    "text": text,
                    "id": doc_id,
                    "retriever_score": result.get("score", 0.0)
                })

        # Sort by score and return top_k
        merged_contexts.sort(key=lambda x: x["retriever_score"], reverse=True) 
        return merged_contexts[:top_k]

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("Were Scott Derrickson and Ed Wood of the same nationality?")  # FYI the answer should not be Abdullah :)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['text']}")