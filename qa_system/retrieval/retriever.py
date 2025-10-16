from typing import List, Dict
from pylate import models, indexes, retrieve
from qa_system.utils import Settings
import os
import json


class Retriever:
    def __init__(self) -> None:
        cfg = Settings()

        # Initialize PLAID index (use existing index)
        self.index = indexes.PLAID(
            index_folder=cfg.index_folder,
            index_name=cfg.index_name,
            override=False
        )

        # Initialize the retriever
        self.retriever = retrieve.ColBERT(index=self.index)

        # Initialize ColBERT model
        self.model = self._init_model()

        # Load document_id -> text mapping from JSON
        self.document_ids_to_sentence = self._load_document_ids_to_sentence()

    def _init_model(self):
        """Initialize ColBERT model using Settings().model_name."""
        cfg = Settings()
        if not hasattr(cfg, "model_name") or not cfg.model_name:
            raise RuntimeError(
                "Settings().model_name is not set. Provide a valid ColBERT model path or name."
            )
        model = models.ColBERT(model_name_or_path=cfg.model_name)
        if model is None:
            raise RuntimeError(
                f"Failed to load ColBERT model from '{cfg.model_name}'."
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

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the top_k most relevant documents for a given query."""
        if self.model is None:
            raise RuntimeError("Model not initialized properly.")

        # Encode the query
        query_emb = self.model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False
        )

        # Retrieve top-k results
        results = self.retriever.retrieve(queries_embeddings=query_emb, k=top_k)

        # If single query, unwrap the list
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]

        # Map results to dictionaries with text field for reranker
        contexts = []
        for result in results:
            doc_id = result["id"]
            text = self.document_ids_to_sentence.get(doc_id, "<text not found>")
            contexts.append({
                "text": text,
                "id": doc_id,
                "retriever_score": result.get("score", 0.0)
            })

        return contexts

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("Who is the dumbest person in the world?")  # FYI the answer should not be Abdullah :)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['text']}")