from pydantic import BaseModel
import os


class Settings(BaseModel):
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    model_name: str = "lightonai/Reason-ModernColBERT"

    # index folder should be a full path relative to repo root
    index_folder: str = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "qa_system",
        "retrieval",
        "index",
    )
    index_name: str = "hotpotqa-colbert-index"

