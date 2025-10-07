from pydantic import BaseModel


class Settings(BaseModel):
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    model_name: str = "lightonai/Reason-ModernColBERT"
    index_folder: str = "index"
    index_name: str = "hotpotqa-colbert-index"

