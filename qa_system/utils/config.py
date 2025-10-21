from pydantic import BaseModel
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Settings(BaseModel):
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    model_name: str = "lightonai/Reason-ModernColBERT"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 16
    reranker_fp16: bool = True
    reranker_max_len: int = 512

    # index folder should be a full path relative to repo root
    index_folder: str = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "qa_system",
        "retrieval",
        "index",
    )
    index_name: str = "hotpotqa-colbert-index"

