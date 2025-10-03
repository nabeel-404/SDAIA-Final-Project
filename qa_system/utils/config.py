from pydantic import BaseModel


class Settings(BaseModel):
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

