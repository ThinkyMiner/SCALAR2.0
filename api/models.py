from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SearchQuery(BaseModel):
    query_text: str = Field(..., min_length=1)
    k: int = Field(5, gt=0, le=100)
    namespace: str = Field("default")
    filter: Optional[Dict[str, Any]] = None
    rerank: bool = Field(False)


class SearchResult(BaseModel):
    chunk_id: int
    source: str
    namespace: str
    page_number: Optional[int] = None
    content: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    namespace: str
