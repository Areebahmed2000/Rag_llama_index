from typing import List, Optional

from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    use_agent: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_id: int

class UploadResponse(BaseModel):
    message: str
    document_count: int
    node_count: int
    files_processed: List[str]