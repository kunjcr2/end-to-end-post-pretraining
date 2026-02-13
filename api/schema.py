from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)


class QueryResponse(BaseModel):
    query: str
    response: str
    tokens: int