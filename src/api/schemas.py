from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

class Source(BaseModel):
    page: str
    topic: str
    preview: str

class QueryRequest(BaseModel):
    query: str = Field(..., example="Explain the chlor-alkali process")
    history: Optional[List[List[str]]] = Field(default_factory=list, description="List of [role, message] pairs")

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: str

class QueryResponse(BaseModel):
    intent: str
    # Response can be a string (explanation) or a list of dicts (quiz)
    response: Union[str, List[QuizQuestion]]
    sources: List[Source]

class HealthResponse(BaseModel):
    status: str
    message: str