from pydantic import BaseModel
from typing import List

class SourceInfo(BaseModel):
    score: float
    summary: str
    text: str
    source_file: str
    chunk_index: int
    page: int

class AskResponse(BaseModel):
    question: str
    generated_answer: str
    answers: List[SourceInfo]

class UploadResponse(BaseModel):
    message: str 