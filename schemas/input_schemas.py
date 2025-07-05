from pydantic import BaseModel
from typing import Literal, Optional, List

class AskRequest(BaseModel):
    question: str
    type: Literal["new", "followup"]
    history: Optional[List[dict]] = None 