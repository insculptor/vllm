from pydantic import BaseModel


## TODO: Add the schema for the GenerateRequest
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = False
