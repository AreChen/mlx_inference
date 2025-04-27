from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str
    max_tokens: int = 64000
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    from_responses: bool = False  # 标记是否为 /v1/responses 端点调用

class ResponseRequest(BaseModel):
    input: Union[str, list]
    model: str
    stream: bool = False
    max_tokens: int = 64000
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None
    functions: Optional[List[Dict[str, Any]]] = None