from typing import Any, List

from pydantic import BaseModel


class PydanticGraph(BaseModel):
    vertices: List[Any]
    edges: List[Any]