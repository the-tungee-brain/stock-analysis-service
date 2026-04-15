from pydantic import BaseModel
from typing import Union


class Quote(BaseModel):
    c: float
    h: float
    l: float
    o: float
    pc: float
    t: int
