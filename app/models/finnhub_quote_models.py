from pydantic import BaseModel


class Quote(BaseModel):
    c: float
    h: float
    l: float
    o: float
    pc: float
    t: int
