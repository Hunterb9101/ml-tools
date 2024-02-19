from typing import Generic, TypeVar

import pydantic as pdt

T = TypeVar("T")

class TrainValTest(pdt.BaseModel, Generic[T]):
    class Config:
        arbitrary_types_allowed = True
    train: T
    val: T
    test: T
