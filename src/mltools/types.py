"""Shared project type containers."""

from typing import Generic, TypeVar

import pydantic as pdt

T = TypeVar("T")


class TrainValTest(pdt.BaseModel, Generic[T]):
    """Store train, validation, and test values."""

    model_config = pdt.ConfigDict(arbitrary_types_allowed=True)
    train: T
    val: T
    test: T
