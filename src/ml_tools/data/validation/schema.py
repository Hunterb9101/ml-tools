from typing import List, Union, Any, Sequence, Optional, Dict

import numpy as np
import pandas as pd

NumericType = Union[float, int]

class SchemaRange:
    def __init__(
        self,
        minval: NumericType = None,
        maxval: NumericType = None,
        include_lb: bool = True,
        include_ub: bool = True
    ):
        self.minval = minval
        self.maxval = maxval
        self.include_lb = include_lb
        self.include_ub = include_ub

        if minval and maxval:
            assert minval < maxval
        if minval is None and maxval is None:
            raise ValueError(f"Minval or Maxval must be set. Got minval={minval} and maxval={maxval}")

    def contains(self, ser: Sequence) -> np.ndarray:
        if not isinstance(ser, np.ndarray):
            arr = np.array(ser)
        else:
            arr = ser
        if not pd.api.types.is_numeric_dtype(arr.dtype):
            colname = ""
            if isinstance(ser, pd.Series):
                colname = f"(in {ser.name})"
            raise NotImplementedError(f"Datatype {arr.dtype}{colname} is not supported.")

        if self.minval is not None:
            lb_cond = (arr >= self.minval) if self.include_lb else (arr > self.minval)
        else:
            lb_cond = pd.Series(np.ones(len(arr)), dtype="bool")
        if self.maxval is not None:
            ub_cond = (arr <= self.maxval) if self.include_ub else (arr < self.maxval)
        else:
            ub_cond = pd.Series(np.ones(len(arr)), dtype="bool")
        return np.array(lb_cond & ub_cond)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minval": self.minval,
            "maxval": self.maxval,
            "include_lb": self.include_lb,
            "include_ub": self.include_ub
        }

    def __eq__(self, other) -> bool:
        return self.to_dict() == self.to_dict() if isinstance(self, type(other)) else False


class SchemaList:
    def __init__(self, vals: List[Any]):
        self.vals = vals

    def contains(self, ser: Sequence) -> np.ndarray:
        if not isinstance(ser, pd.Series):
            ser = pd.Series(ser)
        return np.array(ser.isin(self.vals))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vals": self.vals
        }

    def __eq__(self, other) -> bool:
        return self.to_dict() == self.to_dict() if isinstance(self, type(other)) else False

class SchemaObj:
    def __init__(
        self,
        column: str,
        dtype: str,
        valid_vals: Optional[List[Union[SchemaList, SchemaRange]]] = None,
        nullable: bool = True
    ):
        self.column = column
        self.dtype = dtype
        self.valid_vals = valid_vals if valid_vals else []
        self.nullable = nullable

    def to_dict(self) -> List[Dict[str, Any]]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "valid_vals": [vv.to_dict() for vv in self.valid_vals],
            "nullable": self.nullable
        }


def dict_to_valid_vals(input_dict: Dict[str, Any]) -> Union[SchemaRange, SchemaList]:
    input_dict = input_dict.copy()
    if ("minval" in input_dict.keys() or "maxval" in input_dict.keys()) and "vals" not in input_dict.keys():
        return SchemaRange(**input_dict)
    if ("minval" not in input_dict.keys() and "maxval" not in input_dict.keys()) and "vals" in input_dict.keys():
        return SchemaList(**input_dict)
    raise ValueError(f"Unable to parse valid ranges for {input_dict}. " \
        "Make sure only a range or a list of values are provided per entry."
    )


def dict_to_schema(schema: List[Dict[str, Any]]) -> List[SchemaObj]:
    new_schema = []
    for s in schema:
        if "valid_vals" in s.keys():
            vals = [dict_to_valid_vals(vv) for vv in s["valid_vals"]]
            s["valid_vals"] = vals
        new_schema.append(SchemaObj(**s))
    return new_schema
