"""Synthetic data generation utilities."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


class MockColumn:
    """Describe a synthetic data column."""

    def __init__(
        self,
        name: str,
        distribution: Callable | None = None,
        distribution_kwargs: dict[str, Any] | None = None,
        redundant_of: str | None = None,
        *,
        is_useful: bool = True,
    ):
        self.name = name
        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs or {}
        self.redundant_of = redundant_of
        self.is_useful = is_useful

        if distribution and redundant_of:
            msg = f"Cannot set distribution for redundant column {name}."
            raise ValueError(msg)
        if distribution_kwargs and redundant_of:
            msg = f"Cannot set distribution_kwargs for redundant column {name}."
            raise ValueError(msg)
        if is_useful and redundant_of:
            warnings.warn("is_useful=True for a redundant column. Ignoring.")
            self.is_useful = False
        if not redundant_of and not distribution:
            msg = "Must define a distribution for a non-redundant column"
            raise ValueError(msg)


class DupColumn(MockColumn):
    """Describe a duplicate synthetic data column."""

    def __init__(self, name: str, redundant_of: str):
        super().__init__(
            name=name,
            distribution=None,
            distribution_kwargs=None,
            redundant_of=redundant_of,
            is_useful=False,
        )


class MockManagerClassification:
    """Generate a mock classification dataframe."""

    # TODO: Break this out a little more. Not all mocks need a target column
    # Maybe MockManagerBase, MockManagerDataset, and MockManagerClassification?
    @property
    def columns(self):
        """Return configured columns."""
        return self._columns

    @columns.setter
    def columns(self, value: list[MockColumn | DupColumn]):
        self._columns = value

        self._useful: list[MockColumn | DupColumn] = []
        self._useless: list[MockColumn | DupColumn] = []
        self._redundant = []

        for c in self.columns:
            if c.redundant_of:
                self._redundant.append(c)
            elif c.is_useful and not c.redundant_of:
                self._useful.append(c)
            else:
                self._useless.append(c)

    def __init__(
        self,
        n_rows: int,
        columns: list[MockColumn | DupColumn],
        **kwargs,
    ):
        allowed_kwargs = {"idx_cols", "target_col", "class_balance", "seed"}
        unexpected_kwargs = set(kwargs) - allowed_kwargs
        if unexpected_kwargs:
            msg = f"Unexpected arguments: {sorted(unexpected_kwargs)}"
            raise TypeError(msg)
        idx_cols = kwargs.get("idx_cols")
        target_col = kwargs.get("target_col", "y")
        class_balance = kwargs.get("class_balance", 0.5)
        seed = kwargs.get("seed", 0)

        self.n_rows = n_rows

        self._redundant = []
        self._useful = []
        self._useless = []
        self.columns = columns

        self.idx_cols = idx_cols or []
        self.target_col = target_col
        self.class_balance = class_balance
        self.seed = seed

        if not (0 < class_balance < 1):
            msg = "Class balance outside of 0 to 1 range."
            raise ValueError(msg)

    def _generate_x(self) -> pd.DataFrame:
        """Generate feature columns."""
        np.random.seed(self.seed)
        data = {}
        for i, idx in enumerate(self.idx_cols):
            data[idx] = np.arange(self.n_rows) * (i + 1)
        for c in self._useful + self._useless:
            if not c.distribution:
                msg = "Didn't have a distribution for X"
                raise ValueError(msg)
            data[c.name] = c.distribution(size=self.n_rows, **c.distribution_kwargs)
        for r in self._redundant:
            data[r.name] = data[r.redundant_of]
        return pd.DataFrame.from_dict(data)

    def _generate_y(self, x: pd.DataFrame) -> pd.DataFrame:
        """Generate the target column."""
        np.random.seed(self.seed)
        # TODO: Make this work for useful categoricals / throw better warnings
        data = x[[c.name for c in self._useful]].to_numpy()
        logits = (data - data.mean(axis=0)).sum(axis=1)
        y = (logits < np.percentile(logits, self.class_balance * 100)).astype(int)
        return pd.Series(y, name=self.target_col)

    def generate(self) -> pd.DataFrame:
        """Generate a full synthetic dataframe."""
        x = self._generate_x()
        if self.target_col:
            y = self._generate_y(x)
            x[self.target_col] = y
        return x


def linearly_separable_data(
    n_rows: int = 100,
    **kwargs,
):
    """Generate a linearly separable classification dataframe."""
    allowed_kwargs = {
        "n_useful_cols",
        "n_useless_cols",
        "seed",
        "add_redundant",
        "useful_pfx",
        "useless_pfx",
        "redundant_pfx",
        "idx_cols",
        "target_col",
    }
    unexpected_kwargs = set(kwargs) - allowed_kwargs
    if unexpected_kwargs:
        msg = f"Unexpected arguments: {sorted(unexpected_kwargs)}"
        raise TypeError(msg)
    n_useful_cols = kwargs.get("n_useful_cols", 2)
    n_useless_cols = kwargs.get("n_useless_cols", 2)
    seed = kwargs.get("seed", 0)
    add_redundant = kwargs.get("add_redundant", True)
    useful_pfx = kwargs.get("useful_pfx", "useful")
    useless_pfx = kwargs.get("useless_pfx", "useless")
    redundant_pfx = kwargs.get("redundant_pfx", "redundant")
    idx_cols = kwargs.get("idx_cols")
    target_col = kwargs.get("target_col", "y")

    columns: list[MockColumn | DupColumn] = []
    idx_cols = idx_cols or []
    if add_redundant:
        for i in range(n_useful_cols):
            columns.append(DupColumn(f"{redundant_pfx}{i}", redundant_of=f"{useful_pfx}{i}"))
    for i in range(n_useless_cols):
        columns.append(MockColumn(f"{useless_pfx}{i}", distribution=np.random.normal, is_useful=False))
    useful = [MockColumn(f"{useful_pfx}{i}", distribution=np.random.normal) for i in range(n_useful_cols)]
    columns.extend(useful)

    mmc = MockManagerClassification(
        n_rows=n_rows,
        columns=columns,
        idx_cols=idx_cols,
        target_col=target_col,
        seed=seed,
    )
    df = mmc.generate()
    df = df[[c.name for c in columns] + idx_cols + [target_col]]
    if idx_cols:
        df = df.set_index(idx_cols)
    return df
