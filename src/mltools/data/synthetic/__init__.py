from typing import Dict, List, Any, Optional, Callable, Union
import warnings

import numpy as np
import pandas as pd


class MockColumn():
    def __init__(
        self,
        name: str,
        distribution: Optional[Callable] = None,
        distribution_kwargs: Optional[Dict[str, Any]] = None,
        redundant_of: Optional[str] = None,
        is_useful: bool = True
    ):
        self.name = name
        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs or {}
        self.redundant_of = redundant_of
        self.is_useful = is_useful

        if distribution and redundant_of:
            raise ValueError(f"Cannot set distribution for redundant column {name}.")
        if distribution_kwargs and redundant_of:
            raise ValueError(f"Cannot set distribution_kwargs for redundant column {name}.")
        if is_useful and redundant_of:
            warnings.warn("is_useful=True for a redundant column. Ignoring.")
            self.is_useful= False
        if not redundant_of and not distribution:
            raise ValueError("Must define a distribution for a non-redundant column")


class DupColumn(MockColumn):
    def __init__(self, name: str, redundant_of: str):
        super().__init__(
            name=name,
            distribution=None,
            distribution_kwargs=None,
            redundant_of=redundant_of,
            is_useful=False
        )


class MockManagerClassification:
    # TODO: Break this out a little more. Not all mocks need a target column
    # Maybe MockManagerBase, MockManagerDataset, and MockManagerClassification?
    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value: List[str]):
        self._columns = value

        self._useful: List[Union[MockColumn, DupColumn]] = []
        self._useless: List[Union[MockColumn, DupColumn]] = []
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
        columns: List[Union[MockColumn, DupColumn]],
        idx_cols: Optional[List[str]] = None,
        target_col: Optional[str] = "y",
        class_balance: float = 0.5,
        seed: int = 0
    ):
        self.n_rows = n_rows

        self._redundant = []
        self._useful = []
        self._useless = []
        self.columns = columns

        self.idx_cols = idx_cols or []
        self.target_col = target_col
        self.class_balance = class_balance
        self.seed = seed

        assert 0 < class_balance < 1

    def _generate_X(self) -> pd.DataFrame:
        np.random.seed(self.seed)
        data = {}
        for i, idx in enumerate(self.idx_cols):
            data[idx] = np.arange(self.n_rows) * (i + 1)
        for c in self._useful + self._useless:
            if not c.distribution:
                raise ValueError("Didn't have a distribution for X")
            data[c.name] = c.distribution(size=self.n_rows, **c.distribution_kwargs)
        for r in self._redundant:
            data[r.name] = data[r.redundant_of]
        return pd.DataFrame.from_dict(data)

    def _generate_Y(self, X: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(self.seed)
        # TODO: Make this work for useful categoricals / throw better warnings
        data = X[[c.name for c in self._useful]].values
        logits = (data - data.mean(axis=0)).sum(axis=1)
        y = (logits < np.percentile(logits, self.class_balance * 100)).astype(int)
        return pd.Series(y, name=self.target_col)

    def generate(self) -> pd.DataFrame:
        X = self._generate_X()
        if self.target_col:
            y = self._generate_Y(X)
            X[self.target_col] = y
        return X


def linearly_separable_data(
    n_rows: int = 100,
    n_useful_cols: int = 2,
    n_useless_cols: int = 2,
    seed: int = 0,
    add_redundant: bool = True,
    useful_pfx: str = "useful",
    useless_pfx: str = "useless",
    redundant_pfx: str = "redundant",
    idx_cols: Optional[List[str]] = None,
    target_col: str = "y"
):
    columns: List[Union[MockColumn, DupColumn]] = []
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
