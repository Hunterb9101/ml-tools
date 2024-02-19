"""
A complete configuration is not defined here. However, some common patterns are present.

Model Config: A collection of model-specific configuration values.
ModelPath Config: Manages key model-specific paths for modeling purposes. This
    is used within the `ModelConfig` class.
Path Config: Manages key model-agnostic paths for data and artifacts.
"""
from os.path import join
from typing import Dict, Any

import pydantic as pdt

import mltools.types as mt

class PathConfig(pdt.BaseModel):
    root_path: str = "/tmp"
    tag: str = "prod"

    @pdt.computed_field
    @property
    def data_path(self) -> str:
        return join(self.root_path, "data")

    @pdt.computed_field
    @property
    def raw_data_path(self) -> str:
        """
        Contains raw data files. This should combine data sources down to a
        single file, but that does not mean that it is fully processed.
        """
        return join(self.data_path, "raw", "train.parquet")

    @pdt.computed_field
    @property
    def processed_data_dir(self) -> str:
        """
        Contains the processed data files. These should be split into train, val, and test,
        but still model-agnostic.
        """
        return join(self.data_path, "processed")

    @pdt.computed_field
    @property
    def processed_data_path(self) -> mt.TrainValTest[str]:
        return mt.TrainValTest(
            train=join(self.processed_data_dir, "train.parquet"),
            val=join(self.processed_data_dir, "val.parquet"),
            test=join(self.processed_data_dir, "test.parquet")
        )

    @pdt.computed_field
    @property
    def artifact_dir(self) -> str:
        """
        Model-agnostic artifacts should be saved here, such as processing data
        pipelines.
        """
        return join(self.data_path, "artifacts")


class ModelPathConfig(pdt.BaseModel):
    root_path: str = "/tmp"
    model: str

    @pdt.computed_field
    @property
    def data_path(self) -> str:
        return join(self.root_path, "data")

    @pdt.computed_field
    @property
    def dmatrix_dir(self) -> str:
        """
        Contains the final design matrix files, created from the processed data.
        Any model-specific transformations should be performed here, such as
        feature selection, conversion to a Torch Dataset, etc.
        """
        return join(self.data_path, "dmatrix")

    @pdt.computed_field
    @property
    def dmatrix_path(self) -> mt.TrainValTest[str]:
        return mt.TrainValTest(
            train=join(self.dmatrix_dir, f"train-{self.model}.parquet"),
            val=join(self.dmatrix_dir, f"val-{self.model}.parquet"),
            test=join(self.dmatrix_dir, f"test-{self.model}.parquet")
        )

    @pdt.computed_field
    @property
    def feature_pipeline_path(self) -> str:
        """
        The feature selection and agumentation pipeline, a fit/transform class,
        should be saved as a pickle file here.
        """
        return join(self.model_dir, f"fp-{self.model}.pkl")

    @pdt.computed_field
    @property
    def model_dir(self) -> str:
        return join(self.root_path, "models")

    @pdt.computed_field
    @property
    def model_path(self) -> str:
        """
        The model artifact should be saved here.
        """
        return join(self.model_dir, f"{self.model}.pkl")


class ModelConfig(pdt.BaseModel):
    """
    A collection of model-specific configuration values.
    """
    root_path: str = "/tmp"
    params: Dict[str, Any] = {}
    model: str = 'gbm'

    @pdt.computed_field
    @property
    def paths(self) -> ModelPathConfig:
        return ModelPathConfig(root_path=self.root_path, model=self.model)
