"""
A complete configuration is not defined here. However, some common patterns are present.

Model Config: A collection of model-specific configuration values.
ModelPath Config: Manages key model-specific paths for modeling purposes. This
    is used within the `ModelConfig` class.
Path Config: Manages key model-agnostic paths for data and artifacts.
"""

from pathlib import Path
from tempfile import gettempdir
from typing import Any

import pydantic as pdt

import mltools.types as mt


class PathConfig(pdt.BaseModel):
    """Manage model-agnostic paths for data and artifacts."""

    root_path: str = gettempdir()
    tag: str = "prod"

    @pdt.computed_field
    def data_path(self) -> str:
        """Return the root data directory."""
        return str(Path(self.root_path) / "data")

    @pdt.computed_field
    def raw_data_path(self) -> str:
        """Return the raw training data path.

        This should combine data sources down to a
        single file, but that does not mean that it is fully processed.
        """
        return str(Path(self.root_path) / "data" / "raw" / "train.parquet")

    @pdt.computed_field
    def processed_data_dir(self) -> str:
        """Return the processed data directory.

        These files should be split into train, val, and test,
        but still model-agnostic.
        """
        return str(Path(self.root_path) / "data" / "processed")

    @pdt.computed_field
    def processed_data_path(self) -> mt.TrainValTest[str]:
        """Return train, validation, and test processed data paths."""
        processed_data_dir = Path(self.root_path) / "data" / "processed"
        return mt.TrainValTest(
            train=str(processed_data_dir / "train.parquet"),
            val=str(processed_data_dir / "val.parquet"),
            test=str(processed_data_dir / "test.parquet"),
        )

    @pdt.computed_field
    def artifact_dir(self) -> str:
        """Return the model-agnostic artifact directory.

        Model-agnostic artifacts should be saved here, such as processing data
        pipelines.
        """
        return str(Path(self.root_path) / "data" / "artifacts")


class ModelPathConfig(pdt.BaseModel):
    """Manage model-specific paths for modeling artifacts."""

    root_path: str = gettempdir()
    model: str

    @pdt.computed_field
    def data_path(self) -> str:
        """Return the root data directory."""
        return str(Path(self.root_path) / "data")

    @pdt.computed_field
    def dmatrix_dir(self) -> str:
        """Return the design matrix directory.

        This contains the final design matrix files, created from the processed data.
        Any model-specific transformations should be performed here, such as
        feature selection, conversion to a Torch Dataset, etc.
        """
        return str(Path(self.root_path) / "data" / "dmatrix")

    @pdt.computed_field
    def dmatrix_path(self) -> mt.TrainValTest[str]:
        """Return train, validation, and test design matrix paths."""
        dmatrix_dir = Path(self.root_path) / "data" / "dmatrix"
        return mt.TrainValTest(
            train=str(dmatrix_dir / f"train-{self.model}.parquet"),
            val=str(dmatrix_dir / f"val-{self.model}.parquet"),
            test=str(dmatrix_dir / f"test-{self.model}.parquet"),
        )

    @pdt.computed_field
    def feature_pipeline_path(self) -> str:
        """Return the feature pipeline artifact path.

        The feature selection and augmentation pipeline, a fit/transform class,
        should be saved as a pickle file here.
        """
        return str(Path(self.root_path) / "models" / f"fp-{self.model}.pkl")

    @pdt.computed_field
    def model_dir(self) -> str:
        """Return the model artifact directory."""
        return str(Path(self.root_path) / "models")

    @pdt.computed_field
    def model_path(self) -> str:
        """Return the model artifact path."""
        return str(Path(self.root_path) / "models" / f"{self.model}.pkl")


class ModelConfig(pdt.BaseModel):
    """A collection of model-specific configuration values."""

    root_path: str = gettempdir()
    params: dict[str, Any] = {}
    model: str = "gbm"

    @pdt.computed_field
    def paths(self) -> ModelPathConfig:
        """Return model-specific path configuration."""
        return ModelPathConfig(root_path=self.root_path, model=self.model)
