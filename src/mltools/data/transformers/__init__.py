"""Data transformers that can be used in the feature engineering process.

Transformers follow the scikit-learn fit/transform paradigm.
"""

from mltools.data.transformers.base import BaseTransformer
from mltools.data.transformers.blend import Blender
from mltools.data.transformers.discretizer import Discretizer
from mltools.data.transformers.encoder import TargetEncoder, UnivariateEncoder
from mltools.data.transformers.interaction import Interaction
from mltools.data.transformers.model_imputer import ModelImputer
from mltools.data.transformers.moving import Expanding, MovingAverage
from mltools.data.transformers.outliers import MADMedianOutlierDetector, mad
from mltools.data.transformers.pruners import BasePruner, FeatureSelectionPipeline
from mltools.data.transformers.pruners.moment import MomentPruner
from mltools.data.transformers.pruners.rfe import RFEPruner

__all__ = [
    "BasePruner",
    "BaseTransformer",
    "Blender",
    "Discretizer",
    "Expanding",
    "FeatureSelectionPipeline",
    "Interaction",
    "MADMedianOutlierDetector",
    "ModelImputer",
    "MomentPruner",
    "MovingAverage",
    "RFEPruner",
    "TargetEncoder",
    "UnivariateEncoder",
    "mad",
]
