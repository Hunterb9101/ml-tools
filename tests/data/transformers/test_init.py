from mltools.data import transformers


def test_transformers_exports():
    assert transformers.__all__ == [
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
