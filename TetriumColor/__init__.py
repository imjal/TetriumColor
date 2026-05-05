# Tetrium Color - A color library for Python
from .ColorSpace import ColorSpace, ColorSpaceType, PolyscopeDisplayType
from .Utils.CustomTypes import *
from .Observer import Observer, MaxBasis

try:
    from .ColorSampler import ColorSampler
except ModuleNotFoundError:
    pass

try:
    from .TetraPlate import PseudoIsochromaticPlateGenerator, GaussianBlobGenerator
except ModuleNotFoundError:
    pass

try:
    from .TetraColorPicker import QuestColorGenerator, CircleGridGenerator, AEPsychThresholdContourGenerator
except ModuleNotFoundError:
    pass

try:
    from .ChromaticityAnalysis import (
        EllipsoidFitter, ChromaticityVisualizer, GenotypeClassifier,
        create_full_analysis_report
    )
except ModuleNotFoundError:
    pass
