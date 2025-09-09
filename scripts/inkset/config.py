# inkset_config.py
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class InksetConfig:
    name: str
    data_path: str
    description: str
    analysis_params: Dict[str, Any]


# Define all inksets
INKSET_CONFIGS = {
    "fp_inks": InksetConfig(
        name="FP Inks",
        data_path="../../data/inksets/fp_inks/all_inks.csv",
        description="Fine art printing inks",
        analysis_params={"k_values": [4, 5, 6], "observer": "tetrachromat"}
    ),
    "ansari": InksetConfig(
        name="Ansari Inks",
        data_path="../../data/inksets/ansari/ansari-inks.csv",
        description="Ansari printer inks",
        analysis_params={"k_values": [4, 6], "observer": "tetrachromat"}
    ),
    "screenprinting": InksetConfig(
        name="Screen Printing",
        data_path="../../data/inksets/screenprinting/screenprinting-inks.csv",
        description="Screen printing inks",
        analysis_params={"k_values": [4, 5], "observer": "tetrachromat"}
    ),
    "pantone": InksetConfig(
        name="Pantone",
        data_path="../../data/inksets/pantone/pantone-inks.csv",
        description="Pantone color system",
        analysis_params={"k_values": [4, 5], "observer": "tetrachromat"}
    )
}
