import sys
import os
from datetime import datetime

from TetriumColor.Measurement import MeasurePrimaries

if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    current_date = datetime.now().strftime("%Y-%m-%d")
    directory = f"../../measurements/{current_date}/primaries"
os.makedirs(directory, exist_ok=True)
MeasurePrimaries(directory)
