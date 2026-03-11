import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spatialmt.config import Paths, Dirs, setup_output_dirs, validate_raw_inputs

setup_output_dirs()
validate_raw_inputs()

processed_tpm = pd.read_csv(Paths.processed_tpm, header=0, index_col=0)
print(f"Processed TPM data: {processed_tpm.shape}")