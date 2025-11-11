import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from histogram_utils import load_group_from_file, extract_metrics_, to_dataframe

import glob
parquet_files = glob.glob("histos/*.pkl")
for f_in in parquet_files:
    print(f"Inspecting file {f_in}")
    histo_group = load_group_from_file(f_in)
    for _, histo_base_obj in histo_group:
        extract_metrics_(histo_base_obj)
    df = to_dataframe(histo_group)
    f_out = f_in.replace('histos.pkl','parquet').replace('histos','parquet')
    df.to_parquet(f_out)
