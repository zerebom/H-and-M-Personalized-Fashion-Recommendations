#%%
import gc
import json
import os
import pprint
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from time import time

import cudf
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# %%

root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
output_dir = root_dir / "output"

#%%

path = "submission_03_21_04_16.csv"
df = cudf.read_csv(output_dir / path)

# %%
df.head(30)


# %%
df['prediction'].value_counts()


# %%
