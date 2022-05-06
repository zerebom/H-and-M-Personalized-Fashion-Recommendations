#%%
%load_ext autoreload
%autoreload 2


from collections import defaultdict
import cupy as cp

from cuml import TruncatedSVD
import cupy
import matplotlib.pyplot as plt
from asyncio import SubprocessTransport
import json
import os
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cudf
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


if True:
    sys.path.append("../")
    sys.path.append("/home/kokoro/h_and_m/higu/src")
    from candidacies_dfs import (
        LastBoughtNArticles,
        PopularItemsoftheLastWeeks,
        LastNWeekArticles,
        PairsWithLastBoughtNArticles,
    )
    from eda_tools import visualize_importance
    from features import (
        EmbBlock,
        ModeCategoryBlock,
        TargetEncodingBlock,
        LifetimesBlock,
        SexArticleBlock,
        SexCustomerBlock,
        Dis2HistoryAveVecAndArtVec,
    )
    from metrics import mapk, calc_map12
    import config
    from utils import (
        article_id_int_to_str,
        article_id_str_to_int,
        arts_id_list2str,
        convert_sub_df_customer_id_to_str,
        customer_hex_id_to_int,
        get_var_names,
        jsonKeys2int,
        phase_date,
        read_cdf,
        reduce_mem_usage,
        setup_logger,
        squeeze_pred_df_to_submit_format,
        timer,
    )


root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
exp_name = Path(os.path.basename(__file__)).stem
exp_name = "016_add_similarity"

output_dir = root_dir / "output" / exp_name
log_dir = output_dir / "log" / exp_name

output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"{date.today()}.log"
logger = setup_logger(log_file)
DRY_RUN = False


ArtId = int
CustId = int
ArtIds = List[ArtId]
CustIds = List[CustId]

to_cdf = cudf.DataFrame.from_pandas
from cuml import TruncatedSVD
import cudf

#%%

df = cudf.read_csv(log_dir/"submission_df_0502_0835.csv")
df



# %%
df.dropna()





# %%

# %%
