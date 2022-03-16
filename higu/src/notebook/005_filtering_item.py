#%%
'''
まあ買われないであろうアイテムをあらかじめ除去したい

'''

# %%
import gc
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


if True:
    sys.path.append("../")
    from candidacies import (AbstractCGBlock, BoughtItemsAtInferencePhase,
                             LastNWeekArticles, PopularItemsoftheLastWeeks,
                             candidates_dict2df)
    from eda_tools import visualize_importance
    from features import (AbstractBaseBlock, ModeCategoryBlock,
                          TargetEncodingBlock)
    from metrics import apk, mapk
    from utils import (Categorize, article_id_int_to_str,
                       article_id_str_to_int, arts_id_list2str,
                       customer_hex_id_to_int, phase_date, reduce_mem_usage,
                       timer)


root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
output_dir = root_dir / "output"
DRY_RUN = True
%load_ext autoreload
%autoreload 2

# %%

if not DRY_RUN:
    trans_cdf = cudf.read_parquet(input_dir / 'transactions_train.parquet')
    cust_cdf = cudf.read_parquet(input_dir / 'customers.parquet')
    art_cdf = cudf.read_parquet(input_dir / 'articles.parquet')
else:
    sample = 0.05
    trans_cdf = cudf.read_parquet(
        input_dir / f"transactions_train_sample_{sample}.parquet")
    cust_cdf = cudf.read_parquet(
        input_dir / f'customers_sample_{sample}.parquet')
    art_cdf = cudf.read_parquet(input_dir /
                               f'articles_train_sample_{sample}.parquet')


to_cdf = cudf.DataFrame.from_pandas
s = pd.Series(list(range(1,10)))

s.shift(1).rolling(3).sum()

# %%
trans_cdf

# %%



# %%

cust_id_df =  cust_cdf[['customer_id']]
cust_id_df['key'] = 1
week_df = cudf.DataFrame(list(range(0,104)),columns=['week'])
week_df['key'] = 1

week_cust_prod_df = week_df.merge(cust_id_df,how='outer',on='key').drop('key',axis=1)

size_df = trans_cdf.groupby(['customer_id','week']).size().to_frame('size').reset_index()
week_cust_prod_df = week_cust_prod_df.merge(size_df,how='left',on=['customer_id','week']).fillna(0)
week_cust_prod_df = week_cust_prod_df.sort_values(by=['customer_id','week'],ascending=True).reset_index(drop=True)

week_cust_prod_df

#%%
week_cust_prod_df.groupby('customer_id')['size']

#%%
trans_cdf.groupby(['customer_id','week']).size().rolling(3).sum().shift(1)


# %%
#この調子で特徴量を作成する。yは次のsizeでOK(shift(-1))
week_cust_prod_df['diff3_sum'] = week_cust_prod_df.to_pandas().groupby(['customer_id'])['size'].shift(1).rolling(3).sum()
week_cust_prod_df

# %%
