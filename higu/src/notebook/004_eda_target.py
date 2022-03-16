#%%
'''
予測対象のボリュームに着目したEDAです
https://www.notion.so/EDA-11b37db4b2c24a1699b791cdc9f8d680
'''
import glob
import os
import pprint
import sys
from collections import defaultdict
from pathlib import Path
from time import time

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
import cudf
#%%
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# %%

%load_ext autoreload
%autoreload 2
root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
output_dir = root_dir / "output"

transactions = cudf.read_parquet(input_dir / 'transactions_train.parquet')
customers = cudf.read_parquet(input_dir / 'customers.parquet')
articles = cudf.read_parquet(input_dir / 'articles.parquet')

# %%
last_trans_df = transactions.query("week==104")
last_trans_df
#%%
trans_mat_df = transactions.groupby(['customer_id','week']).size().to_frame(name='size').reset_index().pivot(index='customer_id',columns='week').fillna(0)
trans_mat_df.columns = trans_mat_df.columns.droplevel(0) #multiindexの解除
trans_mat_df = trans_mat_df.reset_index().sort_values(104,ascending=False)
trans_mat_df['sum'] = trans_mat_df.iloc[:, 1:].sum(axis=1)



#%%
#週ごとの購入数
plt.figure(figsize=(10, 20))
sns.heatmap(trans_mat_df.sort_values('sum', ascending=False).iloc[:50,-20:-1].to_pandas(), annot=True, fmt="d",cbar="False",robust=True)
#%%

plt.figure(figsize=(20, 20))
_df = trans_mat_df[trans_mat_df[104]>0].sample(50).drop(columns='sum').to_pandas()
sns.heatmap(_df.iloc[:,-50:], annot=True, fmt="d",cbar="False",robust=True)

#%%

#最終週の購入数ごとのユーザ数
pd.options.plotting.backend = "plotly"
trans_mat_df[104].to_pandas().plot.hist(text_auto=True)


# %%
#購入数と予測数・予測精度への寄与率の関係
pred_num_df=trans_mat_df[104].to_pandas().clip(0,12).value_counts().to_frame('user_cnt').reset_index()
pred_num_df['preds'] = pred_num_df['index'] * pred_num_df['user_cnt']
pred_num_df['ratio'] = round(pred_num_df['preds'] / pred_num_df['preds'].sum(),3)
pred_num_df = pred_num_df.sort_values('index')
pred_num_df.rename({'index':'bought_n_items'})
pred_num_df['r_cumsum'] = pred_num_df['ratio'].cumsum()
pred_num_df

#%%

# %%
