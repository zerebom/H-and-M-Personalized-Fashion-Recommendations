# %%
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

sampled_df1 = cudf.read_csv("sampled_df1.csv").reset_index()
sampled_df2 = cudf.read_csv("sampled_df2.csv").reset_index()
#%%
sampled_df2.groupby(['candidate_block_name','y'])['index'].mean()

#%%
sampled_df1.groupby(['candidate_block_name','y'])['index'].mean()
#%%
cust = int(sampled_df1['customer_id'].sample(1).values[0])

display(sampled_df1.query('customer_id ==@cust').head(50))
display(sampled_df2.query('customer_id ==@cust').head(50))

#%%
sampled_df2.groupby(['candidate_block_name','y']).size()
#%%
sampled_df1.groupby(['candidate_block_name','y']).size()
#%%
sampled_df1.equals(sampled_df2)

#%%
trans_cdf, cust_cdf, df = read_cdf(input_dir, DRY_RUN)




ohe_columns = []
total = 0

for col in df.columns:
    if len(df[col].unique()) <= 500:
        ohe_columns.append(col)
        total += len(df[col].unique())

    print(col, df[col].dtype, len(df[col].unique()))


#%%
n = 20


df = trans_cdf.to_pandas()
df = df.tail(100000)
df

#%%

df.sample(frac=1,random_state=41)

# mdf.groupby('customer_id').apply(lambda x:x.head(5))
#%%

mdf = df.head(1000000)
# index列の用意
mdf = mdf.reset_index()
mdf['rank'] = mdf.groupby('customer_id')['index'].rank(method='first')
mdf.query('rank < 10').drop(columns=['index','rank'])

#%%
mdf.groupby("customer_id").apply(lambda x: x.sample(n=min(n, len(x)), random_state=42))

#%%

df.groupby("customer_id").rank()


#%%
EMB_SIZE = 20

V = cudf.get_dummies(df[ohe_columns], columns=ohe_columns).values
V = cupy.hstack([V.astype("float32")])

svd = TruncatedSVD(n_components=EMB_SIZE, random_state=0)
V = svd.fit_transform(V)
print("Explained variance ratio:", svd.explained_variance_ratio_.sum().item())

# %%


emb_dic = dict(zip(df["article_id"].to_pandas().astype(str).values, V.tolist()))
with open(str(input_dir / "emb/svd.json"), "w") as f:
    json.dump(emb_dic, f, indent=4)
#%%
df = cudf.read_csv("/home/kokoro/h_and_m/higu/output/016_add_similarity/train_df2.csv")



# group = cudf.read_csv("/home/kokoro/h_and_m/higu/output/016_add_similarity/group_df.csv")
#%%

df2 = cudf.read_csv("/home/kokoro/h_and_m/higu/output/016_add_similarity/train_df3.csv")


#%%
df2



# %%
df['y'].sum()



# %%
df[['customer_id','article_id','candidate_block_name']].drop_duplicates().reset_index()

#%%
df

# %%
df.groupby('customer_id')['y'].sum().value_counts()



# %%
cust_id = 2576222976091645
art_id = 399223001
trans_cdf.query(f'customer_id == {cust_id} & article_id == {art_id}')

# %%

trans_cdf.query('customer_id == @cust_id')

# %%

trans_cdf.query(f'article_id == {art_id}')

# %%
trans_cdf[trans_cdf['customer_id']==12816702091124429314]

# %%
df[df['customer_id']== 1402273113592184]

# %%
df[df['y']==1]


# %%

df2.query("y!=0")


# %%

df.query("y!=0")


# %%

df

# %%

df2.tail(20)


# %%
df.tail(20)

# %%

val1 = df2.query("customer_id == 18445412194736247951").sort_values('article_id')['article_id'].values
val2 = df.query("customer_id == 18445412194736247951").sort_values('article_id')['article_id'].values

agg = set(val1.tolist()) & set(val2.tolist())

agg

# %%

df.to_pandas().query("customer_id == 18445412194736247951& article_id in @agg").sort_values('article_id').drop_duplicates(subset=['article_id'])

#%%

df2.to_pandas().query("customer_id == 18445412194736247951& article_id in @agg").sort_values('article_id').drop_duplicates(subset=['article_id'])


# %%
df.groupby('customer_id').size().value_counts()



# %%

df2.groupby('customer_id').size().value_counts()

# %%
