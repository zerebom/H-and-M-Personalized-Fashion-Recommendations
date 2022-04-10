
# %%
import os
import pprint
import sys
from collections import defaultdict
from pathlib import Path
from time import time

import cudf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# %%
if True:
    #sys.path.append("../")
    sys.path.append("/content/drive/MyDrive/competition/h_and_m/higu/src/")
    from utils import (Categorize, article_id_int_to_str,
                       article_id_str_to_int, customer_hex_id_to_int,
                       reduce_mem_usage)

# %%
root_dir = Path("/content/drive/MyDrive/competition/") #Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
output_dir = root_dir / "output"


# %%
# データセットの読み込み
print("データセット読み込み")
cust_df = cudf.read_csv(input_dir / 'customers.csv').to_pandas()
art_df = cudf.read_csv(input_dir / 'articles.csv').to_pandas()
art_df['article_id'] = art_df['article_id'].astype('str')
trans_df = cudf.read_csv(input_dir /
                         "transactions_train.csv"
                         ).to_pandas()



# customersの前処理
cust_df['customer_id'] = customer_hex_id_to_int(cust_df['customer_id'])

for col in ['FN', 'Active', 'age']:
    cust_df[col].fillna(-1, inplace=True)

cust_df["club_member_status"] = Categorize().fit_transform(
    cust_df[['club_member_status']])["club_member_status"]
cust_df["postal_code"] = Categorize().fit_transform(
    cust_df[['postal_code']])["postal_code"]
cust_df["fashion_news_frequency"] = Categorize().fit_transform(
    cust_df[['fashion_news_frequency']])["fashion_news_frequency"]

# %%
# articlesの前処理
print("articlesの前処理")
art_df['article_id'] = article_id_str_to_int(art_df['article_id'])

for col in art_df.columns:
    if art_df[col].dtype == 'object':
        art_df[col] = Categorize().fit_transform(art_df[[col]])[col]


# %%
# データセットのメモリ削減
print("データセット削減")
trans_df = reduce_mem_usage(trans_df)
cust_df = reduce_mem_usage(cust_df)
art_df = reduce_mem_usage(art_df)

# %%
print("tranの処理")
print(trans_df.head(5))
print("str変換")
trans_df['article_id'] = trans_df['article_id'].astype('str')
trans_df['customer_id'] = trans_df['customer_id'].astype('str')
print("str変換終わり")

# transactionsの前処理
trans_df['customer_id'] = customer_hex_id_to_int(trans_df['customer_id'])
trans_df['article_id'] = article_id_str_to_int(trans_df['article_id'])
trans_df['t_dat'] = pd.to_datetime(trans_df['t_dat'], format='%Y-%m-%d')
trans_df['week'] = 104 - (trans_df['t_dat'].max() -
                          trans_df['t_dat']).dt.days // 7
print("ここで落ちてる？")





# %%
# 保存
trans_df.to_parquet(input_dir / 'transactions_train.parquet')
cust_df.to_parquet(input_dir / 'customers.parquet')
art_df.to_parquet(input_dir / 'articles.parquet')


# %%

# 保存(0.05sample)
sample = 0.05
cust_df_sample = cust_df.sample(frac=sample, replace=False)
cust_df_sample_ids = set(cust_df_sample['customer_id'])
trans_df_sample = trans_df[trans_df["customer_id"].isin(
    cust_df_sample_ids)]
art_df_sample_ids = set(trans_df_sample["article_id"])
art_df_sample = art_df[art_df["article_id"].isin(art_df_sample_ids)]


cust_df_sample.to_parquet(
    input_dir / f'customers_sample_{sample}.parquet',
    index=False)
trans_df_sample.to_parquet(
    input_dir / f'transactions_train_sample_{sample}.parquet',
    index=False)
art_df_sample.to_parquet(
    input_dir / f'articles_train_sample_{sample}.parquet',
    index=False)


# %%

# validateの用意
val_week_purchases_by_cust = defaultdict(list)
val_week_purchases_by_cust.update(
    trans_df[trans_df.week == trans_df.week.max()]
    .groupby('customer_id')['article_id']
    .apply(list)
    .to_dict()
)

pd.to_pickle(dict(val_week_purchases_by_cust),
             output_dir / 'val_week_purchases_by_cust.pkl')

# %%

# sample subの用意
# sample_sub = pd.read_csv(input_dir / 'sample_submission.csv')
# valid_gt = customer_hex_id_to_int(sample_sub.customer_id) \
#     .map(val_week_purchases_by_cust) \
#     .apply(lambda xx: ' '.join('0' + str(x) for x in xx))
# sample_sub.prediction = valid_gt
# sample_sub.to_parquet(
#     output_dir /
#     'validation_ground_truth.parquet',
#     index=False)
