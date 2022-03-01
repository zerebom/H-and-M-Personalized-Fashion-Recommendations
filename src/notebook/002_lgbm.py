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
    sys.path.append("../")
    from utils import (Categorize, article_id_int_to_str,
                       article_id_str_to_int, customer_hex_id_to_int,
                       reduce_mem_usage)


# %%
root_dir = Path("/home/kokoro/h_and_m")
input_dir = root_dir / "input"
output_dir = root_dir / "output"
DRY_RUN = True

# %%


if not DRY_RUN:
    transactions = pd.read_parquet(input_dir / 'transactions_train.parquet')
    customers = pd.read_parquet(input_dir / 'customers.parquet')
    articles = pd.read_parquet(input_dir / 'articles.parquet')
else:
    sample = 0.05
    transactions = pd.read_parquet(
        input_dir / f"transactions_train_sample_{sample}.parquet")
    customers = pd.read_parquet(
        input_dir / f'customers_sample_{sample}.parquet')
    articles = pd.read_parquet(input_dir /
                               f'articles_train_sample_{sample}.parquet')


#%%
trans_cdf = cudf.DataFrame.from_pandas(transactions)
art_cdf = cudf.DataFrame.from_pandas(articles)
cust_cdf = cudf.DataFrame.from_pandas(customers)

# %%


#%%
if 'max_week' not in trans_cdf.columns:
    max_week_cdf = trans_cdf.groupby('customer_id')['week'].max().reset_index()
    max_week_cdf.columns = ['customer_id', 'max_week']
    trans_cdf = trans_cdf.merge(max_week_cdf, on='customer_id', how='left')
    trans_cdf['diff_week'] = trans_cdf['max_week'] - trans_cdf['week']
    recent_bought_dic = trans_cdf.\
        loc[trans_cdf['diff_week'] <= 2, ['customer_id', 'article_id']]\
        .to_pandas()\
        .groupby('customer_id')['article_id']\
        .apply(list)\
        .to_dict()  # {'cust_id': ['art_id_1',...]}



# %%

pop_item_dic = {}
popular_items = trans_cdf.loc[trans_cdf['week'] == trans_cdf['week'].max()]['article_id'].value_counts().to_pandas().index[:12].values

all_custs = cust_cdf['customer_id'].unique().to_pandas().values
for cust_id in all_custs:
    pop_item_dic[cust_id] = list(popular_items)
pop_item_dic

# %%

test_week = transactions.week.max()
transactions = transactions[transactions.week > transactions.week.max() - 10]

# %%
c2weeks = transactions.groupby('customer_id')['week'].unique()
c2weeks

# %%
c2weeks2shifted_weeks = {}

for c_id, weeks in c2weeks.items():
    c2weeks2shifted_weeks[c_id] = {}
    for i in range(weeks.shape[0] - 1):
        c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i + 1]
    c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week

# %%
candidates_last_purchase = transactions.copy()
weeks = []
for i, (c_id, week) in enumerate(
        zip(transactions['customer_id'], transactions['week'])):
    weeks.append(c2weeks2shifted_weeks[c_id][week])

candidates_last_purchase.week = weeks

# %%
mean_price = transactions \
    .groupby(['week', 'article_id'])['price'].mean()
sales = transactions \
    .groupby('week')['article_id'].value_counts() \
    .groupby('week').rank(method='dense', ascending=False) \
    .groupby('week').head(12).rename('bestseller_rank')
bestsellers_previous_week = pd.merge(
    sales, mean_price, on=[
        'week', 'article_id']).reset_index()
bestsellers_previous_week.week += 1

# %%
unique_transactions = transactions \
    .groupby(['week', 'customer_id']) \
    .head(1) \
    .drop(columns=['article_id', 'price']) \
    .copy()
# %%
candidates_bestsellers = pd.merge(
    unique_transactions,
    bestsellers_previous_week,
    on='week',
)
test_set_transactions = unique_transactions.drop_duplicates(
    'customer_id').reset_index(drop=True)
# %%

test_set_transactions.week = test_week
candidates_bestsellers_test_week = pd.merge(
    test_set_transactions,
    bestsellers_previous_week,
    on='week'
)

candidates_bestsellers = pd.concat(
    [candidates_bestsellers, candidates_bestsellers_test_week])
candidates_bestsellers.drop(columns='bestseller_rank', inplace=True)
# %%

transactions['purchased'] = 1
data = pd.concat([transactions,
                  candidates_last_purchase,
                  candidates_bestsellers])
data.purchased.fillna(0, inplace=True)

data.purchased.mean()
# %%
data = pd.merge(
    data,
    bestsellers_previous_week[['week', 'article_id', 'bestseller_rank']],
    on=['week', 'article_id'],
    how='left'
)
data = data[data.week != data.week.min()].copy()
data.bestseller_rank.fillna(999, inplace=True)
# %%
data = pd.merge(data, articles, on='article_id', how='left')
data = pd.merge(data, customers, on='customer_id', how='left')
data.sort_values(['week', 'customer_id'], inplace=True)
data.reset_index(drop=True, inplace=True)
# %%
train = data[data.week != test_week]
test = data[data.week == test_week].drop_duplicates(
    ['customer_id', 'article_id', 'sales_channel_id']).copy()
train_baskets = train.groupby(['week', 'customer_id'])[
    'article_id'].count().values
columns_to_use = [
    'article_id',
    'product_type_no',
    'graphical_appearance_no',
    'colour_group_code',
    'perceived_colour_value_id',
    'perceived_colour_master_id',
    'department_no',
    'index_code',
    'index_group_no',
    'section_no',
    'garment_group_no',
    'FN',
    'Active',
    'club_member_status',
    'fashion_news_frequency',
    'age',
    'postal_code',
    'bestseller_rank']
# %%
train_X = train[columns_to_use]
train_y = train['purchased']

test_X = test[columns_to_use]
