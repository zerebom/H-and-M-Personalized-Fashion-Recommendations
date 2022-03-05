# %%
import os
import pprint
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from time import time

import cudf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

if True:
    sys.path.append("../")
    from utils import (Categorize, article_id_int_to_str,
                       article_id_str_to_int, customer_hex_id_to_int,
                       reduce_mem_usage)


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


to_cdf = cudf.DataFrame.from_pandas
trans_cdf = to_cdf(transactions)
art_cdf = to_cdf(articles)
cust_cdf = to_cdf(customers)


# ================================= time =================================
def phase_date(diff_days: int):
    # 最終日からx日前の日付をdatetimeで返す
    last_date = date(year=2020, month=9, day=22)
    return datetime.combine(
        last_date -
        timedelta(
            days=diff_days),
        datetime.min.time())


datetime_dic = {
    "X": {
        "train": {"start_date": phase_date(28), "end_date": phase_date(14)},
        "valid": {"start_date": phase_date(21), "end_date": phase_date(7)},
        "test": {"start_date": phase_date(14), "end_date": phase_date(0)},

    },
    "y": {
        "train": {"start_date": phase_date(14), "end_date": phase_date(7)},
        "valid": {"start_date": phase_date(7), "end_date": phase_date(0)},
    }
}

# ================================= preprocessor =========================


def make_y_cdf(
        transactions: cudf.DataFrame,
        start_date: datetime,
        end_date: datetime):
    y_cdf = transactions.query(
        f'@start_date<t_dat<@end_date')[['customer_id', 'article_id']].drop_duplicates()
    y_cdf['purchased'] = 1
    return y_cdf


def clip_transactions(transactions: cudf.DataFrame, end_date: datetime):
    return transactions.query(f't_dat<@end_date')

# ================================= candidate_generation =================


def candidates_dict2df(candidates_dict):
    tuples = []
    for cust, articles in tqdm(candidates_dict.items()):
        for art in articles:
            tuples.append((cust, art))

    df = pd.DataFrame(tuples)
    df.columns = ['customer_id', 'article_id']
    cdf = to_cdf(df)
    cdf = cdf.drop_duplicates().reset_index(drop=True)
    return cdf


class AbstractCGBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        raise NotImplementedError()

    def get_cudf(self):
        return to_cdf(self.out_df)


class LastNWeekArticles(AbstractCGBlock):
    def __init__(
            self,
            key_col='customer_id',
            item_col='article_id',
            n_weeks=2):
        self.key_col = key_col
        self.item_col = item_col
        self.n_weeks = n_weeks
        self.out_keys = [key_col, item_col]

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        input_df = _input_df.copy()

        week_df = input_df.\
            groupby(self.key_col)['week'].max()\
            .reset_index().\
            rename({'week': 'max_week'}, axis=1)

        input_df = input_df.merge(week_df, on=self.key_col, how='left')
        input_df['diff_week'] = input_df['max_week'] - input_df['week']
        self.out_df = input_df.loc[input_df['diff_week']
                                   <= self.n_weeks, self.out_keys].to_pandas()

        out_dic = self.out_df.groupby(self.key_col)[
            self.item_col].apply(list).to_dict()
        return out_dic


class PopularItemsoftheLastWeeks(AbstractCGBlock):
    def __init__(
            self,
            customer_ids,
            key_col='customer_id',
            item_col='article_id',
            n=12):
        self.customer_ids = customer_ids
        self.key_col = key_col
        self.item_col = item_col
        self.n = n

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        input_df = _input_df.copy()
        pop_item_dic = {}
        popular_items = input_df\
            .loc[input_df['week'] == input_df['week'].max()][self.item_col]\
            .value_counts()\
            .to_pandas()\
            .index[:self.n]\
            .values

        for cust_id in self.customer_ids:
            pop_item_dic[cust_id] = list(popular_items)

        return pop_item_dic
# ================================= feature_Generation ===================


class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        raise NotImplementedError()


class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self,
                 key_col,
                 target_col,
                 agg_list):

        self.key_col = key_col
        self.target_col = target_col
        self.agg_list = agg_list

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        out_df = input_df.groupby(
            self.key_col)[
            self.target_col].agg(
            self.agg_list)
        out_df.columns = ['_'.join([self.target_col, col])
                          for col in out_df.columns]

        out_df.reset_index(inplace=True)

        return out_df


class ModeCategoryBlock(AbstractBaseBlock):
    # key_colごとにtarget_colのmodeを算出
    def __init__(self, key_col, target_col):
        self.key_col = key_col
        self.target_col = target_col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        out_df = input_df.groupby([self.key_col, self.target_col])\
            .size()\
            .reset_index()\
            .sort_values([self.key_col, 0], ascending=False)\
            .groupby(self.key_col)[self.target_col]\
            .nth(1).\
            reset_index().rename({self.target_col: "most_cat_" + self.target_col}, axis=1)
        return out_df  # %%

# %%

def candidate_generation(trans_cdf, art_cdf, cust_cdf):
    customer_ids = list(cust_cdf['customer_id'].to_pandas().unique())
    blocks = [
        *[PopularItemsoftheLastWeeks(customer_ids)],
        # TODO: 全customer_idを含んでるCGが最初じゃないとだめ
        *[LastNWeekArticles(n_weeks=2)]

    ]
    candidates_dict = {}
    for i, block in enumerate(blocks):
        if i == 0:
            candidates_dict = block.fit(trans_cdf)
        else:
            new_dic = block.fit(trans_cdf)
            for key, value in candidates_dict.items():
                if key in new_dic:
                    value.extend(new_dic[key])
    candidates_cdf = candidates_dict2df(candidates_dict)
    return candidates_cdf

def feature_generation(trans_cdf, art_cdf, cust_cdf):
    recent_bought_cdf = cudf.DataFrame.from_pandas(
        trans_cdf.sort_values(['customer_id', 't_dat'], ascending=False)
        .to_pandas()
        .groupby('customer_id')
        .head(30))

    feature_df = recent_bought_cdf\
        .merge(art_cdf, on='article_id', how='left')\
        .merge(cust_cdf, on='customer_id', how='left')

    drop_cols = feature_df.to_pandas().filter(regex='.*_name').columns

    feature_df.drop(columns=drop_cols, inplace=True)
    art_feat_cdf = art_cdf[['article_id']]
    cust_feat_cdf = cust_cdf[['customer_id']]

    item_category_cols = [
        'product_type_no',
        'graphical_appearance_no',
        'colour_group_code',
        'perceived_colour_value_id',
        'perceived_colour_master_id',
        'department_no',
        'index_code',
        'index_group_no',
        'section_no',
        'garment_group_no']

    binary_cols = [
        "FN",
        "Active",
        'club_member_status',
        'fashion_news_frequency']
    agg_list = ['mean', 'max', 'min', 'std', 'median']

    blocks = [
        *[TargetEncodingBlock('article_id', 'age', agg_list)],
        *[TargetEncodingBlock('article_id', col, ['mean']) for col in binary_cols],
        *[TargetEncodingBlock('customer_id', col, agg_list) for col in ['price', 'sales_channel_id']],
        *[ModeCategoryBlock('customer_id', col) for col in item_category_cols],
    ]

    for block in tqdm(blocks):
        if block.key_col == 'article_id':
            art_feat_cdf = art_feat_cdf.merge(
                block.fit(feature_df), how='left', on=block.key_col)
        elif block.key_col == 'customer_id':
            cust_feat_cdf = cust_feat_cdf.merge(
                block.fit(feature_df),
                how='left',
                on=block.key_col)

    return art_feat_cdf, cust_feat_cdf


def make_trainable_data(raw_trans_cdf, art_cdf, cust_cdf, phase):
    X_end_date = datetime_dic['X'][phase]['end_date']

    y_cdf = cudf.DataFrame()
    if phase is not 'test':
        y_start_date = datetime_dic['y'][phase]['start_date']
        y_end_date = datetime_dic['y'][phase]['end_date']
        y_cdf = make_y_cdf(raw_trans_cdf, y_start_date, y_end_date)

    trans_cdf = clip_transactions(raw_trans_cdf, X_end_date)

    print("start candidate generation")
    candidates_cdf = candidate_generation(trans_cdf, art_cdf, cust_cdf)

    print("start feature generation")
    art_feat_cdf, cust_feat_cdf = feature_generation(
        trans_cdf, art_cdf, cust_cdf)

    X = candidates_cdf\
        .merge(cust_feat_cdf, how='left', on='customer_id')\
        .merge(art_feat_cdf, how='left', on='article_id')

    if phase is 'test':
        return X, None

    y = X.merge(y_cdf, how='left', on=['customer_id', 'article_id'])[
            'purchased'].fillna(0).astype(int)
    return X, y


train_X, train_y = make_trainable_data(trans_cdf,art_cdf,cust_cdf,phase='train')
valid_X, valid_y = make_trainable_data(trans_cdf,art_cdf,cust_cdf,phase='valid')
test_X, _ = make_trainable_data(trans_cdf,art_cdf,cust_cdf,phase='test')


# %%


