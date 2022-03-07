# %%
import gc
import os
import pprint
import sys
from collections import defaultdict
from pathlib import Path
from time import time
from datetime import date, datetime, timedelta

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
    from features import (AbstractBaseBlock, ModeCategoryBlock,
                          TargetEncodingBlock)
    from metrics import apk, mapk
    from utils import (Categorize, article_id_int_to_str,arts_id_list2str,
                       article_id_str_to_int, customer_hex_id_to_int,phase_date,
                       reduce_mem_usage)
    from eda_tools import visualize_importance


root_dir = Path("/home/kokoro/h_and_m")
input_dir = root_dir / "input"
output_dir = root_dir / "output"
DRY_RUN = False

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

# %%

def candidate_generation(trans_cdf, art_cdf, cust_cdf,y_cdf=None):
    customer_ids = list(cust_cdf['customer_id'].to_pandas().unique())
    blocks = [
        # *[PopularItemsoftheLastWeeks(customer_ids)],
        # TODO: 全customer_idを含んでるCGが最初じゃないとだめ
        *[LastNWeekArticles(n_weeks=2)]
    ]

    if y_cdf is not None:
        blocks.append(BoughtItemsAtInferencePhase(y_cdf))

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

    _art_cdf = art_cdf.copy()
    _cust_cdf = cust_cdf.copy()

    _cust_cdf = _cust_cdf.merge(cust_feat_cdf, on='customer_id', how='left')
    _art_cdf = _art_cdf.merge(art_feat_cdf, on='article_id', how='left')

    return _art_cdf, _cust_cdf



def make_trainable_data(raw_trans_cdf, art_cdf, cust_cdf, phase):
    X_end_date = datetime_dic['X'][phase]['end_date']

    y_cdf = None
    if phase is not 'test':
        y_start_date = datetime_dic['y'][phase]['start_date']
        y_end_date = datetime_dic['y'][phase]['end_date']
        y_cdf = make_y_cdf(raw_trans_cdf, y_start_date, y_end_date)

    trans_cdf = clip_transactions(raw_trans_cdf, X_end_date)

    print("start candidate generation")
    candidates_cdf = candidate_generation(trans_cdf, art_cdf, cust_cdf,y_cdf)

    print("start feature generation")
    art_feat_cdf, cust_feat_cdf = feature_generation(
        trans_cdf, art_cdf, cust_cdf)

    X = candidates_cdf\
        .merge(cust_feat_cdf, how='left', on='customer_id')\
        .merge(art_feat_cdf, how='left', on='article_id')

    recent_items = trans_cdf.groupby('article_id')['week'].max().reset_index().query('week>96')['article_id'].values
    X = X[X["article_id"].isin(recent_items)]

    key_cols = ['customer_id', 'article_id']
    key_df = X[key_cols]

    if phase is 'test':
        X = X.drop(columns=key_cols)
        return reduce_mem_usage(key_df.to_pandas()), reduce_mem_usage(X.to_pandas()), None

    y = X.merge(y_cdf, how='left', on=key_cols)['purchased'].fillna(0).astype(int)
    print("X_shape:",X.shape, "y_mean:",y.mean())
    X = X.drop(columns=key_cols)
    return reduce_mem_usage(key_df.to_pandas()), reduce_mem_usage(X.to_pandas()), y.to_pandas()



#%%
train_key, train_X, train_y = make_trainable_data(trans_cdf, art_cdf, cust_cdf, phase='train')
valid_key, valid_X, valid_y = make_trainable_data( trans_cdf, art_cdf, cust_cdf, phase='valid')
test_key, test_X, _ = make_trainable_data( trans_cdf, art_cdf, cust_cdf, phase='test')



# %%

param = {
    "objective": "binary",
    "metric": 'auc',
    "verbosity": -1,
    "boosting": "gbdt",
    "is_unbalance":True,
    "seed": 42,
    "learning_rate": 0.1,
    'colsample_bytree': .5,
    'subsample_freq': 3,
    'subsample': .9,
    "n_estimators": 1000,
    'importance_type': 'gain',
    'reg_lambda': 1.5,
    'reg_alpha': .1,
    'max_depth': 6,
    'num_leaves':45
}

cat_features = train_X.select_dtypes('category').columns.tolist()
es = lgb.early_stopping(500, verbose=1)
clf = lgb.LGBMClassifier(**param)
clf.fit(
    train_X,
    train_y,
    eval_set=[(valid_X, valid_y)],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)


# %%

valid_key['pred'] = clf.predict_proba(valid_X)[:, 1]
valid_key['target'] = valid_y
valid_key['pred'].plot.hist()




#%%
valid_map_df = valid_key\
    .sort_values(by=['customer_id', 'pred'], ascending=False)\
    .groupby('customer_id')['article_id']\
    .apply(list)\
    .apply(lambda x: arts_id_list2str(x))\
    .reset_index()\
    .set_index("customer_id")

valid_map_df["target"] = valid_key[valid_key['target'] == 1]\
    .groupby('customer_id')['article_id']\
    .apply(list)\
    .apply(lambda x: arts_id_list2str(x))
valid_map_df['target'] = valid_map_df['target'].replace(np.nan, '')
mapk(
    valid_map_df['article_id'].map(
        lambda x: x.split()), valid_map_df['target'].map(
            lambda x: x.split()), k=12)



# %%



pop = PopularItemsoftheLastWeeks([1])
pop.fit(trans_cdf)
pop_items = list(pop.popular_items)

def fill_pop_items(pred_ids, pop_items):
    pred_ids.extend(pop_items)
    return pred_ids[:12]

test_y = clf.predict_proba(test_X)[:, 1]
test_key['pred'] = test_y
test_key = test_key.sort_values(by=['customer_id', 'pred'], ascending=False)
my_sub_df = test_key.groupby('customer_id')['article_id'].apply(list).apply(lambda x: fill_pop_items(x,pop_items))
my_sub_df = my_sub_df.apply(arts_id_list2str).reset_index()
my_sub_df.columns = ['customer_id', 'prediction']
my_sub_df



#%%

sample_sub_df = pd.read_csv(input_dir/'sample_submission.csv')
hex_id2idx_dic = {v:k for k,v in customer_hex_id_to_int(sample_sub_df['customer_id']).to_dict().items()}

#%%
sub_list = [arts_id_list2str(pop_items) for _ in range(len(sample_sub_df))]
sub_dic = my_sub_df.set_index('customer_id')['prediction'].to_dict()
for hex_id, pred in sub_dic.items():
    sub_list[hex_id2idx_dic[hex_id]] = pred

my_sub_df = sample_sub_df.copy()
my_sub_df['prediction'] = sub_list

# %%
prefix = datetime.now().strftime("%m_%d_%H_%M")
my_sub_df.to_csv(output_dir / f"submission_{prefix}.csv", index=False)
my_sub_df

# %%
visualize_importance([clf], train_X)
