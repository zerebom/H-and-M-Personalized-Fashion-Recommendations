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
    from utils import (Categorize, article_id_int_to_str,
                       article_id_str_to_int, customer_hex_id_to_int,
                       reduce_mem_usage)


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

#%%
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



class BoughtItemsAtInferencePhase(AbstractCGBlock):
    def __init__(
            self,
            y_cdf,
            key_col='customer_id',
            item_col='article_id',
            ):

        self.y_cdf = y_cdf
        self.key_col = key_col
        self.item_col = item_col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        out_df = self.y_cdf\
            .to_pandas()\
            .groupby(self.key_col)[self.item_col]\
            .apply(list).to_dict()

        return out_df


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
        self.popular_items = input_df\
            .loc[input_df['week'] == input_df['week'].max()][self.item_col]\
            .value_counts()\
            .to_pandas()\
            .index[:self.n]\
            .values

        for cust_id in self.customer_ids:
            pop_item_dic[cust_id] = list(self.popular_items)

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
        col_name = "most_cat_" + self.target_col
        out_df = input_df.groupby([self.key_col, self.target_col])\
            .size()\
            .reset_index()\
            .sort_values([self.key_col, 0], ascending=False)\
            .groupby(self.key_col)[self.target_col]\
            .nth(1).\
            reset_index().rename({self.target_col: col_name} , axis=1)
        out_df[col_name] = out_df[col_name].astype('category')
        return out_df  # %%

# ================================= metrics ===================


def mapk(actual: list, predicted: list, k=10) -> float:
    '''
    usage:
    mapk(merged['valid_true'].map(lambda x: x.split()), merged['valid_pred'].map(lambda x: x.split()), k=12)

    '''
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def apk(actual: list, predicted: list, k=10) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

# ================================= utils ===================

def reduce_mem_usage(_df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    df = _df.copy()
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'datetime64[ns]':
          continue

        if str(col_type) in ['object', 'category']:
          df[col] = df[col].astype('category')
          continue

        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] in ['int', 'uin']:
            if c_min > np.iinfo(
                    np.int8).min and c_max < np.iinfo(
                    np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            #     df[col] = df[col].astype(np.float16)
            if c_min > np.finfo(
                    np.float32).min and c_max < np.finfo(
                    np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


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


# %%

def arts_id_list2str(x: list):
    return '0' + ' 0'.join([str(v) for v in x[:12]])

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


# _my_sub_df = my_sub_df.set_index('customer_id')
# _sample_sub_df = sample_sub_df.set_index('customer_id')
# _sample_sub_df.update(_my_sub_df)
# sub_df = _sample_sub_df.reset_index()

my_sub_df

# %%
def visualize_importance(models, feat_train_df):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        print(i)
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importances_
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df], axis=0, ignore_index=True
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.2)))
    sns.boxenplot(
        data=feature_importance_df,
        y="column",
        x="feature_importance",
        order=order,
        ax=ax,
        palette="viridis",
    )
    fig.tight_layout()
    ax.grid()
    return fig, ax
visualize_importance([clf], train_X)
