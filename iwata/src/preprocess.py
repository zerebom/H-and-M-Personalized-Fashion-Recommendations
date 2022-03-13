import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from path import DATA_DIR
# 自作関数
from path import DATA_DIR
from preprocess import *
from utils import *

from utils_logger import set_logger
logger = set_logger("__main__")

def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)

def hex_id_to_int(str):
    return int(str[-16:], 16)

def article_id_str_to_int(series):
    return series.astype('int32')

def article_id_int_to_str(series):
    return '0' + series.astype('str')


class Categorize(BaseEstimator, TransformerMixin):
    def __init__(self, min_examples=0):
        self.min_examples = min_examples
        self.categories = []

    def fit(self, X):
        for i in range(X.shape[1]):
            vc = X.iloc[:, i].value_counts()
            self.categories.append(vc[vc > self.min_examples].index.tolist())
        return self

    def transform(self, X):
        data = {X.columns[i]: pd.Categorical(
            X.iloc[:, i], categories=self.categories[i]).codes for i in range(X.shape[1])}
        return pd.DataFrame(data=data)

@stop_watch(logger)
def preprocess_transactions(df):
    df['customer_id'] = customer_hex_id_to_int(df['customer_id'])
    df['article_id'] = df['article_id'].astype('str')
    df['article_id'] = article_id_str_to_int(df['article_id'])
    df['t_dat'] = pd.to_datetime(df['t_dat'], format='%Y-%m-%d')
    df['week'] = 104 - (df['t_dat'].max() - df['t_dat']).dt.days // 7
    return df

@stop_watch(logger)
def preprocess_customers(df):
    df['customer_id'] = customer_hex_id_to_int(df['customer_id'])

    for col in ['FN', 'Active', 'age']:
        df[col].fillna(-1, inplace=True)

    df["club_member_status"] = Categorize().fit_transform(
        df[['club_member_status']])["club_member_status"]
    df["postal_code"] = Categorize().fit_transform(
        df[['postal_code']])["postal_code"]
    df["fashion_news_frequency"] = Categorize().fit_transform(
        df[['fashion_news_frequency']])["fashion_news_frequency"]
    return df

@stop_watch(logger)
def preprocess_articles(df):
    df['article_id'] = df['article_id'].astype('str')
    df['article_id'] = article_id_str_to_int(df['article_id'])
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = Categorize().fit_transform(df[[col]])[col]
    return df

@stop_watch(logger)
def make_small_data(cust_df, trans_df, art_df, sample_rate):
    df_sample_dict = {}
    df_sample_dict["customer"] = cust_df.sample(frac=sample_rate, replace=False)
    
    cust_df_sample_ids = set(df_sample_dict["customer"])
    df_sample_dict["transaction"] = trans_df[trans_df["customer_id"].isin(cust_df_sample_ids)]
    art_df_sample_ids = set(df_sample_dict["transaction"])
    df_sample_dict["article"] = art_df[art_df["article_id"].isin(art_df_sample_ids)]

    for key in df_sample_dict.keys():
        df_sample_dict[key].to_parquet(DATA_DIR / f'{key}_{sample_rate}.parquet',index=False)
