# helper functions
from pathlib import Path
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from time import time

import numpy as np
import pandas as pd
import cudf
from sklearn.base import BaseEstimator, TransformerMixin

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG

def jsonKeys2int(x):
    """
    文字列keyをintに変換する
    """
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def get_var_names(vars):
    # 変数名を取得する
    names=[]
    for var in vars:
        for k,v in globals().items():
            if id(v) == id(var) and not k.startswith('_'):
                names.append(k)
    return names

def read_cdf(input_dir,DRY_RUN):
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
    return trans_cdf,cust_cdf,art_cdf


def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    #ref: https://nigimitama.hatenablog.jp/entry/2021/01/27/084458
    if not logger.hasHandlers():
        sh = StreamHandler()
        sh.setLevel(DEBUG)
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        fh = FileHandler(log_folder) #fh = file handler
        fh.setLevel(DEBUG)
        fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
    return logger


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix:
        format_str = str(prefix) + format_str
    if suffix:
        format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)

def phase_date(diff_days: int):
    # 最終日からx日前の日付をdatetimeで返す
    last_date = date(year=2020, month=9, day=22)
    return datetime.combine(
        last_date -
        timedelta(
            days=diff_days),
        datetime.min.time())

def arts_id_list2str(x: list):
    return '0' + ' 0'.join([str(v) for v in x[:12]])

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

# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635


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


def squeeze_pred_df_to_submit_format(pred_df:pd.DataFrame, fill_logic=None, args=None)->pd.DataFrame:
    '''
    (cust_id,art_id,pred)の出力形式から提出形式のデータフレームを作成する。
    customer_id毎に、article_idの空白文字区切りの文字列を作成する。

    予測アイテムが12個未満の場合、fill_logicによって埋める。

    pred_df(columns=[customer_id, article_id, pred, target])
    ↓
    submit_format_df(columns=[customer_id, article_ids, preds, targets])
    '''
    if fill_logic is None:
        fill_logic = lambda x: x
        args = None

    submit_format_df = (pred_df
        .sort_values(by=['customer_id', 'pred'], ascending=False)
        .groupby('customer_id')['article_id']
        .apply(list)
        .apply(fill_logic, args=args)
        .apply(arts_id_list2str)
        .reset_index()
        .set_index("customer_id")
        )

    if "target" in pred_df.columns:
        submit_format_df["target"] = (pred_df[pred_df['target'] == 1]
        .groupby('customer_id')['article_id']
        .apply(list)
        .apply(fill_logic, args=args)
        .apply(arts_id_list2str)
        )

        submit_format_df["target"] = submit_format_df["target"].replace(np.nan, '')


    submit_format_df = submit_format_df.rename(columns={'article_id': 'prediction'})

    return submit_format_df


def convert_sub_df_customer_id_to_str(input_dir:Path, sub_fmt_df:pd.DataFrame)-> pd.DataFrame:
    # sub_dfの12桁のcustomer_idを元のcustomer_idに変換する
    sample_df = pd.read_csv(input_dir / 'sample_submission.csv')
    converted_sub_fmt_df = sample_df.copy()

    hex_id2idx_dic = {v: k for k, v in customer_hex_id_to_int(
        sample_df['customer_id']).to_dict().items()}

    sub_list = ['' for _ in range(len(sample_df))]
    sub_dic = sub_fmt_df['prediction'].to_dict()
    for hex_id, pred in sub_dic.items():
        sub_list[hex_id2idx_dic[hex_id]] = pred

    converted_sub_fmt_df['prediction'] = sub_list
    return converted_sub_fmt_df
