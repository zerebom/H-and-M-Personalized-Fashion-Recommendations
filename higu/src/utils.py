# helper functions
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
