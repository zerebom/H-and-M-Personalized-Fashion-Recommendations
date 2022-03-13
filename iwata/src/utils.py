# helper functions
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import time
from functools import wraps
from logging import Logger

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


# ref : https://qiita.com/hisatoshi/items/7354c76a4412dffc4fd7
def stop_watch(logger: Logger): #デコレータに引数を持たせたいのでこの形にする
    def _stop_watch(func):
        """関数の実行時間を出力するデコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 処理開始直前の時間
            start_time = time.time()
            # 処理実行
            result = func(*args, **kwargs)
            # 処理終了直後の時間から処理時間を算出
            elapsed_time = time.time() - start_time

            # 処理時間を出力
            logger.info(f"{func.__name__} : {round(elapsed_time / 60, 2)} min")
            return result
        return wrapper
    return _stop_watch


