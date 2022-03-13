
import cudf

# 自作関数
from path import DATA_DIR
from preprocess import *
from utils import *

from utils_logger import set_logger
logger = set_logger("__main__")

# データセット読み込み
@stop_watch(logger)
def read_dataset():
    trans_df = cudf.read_csv(DATA_DIR / "transactions_train.csv").to_pandas()
    cust_df = cudf.read_csv(DATA_DIR / 'customers.csv').to_pandas()
    art_df = cudf.read_csv(DATA_DIR / 'articles.csv').to_pandas()
    return trans_df,cust_df, art_df

# 前処理
@stop_watch(logger)
def preprocess(trans_df,cust_df, art_df):
    df_dict = {}
    df_dict["transaction"] = preprocess_transactions(trans_df)
    df_dict["customer"] = preprocess_customers(cust_df)
    df_dict["article"] = preprocess_articles(art_df)
    return df_dict

# メモリ削減して保存する
@stop_watch(logger)
def save_df(df_dict):
    for save_name, df in df_dict.items():
        df = reduce_mem_usage(df)
        df.to_parquet(DATA_DIR / f'{save_name}_train.parquet')


if __name__ == '__main__':
    trans_df,cust_df, art_df = read_dataset()
    print(trans_df.head(10))
    df_dict = preprocess(trans_df, cust_df, art_df)
    print(df_dict["transaction"])
    save_df(df_dict)
