from datetime import date, datetime, timedelta
import pandas as pd
from typing import List

# 心くんのutilsから拝借
def phase_date(diff_days: int):
    # 最終日からx日前の日付をdatetimeで返す
    last_date = date(year=2020, month=9, day=22)
    return datetime.combine(
        last_date -
        timedelta(
            days=diff_days),
        datetime.min.time())

# モデルに載せる用の関数
def get_remove_item(df : pd.DataFrame, end_date:datetime, no_buy_days:int) -> List[int]:
    """ n日間買われていない商品はこれからも買われないだろうと考えて、モデル学習 or 推薦から除外するための関数

    Args:
        df (pd.DataFrame): transactions_train.csv を読み込んだdfを指定
        end_date (datetime): 学習期間の1日前を指定
        no_buy_days (int): n日間のnを指定

    Returns:
        List[int]: 
    """

    df["t_dat_datetime"] = pd.to_datetime(df["t_dat"])
    
    # datetimeに変換
    # start_dateはデータの始まりの日にする
    start_date = datetime.datetime.strptime("2018-09-20", "%Y-%m-%d")
    no_buy_days = datetime.timedelta(days=no_buy_days)
    
    # no_buyのラインを決める    
    threshold_date = end_date - no_buy_days
    assert start_date < threshold_date, "開始日より前はだめ"
    
    #期間を限定
    df_old_raw = df[(start_date <= df["t_dat_datetime"]) & (df["t_dat_datetime"] <= end_date)] 
    df_old = df_old_raw.groupby("article_id")["t_dat_datetime"].max().reset_index().rename(columns={"t_dat_datetime":"t_dat_max_datetime"})
    
    df_old["no_buy_flg"] =  (df_old["t_dat_max_datetime"] <= threshold_date).astype(int) # "2019-09-20"が最後の値で、1日前なら、2019-09-19以降買われていないものを知りたい
    
    remove_items = df_old[df_old["no_buy_flg"]==1]["article_id"].tolist() #groupbyされてるからdistinct しなくてOK 
    
    return remove_items
