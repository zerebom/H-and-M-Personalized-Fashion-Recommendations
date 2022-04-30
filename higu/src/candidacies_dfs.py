from typing import Any, Dict, List, Optional, Tuple

import cudf
import nmslib
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import customer_hex_id_to_int
import itertools

to_cdf = cudf.DataFrame.from_pandas


class AbstractCGBlock:
    def fit(self, trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers) -> pd.DataFrame:

        self.out_cdf = self.transform(trans_cdf, art_cdf, cust_cdf, target_customers)
        self.out_cdf["candidate_block_name"] = self.__class__.__name__
        self.out_cdf["candidate_block_name"] = self.out_cdf["candidate_block_name"].astype(
            "category"
        )
        self.inspect(y_cdf, logger)
        out_df = self.out_cdf.to_pandas()
        return out_df

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers) -> pd.DataFrame:
        """
        以下の処理を書いてください
        1. 引数 を加工して、[customer_id, article_id]ペアをレコードにもつself.out_cdf に格納する
        y_cdf, self.out_cdfがある時、inspectが走ります。
        """

        raise NotImplementedError()

    def inspect(self, y_cdf, logger):
        if hasattr(self, "out_cdf") and y_cdf is not None:
            customer_cnt = self.out_cdf["customer_id"].nunique()
            article_cnt = self.out_cdf["article_id"].nunique()
            merged_cdf = self.out_cdf.merge(
                y_cdf, on=["customer_id", "article_id"], how="inner"
            ).drop_duplicates()
            all_pos_cnt = len(y_cdf)

            pos_in_cg_cnt = len(merged_cdf)
            candidate_cnt = len(self.out_cdf)

            pos_ratio = round(pos_in_cg_cnt / candidate_cnt, 3)
            pos_coverage = round(pos_in_cg_cnt / all_pos_cnt, 3)

            logger.info(
                f"""
                候補集合内の正例率: {100*pos_ratio}%({pos_in_cg_cnt}/{candidate_cnt})
                正例カバレッジ率: {100*pos_coverage}%({pos_in_cg_cnt}/{all_pos_cnt})
                """
            )
            logger.info(f"uniques(customer:{customer_cnt}, article: {article_cnt})")
        else:
            logger.info("either y_cdf or out_cdf isn't defined. so skip cg block inspection.")


class LastNWeekArticles(AbstractCGBlock):
    """
    各ユーザごとに最終購買週からn週間前までの購入商品を取得する
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n_weeks=2):
        self.key_col = key_col
        self.item_col = item_col
        self.n_weeks = n_weeks
        self.out_keys = [key_col, item_col]

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = trans_cdf.copy()

        week_df = (
            input_cdf.groupby(self.key_col)["week"]
            .max()
            .reset_index()
            .rename({"week": "max_week"}, axis=1)
        )  # ユーザごとに最後に買った週を取得

        input_cdf = input_cdf.merge(week_df, on=self.key_col, how="left")
        input_cdf["diff_week"] = input_cdf["max_week"] - input_cdf["week"]  # 各商品とユーザの最終購入週の差分
        self.out_cdf = input_cdf.loc[
            input_cdf["diff_week"] <= self.n_weeks, self.out_keys
        ]  # 差分がn週間以内のものを取得

        return self.out_cdf


class LastBoughtNArticles(AbstractCGBlock):
    """
    各ユーザごとに直近の購入商品N個を取得する
    """

    def __init__(self, n, key_col="customer_id", item_col="article_id"):
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.out_keys = [key_col, item_col]

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_df = trans_cdf[self.out_keys].to_pandas()
        self.out_cdf = to_cdf(input_df.groupby("customer_id").tail(self.n))
        return self.out_cdf


class PopularItemsoftheLastWeeks(AbstractCGBlock):
    """
    最終購入週の人気アイテムtopNを返す
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n=12):
        self.key_col = key_col
        self.item_col = item_col
        self.n = n

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = trans_cdf.copy()
        last_week = input_cdf["week"].max()
        input_last_week_cdf = input_cdf.loc[input_cdf["week"] == last_week]

        TopN_articles = list(
            input_last_week_cdf["article_id"].value_counts().tail(self.n).to_pandas().index
        )

        if target_customers is None:
            target_customers = input_cdf["customer_id"].unique().to_pandas().values

        # Ref: https://note.nkmk.me/python-itertools-product/
        self.out_cdf = cudf.DataFrame(
            list(itertools.product(target_customers, TopN_articles)),
            columns=[self.key_col, self.item_col],
        )
        return self.out_cdf


class PairsWithLastBoughtNArticles(AbstractCGBlock):
    """
    各ユーザごとに直近の購入商品N個を取得する
    """

    def __init__(self, n, key_col="customer_id", item_col="article_id"):
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.out_keys = [key_col, item_col]

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        #from: https://www.kaggle.com/code/zerebom/article-id-pairs-in-3s-using-cudf/edit
        pair_df = cudf.read_csv("/home/kokoro/h_and_m/higu/input/pair_df.csv")[["article_id", "pair"]]

        self.out_cdf = to_cdf(
            trans_cdf.to_pandas()
            .groupby("customer_id")[["customer_id", "article_id"]]
            .tail(self.n)
        )

        self.out_cdf = (
            self.out_cdf.merge(pair_df, how="left", on="article_id")
            .drop(columns="article_id")
            .dropna()
            .rename(columns={"pair": "article_id"})
        )
        return self.out_cdf
