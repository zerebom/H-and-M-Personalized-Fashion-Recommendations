from typing import Any, Dict, List, Optional, Tuple

import cudf
import nmslib
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import customer_hex_id_to_int

to_cdf = cudf.DataFrame.from_pandas


class AbstractCGBlock:
    def fit(
        self, trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers
    ) -> pd.DataFrame:

        self.out_cdf = self.transform(trans_cdf, art_cdf, cust_cdf, target_customers)
        self.out_cdf["candidate_block_name"] = self.__class__.__name__
        self.out_cdf["candidate_block_name"] = self.out_cdf[
            "candidate_block_name"
        ].astype("category")
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

            # 候補集合数・候補集合内の正例数・候補集合内の正例率・正例カバレッジ率
            logger.info(
                f"candidate_cnt:{candidate_cnt}, pos_in_cg_cnt: {pos_in_cg_cnt}, pos_in_cg_ratio:{pos_ratio}, pos_coverage:{pos_coverage}"
            )
            logger.info(
                f"customer_uniques: {customer_cnt}, article_uniques: {article_cnt}"
            )
        else:
            logger.info(
                "either y_cdf or out_cdf isn't defined. so skip cg block inspection."
            )


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
        input_cdf["diff_week"] = (
            input_cdf["max_week"] - input_cdf["week"]
        )  # 各商品とユーザの最終購入週の差分
        self.out_cdf = input_cdf.loc[
            input_cdf["diff_week"] <= self.n_weeks, self.out_keys
        ]  # 差分がn週間以内のものを取得

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

        topN = np.array(
            input_last_week_cdf["article_id"]
            .value_counts()
            .to_pandas()
            .index.astype("int32")[: self.n]
        )

        out_df_list = []
        _out_df = pd.DataFrame({"article_id": topN})
        for customer in tqdm(target_customers):
            _out_df["customer_id"] = customer
            out_df_list.append(_out_df.copy())

        self.out_cdf = cudf.from_pandas(pd.concat(out_df_list).reset_index(drop=True))
        return self.out_cdf

    # class PopularItemsoftheLastWeeks(AbstractCGBlock):
    #     """
    #     最終購入週の人気アイテムtopNを返す
    #     TODO: 最終購入数以外の商品も取得できるようにする

    #     """

    #     def __init__(
    #         self, customer_ids, key_col="customer_id", item_col="article_id", n=12
    #     ):
    #         self.customer_ids = customer_ids
    #         self.key_col = key_col
    #         self.item_col = item_col
    #         self.n = n

    #     def transform(self, _input_cdf):
    #         input_cdf = _input_cdf.copy()
    #         pop_item_dic = {}
    #         self.popular_items = (
    #             input_cdf.loc[input_cdf["week"] == input_cdf["week"].max()][self.item_col]
    #             .value_counts()
    #             .to_pandas()
    #             .index[: self.n]
    #             .values
    #         )

    #         for cust_id in self.customer_ids:
    #             pop_item_dic[cust_id] = list(self.popular_items)

    #         return pop_item_dic

    """
    ルールベース候補生成
    """

    def __init__(
        self, key_col="customer_id", item_col="article_id", rule_based_cdf=None
    ):
        self.key_col = key_col
        self.item_col = item_col
        self.rule_based_cdf = rule_based_cdf
        self.out_keys = [key_col, item_col]

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = self.rule_based_cdf  # ルールベースのcdfを読み込み
        self.out_cdf = input_cdf[
            input_cdf["customer_id"].isin(target_customers)
        ]  # bought_cust_ids

        return self.out_cdf
