from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import os
import pickle
from collections import defaultdict
import cudf
import nmslib
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

to_cdf = cudf.DataFrame.from_pandas


class AbstractCGBlock:
    def __init__(self, use_cache):
        self.use_cache = use_cache
        self.name = self.__class__.__name__
        self.cache_dir = Path("/home/ec2-user/kaggle/h_and_m/data/candidates")

    def fit(
        self, trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers, target_week
    ) -> pd.DataFrame:

        file_name = self.cache_dir / f"{self.name}_{target_week}.pkl"

        if os.path.isfile(str(file_name)) and self.use_cache:
            with open(file_name, "rb") as f:
                self.out_cdf = pickle.load(f)
        else:
            if y_cdf is not None:
                target_customers = y_cdf["customer_id"].unique().to_pandas().values
                
            self.out_cdf = self.transform(trans_cdf, art_cdf, cust_cdf, target_customers)
            self.out_cdf["candidate_block_name"] = self.__class__.__name__
            self.out_cdf["candidate_block_name"] = self.out_cdf["candidate_block_name"].astype(
                "category"
            )

            with open(file_name, "wb") as f:
                pickle.dump(self.out_cdf, f)

        if y_cdf is not None:
            self.out_cdf = self.out_cdf.merge(y_cdf, how="left", on=["customer_id", "article_id"])
            self.out_cdf["y"] = self.out_cdf["y"].fillna(0).astype(int)
            self.out_cdf = self.out_cdf.drop_duplicates(
                subset=["customer_id", "article_id", "y"]
            ).reset_index(drop=True)

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

            pos_ratio = pos_in_cg_cnt / candidate_cnt
            pos_coverage = pos_in_cg_cnt / all_pos_cnt

            # logger.info(f" ")
            logger.info(f"候補集合内の正例率: {round(100*pos_ratio, 3)}%({pos_in_cg_cnt}/{candidate_cnt})")
            logger.info(f"正例カバレッジ率: {round(100*pos_coverage,2)}%({pos_in_cg_cnt}/{all_pos_cnt})")
            logger.info(f"ユニーク数: (customer:{customer_cnt}, article: {article_cnt})")
            # logger.info(" ")
        else:
            logger.info("either y_cdf or out_cdf isn't defined. so skip cg block inspection.")


class LastNWeekArticles(AbstractCGBlock):
    """
    各ユーザごとに最終購買週からn週間前までの購入商品を取得する
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n_weeks=2, use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n_weeks = n_weeks
        self.out_keys = [key_col, item_col]
        self.name = self.name + "_" + str(self.n_weeks)

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = trans_cdf[trans_cdf["customer_id"].isin(target_customers)]

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

    def __init__(self, n, key_col="customer_id", item_col="article_id", use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.out_keys = [key_col, item_col]

        self.name = self.name + "_" + str(self.n)

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_df = trans_cdf.loc[
            trans_cdf["customer_id"].isin(target_customers), self.out_keys
        ].to_pandas()
        self.out_cdf = to_cdf(input_df.groupby("customer_id").tail(self.n))
        return self.out_cdf


class PopularItemsoftheLastWeeks(AbstractCGBlock):
    """
    最終購入週の人気アイテムtopNを返す
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n=12, use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.name = self.name + "_" + str(self.n)

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = trans_cdf.copy()
        last_week = input_cdf["week"].max()
        input_last_week_cdf = input_cdf.loc[input_cdf["week"] == last_week]

        TopN_articles = list(
            input_last_week_cdf["article_id"].value_counts().head(self.n).to_pandas().index
        )

        if target_customers is None:
            target_customers = input_cdf["customer_id"].unique().to_pandas().values

        # Ref: https://note.nkmk.me/python-itertools-product/
        self.out_cdf = cudf.DataFrame(
            list(itertools.product(target_customers, TopN_articles)),
            columns=[self.key_col, self.item_col],
        )
        return self.out_cdf

class PopularItemsoftheEachCluster(AbstractCGBlock):
    """
    各クラスターごとに人気アイテムN個を取得する
    """

    def __init__(self, n, key_col="customer_id", item_col="article_id", use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.out_keys = [key_col, item_col]
        self.name = self.name + "_" + str(self.n)

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):

        input_dir = Path("/home/ec2-user/kaggle/h_and_m/output/log/exp18")
        cluster_cdf = cudf.read_csv(input_dir / "customer_kmeans_label.csv")
        cols = ["customer_id", "article_id", "cluster"]

        # 過去半年のtransactionだけ使う
        trans_cdf = trans_cdf.query("week >= 80")

        # 各クラスタごとに人気アイテムを取得([customer_id, article_id, cluster])
        popluar_items_each_cluster_cdf = to_cdf(
            trans_cdf.merge(cluster_cdf, how="left", on="customer_id")[cols]
            .groupby(["article_id", "cluster"])
            .size()
            .to_frame(name="size")
            .reset_index()
            .to_pandas()
            .sort_values(["size", "cluster"], ascending=False)
            .groupby("cluster")
            .head(self.n)
            .reset_index(drop=True)
        )

        cluster_cdf = cluster_cdf[cluster_cdf["customer_id"].isin(target_customers)]
        self.out_cdf = cluster_cdf.merge(popluar_items_each_cluster_cdf, how="left", on="cluster")[
            self.out_keys
        ]
        print(self.out_cdf.shape)

        del popluar_items_each_cluster_cdf

        return self.out_cdf

# class PopularItemsoftheEachAge(AbstractCGBlock):
#     """
#     各年齢ごとに人気アイテムN個を取得する
#     """

#     def __init__(self, n, key_col="customer_id", item_col="article_id", use_cache=True):
#         super().__init__(use_cache)
#         self.key_col = key_col
#         self.item_col = item_col
#         self.n = n
#         self.out_keys = [key_col, item_col]
#         self.name = self.name + "_" + str(self.n)

#     def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):

#         # EachClusterのcluster_cdfっぽいものを作りたい
#         #target_cust_cdf = trans_cdf[trans_cdf["customer_id"].isin(target_customers)].reset_index(drop=True)
#         # target_cust_cdf = trans_cdf.loc[trans_cdf["customer_id"].isin(target_customers), 'customer_id'].reset_index(drop=True)
#         # target_cust_cdf = target_cust_cdf.drop_duplicates()
#         target_cust_cdf = cudf.DataFrame()
#         target_cust_cdf['customer_id'] = target_customers
#         print(target_cust_cdf.shape)

#          # 年のtransactionだけ使う
#         trans_cdf = trans_cdf.query("week >= 80")

#         age_bins = [-100, 0, 20, 30, 40, 50, 60, 100]
#         age_bins_name = ["-1", "0-20", "20-30", "30-40", "40-50", "50-60", "60-"]
#         trans_cdf = trans_cdf.merge(cust_cdf[["customer_id", "age"]], how="left", on="customer_id")
#         # trans_cdf["bins"] = cudf.cut(trans_cdf["age"], bins=age_bins, labels=age_bins_name)
#         # trans_cdf["age_bins"] = cudf.cut(trans_cdf["age"], bins=age_bins, labels=age_bins_name)

#         trans_cdf["age_bins"] = pd.cut(trans_cdf["age"].to_pandas(), bins=age_bins, labels=age_bins_name).values

#         # 各年齢ごとに人気アイテムを取得([customer_id, article_id, age_bins])
#         cols = ["customer_id", "article_id", "age_bins"]
#         popluar_items_each_age_bins = to_cdf(
#             trans_cdf[cols]
#             .groupby(["article_id", "age_bins"])
#             .size()
#             .to_frame(name="size")
#             .reset_index()
#             .to_pandas()
#             .sort_values(["size", "age_bins"], ascending=False)
#             .groupby("age_bins")
#             .head(self.n)
#             .reset_index(drop=True)
#         )
#         # TODO: trans_cdfにmergeすると、transactionがないユーザに候補生成を作れない
#         del trans_cdf
#         target_cust_cdf = target_cust_cdf.merge(cust_cdf[["customer_id", "age"]], how="left", on="customer_id")
#         target_cust_cdf["age_bins"] = pd.cut(target_cust_cdf["age"].to_pandas(), bins=age_bins, labels=age_bins_name).values
#         self.out_cdf = target_cust_cdf.merge(popluar_items_each_age_bins, how="left", on="age_bins")[
#             self.out_keys
#         ]

#         # trans_cdf = trans_cdf.loc[trans_cdf["customer_id"].isin(target_customers), ['customer_id','age_bins']]
#         # trans_cdf = trans_cdf.drop_duplicates(subset=['customer_id'])
#         # self.out_cdf = trans_cdf.merge(popluar_items_each_age_bins, how="left", on="age_bins")[
#         #     self.out_keys
#         # ]
#         print(self.out_cdf.shape)

#         del popluar_items_each_age_bins

#         return self.out_cdf

class PopularItemsoftheEachAge(AbstractCGBlock):
    """
    各年齢ごとに人気アイテムN個を取得する
    """

    def __init__(self, n, key_col="customer_id", item_col="article_id", use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.out_keys = [key_col, item_col]
        self.name = self.name + "_" + str(self.n)

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):

        # 年のtransactionだけ使う
        trans_cdf = trans_cdf.query("week >= 80")

        age_bins = [-100, 0, 20, 30, 40, 50, 60, 100]
        age_bins_name = ["-1", "0-20", "20-30", "30-40", "40-50", "50-60", "60-"]
        trans_cdf = trans_cdf.merge(cust_cdf[["customer_id", "age"]], how="left", on="customer_id")
        # trans_cdf["bins"] = cudf.cut(trans_cdf["age"], bins=age_bins, labels=age_bins_name)
        # trans_cdf["age_bins"] = cudf.cut(trans_cdf["age"], bins=age_bins, labels=age_bins_name)

        trans_cdf["age_bins"] = pd.cut(trans_cdf["age"].to_pandas(), bins=age_bins, labels=age_bins_name).values

        # 各年齢ごとに人気アイテムを取得([customer_id, article_id, age_bins])
        cols = ["customer_id", "article_id", "age_bins"]
        popluar_items_each_age_bins = to_cdf(
            trans_cdf[cols]
            .groupby(["article_id", "age_bins"])
            .size()
            .to_frame(name="size")
            .reset_index()
            .to_pandas()
            .sort_values(["size", "age_bins"], ascending=False)
            .groupby("age_bins")
            .head(self.n)
            .reset_index(drop=True)
        )
        # TODO: trans_cdfにmergeすると、transactionがないユーザに候補生成を作れない
        # target_cust_cdf = raw_cust_cdf[raw_cust_cdf["customer_id"].isin(target_customers)]
        # target_cust_cdf["age_bins"] = pd.cut(target_cust_cdf["age"].to_pandas(), bins=age_bins, labels=age_bins_name).values
        # self.out_cdf = target_cust_cdf.merge(popluar_items_each_age_bins, how="left", on="age_bins")[
        #     self.out_keys
        # ]

        trans_cdf = trans_cdf.loc[trans_cdf["customer_id"].isin(target_customers), ['customer_id','age_bins']]
        trans_cdf = trans_cdf.drop_duplicates(subset=['customer_id'])
        self.out_cdf = trans_cdf.merge(popluar_items_each_age_bins, how="left", on="age_bins")[
            self.out_keys
        ]
        print(self.out_cdf.shape)

        del popluar_items_each_age_bins

        return self.out_cdf
        
class PairsWithLastBoughtNArticles(AbstractCGBlock):
    """
    各ユーザごとに直近の購入商品N個を取得する
    """

    def __init__(self, n, key_col="customer_id", item_col="article_id", use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.out_keys = [key_col, item_col]
        self.name = self.name + "_" + str(self.n)

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        # from: https://www.kaggle.com/code/zerebom/article-id-pairs-in-3s-using-cudf/edit
        # pair_df = cudf.read_parquet("/home/ec2-user/kaggle/h_and_m/data/top1-article-pairs.parquet")[
        #     ["article_id", "pair"]
        # ]
        pair_df = cudf.read_parquet("/home/ec2-user/kaggle/h_and_m/data/top1-article-pairs.parquet")[
            ["article_id", "pair"]
        ]

        trans_cdf = trans_cdf.loc[trans_cdf["customer_id"].isin(target_customers)]

        # ユーザーごとに最後に購入したN個の商品を抽出
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

class MensPopularItemsoftheLastWeeks(AbstractCGBlock):
    """
    最終週の人気MensアイテムtopNを返す
    
    最終的には男性に人気の商品を、女性には女性の人気商品を推薦したいが、
    購買履歴のない人(男女判定できない)には何を推薦すればいいか検討したいのでpending
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n=12, use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.name = self.name + "_" + str(self.n)

    def make_customer_sex_info(self, articles_sex, transactions):
        trans = transactions.merge(articles_sex, on="article_id", how="left")
        assert len(transactions) == len(trans), "mergeうまくいってないです"
        rename_dict = {"women_flg": "women_percentage", "men_flg": "men_percentage"}
        trans = (
            trans.groupby("customer_id")[["women_flg", "men_flg"]]
            .mean()
            .reset_index()
            .rename(columns=rename_dict)
        )
        return trans

    def calc_topn_article(self, input_cdf):
        TopN_articles = list(
            input_cdf["article_id"].value_counts().head(self.n).to_pandas().index
        )
        return  TopN_articles

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = trans_cdf.copy()
        last_week = input_cdf["week"].max()
        input_last_week_cdf = input_cdf.loc[input_cdf["week"] == last_week]

        cust_cdf_w_sex = self.make_customer_sex_info(art_cdf, trans_cdf)
        input_last_week_cdf_w_sex = input_last_week_cdf.merge(cust_cdf_w_sex, on="customer_id", how="left")
        assert len(input_last_week_cdf_w_sex) == len(input_last_week_cdf), "mergeうまくいってないです"

        mens_input_last_week_cdf = input_last_week_cdf_w_sex[input_last_week_cdf_w_sex['men_percentage']>0.7]
        mens_customers = mens_input_last_week_cdf['customer_id'].unique().to_pandas().values
        mens_TopN_articles = self.calc_topn_article(mens_input_last_week_cdf)

        if target_customers is None:
            target_customers = input_cdf["customer_id"].unique().to_pandas().values

        # 男性には男性の商品を推薦

        # Ref: https://note.nkmk.me/python-itertools-product/
        self.out_cdf = cudf.DataFrame(
            list(itertools.product(target_customers, mens_TopN_articles)),
            columns=[self.key_col, self.item_col],
        )
        return self.out_cdf


class EachSexPopularItemsoftheLastWeeks(AbstractCGBlock):
    """
    性別ごとに最終週の人気アイテムtopNを返す
    
    今の実装だと男性とも女性とも判定される人(['men_percentage']>0.7 かつ ['women_percentage']>0.7)に、男性のアイテムも女性のアイテムも候補生成していることになる？？？
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n=12, use_cache=True):
        super().__init__(use_cache)
        self.key_col = key_col
        self.item_col = item_col
        self.n = n
        self.name = self.name + "_" + str(self.n)

    def make_customer_sex_info(self, articles_sex, transactions):
        trans = transactions.merge(articles_sex, on="article_id", how="left")
        assert len(transactions) == len(trans), "mergeうまくいってないです"
        rename_dict = {"women_flg": "women_percentage", "men_flg": "men_percentage"}
        trans = (
            trans.groupby("customer_id")[["women_flg", "men_flg"]]
            .mean()
            .reset_index()
            .rename(columns=rename_dict)
        )
        return trans

    def calc_topn_article(self, input_cdf):
        TopN_articles = list(
            input_cdf["article_id"].value_counts().head(self.n).to_pandas().index
        )
        return  TopN_articles

    def transform(self, trans_cdf, art_cdf, cust_cdf, target_customers):
        input_cdf = trans_cdf.copy()
        last_week = input_cdf["week"].max()
        input_last_week_cdf = input_cdf.loc[input_cdf["week"] == last_week]

        cust_cdf_w_sex = self.make_customer_sex_info(art_cdf, trans_cdf)
        input_last_week_cdf_w_sex = input_last_week_cdf.merge(cust_cdf_w_sex, on="customer_id", how="left")
        assert len(input_last_week_cdf_w_sex) == len(input_last_week_cdf), "mergeうまくいってないです"

        mens_input_last_week_cdf = input_last_week_cdf_w_sex[input_last_week_cdf_w_sex['men_percentage']>0.6]
        mens_customers = mens_input_last_week_cdf['customer_id'].unique().to_pandas().values
        mens_TopN_articles = self.calc_topn_article(mens_input_last_week_cdf)
        
        womens_input_last_week_cdf = input_last_week_cdf_w_sex[input_last_week_cdf_w_sex['women_percentage']>0.6]
        womens_customers = womens_input_last_week_cdf['customer_id'].unique().to_pandas().values
        womens_TopN_articles = self.calc_topn_article(womens_input_last_week_cdf)

        TopN_articles = list(
            input_last_week_cdf["article_id"].value_counts().head(self.n).to_pandas().index
        )

        if target_customers is None:
            target_customers = input_cdf["customer_id"].unique().to_pandas().values

        # target_customersから男性、女性、判定不明を抽出
        # 性別が分かっているユーザー
        sex_known_customers = set(mens_customers) | set(womens_customers) # mens_customers:487
        sex_unknown_customers = set(target_customers) - set(sex_known_customers)
        assert len(target_customers) == (len(mens_customers) + len(womens_customers) + len(sex_unknown_customers)), "集合の計算がおかしいよ!!!!!!"

        """
        print(len(mens_customers) , len(womens_customers) , len(sex_unknown_customers))
        752 55775 1300182
        """

        # Ref: https://note.nkmk.me/python-itertools-product/
        mens_out_cdf = cudf.DataFrame(
            list(itertools.product(mens_customers, mens_TopN_articles)),
            columns=[self.key_col, self.item_col],
        )

        womens_out_cdf = cudf.DataFrame(
            list(itertools.product(womens_customers, mens_TopN_articles)),
            columns=[self.key_col, self.item_col],
        )

        sex_unknown_out_cdf = cudf.DataFrame(
            list(itertools.product(sex_unknown_customers, TopN_articles)),
            columns=[self.key_col, self.item_col],
        )

        self.out_cdf = cudf.concat([mens_out_cdf, womens_out_cdf, sex_unknown_out_cdf]).reset_index(drop=True)
        assert len(target_customers)*self.n == len(self.out_cdf), "cudf.concatがおかしいよ!!!!!!"
        return self.out_cdf