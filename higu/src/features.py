import os
import numpy as np
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import cudf
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data


to_cdf = cudf.DataFrame.from_pandas


class AbstractBaseBlock:
    def __init__(self, use_cache):
        self.use_cache = use_cache
        self.name = self.__class__.__name__
        self.cache_dir = Path("/home/kokoro/h_and_m/higu/input/features")

    def fit(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ) -> cudf.DataFrame:
        file_name = self.cache_dir / f"{self.name}_{target_week}.pkl"

        # キャッシュを使う & ファイルがあるなら読み出し
        if os.path.isfile(str(file_name)) and self.use_cache:
            with open(file_name, "rb") as f:
                feature = pickle.load(f)
            return feature
        # そうでないならtransform実行 & ファイル保存
        else:
            feature = self.transform(
                trans_cdf,
                art_cdf,
                cust_cdf,
                base_cdf,
                y_cdf,
                target_customers,
                logger,
                target_week,
            )

            with open(file_name, "wb") as f:
                pickle.dump(feature, f)

            return feature

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ) -> cudf.DataFrame:
        raise NotImplementedError()


class EmbBlock(AbstractBaseBlock):
    """
    {hoge_id:[List[int]],...}形式のdictをcudfに変換する
    """

    def __init__(self, key_col, emb_dic, prefix, use_cache):
        super().__init__(use_cache)
        self.name = self.name + "_" + self.prefix

        self.key_col = key_col
        self.emb_dic = emb_dic
        self.prefix = prefix

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        out_cdf = cudf.DataFrame(self.emb_dic).T
        out_cdf.columns = [f"{self.prefix}_{col}_emb" for col in out_cdf.columns]
        out_cdf = out_cdf.reset_index().rename({"index": self.key_col}, axis=1)
        out_cdf = out_cdf.iloc[:, :11]

        return out_cdf


class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, key_col, join_col, target_col, agg_list, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col  # 集約単位のkey
        self.join_col = join_col  # target_colを保持するdfのkey
        self.target_col = target_col
        self.agg_list = agg_list

        self.name = self.name + "_" + self.target_col

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        if self.join_col == "article_id":
            input_cdf = trans_cdf.merge(art_cdf[[self.join_col, self.target_col]])
        elif self.join_col == "customer_id":
            input_cdf = trans_cdf.merge(cust_cdf[[self.join_col, self.target_col]])
        else:
            input_cdf = trans_cdf.copy()

        out_cdf = input_cdf.groupby(self.key_col)[self.target_col].agg(self.agg_list)
        out_cdf.columns = ["_".join([self.target_col, col]) for col in out_cdf.columns]

        out_cdf.reset_index(inplace=True)

        return out_cdf


class TargetRollingBlock(TargetEncodingBlock):
    # TODO: 新しいインタフェースにする
    def __init__(self, key_cols, group_col, target_col, agg_list, windows):

        super().__init__(key_cols, target_col, agg_list)
        self.windows = windows
        self.group_col = group_col

    def transform(self, input_cdf: cudf.DataFrame):
        out_cdf = input_cdf[self.key_col].copy()
        for window in self.windows:
            for agg_func in self.agg_list:
                col = f"{self.target_col}_{window}_rolling_{agg_func}"
                out_cdf[col] = (
                    input_cdf.groupby(self.group_col)[self.target_col]
                    .rolling(window, center=False, min_periods=1)
                    .apply(agg_func)
                    .reset_index(0)[self.target_col]
                )

        out_cdf.reset_index(inplace=True, drop=True)
        return out_cdf


class ModeCategoryBlock(AbstractBaseBlock):
    # TODO: 新しいインタフェースにする
    # TODO:train,validでFitとtransformを使い分ける必要がある

    # key_colごとにtarget_colのmodeを算出
    def __init__(self, key_col, target_col):
        self.key_col = key_col
        self.target_col = target_col

    def fit(self, input_cdf):
        return self.transform(input_cdf)

    def transform(self, input_cdf: cudf.DataFrame):
        col_name = "most_cat_" + self.target_col
        out_cdf = (
            input_cdf.groupby([self.key_col, self.target_col])
            .size()
            .reset_index()
            .sort_values([self.key_col, 0], ascending=False)
            .groupby(self.key_col)[self.target_col]
            .nth(1)
            .reset_index()
            .rename({self.target_col: col_name}, axis=1)
        )
        out_cdf = to_cdf(out_cdf.to_pandas().fillna(-1))
        out_cdf[col_name] = out_cdf[col_name].astype("category")
        # out_cdf[col_name].cat.add_categories(-1)
        return out_cdf  # %%


class LifetimesBlock(AbstractBaseBlock):
    def __init__(self, key_col, target_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col  # 'customer_id'
        self.target_col = target_col  # 'price'

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        price_mean_cust_dat = (
            trans_cdf.groupby([self.key_col, "t_dat"])[self.target_col]
            .sum()
            .reset_index()
            .to_pandas()
        )
        out_df = summary_data_from_transaction_data(
            price_mean_cust_dat,
            self.key_col,
            "t_dat",
            observation_period_end=price_mean_cust_dat.t_dat.max(),  # train:2020-09-08, valid: 2020-09-15 , test: 2020-09-22
        ).reset_index()

        out_cdf = cudf.from_pandas(out_df)

        return out_cdf


# 性別
class SexArticleBlock(AbstractBaseBlock):
    # その人の性別が何であるかを判断する
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):

        cols = ["article_id", "index_name", "index_group_name", "section_name"]
        art_cdf = cudf.read_csv(
            "/home/kokoro/h_and_m/higu/input/articles.csv", header=0, usecols=cols
        )
        women_regexp = "women|girl|ladies"
        men_regexp = "boy|men"  # womenが含まれてしまうのであとで処理する
        df = self.make_article_sex_info(art_cdf, women_regexp, men_regexp)
        # 必要なものはそのarticleがwomenなのかmenなのかの情報だけ
        out_cdf = df[["article_id", "women_flg", "men_flg"]]
        return out_cdf

    def make_article_sex_info(self, art_cdf, women_regexp, men_regexp):
        df = art_cdf.copy()
        df["women"] = 0
        df["men"] = 0

        # どこかのcolでwomenっぽい単語があれば女性と判断で良さそう
        for col in ["index_name", "index_group_name", "section_name"]:
            df[col] = df[col].astype(str).str.lower()
            df["women"] += df[col].str.contains(women_regexp).astype(int)
            # womenを含む場合があるので対処
            mens = df[col].str.contains(men_regexp).astype(int) - df[col].str.contains(
                "women"
            ).astype(int)
            df["men"] += mens

        # 0でなければ女性と判断(3列あるので、足し合わせると3になってしまうことがある。その対処)
        df["women_flg"] = (df["women"] > 0).astype(int)
        df["men_flg"] = (df["men"] > 0).astype(int)
        return df


class SexCustomerBlock(AbstractBaseBlock):
    # その人の性別が何であるかを判断する
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.use_cache = use_cache
        self.key_col = key_col

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):

        sex_article = SexArticleBlock("article_id", self.use_cache)
            articles_sex_cdf = sex_article.transform(
            trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
            )

        out_cdf = self.make_customer_sex_info(articles_sex_cdf, trans_cdf)
        return out_cdf

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


class Dis2HistoryAveVecAndArtVec(AbstractBaseBlock):
    # 購買履歴の平均ベクトルと商品ベクトルの距離
    def __init__(self, key_col, art_emb_dic, prefix, use_cache):
        self.key_col = key_col
        self.art_emb_dic = art_emb_dic
        super().__init__(use_cache)
        self.name = self.name + "_" + prefix

        # 一度に平均ベクトルを求めるカスタマー数
        self.ave_vec_batch_size = 100_000

        # 一度に類似度を求めるトランザクション数
        self.cos_sim_batch_size = 10_000_000
        self.emb_dim = 20
        self.prefix = prefix

        # ベクトルをカラムにもったcdf
        self.article_emb_cdf = (
            cudf.DataFrame(self.art_emb_dic)
            .T.reset_index()
            .rename({"index": "article_id"}, axis=1)
        )

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        if base_cdf is None:
            return cudf.DataFrame()

        out_cdf = base_cdf[["article_id", "customer_id"]].copy()
        logger.info("start make ave vec")
        history_ave_vec_dic = self.make_history_ave_vec_dic(trans_cdf)

        logger.info("calc cos sim")
        col = f"sim_{self.prefix}"
        out_cdf[col] = self.calc_cos_sim(base_cdf, history_ave_vec_dic, self.art_emb_dic)

        out_cdf[col] = out_cdf[col].fillna(out_cdf[col].mean().astype(out_cdf[col].dtype))
        out_cdf = out_cdf.drop_duplicates(subset=["article_id", "customer_id"]).reset_index(
            drop=True
        )
        logger.info(out_cdf.shape)
        logger.info(out_cdf.isnull().sum())
        return out_cdf

    def calc_cos_sim(self, base_cdf, ave_vec_dic, art_emb_dic):
        all_arts = base_cdf["article_id"].to_pandas().values
        all_custs = base_cdf["customer_id"].to_pandas().values
        ave_vec_dic = defaultdict(lambda: np.zeros(self.emb_dim), ave_vec_dic)
        art_emb_dic = defaultdict(lambda: np.zeros(self.emb_dim), art_emb_dic)

        sim_list = []
        for i in tqdm(range(0, len(all_arts), self.cos_sim_batch_size)):
            arts = all_arts[i : i + self.cos_sim_batch_size]
            custs = all_custs[i : i + self.cos_sim_batch_size]

            # 辞書からのfetchが遅そう
            art_tensor = torch.Tensor([art_emb_dic[art_idx] for art_idx in arts])
            cust_tensor = torch.Tensor([ave_vec_dic[cust_idx] for cust_idx in custs])
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(art_tensor, cust_tensor)
            sim_list.extend(list(np.array(sim)))

        sim_list = list(np.where(np.array(sim_list) < 0.2, np.nan, np.array(sim_list)))

        print(len(sim_list), len(all_arts), base_cdf.shape)
        return sim_list

    def make_history_ave_vec_dic(self, trans_cdf):
        # trans_cdfから平均ベクトルを計算し、{customer_id: [emb]}の辞書に詰める

        cust_ids = trans_cdf["customer_id"].to_pandas().unique()
        all_cust_cnt = len(cust_ids)
        history_ave_vec_arr = np.empty((all_cust_cnt, self.emb_dim))

        trans_w_art_emb_cdf = trans_cdf[["customer_id", "article_id"]].merge(
            self.article_emb_cdf, how="left", on="article_id"
        )

        for i in tqdm(range(0, all_cust_cnt, self.ave_vec_batch_size)):
            sample_ids = cust_ids[i : i + self.ave_vec_batch_size]
            history_ave_vec_arr[i : i + self.ave_vec_batch_size] = (
                trans_w_art_emb_cdf[trans_w_art_emb_cdf["customer_id"].isin(sample_ids)]
                .drop(columns="article_id")
                .groupby("customer_id")
                .mean()
                .to_pandas()
                .values
            )

        history_ave_vec_dic = {}
        for idx, cust_id in enumerate(cust_ids):
            history_ave_vec_dic[cust_id] = history_ave_vec_arr[idx]

        return history_ave_vec_dic
