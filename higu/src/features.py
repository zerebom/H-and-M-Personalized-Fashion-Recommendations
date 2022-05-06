import os
import numpy as np
import pickle
from collections import defaultdict
#import torch
#import torch.nn as nn
import cudf
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data
from datetime import date
from datetime import datetime
import colorsys



to_cdf = cudf.DataFrame.from_pandas


class AbstractBaseBlock:
    def __init__(self, use_cache):
        self.use_cache = use_cache
        self.name = self.__class__.__name__
        self.cache_dir = Path("/home/ec2-user/kaggle/h_and_m/data/features")

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
        # prefixで中身も変わるので、ファイル名も変える
        self.prefix = prefix
        self.name = self.name + "_" + self.prefix
        self.key_col = key_col
        self.emb_dic = emb_dic

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        out_cdf = cudf.DataFrame(self.emb_dic).T
        out_cdf.columns = [f"{self.prefix}_{col}_emb" for col in out_cdf.columns]
        out_cdf = out_cdf.reset_index().rename({"index": self.key_col}, axis=1)
        out_cdf = out_cdf.iloc[:, :11]

        return out_cdf


# ------ 基本的な特徴量 --------
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
    def __init__(self, key_col, target_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col
        self.target_col = target_col

    # def fit(self, input_cdf):
    #     return self.transform(input_cdf)

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        col_name = "most_cat_" + self.target_col
        out_cdf = (
            trans_cdf.groupby([self.key_col, self.target_col])
            .size()
            .reset_index()
            .sort_values([self.key_col, 0], ascending=False)
            .groupby(self.key_col)[self.target_col]
            .nth(1)
            .reset_index()
            .rename({self.target_col: col_name}, axis=1)
        )
        tmp = out_cdf.to_pandas().fillna(-1)
        tmp[col_name] = tmp[col_name].astype("category")
        out_cdf = to_cdf(tmp)
        # out_cdf = to_cdf(out_cdf.to_pandas().fillna(-1))
        # out_cdf[col_name] = out_cdf[col_name].astype("category")

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
            '/home/ec2-user/kaggle/h_and_m/data/articles.csv', header=0, usecols=cols
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
        # 割と重い & イミュータブルなので先に計算する
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
        # 平均ベクトル計算
        history_ave_vec_dic = self.make_history_ave_vec_dic(trans_cdf)

        logger.info("calc cos sim") # cossim計算
        col = f"sim_{self.prefix}"
        out_cdf[col] = self.calc_cos_sim(base_cdf, history_ave_vec_dic, self.art_emb_dic)

        # nullとduplicate除去
        out_cdf[col] = out_cdf[col].fillna(out_cdf[col].mean().astype(out_cdf[col].dtype))
        out_cdf = out_cdf.drop_duplicates(subset=["article_id", "customer_id"]).reset_index(
            drop=True
        )
        return out_cdf

    def calc_cos_sim(self, base_cdf, ave_vec_dic, art_emb_dic):
        """
        ユーザの平均ベクトルと、アーティクルのベクトルのコサイン類似度をGPUで計算する
        """
        # base_cdfに存在するペアに対して計算する
        all_arts = base_cdf["article_id"].to_pandas().values
        all_custs = base_cdf["customer_id"].to_pandas().values
        # 未知のidが入ってきたら[0,0,...]を返す、
        ave_vec_dic = defaultdict(lambda: np.zeros(self.emb_dim), ave_vec_dic)
        art_emb_dic = defaultdict(lambda: np.zeros(self.emb_dim), art_emb_dic)

        sim_list = []
        # 全部計算するとGPUのメモリエラーになるので、バッチで区切る
        for i in tqdm(range(0, len(all_arts), self.cos_sim_batch_size)):
            arts = all_arts[i : i + self.cos_sim_batch_size]
            custs = all_custs[i : i + self.cos_sim_batch_size]

            # 辞書からのfetchが遅そう
            art_tensor = torch.Tensor([art_emb_dic[art_idx] for art_idx in arts])
            cust_tensor = torch.Tensor([ave_vec_dic[cust_idx] for cust_idx in custs])
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(art_tensor, cust_tensor)
            sim_list.extend(list(np.array(sim)))

        # 未知のidと計算して、類似度0になったやつをnp.nanに置換する
        # もっといい方法があるかも
        sim_list = list(np.where(np.array(sim_list) < 0.2, np.nan, np.array(sim_list)))

        print(len(sim_list), len(all_arts), base_cdf.shape)
        return sim_list

    def make_history_ave_vec_dic(self, trans_cdf):
        # trans_cdfから平均ベクトルを計算し、{customer_id: [emb]}の辞書に詰める

        cust_ids = trans_cdf["customer_id"].to_pandas().unique()
        all_cust_cnt = len(cust_ids)
        # このarrに平均ベクトルを詰めて、後で辞書に変換する
        history_ave_vec_arr = np.empty((all_cust_cnt, self.emb_dim))

        # cudfのカラムに各ベクトルの要素を列方向に詰める
        trans_w_art_emb_cdf = trans_cdf[["customer_id", "article_id"]].merge(
            self.article_emb_cdf, how="left", on="article_id"
        )

        # cudfのgroup byを使って各ユーザの平均ベクトルを高速に計算する
        # cudfのカラムに各ベクトルの要素を列方向に詰める
        # 全部計算するとGPUのメモリエラーになるので、バッチで区切る
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

# ------ articleに関する特徴量 --------
class RepeatSalesCustomerNum5Block(AbstractBaseBlock):
    # 一旦2-5を出そう
    # TODO: ハードコーディング部分を直す？考える
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.use_cache = use_cache
        self.key_col = key_col

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        repeat_num_df = self.repeat_num(trans_cdf)

        out_cdf = art_cdf[["article_id"]].copy()     # 最初に紐づける先を用意する
        # 0件購入はいないので考えない, 2-5までを出す
        for i in np.arange(2,6): # array([1, 2, 3, 4, 5])
            df = repeat_num_df.query(f"sales_num=={i}")[["article_id","repeat_sales_num"]]
            df.columns = ["article_id", f"repeat_in_{i}_sales"]
            out_cdf = out_cdf.merge(df, on="article_id", how="left")
        return out_cdf

    def repeat_num(self, trans_cdf):
        sales_df = trans_cdf.groupby(["customer_id","article_id"]).size().reset_index()
        sales_df.columns = ["customer_id","article_id", "sales"]
        sales_num_df = sales_df.groupby(["article_id", "sales"]).size().reset_index()
        sales_num_df.columns = ["article_id", "sales_num", "count"]

        # リピート回数がn回以上のところを探せるようにcumsum出す
        repeat_num = sales_num_df.sort_values(["article_id", "sales_num"], ascending=False)
        repeat_num["repeat_sales_num"] = repeat_num.groupby("article_id")["count"].cumsum()
        return repeat_num


class ArticleBiasIndicatorBlock(AbstractBaseBlock):
    """
    その商品を買う人に偏りがあるのかどうかといった指標
    その商品はある一部の人から購入されているのか、万人に購入されているのか
    また、購入されている人の中ではどれくらい買われているかを出す
    例：超奇抜なピンクの服 -> ある一部の人から購入されている
    例：万人に購入されている -> 黒い下着
    """
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        bias_cdf = trans_cdf.groupby("article_id")["customer_id"].agg(["nunique","count"]).reset_index()
        bias_cdf.columns =["article_id", "nunique_customers", "sales"]

        bias_cdf["sales_by_buy_users"] = bias_cdf["sales"] / bias_cdf["nunique_customers"] # その商品購入する人たちの中ではどれくらい買われるか？
        out_cdf = bias_cdf[["article_id", "sales_by_buy_users", "nunique_customers"]]
        return out_cdf

class FirstBuyDateBlock(AbstractBaseBlock):
    def __init__(self, key_col, end_date, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col
        self.end_date = end_date

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        end_date_datetime = datetime.strptime(self.end_date, '%Y-%m-%d')

        # 初日を出す
        date_item_first_purchased = trans_cdf.groupby("article_id")["t_dat"].min().reset_index()
        date_item_first_purchased.columns = ["article_id", "first_purchased_day"]
        date_item_first_purchased["first_purchased_day"] = cudf.to_datetime(date_item_first_purchased["first_purchased_day"])

        date_item_first_purchased["days_from_first_purchased"] = (end_date_datetime - date_item_first_purchased["first_purchased_day"]).dt.days
        del date_item_first_purchased["first_purchased_day"]

        return date_item_first_purchased


class SalesPerTimesForArticleBlock(AbstractBaseBlock):
    def __init__(self, key_col, groupby_cols_dict, agg_list, use_cache):
        super().__init__(use_cache)
        self.use_cache = use_cache
        self.key_col = key_col
        self.groupby_cols_dict = groupby_cols_dict
        self.agg_list = agg_list

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        trans_cdf = self.preprocess(trans_cdf.copy()) # 直接変更しないようにする

        out_cdf = art_cdf[["article_id"]].copy()     # 最初に紐づける先を用意する
        for groupby_name, groupby_cols in self.groupby_cols_dict.items():
            sales_cdf = self.make_agg_sales(trans_cdf, groupby_name, groupby_cols)
            out_cdf = out_cdf.merge(sales_cdf, on="article_id", how="left")
        return out_cdf

    def preprocess(self, trans_cdf: cudf.DataFrame):
        # 色々な日時情報を持ってくる
        #trans_cdf["t_dat_datetime"] = pd.to_datetime(trans_cdf["t_dat"].to_array())
        trans_cdf["t_dat_datetime"] = cudf.to_datetime(trans_cdf["t_dat"])
        trans_cdf["year"] = trans_cdf["t_dat_datetime"].dt.year
        trans_cdf["month"] = trans_cdf["t_dat_datetime"].dt.month
        trans_cdf["day"] = trans_cdf["t_dat_datetime"].dt.day
        trans_cdf["dayofweek"] = trans_cdf["t_dat_datetime"].dt.dayofweek # 月曜日が0, 日曜日が6 ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html
        return trans_cdf

    def make_agg_sales(self, trans_cdf, groupby_name, groupby_cols):
        groupby_cols = groupby_cols + ["article_id"]
        sales_by_groupby_cols = trans_cdf.groupby(groupby_cols).size().reset_index()
        sales_by_groupby_cols.columns = groupby_cols + ["sales"]

        sales_cdf = sales_by_groupby_cols.groupby("article_id")["sales"].agg(self.agg_list).reset_index()
        sales_cdf.columns = ["article_id"] +  [f"sales_{groupby_name}_{i}" for i in self.agg_list]
        return sales_cdf


# ------ userに関する特徴量 --------
class SalesPerDayCustomerBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_list, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col
        self.agg_list = agg_list

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        trans_sales_cdf = self.sales_day(trans_cdf)
        out_cdf = trans_sales_cdf.groupby("customer_id")["all_sales_per_day"].agg(self.agg_list).reset_index()
        out_cdf.columns = ["customer_id"] + [f"sales_in_day_{col}" for col in out_cdf.columns[1:]] # 1列目はcustomer_id

        return out_cdf

    def sales_day(self, trans_cdf):
        sales_per_day = trans_cdf.groupby(["t_dat"]).size().reset_index()
        sales_per_day.columns = ["t_dat", "all_sales_per_day"]

        sales_cust_day = trans_cdf.drop_duplicates(subset=["t_dat","customer_id"])[["t_dat","customer_id"]]
        trans_sales_cdf = sales_cust_day.merge(sales_per_day, on="t_dat", how="left")
        return trans_sales_cdf


class RepeatCustomerBlock(AbstractBaseBlock):
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        trans_cdf = trans_cdf.copy()
        t_dat_df = self.preprocess(trans_cdf)

        # TODO : 他にも活かし方あるかも？
        repeat_customer_df = t_dat_df.groupby("customer_id")["repeat_flg"].agg(["sum","mean"]).reset_index()
        repeat_customer_df.columns = ["customer_id", "cust_repeat_sum", "cust_repeat_mean"]
        repeat_customer_cdf = cudf.from_pandas(repeat_customer_df)
        return repeat_customer_cdf

    def preprocess(self, trans_cdf):
        #trans_cdf["t_dat_datetime"] = pd.to_datetime(trans_cdf["t_dat"].to_array())
        trans_cdf["t_dat_datetime"] = cudf.to_datetime(trans_cdf["t_dat"])
        # customerがある商品をどれくらいの頻度で買うか, 当日の場合はリピ買いと判定しない
        t_dat_cdf = trans_cdf.sort_values(['customer_id','article_id','t_dat_datetime']).drop_duplicates(subset=['customer_id','article_id','t_dat_datetime'],keep='first')[['customer_id','article_id','t_dat_datetime']].reset_index()
        t_dat_df = t_dat_cdf.to_pandas()
        t_dat_df["shift_t_dat_datetime"] = t_dat_df.groupby(["customer_id","article_id"])['t_dat_datetime'].shift(1)
        t_dat_df["day_diff"] = (t_dat_df["t_dat_datetime"] - t_dat_df["shift_t_dat_datetime"] ).dt.days
        t_dat_df["repeat_flg"] = 1 - t_dat_df["shift_t_dat_datetime"].isna().astype(int)
        return t_dat_df



class MostFreqBuyDayofWeekBlock(AbstractBaseBlock):
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        trans_cdf = trans_cdf.copy() # 直接変更しないようにする
        #trans_cdf["t_dat_datetime"] = pd.to_datetime(trans_cdf["t_dat"].to_array())
        trans_cdf["t_dat_datetime"] = cudf.to_datetime(trans_cdf["t_dat"])
        trans_cdf["dayofweek"] = trans_cdf["t_dat_datetime"].dt.dayofweek # 月曜日が0, 日曜日が6 ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html

        sales_by_dayofweek = trans_cdf.groupby(["customer_id", "dayofweek"]).size().reset_index()
        sales_by_dayofweek.columns = ["customer_id", "dayofweek", "sales"]

        # 大小関係は関係ないので、strにする
        sales_by_dayofweek["dayofweek"] = sales_by_dayofweek["dayofweek"].astype(str)

        out_cdf = sales_by_dayofweek.sort_values(["customer_id", "sales"], ascending=False).drop_duplicates(subset=['customer_id'],keep='first')[["customer_id","dayofweek"]]
        return out_cdf

class CustomerBuyIntervalBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_list, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col
        self.agg_list = agg_list

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        trans_cdf = trans_cdf.copy() # 直接変更しないようにする
        #trans_cdf["t_dat_datetime"] = pd.to_datetime(trans_cdf["t_dat"].to_array())
        trans_cdf["t_dat_datetime"] = cudf.to_datetime(trans_cdf["t_dat"])
        t_dat_df = self.make_day_diff(trans_cdf)

        # ここで連続するかどうか
        countinue_buy_df = self.continue_buy(t_dat_df)
        day_diff_describe_df = self.interval_describe(t_dat_df)

        # 作成したものをくっつけて, cdfにする
        coutinue_buy_cdf = cudf.from_pandas(countinue_buy_df)
        day_diff_describe_cdf = cudf.from_pandas(day_diff_describe_df)

        out_cdf = coutinue_buy_cdf.merge(day_diff_describe_cdf, on="customer_id", how="left")
        return out_cdf

    def make_day_diff(self,trans_cdf):
        t_dat_cdf = trans_cdf.sort_values(['customer_id',"t_dat_datetime"]).drop_duplicates(subset=['customer_id',"t_dat_datetime"],keep='first')[['customer_id',"t_dat_datetime"]].reset_index()
        t_dat_df = t_dat_cdf.to_pandas()
        t_dat_df["shift_t_dat_datetime"] = t_dat_df.groupby("customer_id")["t_dat_datetime"].shift(1)
        t_dat_df["day_diff"] = (t_dat_df["t_dat_datetime"] - t_dat_df["shift_t_dat_datetime"] ).dt.days
        return t_dat_df

    def continue_buy(self, t_dat_df):
        t_dat_df["continue_buy_flg"] = (t_dat_df["day_diff"] == 1).astype(int)
        t_dat_df = t_dat_df.sort_values(['customer_id','continue_buy_flg'], ascending=False).drop_duplicates(subset=['customer_id'],keep='first')[['customer_id','continue_buy_flg']]
        return t_dat_df

    def interval_describe(self, t_dat_df):
        day_diff_describe = t_dat_df.groupby("customer_id")["day_diff"].agg(self.agg_list).reset_index()
        day_diff_describe.columns = ["customer_id"] + [f"buy_interval_{i}" for i in self.agg_list]
        return day_diff_describe


# ------ postal codeに関する特徴量 --------
class PostalCodeBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_cust_cols, agg_trans_cols, agg_list, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col
        self.agg_cust_cols = agg_cust_cols
        self.agg_trans_cols = agg_trans_cols
        self.agg_list = agg_list

    def transform(self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week):
        # 準備としてtransにpostal codeをjoinさせる
        trans_postal_cdf = trans_cdf.merge(cust_cdf[["customer_id","postal_code"]], on="customer_id", how="left")

        # agg_colsの計算(cust info)
        out_agg_cust_cdf = cust_cdf.groupby("postal_code")[self.agg_cust_cols].agg(self.agg_list).reset_index()
        out_agg_cust_cdf.columns = ["postal_code"] + [f"postal_code_{i[0]}_{i[1]}" for i in list(out_agg_cust_cdf.columns[1:])] # 1列はpostal_code

        # agg_colsの計算(trans info)
        out_agg_trans_cdf = trans_postal_cdf.groupby("postal_code")[self.agg_trans_cols].agg(self.agg_list).reset_index()
        out_agg_trans_cdf.columns = ["postal_code"] + [f"postal_code_{i[0]}_{i[1]}" for i in list(out_agg_trans_cdf.columns[1:])] # 1列はpostal_code


        # 性別特徴量
        out_sex_cdf = self.sex_per_postal_code(trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger)

        # 人気商品
        most_popular_article_cdf = self.most_popular_item(trans_postal_cdf)

        out_cdf = (
            out_agg_cust_cdf
            .merge(out_agg_trans_cdf, on="postal_code", how="left")
            .merge(out_sex_cdf, on="postal_code", how="left")
            .merge(most_popular_article_cdf, on="postal_code", how="left")
        )
        return out_cdf

    def sex_per_postal_code(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
        # cust_cdfに入っているなら、それを呼び出す
        if "men_percentage" in cust_cdf:
            cust_sex_cdf = cust_cdf[["customer_id","postal_code","women_percentage","men_percentage"]].copy()
        else:
            # 特徴量を作成する
            cust_sex = SexCustomerBlock("customer_id")
            cust_sex_cdf_no_postal = cust_sex.transform(trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger)

            # postal_code情報を追加する
            cust_sex_cdf = cust_cdf[["customer_id","postal_code"]].merge(cust_sex_cdf_no_postal, on="customer_id", how="left")
        # 計算
        out_sex_cdf = cust_sex_cdf.groupby("postal_code")[["women_percentage","men_percentage"]].agg(self.agg_list).reset_index()
        out_sex_cdf.columns = ["postal_code"] + [f"postal_code_{i[0]}_{i[1]}" for i in list(out_sex_cdf.columns[1:])] # 1列はpostal_code
        return out_sex_cdf

    def most_popular_item(self, trans_postal_cdf):
        popular_article_cdf = trans_postal_cdf.groupby(["postal_code","article_id"]).size().reset_index()
        popular_article_cdf.columns = ["postal_code","article_id", "sales_in_postal_code"]
        # 一番人気の商品を入れる
        # TODO: もう少し工夫しても良いかも, 同率1位は考えられていない
        most_popular_article_cdf = popular_article_cdf.sort_values("sales_in_postal_code",ascending=False).drop_duplicates(subset=["postal_code"])[["postal_code","article_id"]]
        most_popular_article_cdf.columns = ["postal_code","most_popular_article_in_postal"]
        return most_popular_article_cdf

# -----------色に関係する特徴量 -------------
class ColorArticleBlock(AbstractBaseBlock):
    def __init__(self, key_col, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        cols = ["article_id", "perceived_colour_value_name", "perceived_colour_master_name"]
        # art_cdf = cudf.read_csv(
        #     "/home/ec2-user/kaggle/h_and_m/data/articles.csv", header=0, usecols=cols
        # )
        art_cdf = cudf.read_csv(
            "/home/ec2-user/kaggle/h_and_m/data/articles.csv", header=0, usecols=cols
        )
        # cudf.read_csv(
        #     '../input/h-and-m-personalized-fashion-recommendations/articles.csv', header=0, usecols=cols
        # )

        # 辞書を作成(created by mana)
        # 暗い方0、明るい方1
        perceived_dict = {
            "Dark": 0,
            "Light": 1,
            "Dusty Light": 0.5,
            "Bright": 1,
            "Medium Dusty": 0.3,
            "Medium": 0.5,
            "Undefined": None,
            "Unknown": None,
        }
        # ref : https://note.cman.jp/color/base_color.cgi
        # Khaki green, 'Metal', 'Mole', 'Lilac Purple', 'Yellowish Green', 'Bluish Green' はググった
        # rgb
        perceived_master_rgb_dict = {
            "Black": [0, 0, 0],
            "Blue": [0, 0, 255],
            "White": [255, 255, 255],
            "Pink": [255, 192, 203],
            "Grey": [128, 128, 128],
            "Red": [255, 0, 0],
            "Beige": [245, 245, 220],
            "Green": [0, 128, 0],
            "Khaki green": [138, 134, 93],
            "Yellow": [255, 255, 0],
            "Orange": [255, 165, 0],
            "Brown": [165, 42, 42],
            "Metal": None,
            "Turquoise": [64, 224, 208],
            "Mole": None,
            "Lilac Purple": [220, 208, 255],
            "Unknown": None,
            "undefined": None,
            "Yellowish Green": [154, 205, 50],
            "Bluish Green": [13, 152, 186],
        }

        # rgbを変換する
        perceived_master_hls_dict = self.rgb2other_dict(
            perceived_master_rgb_dict, colorsys.rgb_to_hls
        )
        perceived_master_yiq_dict = self.rgb2other_dict(
            perceived_master_rgb_dict, colorsys.rgb_to_yiq
        )
        perceived_master_hsv_dict = self.rgb2other_dict(
            perceived_master_rgb_dict, colorsys.rgb_to_hsv
        )

        # 辞書作成
        for color_type in ["rgb", "hls", "yiq", "hsv"]:
            exec(
                f"perceived_master_{color_type}_dict_0,perceived_master_{color_type}_dict_1, perceived_master_{color_type}_dict_2 = self.make_3_dict(perceived_master_{color_type}_dict)"
            )

        # 列を変換
        # 明るいと1になる
        art_cdf["colour_light"] = art_cdf["perceived_colour_value_name"].map(perceived_dict)
        # 新しい情報を追加する(リストは展開した状態で渡す)
        for color_type in ["rgb", "hls", "yiq", "hsv"]:
            for dict_i in range(3):
                exec(
                    f'art_cdf["colour_master_{color_type}_{dict_i}"] = art_cdf["perceived_colour_master_name"].map(perceived_master_{color_type}_dict_{dict_i})'
                )

        # 使用する列を取得
        art_color_3dimension = [
            "colour_master_rgb_0",
            "colour_master_rgb_1",
            "colour_master_rgb_2",
            "colour_master_hls_0",
            "colour_master_hls_1",
            "colour_master_hls_2",
            "colour_master_yiq_0",
            "colour_master_yiq_1",
            "colour_master_yiq_2",
            "colour_master_hsv_0",
            "colour_master_hsv_1",
            "colour_master_hsv_2",
        ]
        art_color_cols = ["colour_light"] + art_color_3dimension

        out_cdf = art_cdf[["article_id"] + art_color_cols]
        return out_cdf

    def rgb2other_dict(self, use_dict, change_function):
        return_dict = dict()
        for k, v in use_dict.items():
            if v is not None:
                r, g, b = v
                return_value = list(change_function(r / 255, g / 255, b / 255))
                return_dict[k] = return_value
        return return_dict

    def make_3_dict(self, use_dict):
        perceived_master_dict_list = []

        for dict_i in range(3):
            add_dict = {k: v[dict_i] for k, v in use_dict.items() if v is not None}
            perceived_master_dict_list.append(add_dict)

        return perceived_master_dict_list


class ColorCustomerBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_list, use_cache):
        super().__init__(use_cache)
        self.key_col = key_col
        self.agg_list = agg_list

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
    ):
        # art_cdfを変換
        color_article = ColorArticleBlock("article_id", self.use_cache)
        art_cdf_color = color_article.transform(
            trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, logger, target_week
        )

        # 使用する列
        art_color_3dimension = [
            "colour_master_rgb_0",
            "colour_master_rgb_1",
            "colour_master_rgb_2",
            "colour_master_hls_0",
            "colour_master_hls_1",
            "colour_master_hls_2",
            "colour_master_yiq_0",
            "colour_master_yiq_1",
            "colour_master_yiq_2",
            "colour_master_hsv_0",
            "colour_master_hsv_1",
            "colour_master_hsv_2",
        ]
        art_color_cols = ["colour_light"] + art_color_3dimension

        # ここからtransactionに紐付けて個人の情報量にする
        trans_cdf = trans_cdf.query('week >= 80')
        trans_cdf_color = trans_cdf[["customer_id", "article_id"]].merge(
            art_cdf_color[["article_id"] + art_color_cols], on="article_id", how="left"
        )

        out_cdf = (
            trans_cdf_color.groupby("customer_id")[art_color_cols].agg(self.agg_list).reset_index()
        )
        out_cdf.columns = ["customer_id"] + [
            f"{col_name}_{agg}" for col_name in art_color_cols for agg in self.agg_list
        ]
        return out_cdf