import cudf
import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data


to_cdf = cudf.DataFrame.from_pandas


class AbstractBaseBlock:
    def fit(self, trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers) -> cudf.DataFrame:
        return self.transform(trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger)

    def transform(
        self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger
    ) -> cudf.DataFrame:
        raise NotImplementedError()


class EmbBlock(AbstractBaseBlock):
    """
    {hoge_id:[List[int]],...}形式のdictをcudfに変換する
    """

    def __init__(self, key_col, emb_dic, prefix):
        self.key_col = key_col
        self.emb_dic = emb_dic
        self.prefix = prefix

    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
        out_cdf = cudf.DataFrame(self.emb_dic).T
        out_cdf.columns = [f"{self.prefix}_{col}_emb" for col in out_cdf.columns]
        out_cdf = out_cdf.reset_index().rename({"index": self.key_col}, axis=1)
        out_cdf = out_cdf.iloc[:, :11]

        return out_cdf


class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, key_col, join_col, target_col, agg_list):

        self.key_col = key_col  # 集約単位のkey
        self.join_col = join_col  # target_colを保持するdfのkey
        self.target_col = target_col
        self.agg_list = agg_list

    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
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
    def __init__(self, key_col, target_col):

        self.key_col = key_col  # 'customer_id'
        self.target_col = target_col  # 'price'

    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
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
    def __init__(self, key_col):
        self.key_col = key_col

    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
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
    def __init__(self, key_col):
        self.key_col = key_col

    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):

        # art_cdfに入っているなら、それを呼び出す
        if "men_flg" in art_cdf:
            articles_sex_cdf = art_cdf[["article_id", "women_flg", "men_flg"]].copy()
        else:
            sex_article = SexArticleBlock("article_id")
            articles_sex_cdf = sex_article.transform(
                trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger
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

class RepeatSalesCustomerNum5Block(AbstractBaseBlock):
    # 一旦2-5を出そう
    # TODO: ハードコーディング部分を直す？考える
    def __init__(self, key_col):
        self.key_col = key_col
        
    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
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

class SalesPerDayBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_list):
        self.key_col = key_col
        self.agg_list = agg_list
    
    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
        trans_sales_cdf = self.sales_day(trans_cdf)
        out_cdf = trans_sales_cdf.groupby("customer_id")["all_sales_per_day"].agg(self.agg_list)
        out_cdf.columns = [f"sales_in_day_{col}" for col in out_cdf.columns]
        out_cdf.reset_index(inplace=True)
        return out_cdf 

    def sales_day(self, trans_cdf):
        sales_per_day = trans_cdf.groupby(["t_dat"]).size().reset_index()
        sales_per_day.columns = ["t_dat", "all_sales_per_day"]

        sales_cust_day = trans_cdf.drop_duplicates(subset=["t_dat","customer_id"])[["t_dat","customer_id"]]    
        trans_sales_cdf = sales_cust_day.merge(sales_per_day, on="t_dat", how="left")
        return trans_sales_cdf


class PostalCodeBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_cust_cols, agg_trans_cols, agg_list):
        self.key_col = key_col
        self.agg_cust_cols = agg_cust_cols
        self.agg_trans_cols = agg_trans_cols
        self.agg_list = agg_list
    
    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
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


class SalesPerTimesInfBlock(AbstractBaseBlock):
    def __init__(self, key_col, groupby_cols_dict, agg_list):
        self.key_col = key_col
        self.groupby_cols_dict = groupby_cols_dict
        self.agg_list = agg_list
        
    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):        
        trans_cdf = self.preprocess(trans_cdf.copy()) # 直接変更しないようにする
        
        out_cdf = art_cdf[["article_id"]].copy()     # 最初に紐づける先を用意する
        
        for groupby_name, groupby_cols in self.groupby_cols_dict.items():
            max_sales_cdf = self.max_sales(trans_cdf, groupby_name, groupby_cols)
            out_cdf = out_cdf.merge(max_sales_cdf, on="article_id", how="left")           
        return out_cdf 
    
    def preprocess(self, trans_cdf: cudf.DataFrame):
        # 色々な日時情報を持ってくる
        trans_cdf["t_dat_datetime"] = pd.to_datetime(trans_cdf["t_dat"].to_array())
        trans_cdf["year"] = trans_cdf["t_dat_datetime"].dt.year
        trans_cdf["month"] = trans_cdf["t_dat_datetime"].dt.month
        trans_cdf["day"] = trans_cdf["t_dat_datetime"].dt.day
        trans_cdf["dayofweek"] = trans_cdf["t_dat_datetime"].dt.dayofweek # 月曜日が0, 日曜日が6 ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html
        return trans_cdf

    def max_sales(self, trans_cdf, groupby_name, groupby_cols):
        groupby_cols = groupby_cols + ["article_id"]
        sales_by_groupby_cols = trans_cdf.groupby(groupby_cols).size().reset_index()
        sales_by_groupby_cols.columns = groupby_cols + ["sales"]

        sales_cdf = sales_by_groupby_cols.groupby("article_id")["sales"].agg(self.agg_list).reset_index()
        sales_cdf.columns = ["article_id"] +  [f"sales_{groupby_name}_{i}" for i in agg_list]
        return sales_cdf


class MostFreqBuyDayofWeekBlock(AbstractBaseBlock):
    def __init__(self, key_col):
        self.key_col = key_col
        
    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
        trans_cdf = trans_cdf.copy() # 直接変更しないようにする
        trans_cdf["dayofweek"] = trans_cdf["t_dat_datetime"].dt.dayofweek # 月曜日が0, 日曜日が6 ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html
        
        sales_by_dayofweek = trans_cdf.groupby(["customer_id", "dayofweek"]).size().reset_index()
        sales_by_dayofweek.columns = ["customer_id", "dayofweek", "sales"]
        
        # 大小関係は関係ないので、strにする
        sales_by_dayofweek["dayofweek"] = sales_by_dayofweek["dayofweek"].astype(str)
        
        out_cdf = sales_by_dayofweek.sort_values(["customer_id", "sales"], ascending=False).drop_duplicates(subset=['customer_id'],keep='first')[["customer_id","dayofweek"]]  
        return out_cdf 

class UseBuyIntervalBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_list):
        self.key_col = key_col
        self.agg_list = agg_list
        
    def transform(self, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, logger):
        t_dat_df = self.make_day_diff(trans_cdf.copy()) # 直接変更しないようにする
        
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
        t_dat_df["shift_t_dat_datetime"] = t_dat_df.groupby("customer_id")[col].shift(1)
        t_dat_df["day_diff"] = (t_dat_df["t_dat_datetime"] - t_dat_df["shift_t_dat_datetime"] ).dt.days
        return t_dat_df
    
    def continue_buy(self, t_dat_df):
        t_dat_df["continue_buy_flg"] = (t_dat_df["day_diff"] == 1).astype(int)
        t_dat_df = t_dat_df.sort_values(['customer_id','continue_buy_flg'], ascending=False).drop_duplicates(subset=['customer_id'],keep='first')[['customer_id','continue_buy_flg']]
        return t_dat_df
    
    def interval_describe(self, t_dat_df):
        day_diff_describe = t_dat_df.groupby("customer_id")["day_diff"].agg(agg_list).reset_index()
        day_diff_describe.columns = ["customer_id"] + [f"buy_interval_{i}" for i in agg_list]
        return day_diff_describe