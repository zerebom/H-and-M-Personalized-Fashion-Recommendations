import cudf
import pandas as pd

to_cdf = cudf.DataFrame.from_pandas


class AbstractBaseBlock:
    def fit(self, input_cdf: cudf.DataFrame, y=None) -> cudf.DataFrame:
        return self.transform(input_cdf)

    def transform(self, input_cdf: cudf.DataFrame) -> cudf.DataFrame:
        raise NotImplementedError()


class EmbBlock(AbstractBaseBlock):
    """
    {hoge_id:[List[int]],...}形式のdictをcudfに変換する
    """

    def __init__(self, key_col, emb_dic):
        self.key_col = key_col
        self.emb_dic = emb_dic

    def transform(self, input_cdf: cudf.DataFrame):
        _ = input_cdf  # input_cdfはinterfaceを揃えるためだけに呼ばれてる(悲しいね)

        out_cdf = cudf.DataFrame(self.emb_dic).T
        out_cdf.columns = [f"{col}_emb" for col in out_cdf.columns]
        out_cdf = out_cdf.reset_index().rename({"index": self.key_col}, axis=1)
        out_cdf = out_cdf.iloc[:, :11]

        return out_cdf


class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self, key_col, target_col, agg_list):

        self.key_col = key_col
        self.target_col = target_col
        self.agg_list = agg_list

    def transform(self, input_cdf: cudf.DataFrame):
        out_cdf = input_cdf.groupby(self.key_col)[self.target_col].agg(self.agg_list)
        out_cdf.columns = ["_".join([self.target_col, col]) for col in out_cdf.columns]

        out_cdf.reset_index(inplace=True)

        return out_cdf


class TargetRollingBlock(TargetEncodingBlock):
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
        out_cdf[col_name] = out_cdf[col_name].astype("category")
        return out_cdf  # %%


# 性別
class SexArticleBlock(AbstractBaseBlock):
    # その人の性別が何であるかを判断する
    def __init__(self, key_col):
        self.key_col = key_col
    
    def fit(self, art_cdf):
        return self.transform(art_cdf)

    def transform(self, art_cdf: cudf.DataFrame):
        women_regexp = "women|girl|ladies"
        men_regexp = "boy|men" # womenが含まれてしまうのであとで処理する
        df = self.make_article_sex_info(art_cdf, women_regexp, men_regexp)
        # 必要なものはそのarticleがwomenなのかmenなのかの情報だけ
        out_cdf = df[["article_id","women_flg","men_flg"]]
        return out_cdf 
        
    def make_article_sex_info(self, art_cdf, women_regexp, men_regexp):
        df = art_cdf
        df["women"] = 0
        df["men"] = 0

        # どこかのcolでwomenっぽい単語があれば女性と判断で良さそう
        for col in ["index_name", "index_group_name","section_name"]:
            df[col] = df[col].astype(str).str.lower()
            df["women"] += df[col].str.contains(women_regexp).astype(int) 
            # womenを含む場合があるので対処
            mens = df[col].str.contains(men_regexp).astype(int) - df[col].str.contains("women").astype(int)
            df["men"] += mens
        
        # 0でなければ女性と判断(3列あるので、足し合わせると3になってしまうことがある。その対処)
        df["women_flg"] = (df["women"] > 0).astype(int)
        df["men_flg"] = (df["men"] > 0).astype(int)
        return df

class SexCustomerBlock(AbstractBaseBlock):
    # その人の性別が何であるかを判断する
    def __init__(self, key_col):
        self.key_col = key_col
    
    def fit(self, trans_cdf, art_cdf):
        return self.transform(trans_cdf, art_cdf)

    def transform(self, trans_cdf: cudf.DataFrame, art_cdf: cudf.DataFrame):
        sex_article = SexArticleBlock("article_id")
        articles_sex_cdf = sex_article.transform(art_cdf)
        out_cdf = self.make_customer_sex_info(articles_sex_cdf, trans_cdf)
        return out_cdf 

    def make_customer_sex_info(self,articles_sex, transactions):
        trans = transactions.merge(articles_sex,on="article_id", how="left")
        assert len(transactions) == len(trans), "mergeうまくいってないです"
        rename_dict = {"women_flg":"women_percentage", "men_flg":"men_percentage"}
        trans = trans.groupby("customer_id")[["women_flg","men_flg"]].mean().reset_index().rename(columns=rename_dict)
        return trans

class RepeatSalesCustomerNum5(AbstractBaseBlock):
    # 一旦2-5を出そう
    # TODO: ハードコーディング部分を直す？考える
    def __init__(self, key_col):
        self.key_col = key_col
        
    def transform(self, input_cdf: cudf.DataFrame):
        repeat_num_df = self.repeat_num(input_cdf)
        
        out_cdf = input_cdf[["article_id"]].drop_duplicates()     # 最初に紐づける先を用意する
        # 0件購入はいないので考えない, 2-5までを出す
        for i in np.arange(2,6): # array([1, 2, 3, 4, 5])
            df = repeat_num_df.query(f"sales_num=={i}")[["article_id","repeat_sales_num"]]
            df.columns = ["article_id", f"repeat_in_{i}_sales"]
            out_cdf = out_cdf.merge(df, on="article_id", how="left")
        return out_cdf 

    def repeat_num(self, input_cdf: cudf.DataFrame):
        sales_df = input_cdf.groupby(["customer_id","article_id"]).size().reset_index()
        sales_df.columns = ["customer_id","article_id", "sales"]
        sales_num_df = sales_df.groupby(["article_id", "sales"]).size().reset_index()
        sales_num_df.columns = ["article_id", "sales_num", "count"]

        # リピート回数がn回以上のところを探せるようにcumsum出す
        repeat_num = sales_num_df.sort_values(["article_id", "sales_num"], ascending=False)
        repeat_num["repeat_sales_num"] = repeat_num.groupby("article_id")["count"].cumsum()
        return repeat_num


class SalesPerDay(AbstractBaseBlock):
    def __init__(self, key_col, agg_list):
        self.key_col = key_col
        self.agg_list = agg_list
    
    def transform(self, input_cdf: cudf.DataFrame):
        trans_sales = self.sales_day(trans_cdf=input_cdf)
        out_cdf = trans_sales.groupby("customer_id")["all_sales_per_day"].agg(self.agg_list)
        out_cdf.columns = [f"sales_in_day_{col}" for col in out_cdf.columns]
        out_cdf.reset_index(inplace=True)
        return out_cdf 

    def sales_day(self, trans_cdf):
        sales_per_day = trans_cdf.groupby(["t_dat"]).size().reset_index()
        sales_per_day.columns = ["t_dat", "all_sales_per_day"]

        sales_cust_day = trans_cdf.drop_duplicates(subset=["t_dat","customer_id"])[["t_dat","customer_id"]]    
        trans_sales = sales_cust_day.merge(sales_per_day, on="t_dat", how="left")
        return trans_sales

class PostalCodeBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_cols , agg_list):
        self.key_col = key_col
        self.agg_cols = agg_cols
        self.agg_list = agg_list
        
    def fit(self, input_cdf: cudf.DataFrame, customer_cdf_add_sex: cudf.DataFrame):
        return self.transform(input_cdf, customer_cdf_add_sex)
    
    def transform(self, input_cdf: cudf.DataFrame, customer_cdf_add_sex: cudf.DataFrame):
        # agg_colsの計算
        out_agg_cdf = input_cdf.groupby("postal_code")[self.agg_cols].agg(self.agg_list).reset_index()
        out_agg_cdf.columns = ["postal_code"] + [f"postal_code_{i[0]}_{i[1]}" for i in list(out_agg_cdf.columns[1:])] # 1列はpostal_code
        
        # 性別特徴量
        out_sex_cdf = cust_cdf_add_sex.groupby("postal_code")[["women_percentage","men_percentage"]].agg(self.agg_list).reset_index()
        out_sex_cdf.columns = ["postal_code"] + [f"postal_code_{i[0]}_{i[1]}" for i in list(out_sex_cdf.columns[1:])] # 1列はpostal_code
        
        # 人気商品
        popular_article = input_cdf.groupby(["postal_code","article_id"]).size().reset_index()
        popular_article.columns = ["postal_code","article_id", "sales_in_postal_code"]
        # 一番人気の商品を入れる # TODO: もう少し工夫しても良いかも, 同率1位は考えられていない
        most_popular_article = popular_article.sort_values("sales_in_postal_code",ascending=False).drop_duplicates(subset=["postal_code"])[["postal_code","article_id"]]
        most_popular_article.columns = ["postal_code","most_popular_article_in_postal"]
        
        out_cdf = out_agg_cdf.merge(out_sex_cdf, on="postal_code", how="left").merge(most_popular_article, on="postal_code", how="left")
        return out_cdf 


class MaxSales(AbstractBaseBlock):
    def __init__(self, key_col):
        self.key_col = key_col
        
    def fit(self, input_cdf: cudf.DataFrame, groupby_cols_dict):
        return self.transform(input_cdf, groupby_cols_dict)
    
    def transform(self, input_cdf: cudf.DataFrame, groupby_cols_dict):
        input_cdf = self.preprocess(input_cdf)
        
        out_cdf = input_cdf[["article_id"]].drop_duplicates()     # 最初に紐づける先を用意する
        for groupby_name, groupby_cols in groupby_cols_dict.items():
            max_sales_cdf = self.max_sales(input_cdf, groupby_name, groupby_cols)
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
        # display(sales_by_groupby_cols.sort_values("article_id").head(5))

        max_sales = sales_by_groupby_cols.groupby("article_id")["sales"].max().reset_index()
        max_sales.columns = ["article_id", f"max_sales_per_{groupby_name}"]
        return max_sales

class MostFreqBuyDayofWeek(AbstractBaseBlock):
    def __init__(self, key_col):
        self.key_col = key_col
        
    def transform(self, input_cdf: cudf.DataFrame):
        sales_by_dayofweek = input_cdf.groupby(["customer_id", "dayofweek"]).size().reset_index()
        sales_by_dayofweek.columns = ["customer_id", "dayofweek", "sales"]
        out_cdf = sales_by_dayofweek.sort_values(["customer_id", "sales"], ascending=False).drop_duplicates(subset=['customer_id'],keep='first')[["customer_id","dayofweek"]]       
        return out_cdf 

class UseBuyIntervalBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_list):
        self.key_col = key_col
        self.agg_list = agg_list
        
    def transform(self, input_cdf: cudf.DataFrame):
        t_dat_df = self.make_day_diff(input_cdf)
        # ここで作成
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