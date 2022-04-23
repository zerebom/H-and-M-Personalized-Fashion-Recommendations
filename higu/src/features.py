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

class SexBlock(AbstractBaseBlock):
    # その人の性別が何であるかを判断する
    def __init__(self, key_col):
        self.key_col = key_col
    
    def fit(self, trans_cdf, art_cdf):
        return self.transform(trans_cdf, art_cdf)

    def transform(self, trans_cdf: cudf.DataFrame, art_cdf: cudf.DataFrame):
        women_regexp = "women|girl|ladies"
        men_regexp = "boy|men" # womenが含まれてしまうのであとで処理する
        articles_sex_cdf = self.make_article_sex_info(art_cdf, women_regexp, men_regexp)
        out_cdf = self.make_customer_sex_info(articles_sex_cdf, trans_cdf)
        return out_cdf 

    def make_article_sex_info(df, women_regexp, men_regexp):
        df["women"] = 0
        df["men"] = 0

        # どこかのcolでwomenっぽい単語があれば女性と判断で良さそう
        for col in ["index_name", "index_group_name","section_name"]:
            df[col] = df[col].str.lower()
            df["women"] += df[col].str.contains(women_regexp).astype(int) 
            # womenを含む場合があるので対処
            mens = df[col].str.contains(men_regexp).astype(int) - df[col].str.contains("women").astype(int)
            df["men"] += mens

        # 0でなければ女性と判断(3列あるので、足し合わせると3になってしまうことがある。その対処)
        df["women_flg"] = (df["women"] > 0).astype(int)
        df["men_flg"] = (df["men"] > 0).astype(int)

        # 必要なものはそのarticleがwomenなのかmenなのかの情報だけ
        return_df = df[["article_id","women_flg","men_flg"]]
        return return_df

    def make_customer_sex_info(articles_sex, transactions):
        trans = transactions.merge(articles_sex,on="article_id", how="left")
        assert len(transactions) == len(trans), "mergeうまくいってないです"
        rename_dict = {"women_flg":"women_percentage", "men_flg":"men_percentage"}
        trans = trans.groupby("customer_id")[["women_flg","men_flg"]].mean().reset_index().rename(columns=rename_dict)
        return trans


class RepeatSalesUserNum5(AbstractBaseBlock):
    # 一旦2-5を出そう
    # TODO: ハードコーディング部分を直す？考える
    def __init__(self, key_col):
        self.key_col = key_col
    
    def fit(self, trans_cdf, art_cdf):
        return self.transform(trans_cdf, art_cdf)

    def transform(self, trans_cdf: cudf.DataFrame, art_cdf: cudf.DataFrame):
        repeat_num_df = self.repeat_num(trans_cdf)
        
        out_cdf = art_cdf[["article_id"]] # 最初に紐づける先を用意する
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