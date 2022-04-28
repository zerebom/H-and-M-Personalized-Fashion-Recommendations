import cudf
import pandas as pd
import lifetimes
from lifetimes.utils import summary_data_from_transaction_data
from cuml import TruncatedSVD
from cuml.feature_extraction.text import TfidfVectorizer


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
    
class LifetimesBlock(AbstractBaseBlock):
    def __init__(self, key_col, target_col):

        self.key_col = key_col # 'customer_id'
        self.target_col = target_col # 'price'

    def transform(self, input_cdf: cudf.DataFrame):
        price_mean_cust_dat= input_cdf.groupby([self.key_col, 't_dat'])[self.target_col].sum().reset_index().to_pandas()
        out_df = summary_data_from_transaction_data(
            price_mean_cust_dat, self.key_col, 't_dat', 
            observation_period_end=price_mean_cust_dat.t_dat.max()# train:2020-09-08, valid: 2020-09-15 , test: 2020-09-22 
        ).reset_index()
        
        out_cdf = cudf.from_pandas(out_df)

        return out_cdf
    
    
class CustItemEmbBlock(AbstractBaseBlock):
    def __init__(self, key_col, enb_size, article_cdf):

        self.key_col = key_col # 'customer_id'
        self.enb_size = enb_size
        self.article_cdf = article_cdf # parquetはtxtが空なのでローデータのcsvを使用
        
    def transform(self, input_cdf: cudf.DataFrame):
        input_cdf = input_cdf.merge(self.article_cdf, on = 'article_id')
        
        # customerごとにtextを集約, これで顧客ごとに買う系統の服の情報が集まるイメージ
        customer_desc_df = input_cdf.to_pandas().groupby(self.key_col)['text'].apply(list).apply(' '.join).reset_index()
        customer_desc_cdf = cudf.from_pandas(customer_desc_df)
        del customer_desc_df
        
        # TFIDF & TruncatedSVD
        tfidf = TfidfVectorizer(min_df=3)
        V_desc = tfidf.fit_transform(customer_desc_cdf["text"].fillna("nodesc"))
        svd = TruncatedSVD(n_components=self.enb_size, random_state=0)
        svd.fit(V_desc.todense())
        V_desc = svd.transform(V_desc.todense())
        
        out_cdf = cudf.DataFrame(V_desc)
        out_cdf.columns = [f"cust_itme_desc{col}" for col in out_cdf.columns] 
        out_cdf['customer_id'] = customer_desc_cdf['customer_id']

        return out_cdf
    
class ItemBiasIndicatorBlock(AbstractBaseBlock):
    """
    その商品を買う人に偏りがあるのかどうかといった指標
    その商品はある一部の人から購入されているのか、万人に購入されているのか
    例: 超奇抜なピンクの服 -> ある一部の人から購入されている
    例: 万人に購入されている -> 黒い下着
    """
    def __init__(self, key_col, target_col):

        self.key_col = key_col # 'article_id'
        self.target_col = target_col # 'customer_id'
        
    def transform(self, input_cdf: cudf.DataFrame):
        nunique_user_by_article = input_cdf.groupby(self.key_col)[self.target_col].nunique().reset_index()
        nunique_user_by_article.columns =[self.key_col, "nunique_user_by_article"]

        # すでに作成済み？？？
        count_user_by_article = input_cdf.groupby(self.key_col)[self.target_col].count().reset_index()
        count_user_by_article.columns =[self.key_col, "count_user_by_article"]

        out_cdf = nunique_user_by_article.merge(count_user_by_article, on=self.key_col)
        out_cdf["suitable_for_everyone"] = out_cdf["nunique_user_by_article"] / out_cdf["count_user_by_article"]
        return out_cdf