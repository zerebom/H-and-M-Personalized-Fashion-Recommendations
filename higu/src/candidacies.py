from typing import Any, Dict, List, Tuple

import cudf
import nmslib
import numpy as np
import pandas as pd
from tqdm import tqdm

to_cdf = cudf.DataFrame.from_pandas

def candidates_dict2df(candidates_dict):
    '''
    {'customer_id_a': [article_id_1, article_id_2],'customer_id_b': [article_id_3, article_id_4]}
    -> pd.DataFrame([[customer_id_a, article_id_1], [customer_id_a, article_id_2],]...)
    '''
    tuples = []
    for cust, articles in tqdm(candidates_dict.items()):
        for art in articles:
            tuples.append((cust, art))

    df = pd.DataFrame(tuples)
    df.columns = ['customer_id', 'article_id']
    cdf = to_cdf(df)
    cdf = cdf.drop_duplicates().reset_index(drop=True)
    return cdf


def create_nmslib_index(
    vector_dict: Dict[Any, np.ndarray],
    efSearch: int,
    method: str = "hnsw",
    space: str = "cosinesimil",
    post: int = 2,
    efConstruction: int = 300,  # efConstruction の値を大きくするほど検索精度が向上するが、インデックス作成時間が伸びる
    M: int = 30,  # M の値をある程度まで大きくすると、再現性が向上して検索時間が短くなるが、インデックス作成時間が伸びる (5 ~ 100 が妥当な範囲)
) -> Tuple[Any, List[Any]]:
    print("train NMSLib")
    nmslib_index = nmslib.init(method=method, space=space)
    embs = np.stack(list(vector_dict.values()))
    keys = list(vector_dict.keys())
    nmslib_index.addDataPointBatch(embs)

    index_time_params = {"M": M, "efConstruction": efConstruction, "post": post}
    nmslib_index.createIndex(index_time_params, print_progress=True)

    # efSearch を大きい値に設定しないと knnQuery で指定した数(k)より少ない結果が返ってきてしまう
    # https://github.com/nmslib/nmslib/issues/172#issuecomment-281376808
    # max(k, efSearch) を設定すると良いらしい
    query_time_params = {"efSearch": efSearch}
    nmslib_index.setQueryTimeParams(query_time_params)
    return nmslib_index, keys

class AbstractCGBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        raise NotImplementedError()

    def get_cudf(self):
        return to_cdf(self.out_df)


class ArticlesSimilartoThoseUsersHavePurchased(AbstractCGBlock):
    """
        transactionとアイテムのエンべディングを使って、
        ユーザが過去の購入商品と似ている商品を取得する
    """

    def __init__(
        self,
        article_emb_dict: Dict[Any, np.ndarray],
        max_neighbors: int,
        k_neighbors: int,
        target_articles: List[int],
        target_customers: List[int]

    ) -> None:
        # NMSLib に検索対象となるベクトルを登録する
        nmslib_index, registered_article_ids = create_nmslib_index(
            vector_dict=article_emb_dict, efSearch=max_neighbors
        )
        self.nmslib_index = nmslib_index
        self.lib_idx_2_art_id = { i: art_id for i, art_id in enumerate(registered_article_ids)}
        self.max_neighbors = max_neighbors
        self.k_neighbors = k_neighbors
        self.article_emb_dict = article_emb_dict
        self.target_articles = target_articles
        self.target_customers = target_customers

    def transform(self, _trans_cdf):
        trans_df = _trans_cdf.to_pandas()
        log_dict = (
            trans_df
            .groupby(["customer_id"])["article_id"]
            .agg(list)
            .to_dict()
        )
        self.target_customers = list(set(self.target_customers) & set(list(trans_df['customer_id'].unique())))
        target_articles_set = set(self.target_articles)
        candidates_dict = {}
        for customer_id in tqdm(self.target_customers):
            customer_bought_article_ids = log_dict[customer_id]
            candidate_ids = []

            # ユーザの購入商品ごとに近傍アイテムを取得
            for article_id in customer_bought_article_ids:
                article_emb = self.article_emb_dict[article_id]
                scoutee_neighbor_ids, _ = self.nmslib_index.knnQuery(
                    article_emb, k=self.k_neighbors
                )
                candidate_ids.extend(list(scoutee_neighbor_ids))

            # lib_idx_2_art_id でnsmlibのindex順を article_id に変換
            candidate_ids = [self.lib_idx_2_art_id[i] for i in candidate_ids]

            # drop duplicates & 指定したarticle_ids 以外除去 & max_neighborsでlimit
            candidate_ids = list(set(candidate_ids)&target_articles_set)[:self.max_neighbors]
            candidates_dict[customer_id] = candidate_ids

        return candidates_dict


class LastNWeekArticles(AbstractCGBlock):
    """
    各ユーザごとに最終購買週からn週間前までの購入商品を取得する
    """
    def __init__(
            self,
            key_col='customer_id',
            item_col='article_id',
            n_weeks=2):
        self.key_col = key_col
        self.item_col = item_col
        self.n_weeks = n_weeks
        self.out_keys = [key_col, item_col]

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        input_df = _input_df.copy()

        week_df = input_df.\
            groupby(self.key_col)['week'].max()\
            .reset_index().\
            rename({'week': 'max_week'}, axis=1) #ユーザごとに最後に買った週を取得

        input_df = input_df.merge(week_df, on=self.key_col, how='left')
        input_df['diff_week'] = input_df['max_week'] - input_df['week'] #各商品とユーザの最終購入週の差分
        self.out_df = input_df.loc[input_df['diff_week']
                                   <= self.n_weeks, self.out_keys].to_pandas() #差分がn週間以内のものを取得

        out_dic = self.out_df.groupby(self.key_col)[
            self.item_col].apply(list).to_dict() #ユーザごとに購入商品をリストに変換
        return out_dic



class BoughtItemsAtInferencePhase(AbstractCGBlock):
    '''
    推論期間に購入した商品をCandidateとして返す
    (リークだが、正例を増やす最も手っ取り早い方法)

    '''
    def __init__(
            self,
            y_cdf,
            key_col='customer_id',
            item_col='article_id',
            ):

        self.y_cdf = y_cdf
        self.key_col = key_col
        self.item_col = item_col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        out_df = self.y_cdf\
            .to_pandas()\
            .groupby(self.key_col)[self.item_col]\
            .apply(list).to_dict()

        return out_df


class PopularItemsoftheLastWeeks(AbstractCGBlock):
    '''
    最終購入週の人気アイテムtopNを返す
    TODO: 最終購入数以外の商品も取得できるようにする

    '''

    def __init__(
            self,
            customer_ids,
            key_col='customer_id',
            item_col='article_id',
            n=12):
        self.customer_ids = customer_ids
        self.key_col = key_col
        self.item_col = item_col
        self.n = n

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        input_df = _input_df.copy()
        pop_item_dic = {}
        self.popular_items = input_df\
            .loc[input_df['week'] == input_df['week'].max()][self.item_col]\
            .value_counts()\
            .to_pandas()\
            .index[:self.n]\
            .values

        for cust_id in self.customer_ids:
            pop_item_dic[cust_id] = list(self.popular_items)

        return pop_item_dic
