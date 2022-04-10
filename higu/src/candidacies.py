from typing import Any, Dict, List, Optional, Tuple

import cudf
import nmslib
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import customer_hex_id_to_int

to_cdf = cudf.DataFrame.from_pandas


def candidates_dict2df(candidates_dict) -> cudf.DataFrame:
    """
    {'customer_id_a': [article_id_1, article_id_2],'customer_id_b': [article_id_3, article_id_4]}
    -> pd.DataFrame([[customer_id_a, article_id_1], [customer_id_a, article_id_2],]...)
    """
    tuples = []
    for cust, articles in tqdm(candidates_dict.items()):
        for art in articles:
            tuples.append((cust, art))

    df = pd.DataFrame(tuples)
    df.columns = ["customer_id", "article_id"]
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
    M: int = 60,  # M の値をある程度まで大きくすると、再現性が向上して検索時間が短くなるが、インデックス作成時間が伸びる (5 ~ 100 が妥当な範囲)
) -> Tuple[Any, List[Any]]:
    """
    近似を探訪する

    Args:
        vector_dict (Dict[Any, np.ndarray]): _description_
        efSearch (int): _description_
        method (str, optional): _description_. Defaults to "hnsw".
        space (str, optional): _description_. Defaults to "cosinesimil".
        post (int, optional): _description_. Defaults to 2.
        efConstruction (int, optional): _description_. Defaults to 300.

    Returns:
        Tuple[Any, List[Any]]: _description_
    """
    print("train NMSLib")
    # initialize a new index
    nmslib_index = nmslib.init(method=method, space=space)

    # matixで渡さないといけない
    embs = np.stack(list(vector_dict.values()))
    keys = list(vector_dict.keys())

    # データを渡す 
    nmslib_index.addDataPointBatch(embs)

    # ここなにしてるんだ、、？
    index_time_params = {"M": M, "efConstruction": efConstruction, "post": post}
    nmslib_index.createIndex(index_time_params, print_progress=True)

    # efSearch を大きい値に設定しないと knnQuery で指定した数(k)より少ない結果が返ってきてしまう
    # https://github.com/nmslib/nmslib/issues/172#issuecomment-281376808
    # max(k, efSearch) を設定すると良いらしい
    query_time_params = {"efSearch": efSearch}

    # ここで探す(？)
    nmslib_index.setQueryTimeParams(query_time_params)
    return nmslib_index, keys

"""" 
抽象クラス
"""

class AbstractCGBlock:
    def fit(
        self, input_cdf: cudf.DataFrame, logger, y_cdf: Optional[cudf.DataFrame] = None
    ) -> Dict[int, List[int]]:

        out_dic = self.transform(input_cdf)
        # inspect走るようにしてる
        self.inspect(y_cdf, logger)

        return out_dic

    def transform(self, input_cdf: cudf.DataFrame) -> Dict[int, List[int]]:
        '''
        以下の処理を書いてください
        1. input_cdf を加工して、[customer_id,article_id]ペアをレコードにもつself.out_cdf に格納する
        2. self.out_cdfに self.cdf2dicに渡して[customer_id,List[article_id]]を持つ辞書形式にする

        y_cdf,self.out_cdfがある時、inspectが走ります。
        '''

        raise NotImplementedError()

    def inspect(self, y_cdf, logger):
        if hasattr(self, "out_cdf") and y_cdf is not None:
            customer_cnt = self.out_cdf["customer_id"].nunique()
            article_cnt = self.out_cdf["article_id"].nunique()
            merged_cdf = self.out_cdf.merge(
                y_cdf, on=["customer_id", "article_id"], how="inner"
            )
            all_pos_cnt = len(y_cdf)

            pos_in_cg_cnt = len(merged_cdf)
            candidate_cnt = len(self.out_cdf)

            pos_ratio = round(pos_in_cg_cnt / candidate_cnt, 3)
            pos_coverage = round(pos_in_cg_cnt / all_pos_cnt, 3)

            # 候補集合数・候補集合内の正例数・候補集合内の正例率・正例カバレッジ率
            # pos_ratioが0.5になるのが原理上は理想(難易度高すぎで無理だけど)
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

    @staticmethod
    def cdf2dic(cdf, key_col, item_col):
        out_dic = (
            cdf.to_pandas().groupby(key_col)[item_col].apply(list).to_dict()
        )  # ユーザごとに商品をリストに変換
        return out_dic

""" 
正例を増やす
"""

class BoughtItemsAtInferencePhase(AbstractCGBlock):
    """
    推論期間に購入した商品をCandidateとして返す
    (リークだが、正例を増やす最も手っ取り早い方法)

    """

    def __init__(
        self,
        y_cdf,
        key_col="customer_id",
        item_col="article_id",
    ):

        self.y_cdf = y_cdf
        self.key_col = key_col
        self.item_col = item_col

    def transform(self, _input_cdf):
        self.out_cdf = self.y_cdf.copy()
        out_dic = self.cdf2dic(self.out_cdf, self.key_col, self.item_col)

        return out_dic


"""
ルールベース
"""
class RuleBase(AbstractCGBlock):
    """
    ルールベース

    """

    def __init__(
        self,
        file_path,
        target_customers,
        key_col="customer_id",
        item_col="article_id",
    ):

        self.target_customers = target_customers
        self.file_path = file_path
        self.key_col = key_col
        self.item_col = item_col

    def transform(self, _input_cdf):
        df = cudf.read_csv(self.file_path).to_pandas()
        tqdm.pandas()

        df['customer_id'] = customer_hex_id_to_int(df['customer_id'])
        df['prediction'] = df['prediction'].progress_apply(self.split)

        df = df.explode('prediction').dropna().reset_index(drop=True)
        df['prediction'] = df['prediction'].astype(int)
        df.columns = [self.key_col,self.item_col]
        df = df[df['customer_id'].isin(self.target_customers)]

        self.out_cdf = to_cdf(df)
        out_dic = self.cdf2dic(self.out_cdf, self.key_col, self.item_col)

        return out_dic

    @staticmethod
    def split(x):
        if type(x) is str:
            return x.split(' ')
        else:
            return []


""" 
似ている商品を取得
"""

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
        target_customers: List[int],
    ) -> None:
        # NMSLib に検索対象となるベクトルを登録する
        nmslib_index, registered_article_ids = create_nmslib_index(
            vector_dict=article_emb_dict, efSearch=max_neighbors
        )
        self.nmslib_index = nmslib_index
        self.lib_idx_2_art_id = {
            i: art_id for i, art_id in enumerate(registered_article_ids)
        }
        self.max_neighbors = max_neighbors
        self.k_neighbors = k_neighbors
        self.article_emb_dict = article_emb_dict
        self.target_articles = target_articles
        self.target_customers = target_customers

    def transform(self, _trans_cdf):
        # 絞られた期間のtrans_dfを取得する
        trans_df = _trans_cdf.to_pandas()
        # 過去の購買商品を出す
        log_dict = trans_df.groupby(["customer_id"])["article_id"].agg(list).to_dict()

        # 対象となるユーザー + 絞られた期間に購買したユーザー
        self.target_customers = list(
            set(self.target_customers) & set(list(trans_df["customer_id"].unique()))
        )

        # 対象商品を取得する
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
            candidate_ids = list(set(candidate_ids) & target_articles_set)[
                : self.max_neighbors
            ]
            candidates_dict[customer_id] = candidate_ids
            # TODO: candidates_dict→ out_cdfにする処理が書けるとinspectできます。力尽きて書いてない

        return candidates_dict


class LastNWeekArticles(AbstractCGBlock):
    """
    各ユーザごとに最終購買週からn週間前までに他の人が購入した購入商品を取得
    """

    def __init__(self, key_col="customer_id", item_col="article_id", n_weeks=2):
        self.key_col = key_col
        self.item_col = item_col
        self.n_weeks = n_weeks
        self.out_keys = [key_col, item_col]

    def transform(self, _input_cdf):
        input_cdf = _input_cdf.copy()

        week_df = (
            input_cdf.groupby(self.key_col)["week"]
            .max()
            .reset_index()
            .rename({"week": "max_week"}, axis=1)
        )  # ユーザごとに最後に買った週を取得

        # ここがclipの期間になってる？
        input_cdf = input_cdf.merge(week_df, on=self.key_col, how="left")
        input_cdf["diff_week"] = (
            input_cdf["max_week"] - input_cdf["week"]
        )  # 各商品とユーザの最終購入週の差分
        self.out_cdf = input_cdf.loc[
            input_cdf["diff_week"] <= self.n_weeks, self.out_keys
        ]  # 差分がn週間以内のものを取得

        out_dic = self.cdf2dic(self.out_cdf, self.key_col, self.item_col)
        return out_dic



""" 
人気商品系
"""

class PopularItemsoftheLastWeeks(AbstractCGBlock):
    """
    最終購入週の人気アイテムtopNを返す
    TODO: 最終購入数以外の商品も取得できるようにする

    """

    def __init__(
        self, customer_ids, key_col="customer_id", item_col="article_id", n=12
    ):
        self.customer_ids = customer_ids
        self.key_col = key_col
        self.item_col = item_col
        self.n = n

    def transform(self, _input_cdf):
        input_cdf = _input_cdf.copy()
        pop_item_dic = {}
        self.popular_items = (
            input_cdf.loc[input_cdf["week"] == input_cdf["week"].max()][self.item_col]
            .value_counts()
            .to_pandas()
            .index[: self.n]
            .values
        )

        for cust_id in self.customer_ids:
            pop_item_dic[cust_id] = list(self.popular_items)

        return pop_item_dic


class LastWeekPopularArticles(AbstractCGBlock):
    """
    各ユーザごとに最終購買週の人気商品を返す
    """

    def __init__(self, key_col="customer_id", item_col="article_id"):
        self.key_col = key_col
        self.item_col = item_col
        self.out_keys = [key_col, item_col]

    def transform(self, _input_cdf):
        input_cdf = _input_cdf.copy()

        week_df = (
            input_cdf.groupby(self.key_col)["t_dat"]
            .max()
            .reset_index()
            .rename({"t_dat": "max_t_dat"}, axis=1)
        )  # ユーザごとに最後に買った日を取得

        input_cdf = input_cdf.merge(week_df, on=self.key_col, how="left")
        input_cdf["diff_day"] = (
            input_cdf["max_t_dat"] - input_cdf["t_dat"]
        )  # 各商品とユーザの最終購入週の差分
        self.out_cdf = input_cdf.loc[
            input_cdf["diff_day"] <= 0, self.out_keys
        ]  # 差分が0日以内のものを取得

        out_dic = self.cdf2dic(self.out_cdf, self.key_col, self.item_col)
        return out_dic
