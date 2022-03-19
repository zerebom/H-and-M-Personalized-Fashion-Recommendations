
#%%

import json
import plotly.express as px
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cudf
import nmslib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

if True:
    sys.path.append("../")

root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
output_dir = root_dir / "output"
DRY_RUN = True

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.plotting.backend = "plotly"

%matplotlib inline
%load_ext autoreload
%autoreload 2
#%%
def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

with open(str(input_dir/ 'emb/customer_emb.json')) as f:
    emb_dic = json.load(f,object_hook=jsonKeys2int)


#%%
DRY_RUN = False

if not DRY_RUN:
    trans_cdf = cudf.read_parquet(input_dir / 'transactions_train.parquet')
    cust_cdf = cudf.read_parquet(input_dir / 'customers.parquet')
    art_cdf = cudf.read_parquet(input_dir / 'articles.parquet')
else:
    sample = 0.05
    trans_cdf = cudf.read_parquet(
        input_dir / f"transactions_train_sample_{sample}.parquet")
    cust_cdf = cudf.read_parquet(
        input_dir / f'customers_sample_{sample}.parquet')
    art_cdf = cudf.read_parquet(input_dir /
                               f'articles_train_sample_{sample}.parquet')
# %%

target_customers = trans_cdf['customer_id'].drop_duplicates().to_pandas().values
interaction_data = trans_cdf.copy().to_pandas()

#%%
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

class CandidateRetrieval:

    def __init__(
        self,
        article_emb_dict: Dict[Any, np.ndarray],
        max_neighbors: int,

    ) -> None:
        # NMSLib に検索対象となるベクトルを登録する
        nmslib_index, registered_article_ids = create_nmslib_index(
            vector_dict=article_emb_dict, efSearch=max_neighbors
        )
        self.nmslib_index = nmslib_index
        self.lib_idx_2_art_id = { i: art_id for i, art_id in enumerate(registered_article_ids)}
        self.max_neighbors = max_neighbors

    def search(
        self,
        n_neighbors: int,
        target_customers: List[int],
        target_articles: List[int],
        interaction_data: pd.DataFrame,
        article_emb_dict: Dict[Any, np.ndarray],
    ) -> Dict[int, List[int]]:
        """推論候補となるアイテムを検索する

        ユーザが購入したアイテムの近傍に位置するアイテム群の和集合を、そのユーザに対する推論候補集合としている。
        """
        log_dict = (
            interaction_data
            .groupby(["customer_id"])["article_id"]
            .agg(list)
            .to_dict()
        )
        target_articles_set = set(target_articles)
        candidates_dict = {}
        for customer_id in tqdm(target_customers):
            customer_bought_article_ids = log_dict[customer_id]
            candidate_ids = []

            # ユーザの購入商品ごとに近傍アイテムを取得
            for article_id in customer_bought_article_ids:
                article_emb = article_emb_dict[article_id]
                scoutee_neighbor_ids, _ = self.nmslib_index.knnQuery(
                    article_emb, k=n_neighbors
                )
                candidate_ids.extend(list(scoutee_neighbor_ids))

            # lib_idx_2_art_id でnsmlibのindex順を article_id に変換
            candidate_ids = [self.lib_idx_2_art_id[i] for i in candidate_ids]

            # drop duplicates & 指定したarticle_ids 以外除去 & max_neighborsでlimit
            candidate_ids = list(set(candidate_ids)&target_articles_set)[:self.max_neighbors]
            candidates_dict[customer_id] = candidate_ids

        return candidates_dict

#%%
candidate_retrieval = CandidateRetrieval(emb_dic, 100)


#%%

recent_bought_art_ids = interaction_data.query('week>=100')['article_id'].unique()
candidates_dict = candidate_retrieval.search(
    n_neighbors=10,
    target_customers=target_customers,
    target_articles = recent_bought_art_ids,
    interaction_data=interaction_data,
    article_emb_dict=emb_dic,
)
#%%


with open(str(input_dir / 'emb/customer_to_similar_item.json'), 'w') as f:
    json.dump(emb_dic, f, indent=4)

# %%
