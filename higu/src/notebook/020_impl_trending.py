# %%

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from asyncio import SubprocessTransport
import json
import os
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cudf
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


if True:
    sys.path.append("../")
    sys.path.append("/home/kokoro/h_and_m/higu/src")
    # sys.path.append("/home/ec2-user/kaggle/h_and_m/higu/src")
    from candidacies_dfs import (
        LastBoughtNArticles,
        PopularItemsoftheLastWeeks,
        LastNWeekArticles,
        PairsWithLastBoughtNArticles,
        MensPopularItemsoftheLastWeeks,
        EachSexPopularItemsoftheLastWeeks,
    )
    from eda_tools import visualize_importance
    from features import (
        EmbBlock,
        ModeCategoryBlock,
        TargetEncodingBlock,
        LifetimesBlock,
        SexArticleBlock,
        SexCustomerBlock,
        # Dis2HistoryAveVecAndArtVec,
        RepeatCustomerBlock,
        MostFreqBuyDayofWeekBlock,
        CustomerBuyIntervalBlock,
        RepeatSalesCustomerNum5Block,
        ArticleBiasIndicatorBlock,
        FirstBuyDateBlock,
        SalesPerTimesForArticleBlock,
        ColorCustomerBlock,
    )
    from metrics import mapk, calc_map12
    import config
    from utils import (
        article_id_int_to_str,
        article_id_str_to_int,
        arts_id_list2str,
        convert_sub_df_customer_id_to_str,
        customer_hex_id_to_int,
        get_var_names,
        jsonKeys2int,
        phase_date,
        read_cdf,
        reduce_mem_usage,
        setup_logger,
        squeeze_pred_df_to_submit_format,
        timer,
    )

root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
# input_dir = root_dir / "data"
exp_name = Path(os.path.basename(__file__)).stem
output_dir = root_dir / "output" / exp_name
log_dir = output_dir / "log"
emb_dir = input_dir / "emb"

output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"{date.today()}.log"
logger = setup_logger(log_file)
DRY_RUN = False
# USE_CACHE = not DRY_RUN
USE_CACHE = True


ArtId = int
CustId = int
ArtIds = List[ArtId]
CustIds = List[CustId]

to_cdf = cudf.DataFrame.from_pandas

#%%

# %load_ext autoreload
# %autoreload 2



raw_trans_cdf, raw_cust_cdf, raw_art_cdf = read_cdf(input_dir, DRY_RUN)

# %%


def feature_generation(
    blocks, trans_cdf, art_cdf, cust_cdf, base_cdf, y_cdf, target_customers, target_week
):
    """
    feature_generation blocksを使って、特徴量を作成する。
    art_id, cust_idをkeyに持つdataframeを返す。
    """

    art_feat_cdf = art_cdf[["article_id"]]  # base
    cust_feat_cdf = cust_cdf[["customer_id"]]

    if base_cdf is not None:
        pair_feat_cdf = base_cdf[["customer_id", "article_id"]]
    else:
        pair_feat_cdf = cudf.DataFrame()

    for block in blocks:
        with timer(logger=logger, prefix="fit {} ".format(block.__class__.__name__)):

            feature_cdf = block.fit(
                trans_cdf,
                art_cdf,
                cust_cdf,
                base_cdf,
                y_cdf,
                target_customers,
                logger,
                target_week,
            )

            if block.key_col == "article_id":
                art_feat_cdf = art_feat_cdf.merge(feature_cdf, how="left", on=block.key_col)
            elif block.key_col == "customer_id":
                cust_feat_cdf = cust_feat_cdf.merge(feature_cdf, how="left", on=block.key_col)
            elif block.key_col == ["customer_id", "article_id"]:
                if base_cdf is not None:
                    pair_feat_cdf = pair_feat_cdf.merge(feature_cdf, how="left", on=block.key_col)

    cust_cdf = cust_cdf.merge(cust_feat_cdf, on="customer_id", how="left").to_pandas()
    art_df = art_cdf.merge(art_feat_cdf, on="article_id", how="left").to_pandas()
    pair_feat_df = pair_feat_cdf.to_pandas()
    return art_df, cust_cdf, pair_feat_df


agg_list = ["mean", "max", "min", "std", "median"]
groupby_cols_dict = {
    "year_month_day": ["year", "month", "day"],
    "year_month": ["year", "month"],
    "day": ["day"],
}
agg_list_for_sales = ["max", "sum", "min"]
agg_list_for_color = ["mean", "std"]

customer_feat_blocks = [
    *[SexCustomerBlock("customer_id", USE_CACHE)],
    *[LifetimesBlock("customer_id", "price", USE_CACHE)],
    *[RepeatCustomerBlock("customer_id", USE_CACHE)],
    *[MostFreqBuyDayofWeekBlock("customer_id", USE_CACHE)],
    *[CustomerBuyIntervalBlock("customer_id", agg_list, USE_CACHE)],
    *[ColorCustomerBlock("customer_id", agg_list_for_color, False)],
]

_, cust_cdf, _ = feature_generation(
    customer_feat_blocks, raw_trans_cdf, raw_art_cdf, raw_cust_cdf, None, None, None, None
)
#%%


#%%
from cuml.cluster import KMeans

# %%

cust_cdf = cust_cdf.fillna(cust_cdf.mean())
X = cust_cdf.drop(columns=["customer_id", "postal_code", "dayofweek"])


scaler = StandardScaler()
X = scaler.fit_transform(X)
k_means = KMeans(n_clusters=300, random_state=0, verbose=True)
cust_cdf["cluster"] = k_means.fit_predict(X)
cust_cdf[["customer_id", "cluster"]].to_csv(input_dir / "customer_kmeans_label.csv", index=False)
#%%
cust_cdf.query('cluster==1').mean()
#%%

cust_cdf.query('cluster==100').mean()

# %%
# cluster_cdf = cudf.read_csv(input_dir / "customer_kmeans_label.csv")
# cols = ["customer_id", "article_id", "cluster"]
# popluar_items_each_cluster_cdf = to_cdf(
#     raw_trans_cdf.merge(cluster_cdf, how="left", on="customer_id")[cols]
#     .groupby(["article_id", "cluster"])
#     .size()
#     .to_frame(name="size")
#     .reset_index()
#     .to_pandas()
#     .sort_values(["size", "cluster"], ascending=False)
#     .groupby("cluster")
#     .head(50)
#     .reset_index(drop=True)
# )

# out_cdf = cluster_cdf.merge(popluar_items_each_cluster_cdf, how="left", on="label")[
#     ["customer_id", "article_id"]
# ]
