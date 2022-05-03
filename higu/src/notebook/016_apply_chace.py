# %%
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict

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
    #sys.path.append("/home/kokoro/h_and_m/higu/src")
    sys.path.append("/home/ec2-user/kaggle/h_and_m/higu/src")
    from candidacies_dfs import (
        LastBoughtNArticles,
        PopularItemsoftheLastWeeks,
        LastNWeekArticles,
        PairsWithLastBoughtNArticles,
        MensPopularItemsoftheLastWeeks,
        EachSexPopularItemsoftheLastWeeks
    )
    from eda_tools import visualize_importance
    from features import (
        EmbBlock,
        ModeCategoryBlock,
        TargetEncodingBlock,
        LifetimesBlock,
        SexArticleBlock,
        SexCustomerBlock,
        #Dis2HistoryAveVecAndArtVec,
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


root_dir = Path("/home/ec2-user/kaggle/h_and_m")
input_dir = root_dir / "data"
#exp_name = Path(os.path.basename(__file__)).stem
exp_name = "exp15"
exp_disc = "性別ごとにtopN商品を推薦する候補生成を追加"
output_dir = root_dir / "output"
log_dir = output_dir / "log" / exp_name

output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"{date.today()}.log"
logger = setup_logger(log_file)
DRY_RUN = False
#USE_CACHE = not DRY_RUN
USE_CACHE = True


ArtId = int
CustId = int
ArtIds = List[ArtId]
CustIds = List[CustId]

to_cdf = cudf.DataFrame.from_pandas


# %%

raw_trans_cdf, raw_cust_cdf, raw_art_cdf = read_cdf(input_dir, DRY_RUN)
# with open(str(input_dir / "emb/article_desc_bert.json")) as f:
#     article_desc_bert_dic = json.load(f, object_hook=jsonKeys2int)


# with open(str(input_dir / "emb/svd.json")) as f:
#     article_svd_dic = json.load(f, object_hook=jsonKeys2int)

# with open(str(input_dir / "emb/resnet-18_umap_10.json")) as f:
#     resnet18_umap_10_dic = json.load(f, object_hook=jsonKeys2int)


#%%

candidate_blocks = [
    *[PairsWithLastBoughtNArticles(n=30, use_cache=USE_CACHE)],
    #*[PopularItemsoftheLastWeeks(n=30, use_cache=USE_CACHE)],
    *[LastBoughtNArticles(n=30, use_cache=USE_CACHE)],
    *[LastNWeekArticles(n_weeks=2, use_cache=USE_CACHE)],
    *[EachSexPopularItemsoftheLastWeeks(n=30, use_cache=USE_CACHE)],
]

# inferenceはバッチで回すのでcacheは使わない
candidate_blocks_test = [
    *[PairsWithLastBoughtNArticles(n=30, use_cache=False)],
    #*[PopularItemsoftheLastWeeks(n=30, use_cache=False)],
    *[LastBoughtNArticles(n=30, use_cache=False)],
    *[LastNWeekArticles(n_weeks=2, use_cache=False)],
    *[EachSexPopularItemsoftheLastWeeks(n=30, use_cache=False)],
]


# candidate_blocks = [
#     *[PairsWithLastBoughtNArticles(n=60, use_cache=USE_CACHE)],
#     *[PopularItemsoftheLastWeeks(n=60, use_cache=USE_CACHE)],
#     *[LastBoughtNArticles(n=60, use_cache=USE_CACHE)],
#     *[LastNWeekArticles(n_weeks=4, use_cache=USE_CACHE)],
# ]

# candidate_blocks_test = [
#     *[PairsWithLastBoughtNArticles(n=60, use_cache=USE_CACHE)],
#     *[PopularItemsoftheLastWeeks(n=60, use_cache=USE_CACHE)],
#     *[LastBoughtNArticles(n=60, use_cache=USE_CACHE)],
#     *[LastNWeekArticles(n_weeks=4, use_cache=USE_CACHE)],
# ]


agg_list = ["mean", "max", "min", "std", "median"]


article_feature_blocks = [
    *[SexArticleBlock("article_id", USE_CACHE)],
    # *[
    #     EmbBlock("article_id", article_desc_bert_dic, "bert"),
    # ],
    # *[
    #     EmbBlock("article_id", resnet18_umap_10_dic, "res18"),
    # ],
]

transaction_feature_blocks = [
    # *[Dis2HistoryAveVecAndArtVec(["customer_id", "article_id"], article_svd_dic, "svd")],
    *[SexCustomerBlock("customer_id", USE_CACHE)],
    *[LifetimesBlock("customer_id", "price", USE_CACHE)],
    *[TargetEncodingBlock("article_id", "customer_id", "age", agg_list + ["count"], USE_CACHE)],
    *[
        TargetEncodingBlock("article_id", "customer_id", col, ["mean"], USE_CACHE)
        for col in ["FN", "Active", "club_member_status", "fashion_news_frequency"]
    ],
    *[
        TargetEncodingBlock("article_id", ["customer_id", "article_id"], col, agg_list, USE_CACHE)
        for col in ["price", "sales_channel_id"]
    ],
]


static_feature_test_blocks = [
    *[SexCustomerBlock("customer_id", USE_CACHE)],
    *[LifetimesBlock("customer_id", "price", USE_CACHE)],
    *[TargetEncodingBlock("article_id", "customer_id", "age", agg_list + ["count"], USE_CACHE)],
    *[
        TargetEncodingBlock("article_id", "customer_id", col, ["mean"], USE_CACHE)
        for col in ["FN", "Active", "club_member_status", "fashion_news_frequency"]
    ],
    *[
        TargetEncodingBlock("article_id", ["customer_id", "article_id"], col, agg_list, USE_CACHE)
        for col in ["price", "sales_channel_id"]
    ],
]

transaction_feature_test_blocks = [
    # *[Dis2HistoryAveVecAndArtVec(["customer_id", "article_id"], article_desc_bert_dic, "bert")]
]
#%%


def make_y_data(transactions: cudf.DataFrame, target_week) -> Tuple[cudf.DataFrame, pd.DataFrame]:
    y_cdf = transactions.query(f"week == {target_week}")[
        ["customer_id", "article_id"]
    ].drop_duplicates()
    y_cdf["y"] = 1
    y_cdf = y_cdf[["customer_id", "article_id", "y"]]
    y_df = y_cdf.to_pandas()
    return y_cdf, y_df


def candidate_generation(
    blocks, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers, target_week
) -> pd.DataFrame:

    """
    candidate_generation blocksを使って、推論候補対象を作成する。
    (art_id, cust_id)をkeyに持つdataframeを返す。
    """
    candidates_df = pd.DataFrame()
    for i, block in enumerate(blocks):
        logger.info("↓fit {} ".format(block.__class__.__name__))

        with timer(logger=logger, prefix="↑fitted {} ".format(block.__class__.__name__)):
            if i == 0:
                candidates_df = block.fit(
                    trans_cdf=trans_cdf,
                    art_cdf=art_cdf,
                    cust_cdf=cust_cdf,
                    logger=logger,
                    y_cdf=y_cdf,
                    target_customers=target_customers,
                    target_week=target_week,
                )
            else:
                new_df = block.fit(
                    trans_cdf=trans_cdf,
                    art_cdf=art_cdf,
                    cust_cdf=cust_cdf,
                    logger=logger,
                    y_cdf=y_cdf,
                    target_customers=target_customers,
                    target_week=target_week,
                )
                candidates_df = pd.concat([new_df, candidates_df])

        logger.info(f" ")
    candidates_df.reset_index(drop=True, inplace=True)
    return candidates_df


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

    cust_df = cust_cdf.merge(cust_feat_cdf, on="customer_id", how="left").to_pandas()
    art_df = art_cdf.merge(art_feat_cdf, on="article_id", how="left").to_pandas()
    pair_feat_df = pair_feat_cdf.to_pandas()
    return art_df, cust_df, pair_feat_df


#%%


def negative_sampling(base_df, logger, n):
    positive_df = base_df.query("y==1")
    positive_customer_ids = positive_df["customer_id"].unique()

    # postiveが存在するcustomer_idのみに絞る。不例の数は全ユーザ２０件
    negative_df = base_df.query("y==0")
    negative_df = negative_df[negative_df["customer_id"].isin(positive_customer_ids)]

    # なぜか精度が落ちるサンプル方法
    # negative_df = negative_df.sample(frac=1, random_state=42).reset_index()
    # negative_df = negative_df.reset_index()
    # negative_df["rank"] = negative_df.groupby("customer_id")["index"].rank(method="first").values
    # negative_df = negative_df.query("rank <= @n").drop(columns=["index", "rank"])

    negative_df = negative_df.groupby("customer_id").apply(
        lambda x: x.sample(n=min(n, len(x)), random_state=42)
    )

    try:
        logger.info(
            f"base_df内の正例率: {round(100 * (len(positive_df)/len(negative_df)),3)}% ({len(positive_df)}/{len(negative_df)})"
        )
    except:
        pass

    sampled_df = pd.concat([positive_df, negative_df])

    return sampled_df


def make_train_valid_df(
    trans_cdf, art_cdf, cust_cdf, target_weeks, candidate_blocks, feature_blocks
):
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for target_week in tqdm(target_weeks):
        phase = "validate" if target_week == 104 else "train"
        clipped_trans_cdf = trans_cdf.query(f"week < @target_week")  # 予測日の前週までのトランザクションが使える
        y_cdf, _ = make_y_data(trans_cdf, target_week)

        base_df = candidate_generation(
            candidate_blocks,
            trans_cdf=clipped_trans_cdf,
            art_cdf=art_cdf,
            cust_cdf=cust_cdf,
            y_cdf=y_cdf,
            target_customers=None,
            target_week=target_week,
        )

        n = 50 if phase == "train" else 200
        base_df = negative_sampling(base_df, logger, n)
        base_df = base_df.sample(frac=1, random_state=42)
        base_df = base_df.reset_index(drop=True)
        base_df = base_df.drop_duplicates(
            subset=["customer_id", "article_id", "candidate_block_name"], keep="first"
        ).reset_index(drop=True)

        base_cdf = to_cdf(base_df)
        art_feat_df, cust_feat_df, pair_feat_df = feature_generation(
            feature_blocks,
            trans_cdf=clipped_trans_cdf,
            art_cdf=art_cdf,
            cust_cdf=cust_cdf,
            base_cdf=base_cdf,
            y_cdf=y_cdf,
            target_customers=None,
            target_week=target_week,
        )

        # 特徴量作成
        base_df = base_df.merge(cust_feat_df, how="left", on="customer_id")
        base_df = base_df.merge(art_feat_df, how="left", on="article_id")

        # 同じcust,artのログで重複が生まれる
        base_df = base_df.merge(pair_feat_df, how="left", on=["article_id", "customer_id"])
        base_df = base_df.drop_duplicates(
            subset=["article_id", "customer_id", "candidate_block_name"]
        ).reset_index(drop=True)

        base_df["target_week"] = target_week

        base_df = reduce_mem_usage(base_df)
        if phase == "train":
            # 学習データは複数予測週があるのでconcatする
            train_df = pd.concat([train_df, base_df])
        elif phase == "validate":
            valid_df = base_df.copy()

    query_keys = ["customer_id", "target_week"]
    train_df.sort_values(query_keys, inplace=True)
    valid_df.sort_values(query_keys, inplace=True)

    train_groups = train_df.groupby(query_keys).size().values
    valid_groups = valid_df.groupby(query_keys).size().values

    return train_df, valid_df, train_groups, valid_groups


#%%


art_df_w_feat, _, _ = feature_generation(
    article_feature_blocks,
    raw_trans_cdf,
    raw_art_cdf,
    raw_cust_cdf,
    base_cdf=None,
    y_cdf=None,
    target_customers=None,
    target_week="None",
)

drop_article_columns = list(raw_art_cdf.columns)
drop_article_columns.remove("article_id")
art_df_w_feat = art_df_w_feat.drop(drop_article_columns, axis=1)

art_cdf_w_feat = to_cdf(art_df_w_feat)
del art_df_w_feat

#%%


target_weeks = [104, 103, 102, 101, 100]  # test_yから何週間離れているか
# target_weeks = [104, 103]  # test_yから何週間離れているか

#%%
if DRY_RUN:
    tmp = raw_trans_cdf.query(f"week in {target_weeks}").to_pandas()
    sample_ids = (
        (tmp.groupby(["customer_id"])["week"].nunique() == 2)
        .to_frame()
        .reset_index()
        .query(f"week == True")["customer_id"]
        .values[:10000]
    )
    trans_cdf = raw_trans_cdf[raw_trans_cdf["customer_id"].isin(list(sample_ids))]
    train_df, valid_df, train_groups, valid_groups = make_train_valid_df(
        trans_cdf,
        art_cdf_w_feat,
        raw_cust_cdf,
        target_weeks,
        candidate_blocks,
        transaction_feature_blocks,
    )
else:
    train_df, valid_df, train_groups, valid_groups = make_train_valid_df(
        raw_trans_cdf,
        art_cdf_w_feat,
        raw_cust_cdf,
        target_weeks,
        candidate_blocks,
        transaction_feature_blocks,
    )
#%%


#%%
# train_df.to_csv(output_dir / "train_df3.csv", index=False)
# valid_df.to_csv(output_dir / "valid_df.csv", index=False)
# group_df = pd.DataFrame()

# group_df["group"] = train_groups
# group_df.to_csv(output_dir / "group_df2.csv", index=False)

drop_cols = ["customer_id", "article_id", "target_week", "y"]
train_X = train_df.drop(drop_cols, axis=1)
valid_X = valid_df.drop(drop_cols, axis=1)


#%%


RANK_PARAM = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [12],
    "verbosity": -1,
    "boosting": "gbdt",
    "is_unbalance": True,
    "seed": 42,
    "learning_rate": 0.05,
    "colsample_bytree": 0.5,
    "subsample_freq": 3,
    "subsample": 0.9,
    "n_estimators": 10000,
    "importance_type": "gain",
    "reg_lambda": 1.5,
    "reg_alpha": 0.1,
    "max_depth": 6,
    "num_leaves": 45,
}


#%%
def train_rank_lgb(
    train_X, train_y, valid_X, valid_y, param, logger, early_stop_round, log_period
):

    lgb.register_logger(logger)
    clf = lgb.LGBMRanker(**param)

    clf.fit(
        train_X,
        train_y,
        group=train_groups,
        eval_group=[valid_groups],
        eval_set=[(valid_X, valid_y)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stop_round),
            lgb.log_evaluation(period=log_period),
        ],
    )
    val_pred = clf.predict(valid_X)
    return clf, val_pred


clf, val_pred = train_rank_lgb(
    train_X,
    train_df["y"],
    valid_X,
    valid_df["y"],
    RANK_PARAM,
    logger,
    early_stop_round=100,
    log_period=10,
)


#%%
valid_df["prediction"] = val_pred
mapk_val, valid_true = calc_map12(valid_df, logger, input_dir / "valid_true_after0916.csv")


#%%
fig, ax = visualize_importance([clf], train_X)
fig.savefig(log_dir / "feature_importance.png")

lgbm_path = output_dir / "lgbm.txt"
clf.booster_.save_model(lgbm_path)

with open(lgbm_path, "wb") as f:
    pickle.dump(clf, f)

plt.show()


# %%
target_week = 105

if DRY_RUN:
    BATCH_USER_SIZE = 500_000
else:
    BATCH_USER_SIZE = 200_000

clipped_trans_cdf = raw_trans_cdf.query(f"week < @target_week")

#%%
submission_df = pd.read_csv(input_dir / "sample_submission.csv")
submission_df["customer_id"] = customer_hex_id_to_int(submission_df["customer_id"])
sub_customer_ids = submission_df["customer_id"].unique()


#%%

# ここに特徴生成 バッチ推論ではclipped_trans_cdfが同様のため先に特徴は作る
art_feat_df, cust_feat_df, _ = feature_generation(
    static_feature_test_blocks,
    raw_trans_cdf,
    art_cdf_w_feat,
    raw_cust_cdf,
    base_cdf=None,
    y_cdf=None,
    target_customers=None,
    target_week="None",
)

art_feat_df = reduce_mem_usage(art_feat_df)
cust_feat_df = reduce_mem_usage(cust_feat_df)


#%%


preds_list = []
preds_dic = {}
for bucket in tqdm(range(0, len(sub_customer_ids), BATCH_USER_SIZE)):
    batch_customer_ids = sub_customer_ids[bucket : bucket + BATCH_USER_SIZE]
    batch_trans_cdf = raw_trans_cdf[raw_trans_cdf["customer_id"].isin(batch_customer_ids)]
    # if len(batch_trans_cdf)>0:

    batch_base_df = candidate_generation(
        candidate_blocks_test,
        batch_trans_cdf,
        art_cdf_w_feat,
        raw_cust_cdf,
        y_cdf=None,
        target_customers=batch_customer_ids,
        target_week=105,
    )

    logger.info("fit feature")
    batch_base_cdf = to_cdf(batch_base_df)
    _, _, pair_feat_df = feature_generation(
        transaction_feature_test_blocks,
        raw_trans_cdf,
        art_cdf_w_feat,
        raw_cust_cdf,
        base_cdf=batch_base_cdf,
        y_cdf=None,
        target_customers=None,
        target_week=105,
    )

    logger.info("preprocess")
    batch_base_df = batch_base_df.merge(cust_feat_df, how="left", on="customer_id")
    batch_base_df = batch_base_df.merge(art_feat_df, how="left", on="article_id")

    batch_base_df = batch_base_df.merge(pair_feat_df, how="left", on=["article_id", "customer_id"])
    batch_base_df = batch_base_df.drop_duplicates(
        subset=["article_id", "customer_id", "candidate_block_name"]
    ).reset_index(drop=True)

    # batch_base_df = reduce_mem_usage(batch_base_df)
    # 不要?
    drop_cols = ["customer_id", "article_id"]
    batch_base_df.sort_values(["customer_id"], inplace=True)
    batch_test_X = batch_base_df.drop(drop_cols, axis=1)

    batch_test_X["candidate_block_name"] = batch_test_X["candidate_block_name"].astype("category")

    # 推論

    logger.info("predict")
    batch_test_pred = clf.predict(batch_test_X)
    batch_base_df["pred"] = batch_test_pred

    logger.info("postprocess")
    # subの作成
    batch_submission_df = batch_base_df[["customer_id", "article_id", "pred"]]
    batch_submission_df.sort_values(["customer_id", "pred"], ascending=False, inplace=True)

    batch_submission_df.drop_duplicates(
        subset=["customer_id", "article_id"], keep="first", inplace=True
    )

    batch_submission_df = batch_submission_df.groupby("customer_id").head(12)

    batch_submission_df = batch_submission_df.groupby("customer_id")[["article_id"]].aggregate(
        lambda x: x.tolist()
    )
    batch_submission_df["article_id"] = batch_submission_df["article_id"].apply(
        lambda x: " ".join(["0" + str(k) for k in x])
    )

    preds_list.append(batch_submission_df)
#%%
preds_df = pd.concat(preds_list).reset_index()
preds_df

#%%
# submission_df["customer_id"] = customer_hex_id_to_int(submission_df["customer_id"])
submission_df = submission_df.merge(preds_df, on="customer_id", how="left")

# customer_idをint→strに変換している
submission_df["customer_id"] = pd.read_csv(input_dir / "sample_submission.csv")["customer_id"]
if "prediction" in submission_df.columns:
    submission_df.drop(columns="prediction", inplace=True)
submission_df.rename(columns={"article_id": "prediction"}, inplace=True)

prefix = datetime.now().strftime(f"%m%d_%H%M")
submission_df.to_csv(output_dir / f"submission_df_{prefix}.csv", index=False)
