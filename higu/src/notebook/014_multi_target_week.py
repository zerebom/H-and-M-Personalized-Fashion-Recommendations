#%%
# %load_ext autoreload
# %autoreload 2

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
    from candidacies_dfs import (
        LastNWeekArticles,
    )
    from eda_tools import visualize_importance
    from features import EmbBlock, ModeCategoryBlock, TargetEncodingBlock
    from metrics import mapk
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
exp_name = Path(os.path.basename(__file__)).stem
output_dir = root_dir / "output" / exp_name
log_dir = output_dir / "log" / exp_name

output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"{date.today()}.log"
logger = setup_logger(log_file)
DRY_RUN = False


ArtId = int
CustId = int
ArtIds = List[ArtId]
CustIds = List[CustId]

to_cdf = cudf.DataFrame.from_pandas

# %%
raw_trans_cdf, raw_cust_cdf, raw_art_cdf = read_cdf(input_dir, DRY_RUN)

#%%


agg_list = ["mean", "max", "min", "std", "median"]
feature_blocks = [
    *[
        TargetEncodingBlock(
            "article_id",
            "age",
            agg_list + ["count"],
        )
    ],
    *[
        TargetEncodingBlock("article_id", col, ["mean"])
        for col in ["FN", "Active", "club_member_status", "fashion_news_frequency"]
    ],
    *[TargetEncodingBlock("customer_id", col, agg_list) for col in ["price", "sales_channel_id"]],
]

candidate_blocks = [
    # *[PopularItemsoftheLastWeeks(customer_ids)],
    *[LastNWeekArticles(n_weeks=2)]
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
    blocks, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers=None
) -> pd.DataFrame:

    """
    candidate_generation blocksを使って、推論候補対象を作成する。
    (art_id, cust_id)をkeyに持つdataframeを返す。
    """

    for i, block in enumerate(blocks):
        with timer(logger=logger, prefix="↑fitted {} ".format(block.__class__.__name__)):
            if i == 0:
                candidates_df = block.fit(
                    trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers
                )
            else:
                new_df = block.fit(trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers)
                candidates_df = pd.concat([new_df, candidates_df])

    candidates_df.reset_index(drop=True, inplace=True)
    y_df = y_cdf.to_pandas()
    candidates_df["y"] = (
        candidates_df.merge(y_df, how="left", on=["customer_id", "article_id"])["y"]
        .fillna(0)
        .astype(int)
    )
    return candidates_df


def feature_generation(blocks, trans_cdf, art_cdf, cust_cdf):
    """
    feature_generation blocksを使って、特徴量を作成する。
    art_id, cust_idをkeyに持つdataframeを返す。
    """

    # CHECK: 割とmem消費の激しい処理。メモリエラーにならないか注視する
    # TODO: このままの実装だと、transactionの存在しない、art,custの特徴量を取得できない
    # ex):類似アイテムで取得してきたcandidateに対して、transactionがない場合
    trans_w_art_cust_info_cdf = trans_cdf.merge(art_cdf, on="article_id", how="left").merge(
        cust_cdf, on="customer_id", how="left"
    )

    drop_cols = trans_w_art_cust_info_cdf.to_pandas().filter(regex=".*_name").columns
    trans_w_art_cust_info_cdf.drop(columns=drop_cols, inplace=True)
    art_feat_cdf = art_cdf[["article_id"]]  # base
    cust_feat_cdf = cust_cdf[["customer_id"]]

    for block in blocks:
        with timer(logger=logger, prefix="fit {} ".format(block.__class__.__name__)):

            # TODO: EmbBlockなどはtrans_w_art_cust_info_cdfを使いたくない
            feature_cdf = block.fit(trans_w_art_cust_info_cdf)

            if block.key_col == "article_id":
                art_feat_cdf = art_feat_cdf.merge(feature_cdf, how="left", on=block.key_col)
            elif block.key_col == "customer_id":
                cust_feat_cdf = cust_feat_cdf.merge(feature_cdf, how="left", on=block.key_col)

    cust_df = cust_cdf.merge(cust_feat_cdf, on="customer_id", how="left").to_pandas()
    art_df = art_cdf.merge(art_feat_cdf, on="article_id", how="left").to_pandas()

    return art_df, cust_df, None


#%%


def negative_sampling(base_df):
    positive_df = base_df.query("y==1")
    positive_customer_ids = positive_df["customer_id"].unique()

    # postiveが存在するcustomer_idのみに絞る。不例の数は全ユーザ２０件
    negative_df = base_df.query("y==0")
    negative_df = negative_df[negative_df["customer_id"].isin(positive_customer_ids)]
    negative_df = negative_df.groupby("customer_id").apply(
        lambda x: x.sample(n=min(20, len(x)), random_state=42)
    )
    print(positive_df.shape, negative_df.shape)

    sampled_df = pd.concat([positive_df, negative_df])
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled_df


def make_train_valid_df(trans_cdf, art_cdf, cust_cdf, target_weeks, train_duration_weeks=8):
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for target_week in tqdm(target_weeks):
        phase = "validate" if target_week == 104 else "train"
        clipped_trans_cdf = trans_cdf.query(f"week < @target_week")  # 予測日の前週までのトランザクションが使える
        y_cdf, y_df = make_y_data(trans_cdf, target_week)

        # いろんなメソッドを使って候補生成
        base_df = candidate_generation(
            candidate_blocks,
            clipped_trans_cdf,
            art_cdf,
            cust_cdf,
            y_cdf,
            target_customers=None,
        )

        base_df = negative_sampling(base_df)  # 適当に間引くはず

        art_feat_df, cust_feat_df, pair_feat_df = feature_generation(
            feature_blocks,
            clipped_trans_cdf,
            art_cdf,
            cust_cdf,  # base_dfに入っているuser_id,article_idが必要
        )

        # 特徴量作成
        base_df = base_df.merge(cust_feat_df, how="left", on="customer_id")
        base_df = base_df.merge(art_feat_df, how="left", on="article_id")
        # base_df = base_df.merge(pair_feat_df, how="left", on=["article_id", "customer_id"])
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
target_weeks = [104, 103, 102, 101, 100]  # test_yから何週間離れているか
DRY_RUN = False

if DRY_RUN:
    tmp = raw_trans_cdf.query(f"week in {target_weeks}").to_pandas()
    sample_ids = (
        (tmp.groupby(["customer_id"])["week"].nunique() == 3)
        .to_frame()
        .reset_index()
        .query(f"week == True")["customer_id"]
        .values[:10]
    )

    trans_cdf = raw_trans_cdf.query(f"customer_id in {list(sample_ids)}")
    train_df, valid_df, train_groups, valid_groups = make_train_valid_df(
        trans_cdf, raw_art_cdf, raw_cust_cdf, target_weeks, train_duration_weeks=8
    )
else:
    train_df, valid_df, train_groups, valid_groups = make_train_valid_df(
        raw_trans_cdf, raw_art_cdf, raw_cust_cdf, target_weeks, train_duration_weeks=8
    )


#%%
# train_df.to_csv(output_dir / "train_df.csv", index=False)
# valid_df.to_csv(output_dir / "valid_df.csv", index=False)
# group_df = pd.DataFrame()

# group_df["group"] = train_groups
# group_df.to_csv(output_dir / "group_df.csv", index=False)

drop_cols = ["customer_id", "article_id", "target_week", "y"]
train_X = train_df.drop(drop_cols, axis=1)
valid_X = valid_df.drop(drop_cols, axis=1)

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
    config.RANK_PARAM,
    logger,
    early_stop_round=100,
    log_period=10,
)


#%%
valid_df["prediction"] = val_pred
mapk_val, valid_true = calc_map12(valid_df, logger, input_dir / "valid_true_after0916.csv")

# %%


fig, ax = visualize_importance([clf], train_X)
fig.savefig(log_dir / "feature_importance.png")

lgbm_path = output_dir / "lgbm.txt"
clf.booster_.save_model(lgbm_path)

with open(lgbm_path, "wb") as f:
    pickle.dump(clf, f)

plt.show()


# %%
DRY_RUN = False
target_week = 105

if DRY_RUN:
    BATCH_SIZE = 100_000
else:
    BATCH_SIZE = 10_000

clipped_trans_cdf = raw_trans_cdf.query(f"week < @target_week")

#%%
submission_df = pd.read_csv(input_dir / "sample_submission.csv")
submission_df["customer_id"] = customer_hex_id_to_int(submission_df["customer_id"])
sub_customer_ids = submission_df["customer_id"].unique()

drop_cols = ["customer_id", "article_id"]

preds_list = []
preds_dic = {}

#%%
# ここに特徴生成 バッチ推論ではclipped_trans_cdfが同様のため先に特徴は作る
art_feat_df, cust_feat_df, pair_feat_df = feature_generation(
    feature_blocks,
    clipped_trans_cdf,  # 特徴生成なので全ユーザーのtransactionが欲しい。けどメモリのために期間は絞る
    raw_art_cdf,
    raw_cust_cdf,  # base_dfに入っているuser_id,article_idが必要
)
art_feat_df = reduce_mem_usage(art_feat_df)
cust_feat_df = reduce_mem_usage(cust_feat_df)
candidate_blocks_test = [
    # *[PopularItemsoftheLastWeeks(customer_ids)],
    *[LastNWeekArticles(n_weeks=2)]
]

#%%
for bucket in tqdm(range(0, len(sub_customer_ids), BATCH_SIZE)):
    batch_customer_ids = sub_customer_ids[bucket : bucket + BATCH_SIZE]
    batch_trans_cdf = raw_trans_cdf[raw_trans_cdf["customer_id"].isin(batch_customer_ids)]
    # if len(batch_trans_cdf)>0:

    # (要議論) speed upのためにclipped_trans_cdfに存在するart, custだけの情報を使う
    buyable_art_ids = batch_trans_cdf["article_id"].unique()
    buyable_art_cdf = clipped_trans_cdf[clipped_trans_cdf["article_id"].isin(buyable_art_ids)]
    bought_cust_ids = batch_trans_cdf["customer_id"].to_pandas().unique()
    bought_cust_cdf = clipped_trans_cdf[clipped_trans_cdf["customer_id"].isin(bought_cust_ids)]

    # ここに候補生成
    batch_base_df = candidate_generation(
        candidate_blocks_test,
        batch_trans_cdf,  # 該当ユーザーのtransactionが欲しいのでこれでOK
        buyable_art_cdf,
        bought_cust_cdf,
        y_cdf=None,
        target_customers=batch_customer_ids,
    )
    batch_base_df = batch_base_df.drop_duplicates(
        subset=["customer_id", "article_id"], keep="last"
    )
    # batch_base_df = reduce_mem_usage(batch_base_df)

    batch_base_df = batch_base_df.merge(cust_feat_df, how="left", on="customer_id")
    batch_base_df = batch_base_df.merge(art_feat_df, how="left", on="article_id")

    # batch_base_df = reduce_mem_usage(batch_base_df)
    # 不要?
    batch_base_df.sort_values(["customer_id"], inplace=True)
    batch_test_X = batch_base_df.drop(drop_cols, axis=1)

    # 推論
    batch_test_pred = clf.predict(batch_test_X)
    batch_base_df["pred"] = batch_test_pred

    # subの作成
    batch_submission_df = batch_base_df[["customer_id", "article_id", "pred"]]
    batch_submission_df.sort_values(["customer_id", "pred"], ascending=False, inplace=True)

    batch_submission_df = batch_submission_df.groupby("customer_id").head(12)

    batch_submission_df = batch_submission_df.groupby("customer_id")[["article_id"]].aggregate(
        lambda x: x.tolist()
    )
    batch_submission_df["article_id"] = batch_submission_df["article_id"].apply(
        lambda x: " ".join(["0" + str(k) for k in x])
    )

    preds_list.append(batch_submission_df)

preds_df = pd.concat(preds_list).reset_index()

#%%
submission_df["customer_id"] = customer_hex_id_to_int(submission_df["customer_id"])
submission_df = submission_df.merge(preds_df, on="customer_id", how="left")

# customer_idをint→strに変換している
submission_df["customer_id"] = pd.read_csv(input_dir / "sample_submission.csv")["customer_id"]
# submission_df.drop(columns="prediction", inplace=True)
#%%
submission_df.rename(columns={"article_id": "prediction"}, inplace=True)
display(submission_df.head())

submission_df.to_csv(log_dir / "submission_df.csv", index=False)

# %%
submission_df.head()


# %%
