#%%
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import cudf
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

if True:
    sys.path.append("../")
    from candidacies import (
        ArticlesSimilartoThoseUsersHavePurchased,
        BoughtItemsAtInferencePhase,
        LastNWeekArticles,
        PopularItemsoftheLastWeeks,
        candidates_dict2df,
    )
    from eda_tools import visualize_importance
    from features import EmbBlock, ModeCategoryBlock, TargetEncodingBlock
    from metrics import mapk
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
exp_name = "009_lgbm"
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
# %%


# ==================================================================================
# ================================= データのロード =================================
# ==================================================================================
trans_cdf, cust_cdf, art_cdf = read_cdf(input_dir, DRY_RUN)

with open(str(input_dir / "emb/article_emb.json")) as f:
    article_emb_dic = json.load(f, object_hook=jsonKeys2int)

target_articles = trans_cdf.query("week>=95")["article_id"].unique().to_pandas().values
preds_of_not_purchase_user_cdf = cudf.read_parquet(
    input_dir / "005_preds_of_not_purchase_user.parquet"
)
purchase_user_ids = (
    preds_of_not_purchase_user_cdf.query("pred>=0.4")["customer_id"]
    .unique()
    .to_pandas()
    .values
)

# %%
#
# ==============================================================================
# ================================= 関数・変数の定義 ===========================
# ==============================================================================


# ================================= time =================================

datetime_dic = {
    "X": {
        "train": {"start_date": phase_date(35), "end_date": phase_date(14)},
        "valid": {"start_date": phase_date(28), "end_date": phase_date(7)},
        "test": {"start_date": phase_date(21), "end_date": phase_date(0)},
    },
    "y": {
        "train": {"start_date": phase_date(14), "end_date": phase_date(7)},
        "valid": {"start_date": phase_date(7), "end_date": phase_date(0)},
    },
}

# ================================= preprocessor =========================


def make_y_cdf(transactions: cudf.DataFrame, start_date: datetime, end_date: datetime):
    y_cdf = transactions.query(f"@start_date<t_dat<@end_date")[
        ["customer_id", "article_id"]
    ].drop_duplicates()
    y_cdf["purchased"] = 1
    return y_cdf


def clip_transactions(transactions: cudf.DataFrame, end_date: datetime):
    return transactions.query(f"t_dat<@end_date")


# ------------------------------blocks&params----------------------------------

candidate_blocks = [
    # *[PopularItemsoftheLastWeeks(customer_ids)],
    *[LastNWeekArticles(n_weeks=2)],
    *[
        ArticlesSimilartoThoseUsersHavePurchased(
            article_emb_dic,
            50,
            10,
            target_articles,
            purchase_user_ids,
        )
    ],
]

agg_list = ["mean", "max", "min", "std", "median"]
feature_blocks = [
    EmbBlock("article_id", article_emb_dic),
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
    *[
        TargetEncodingBlock("customer_id", col, agg_list)
        for col in ["price", "sales_channel_id"]
    ],
    *[
        ModeCategoryBlock("customer_id", col)
        for col in [
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
    ],
]


param = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "boosting": "gbdt",
    "is_unbalance": True,
    "seed": 42,
    "learning_rate": 0.1,
    "colsample_bytree": 0.5,
    "subsample_freq": 3,
    "subsample": 0.9,
    "n_estimators": 1000,
    "importance_type": "gain",
    "reg_lambda": 1.5,
    "reg_alpha": 0.1,
    "max_depth": 6,
    "num_leaves": 45,
}


# ================================= データ処理パイプライン =============================


def candidate_generation(blocks, trans_cdf, art_cdf, cust_cdf, y_cdf=None):

    """
    candidate_generation blocksを使って、推論候補対象を作成する。
    (art_id, cust_id)をkeyに持つdataframeを返す。
    """

    customer_ids = list(cust_cdf["customer_id"].to_pandas().unique())

    if y_cdf is not None:
        blocks.append(BoughtItemsAtInferencePhase(y_cdf))

    candidates_dict = {}
    candidates_df = None

    for i, block in enumerate(blocks):
        with timer(logger=logger, prefix="fit {} ".format(block)):
            if i == 0:
                candidates_dict = block.fit(trans_cdf)
            else:
                new_dic = block.fit(trans_cdf)
                for key, value in candidates_dict.items():
                    if key in new_dic:
                        value.extend(new_dic[key])
        candidates_df = candidates_dict2df(candidates_dict).to_pandas()
    return candidates_df


def feature_generation(blocks, trans_cdf, art_cdf, cust_cdf):
    """
    feature_generation blocksを使って、特徴量を作成する。
    art_id, cust_idをkeyに持つdataframeを返す。
    """
    recent_bought_cdf = cudf.DataFrame.from_pandas(
        trans_cdf.sort_values(["customer_id", "t_dat"], ascending=False)
        .to_pandas()
        .groupby("customer_id")
        .head(30)
    )

    feature_df = recent_bought_cdf.merge(art_cdf, on="article_id", how="left").merge(
        cust_cdf, on="customer_id", how="left"
    )

    drop_cols = feature_df.to_pandas().filter(regex=".*_name").columns

    feature_df.drop(columns=drop_cols, inplace=True)
    art_feat_cdf = art_cdf[["article_id"]]
    cust_feat_cdf = cust_cdf[["customer_id"]]

    for block in tqdm(blocks):
        with timer(logger=logger, prefix="fit {} ".format(block)):

            feature_cdf = block.fit(feature_df)

            if block.key_col == "article_id":
                art_feat_cdf = art_feat_cdf.merge(
                    feature_cdf, how="left", on=block.key_col
                )
            elif block.key_col == "customer_id":
                cust_feat_cdf = cust_feat_cdf.merge(
                    feature_cdf, how="left", on=block.key_col
                )

    cust_df = cust_cdf.merge(cust_feat_cdf, on="customer_id", how="left").to_pandas()
    art_df = art_cdf.merge(art_feat_cdf, on="article_id", how="left").to_pandas()

    return art_df, cust_df


def make_trainable_data(
    candidate_blocks, feature_blocks, raw_trans_cdf, art_cdf, cust_cdf, phase
):

    """
    CG→FEを経て学習可能なX ,y, keyを作成する
    """

    X_end_date = datetime_dic["X"][phase]["end_date"]
    logger.info(f"make {phase} start.")

    key_cols = ["customer_id", "article_id"]
    y_cdf, y_df, y = None, None, None

    # 学習するtransaction期間を絞る
    trans_df = clip_transactions(raw_trans_cdf, X_end_date).to_pandas()

    # Xの作成
    print("start candidate generation")  # CG
    candidates_df = candidate_generation(
        candidate_blocks, trans_cdf, art_cdf, cust_cdf, y_cdf
    )

    print("start feature generation")  # FE
    art_feat_df, cust_feat_df = feature_generation(
        feature_blocks, trans_cdf, art_cdf, cust_cdf
    )

    # 予測対象ペアに特徴量をmergeする
    X = candidates_df.merge(cust_feat_df, how="left", on="customer_id").merge(
        art_feat_df, how="left", on="article_id"
    )

    # 最近購入されてないアイテムを取り除く
    recent_items = (
        trans_df.groupby("article_id")["week"]
        .max()
        .reset_index()
        .query("week>96")["article_id"]
        .values
    )

    X = X[X["article_id"].isin(recent_items)]

    # yの作成
    if phase is not "test":
        y_start_date = datetime_dic["y"][phase]["start_date"]
        y_end_date = datetime_dic["y"][phase]["end_date"]
        y_df = make_y_cdf(raw_trans_cdf, y_start_date, y_end_date).to_pandas()
        y = X.merge(y_df, how="left", on=key_cols)["purchased"].fillna(0).astype(int)
        logger.info(f"X_shape:, {X.shape}, y_mean: {y.mean()}")

    # keyの作成
    key_df = X[key_cols]
    X = X.drop(columns=key_cols)

    return reduce_mem_usage(key_df), reduce_mem_usage(X), y


def train_lgb(
    train_X, train_y, valid_X, valid_y, param, logger, early_stop_round, log_period
):

    lgb.register_logger(logger)
    clf = lgb.LGBMClassifier(**param)
    clf.fit(
        train_X,
        train_y,
        eval_set=[(valid_X, valid_y)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stop_round),
            lgb.log_evaluation(period=log_period),
        ],
    )
    val_pred = clf.predict_proba(valid_X)[:, 1]
    return clf, val_pred


# %%

# ==============================================================================
# ================================= データ処理開始 =============================
# ==============================================================================

# データの作成
phases = ["train", "valid", "test"]
data_dic = {}

for phase in phases:
    data_dic[phase] = {}

    key, X, y = make_trainable_data(
        candidate_blocks, feature_blocks, trans_cdf, art_cdf, cust_cdf, phase=phase
    )

    data_dic[phase]["key"] = key
    data_dic[phase]["X"] = X
    data_dic[phase]["y"] = y


#%%
# 中間生成物の保存
if not DRY_RUN:
    for phase, phase_dic in data_dic.items():
        for df_name, df in phase_dic.items():
            df.to_pickle(output_dir / f"{phase}_{df_name}.pkl")

#%%

# 学習
clf, val_pred = train_lgb(
    data_dic["train"]["X"],
    data_dic["train"]["y"],
    data_dic["valid"]["X"],
    data_dic["valid"]["y"],
    param,
    logger,
    early_stop_round=100,
    log_period=100,
)

# %%

# 推論の作成
valid_key = data_dic["valid"]["key"]
test_key = data_dic["test"]["key"]
valid_y = data_dic["valid"]["y"]
test_X = data_dic["test"]["X"]

valid_key["pred"] = val_pred
valid_key["target"] = valid_y
valid_key["pred"].plot.hist()

test_y = clf.predict_proba(test_X)[:, 1]
test_key["pred"] = test_y


#%%


def get_pop_items(trans_cdf):
    pop = PopularItemsoftheLastWeeks([1])
    pop.fit(trans_cdf)
    pop_items = list(pop.popular_items)
    return pop_items


def fill_pop_items(pred_ids: ArtIds, pop_items: ArtIds) -> ArtIds:
    pred_ids.extend(pop_items)
    return pred_ids[:12]


# 推論を提出形式に変換
val_sub_fmt_df = squeeze_pred_df_to_submit_format(valid_key)

# 人気アイテムで埋める
pop_items = get_pop_items(trans_cdf)
test_sub_fmt_df = squeeze_pred_df_to_submit_format(
    test_key, fill_logic=fill_pop_items, args=(pop_items,)
)

# 12桁idを文字列idに変換
converted_sub_fmt_df = convert_sub_df_customer_id_to_str(input_dir, test_sub_fmt_df)


# 保存
prefix = datetime.now().strftime("%m_%d_%H_%M")
converted_sub_fmt_df.to_csv(output_dir / f"submission_{prefix}.csv", index=False)


#%%
# 精度の確認
mapk_val = mapk(
    val_sub_fmt_df["prediction"].map(lambda x: x.split()),
    val_sub_fmt_df["target"].map(lambda x: x.split()),
    k=12,
)

logger.info(f"mapk:{mapk_val}")

# %%
fig, ax = visualize_importance([clf], data_dic["train"]["X"])
fig.savefig(log_dir / "feature_importance.png")

# %%
