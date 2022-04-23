#%%

%load_ext autoreload
%autoreload 2

from asyncio import SubprocessTransport
import json
import os
import pickle
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
    from candidacies_dfs import (
        LastNWeekArticles,
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

TARGET_ARTICLE_WEEK = 103
PERCHASE_USER_THRESHOLD = 0.4
to_cdf = cudf.DataFrame.from_pandas

# %%

raw_trans_cdf, raw_cust_cdf, raw_art_cdf = read_cdf(input_dir, DRY_RUN)
week = 7

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

candidate_blocks = [
    # *[PopularItemsoftheLastWeeks(customer_ids)],
    *[LastNWeekArticles(n_weeks=2)]
    ]

rank_param = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": list(range(1, 13)),
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
#%%

def make_y_cdf(
    transactions: cudf.DataFrame, target_week
) -> cudf.DataFrame:
    y_cdf = transactions.query(f"week == {target_week}")[
        ["customer_id", "article_id"]
    ].drop_duplicates()
    y_cdf["y"] = 1
    return y_cdf[["customer_id", "article_id","y"]]

def candidate_generation(blocks, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers = None) -> pd.DataFrame:

    """
    candidate_generation blocksを使って、推論候補対象を作成する。
    (art_id, cust_id)をkeyに持つdataframeを返す。
    """

    candidates_df = None

    for i, block in enumerate(blocks):
        with timer(logger=logger, prefix="↑fitted {} ".format(block.__class__.__name__)):
            if i == 0:
                candidates_df = block.fit(trans_cdf, art_cdf, cust_cdf, logger, y_cdf,target_customers)
            else:
                new_df = block.fit(trans_cdf, art_cdf, cust_cdf, logger, y_cdf,target_customers)
                candidates_df = pd.concat([new_df, candidates_df])
    return candidates_df


def feature_generation_with_transaction(blocks, trans_cdf, art_cdf, cust_cdf):
    """
    feature_generation blocksを使って、特徴量を作成する。
    art_id, cust_idをkeyに持つdataframeを返す。
    """

    # CHECK: 割とmem消費の激しい処理。メモリエラーにならないか注視する
    # TODO: このままの実装だと、transactionの存在しない、art,custの特徴量を取得できない
    # ex):類似アイテムで取得してきたcandidateに対して、transactionがない場合
    trans_w_art_cust_info_cdf = (trans_cdf.
                merge(art_cdf, on="article_id", how="left").
                merge(cust_cdf, on="customer_id", how="left")
    )

    drop_cols = trans_w_art_cust_info_cdf.to_pandas().filter(regex=".*_name").columns
    trans_w_art_cust_info_cdf.drop(columns=drop_cols, inplace=True)
    art_feat_cdf = art_cdf[["article_id"]] # base
    cust_feat_cdf = cust_cdf[["customer_id"]]

    for block in blocks:
        with timer(logger=logger, prefix="fit {} ".format(block.__class__.__name__)):

            # TODO: EmbBlockなどはtrans_w_art_cust_info_cdfを使いたくない
            feature_cdf = block.fit(trans_w_art_cust_info_cdf)

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

    return art_df, cust_df, None

#%%

def negative_sampling(base_df):
    positive_df = base_df.query('y==1')
    positive_customer_ids = positive_df['customer_id'].unique()

    #postiveに存在するcustomer_idのみに絞る。不例の数は全ユーザ２０件
    # negative_df = (base_df.
    #                 query('y==0').
    #                 # query(f'customer_id in {positive_customer_ids}').
    #                 groupby('customer_id').
    #                 apply(lambda x: x.sample(n=min(20, len(x)), random_state=42))
    #                 )

    negative_df = base_df.query('y==0')
    negative_df = negative_df[negative_df['customer_id'].isin(positive_customer_ids)]
    negative_df = negative_df.groupby('customer_id').apply(lambda x: x.sample(n=min(20, len(x)), random_state=42))
    print(positive_df.shape, negative_df.shape)

    sampled_df = pd.concat([positive_df, negative_df])
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled_df

def make_train_valid_df(trans_cdf, art_cdf, cust_cdf, target_weeks, train_duration_weeks=8):
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for target_week in tqdm(target_weeks):
        phase = "validate" if target_week == 104 else "train"
        # import pdb;pdb.set_trace()

        clipped_trans_cdf = trans_cdf.query(f"@target_week - {train_duration_weeks} <= week < @target_week") #予測日の前週までのトランザクションが使える

        #(要議論) speed upのためにclipped_trans_cdfに存在するart, custだけの情報を使う
        buyable_art_ids = clipped_trans_cdf['article_id'].unique()
        buyable_art_cdf = art_cdf[art_cdf["article_id"].isin(buyable_art_ids)]
        # buyable_art_cdf = art_cdf.query(f"article_id in {list(buyable_art_ids)}")

        bought_cust_ids = clipped_trans_cdf['customer_id'].to_pandas().unique()
        bought_cust_cdf = cust_cdf[cust_cdf["customer_id"].isin(bought_cust_ids)]
        # bought_cust_cdf = cust_cdf.query(f"customer_id in {list(bought_cust_ids)}")


        y_cdf = make_y_cdf(trans_cdf, target_week)
        y_df = y_cdf.to_pandas()

        #いろんなメソッドを使って候補生成
        base_df = candidate_generation(
            candidate_blocks, clipped_trans_cdf, buyable_art_cdf, bought_cust_cdf, y_cdf, target_customers = None
        )

        base_df['y'] = base_df.merge(y_df, how='left',on=['customer_id', 'article_id'])["y"].fillna(0).astype(int)
        base_df = negative_sampling(base_df) # 適当に間引くはず

        art_feat_df, cust_feat_df, pair_feat_df = feature_generation_with_transaction(
        feature_blocks, clipped_trans_cdf, buyable_art_cdf, bought_cust_cdf #base_dfに入っているuser_id,article_idが必要
    )

        # 特徴量作成
        base_df = base_df.merge(cust_feat_df, how="left", on="customer_id")
        base_df = base_df.merge(art_feat_df, how="left", on="article_id")
        # base_df = base_df.merge(pair_feat_df, how="left", on=["article_id", "customer_id"])
        base_df["target_week"] = target_week

        # base_df.columns ==  [
        #     "customer_id",
        #     "article_id",
        #     "candidate_block_name",
        #     "target_week",
        #     "target",
        # ] + [feature_columns]

        base_df = reduce_mem_usage(base_df)
        if phase == "train":
            # 学習データは複数予測週があるのでconcatする
            train_df = pd.concat([train_df, base_df])
        elif phase == "validate":
            valid_df = base_df.copy()

    # transactionによらない特徴量はゆくゆくはここで実装したい (ex. embbeding特徴量)
    # train_df = feature_generation_with_article(train_df, article_cdf)
    # valid_df = feature_generation_with_article(valid_df, article_cdf)
    # train_df = feature_generation_with_customer(train_df, customer_cdf)
    # valid_df = feature_generation_with_customer(valid_df, customer_cdf)

    train_df.sort_values(['customer_id', 'target_week'], inplace=True)
    valid_df.sort_values(['customer_id', 'target_week'], inplace=True)

    train_groups = train_df.groupby(['customer_id','target_week']).size().values
    valid_groups = valid_df.groupby(['customer_id','target_week']).size().values

    return train_df, valid_df, train_groups, valid_groups


#%%
target_weeks = [104,103,102,101,100]  # test_yから何週間離れているか
tmp = raw_trans_cdf.query(f"week in {target_weeks}").to_pandas()
sample_ids = (tmp.groupby(['customer_id'])['week'].nunique() == 3).to_frame().reset_index().query(f'week == True')['customer_id'].values[:10]

trans_cdf = raw_trans_cdf.query(f"customer_id in {list(sample_ids)}")

# train_df, valid_df, train_groups, valid_groups = make_train_valid_df(trans_cdf, raw_art_cdf, raw_cust_cdf, target_weeks, train_duration_weeks=8)
train_df, valid_df, train_groups, valid_groups = make_train_valid_df(raw_trans_cdf, raw_art_cdf, raw_cust_cdf, target_weeks, train_duration_weeks=8)


#%%



#%%
train_df.to_csv(output_dir/'train_df.csv',index=False)
valid_df.to_csv(output_dir/'valid_df.csv',index=False)
group_df = pd.DataFrame()

group_df['group'] = train_groups
group_df.to_csv(output_dir/'group_df.csv',index=False)

#%%
train_df.isnull().sum().values
#%%

train_df.select_dtypes('object')
#%%
train_df.loc[]

#%%
# raw_art_cdf,
# raw_cust_cdf

#%%

# %%
from lightgbm.sklearn import LGBMRanker

ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    importance_type='gain',
    verbose=-1
)


drop_cols = [
    "customer_id",
    "article_id",
    "target_week",
    "y"]


train_X = train_df.drop(drop_cols, axis=1)
valid_X = valid_df.drop(drop_cols, axis=1)


#%%

ranker = ranker.fit(
    train_X,
    train_df["y"],
    group=train_groups,
    eval_set=[(valid_X, valid_df["y"])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    eval_group=[list(valid_groups)]
)

# %%

train_df


#%%
def shuffle_by_query(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    # df = df.sort_values(by=["keyword", "user_id"], ascending=True)
    df = df.groupby(["customer_id","target_week"]).apply(
        lambda x: x.sample(frac=1, random_state=42)
    )
    df.reset_index(drop=True, inplace=True)
    return df
#%%
train_df.head(50)

#%%
# shuffle_by_query(train_df).head(50)

train_df.groupby(["customer_id","target_week"]).apply(
        lambda x: x.sample(frac=1, random_state=42)
    )
#%%
train_df

#%%
train_df.groupby(["customer_id","target_week"]).sample(frac=1)
#%%
train_df['query_id'] = train_df["customer_id"].astype(str) + "_" + train_df["target_week"].astype(str)
#%%
train_df.groupby(['query_id']).(
        lambda x: x.sample(frac=1, random_state=42)
)



#%%
#%%

rank_param = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [12],
    "verbosity": -1,
    "boosting": "gbdt",
    # "is_unbalance": True,
    "seed": 42,
    "learning_rate": 0.01,
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
    train_X, train_df['y'], valid_X, valid_df['y'], rank_param, logger, early_stop_round=100, log_period=10
)

# %%
