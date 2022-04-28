#%load_ext autoreload
#%autoreload 2

from asyncio import SubprocessTransport
import json
import os,shutil
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
    sys.path.append("/home/ec2-user/h_and_m/src/")
    from candidacies_dfs import (
        LastNWeekArticles,RuleBased,PopularItemsoftheLastWeeks,BoughtItemsAtInferencePhase
    )
    from eda_tools import visualize_importance
    from features import EmbBlock, ModeCategoryBlock, TargetEncodingBlock, LifetimesBlock, CustItemEmbBlock, ItemBiasIndicatorBlock
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
        pre_process_article_cdf,
    )


root_dir = Path("/home/ec2-user/h_and_m/")
input_dir = Path("/home/ec2-user/h_and_m/data/inputs")
exp_name = "exp11"
exp_disc = "購買履歴を5ヶ月使用"

output_dir = root_dir / "output" 
log_dir = output_dir / "log" / exp_name

output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{date.today()}.log"
logger = setup_logger(log_file)
logger.info(f"{exp_disc}")
shutil.copy('/home/ec2-user/h_and_m/src/notebook/014_multi_target_week.py',log_dir)

DRY_RUN = False
TRAIN_DURATION_WEEKS = 4 * 5


ArtId = int
CustId = int
ArtIds = List[ArtId]
CustIds = List[CustId]

to_cdf = cudf.DataFrame.from_pandas


raw_trans_cdf, raw_cust_cdf, raw_art_cdf = read_cdf(input_dir, DRY_RUN)

# parquetはtxtが空なのでローデータのcsvを使用
articles_cdf_text = cudf.read_csv('/home/ec2-user/h_and_m/data/articles.csv')
articles_cdf_text = pre_process_article_cdf(articles_cdf_text)

with open("/home/ec2-user/h_and_m/data/inputs/image_embedding/resnet-18_umap_10.json") as f:
    article_emb_resnet18_umap10_dic = json.load(f, object_hook=jsonKeys2int)

valid_true = pd.read_csv('/home/ec2-user/h_and_m/data/inputs/valid_true_after0916.csv')[["customer_id", "valid_true"]]



agg_list = ["mean", "max", "min", "std", "median"]
feature_blocks = [
    EmbBlock("article_id", article_emb_resnet18_umap10_dic),
    LifetimesBlock("customer_id", "price"),
    #CustItemEmbBlock("customer_id", 20, articles_cdf_text),
    ItemBiasIndicatorBlock("article_id", "customer_id"),
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
            #"product_type_no",
            "graphical_appearance_no",
            #"colour_group_code",
            #"perceived_colour_value_id",
            "perceived_colour_master_id",
            #"department_no",
            "index_code",
            "index_group_no",
            #"section_no",
            "garment_group_no",
        ]
    ]
]

rule_based_cdf = cudf.read_csv("/home/ec2-user/h_and_m/data/inputs/rulebased_cg_ensemble_int-cust-0p0236_N30_test.csv")
candidate_blocks = [
    *[PopularItemsoftheLastWeeks(n=30)],
    *[LastNWeekArticles(n_weeks=4)],
    #*[RuleBased(rule_based_cdf=rule_based_cdf)],
    # *[BoughtItemsAtInferencePhase()]
]


candidate_blocks_valid = [
    *[PopularItemsoftheLastWeeks(n=30)],
    *[LastNWeekArticles(n_weeks=4)],
    #*[RuleBased(rule_based_cdf=rule_based_cdf)],
    *[BoughtItemsAtInferencePhase()]
]

candidate_blocks_test = [
    *[PopularItemsoftheLastWeeks(n=300)],
    *[LastNWeekArticles(n_weeks=8)],
    *[RuleBased(rule_based_cdf=rule_based_cdf)],
]


def make_y_cdf(transactions: cudf.DataFrame, target_week) -> cudf.DataFrame:
    y_cdf = transactions.query(f"week == {target_week}")[
        ["customer_id", "article_id"]
    ].drop_duplicates()
    y_cdf["y"] = 1
    return y_cdf[["customer_id", "article_id", "y"]]

def candidate_generation(
    blocks, trans_cdf, art_cdf, cust_cdf, y_cdf, target_customers=None, 
) -> pd.DataFrame:

    """
    candidate_generation blocksを使って、推論候補対象を作成する。
    (art_id, cust_id)をkeyに持つdataframeを返す。
    """

    candidates_df = None

    for i, block in enumerate(blocks):
        with timer(
            logger=logger, prefix="↑fitted {} ".format(block.__class__.__name__)
        ):
            if i == 0:
                candidates_df = block.fit(
                    trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers
                )
            else:
                new_df = block.fit(
                    trans_cdf, art_cdf, cust_cdf, logger, y_cdf, target_customers
                )
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
    trans_w_art_cust_info_cdf = trans_cdf.merge(
        art_cdf, on="article_id", how="left"
    ).merge(cust_cdf, on="customer_id", how="left")

    drop_cols = trans_w_art_cust_info_cdf.to_pandas().filter(regex=".*_name").columns
    trans_w_art_cust_info_cdf.drop(columns=drop_cols, inplace=True)
    art_feat_cdf = art_cdf[["article_id"]]  # base
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

import time
def negative_sampling(base_df, phase):
    positive_df = base_df.query("y==1")
    positive_customer_ids = positive_df["customer_id"].unique()

    # postiveが存在するcustomer_idのみに絞る。不例の数は全ユーザ２０件
    negative_df = base_df.query("y==0")
    negative_df = negative_df[negative_df["customer_id"].isin(positive_customer_ids)]
    time_sta = time.time()
    negative_df = negative_df.groupby("customer_id").apply(
        lambda x: x.sample(n=min(10, len(x)), random_state=42)
    )
    print("groupby - sampleにかかった時間", time.time()- time_sta)
    print('positive_df.shape', positive_df.shape, 'negative_df.shape', negative_df.shape)

    sampled_df = pd.concat([positive_df, negative_df])
    time_sta = time.time()
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("sampleにかかった時間", time.time()- time_sta)
    return sampled_df


def make_train_valid_df(
    trans_cdf, art_cdf, cust_cdf, target_weeks, train_duration_weeks=8
):
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for target_week in tqdm(target_weeks):
        phase = "validate" if target_week == 104 else "train"
        # import pdb;pdb.set_trace()

        clipped_trans_cdf = trans_cdf.query(
            f"@target_week - {train_duration_weeks} <= week < @target_week"
        )  # 予測日の前週までのトランザクションが使える

        # (要議論) speed upのためにclipped_trans_cdfに存在するart, custだけの情報を使う
        buyable_art_ids = clipped_trans_cdf["article_id"].unique()
        buyable_art_cdf = art_cdf[art_cdf["article_id"].isin(buyable_art_ids)]
        # buyable_art_cdf = art_cdf.query(f"article_id in {list(buyable_art_ids)}")

        bought_cust_ids = clipped_trans_cdf["customer_id"].to_pandas().unique()
        bought_cust_cdf = cust_cdf[cust_cdf["customer_id"].isin(bought_cust_ids)]
        # bought_cust_cdf = cust_cdf.query(f"customer_id in {list(bought_cust_ids)}")

        y_cdf = make_y_cdf(trans_cdf, target_week)
        y_df = y_cdf.to_pandas()

        target_customers = np.unique(np.append(y_df["customer_id"].unique(), bought_cust_ids))
        # いろんなメソッドを使って候補生成
        if phase=='validate':
            base_df = candidate_generation(
                candidate_blocks_valid, # valid用
                clipped_trans_cdf,
                buyable_art_cdf,
                bought_cust_cdf,
                y_cdf,
                target_customers=target_customers, # 正例のcustomerに対する負例も取ってくる
            )
        else:
            base_df = candidate_generation(
                candidate_blocks,
                clipped_trans_cdf,
                buyable_art_cdf,
                bought_cust_cdf,
                y_cdf,
                target_customers=target_customers,# 正例のcustomerに対する負例も取ってくる
            )

        print(base_df.shape)
        base_df = base_df.drop_duplicates(subset=["article_id", 'customer_id'])
        print(base_df.shape)

        base_df["y"] = (
            base_df.merge(y_df, how="left", on=["customer_id", "article_id"])["y"]
            .fillna(0)
            .astype(int)
        )
        
        base_df = negative_sampling(base_df, phase)  # 適当に間引くはず

        art_feat_df, cust_feat_df, pair_feat_df = feature_generation_with_transaction(
            feature_blocks,
            clipped_trans_cdf,
            buyable_art_cdf,
            bought_cust_cdf,  # base_dfに入っているuser_id,article_idが必要
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
            base_df = base_df.sample(frac=1, random_state=42).reset_index(drop=True)
            train_df = pd.concat([train_df, base_df])
        elif phase == "validate":
            base_df = base_df.sample(frac=1, random_state=42).reset_index(drop=True)
            valid_df = base_df.copy()

    # transactionによらない特徴量はゆくゆくはここで実装したい (ex. embbeding特徴量)
    # train_df = feature_generation_with_article(train_df, article_cdf)
    # valid_df = feature_generation_with_article(valid_df, article_cdf)
    # train_df = feature_generation_with_customer(train_df, customer_cdf)
    # valid_df = feature_generation_with_customer(valid_df, customer_cdf)

    train_df.sort_values(["customer_id", "target_week"], inplace=True)
    valid_df.sort_values(["customer_id", "target_week"], inplace=True)

    train_groups = train_df.groupby(["customer_id", "target_week"]).size().values
    valid_groups = valid_df.groupby(["customer_id", "target_week"]).size().values

    return train_df, valid_df, train_groups, valid_groups


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
    val_pred = clf.predict(valid_X, group=valid_groups)
    return clf, val_pred

def calc_map12(valid_df, logger, path_to_valid_true):
    valid_df.sort_values(['customer_id', 'prediction'], ascending=False, inplace = True)
    valid_pred = (
        valid_df
        .groupby('customer_id')[['article_id']]
        .aggregate(lambda x: x.tolist())
    )
    valid_pred['prediction'] = valid_pred['article_id'].apply(lambda x: ' '.join(['0'+str(k) for k in x]))

    valid_true = pd.read_csv(path_to_valid_true)[["customer_id", "valid_true"]]
    valid_true['customer_id'] = customer_hex_id_to_int(valid_true['customer_id'])
    valid_true = valid_true.merge(valid_pred.reset_index(), how="left", on="customer_id")

    mapk_val = round(
        mapk(
            valid_true["valid_true"].map(lambda x: x.split()),
            valid_true["prediction"].astype(str).map(lambda x: x.split()),
            k=12,
        ),
        5,
    )
    print(f"validation map@12: {mapk_val}")
    logger.info(f"validation map@12: {mapk_val}")

    return mapk_val, valid_true


""" ----------------------------------------------------------------前処理----------------------------------------------------------------"""
target_weeks = [104, 103]#102 , 101]  # test_yから何週間離れているか

train_df, valid_df, train_groups, valid_groups = make_train_valid_df(
        raw_trans_cdf, raw_art_cdf, raw_cust_cdf, target_weeks, train_duration_weeks=TRAIN_DURATION_WEEKS
    )

# train_df[["customer_id", "article_id", "target_week", "y"]].to_csv(log_dir / "train_df.csv", index=False)
# valid_df[["customer_id", "article_id", "target_week", "y"]].to_csv(log_dir / "valid_df.csv", index=False)

# train_group_df = pd.DataFrame()
# train_group_df["group"] = train_groups
# train_group_df.to_csv(output_dir / "train_group_df.csv", index=False)

# valid_group_df = pd.DataFrame()
# valid_group_df["group"] = valid_groups
# valid_group_df.to_csv(output_dir / "valid_group_df.csv", index=False)

drop_cols = ["customer_id", "article_id", "target_week", "y"]
train_X = train_df.drop(drop_cols, axis=1)
valid_X = valid_df.drop(drop_cols, axis=1)

""" ----------------------------------------------------------------学習----------------------------------------------------------------"""
rank_param = {
    "objective": "lambdarank",
    "eval_metric": "map",
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


clf, val_pred = train_rank_lgb(
    train_X,
    train_df["y"],
    valid_X,
    valid_df["y"],
    rank_param,
    logger,
    early_stop_round=100,
    log_period=10,
)

lgbm_path = output_dir / "lgbm.pickle"
with open(lgbm_path, "wb") as f:
    pickle.dump(clf, f)


valid_df['prediction'] = val_pred
mapk_val, valid_true = calc_map12(valid_df, logger, '/home/ec2-user/h_and_m/data/inputs/valid_true_after0916.csv')

# train_df[["customer_id", "article_id", "target_week", "y"]].to_csv(log_dir / "train_df.csv", index=False)
# valid_df[["customer_id", "article_id", "target_week", "y", 'prediction']].sort_values(['customer_id', 'prediction'], ascending=False).to_csv(log_dir / "valid_df.csv", index=False)
# valid_true.to_csv(log_dir / "valid_true.csv", index=False)


fig, ax = visualize_importance([clf], train_X)
fig.savefig(log_dir / "feature_importance.png")


""" ----------------------------------------------------------------推論コード----------------------------------------------------------------"""
lgbm_path = output_dir / "lgbm.pickle"
clf = pickle.load(open(lgbm_path, 'rb'))

#sub_df = pd.read_csv("/home/ubuntu/comp/hm/data/submission.csv")
#sub_df['customer_id'] = customer_hex_id_to_int(sub_df['customer_id'])

target_week = 105
if DRY_RUN:
    train_duration_weeks=2
    BATCH_SIZE = 100_000
else:
    train_duration_weeks=TRAIN_DURATION_WEEKS
    BATCH_SIZE = 10_000

clipped_trans_cdf = raw_trans_cdf.query(f"@target_week - {train_duration_weeks} <= week < @target_week" )

#sub_customer_ids = clipped_trans_cdf["customer_id"].to_pandas().unique()
submission_df = pd.read_csv("/home/ec2-user/h_and_m/data/sample_submission.csv")
submission_df['customer_id'] = customer_hex_id_to_int(submission_df['customer_id'])
sub_customer_ids = submission_df["customer_id"].unique()

drop_cols = ["customer_id", "article_id"]

preds_list = []
preds_dic ={}

# ここに特徴生成 バッチ推論ではclipped_trans_cdfが同様のため先に特徴は作る
art_feat_df, cust_feat_df, pair_feat_df = feature_generation_with_transaction(
        feature_blocks,
        clipped_trans_cdf, # 特徴生成なので全ユーザーのtransactionが欲しい。けどメモリのために期間は絞る
        raw_art_cdf,
        raw_cust_cdf,  # base_dfに入っているuser_id,article_idが必要
)
art_feat_df = reduce_mem_usage(art_feat_df)
cust_feat_df = reduce_mem_usage(cust_feat_df)

for bucket in tqdm(range(0, len(sub_customer_ids), BATCH_SIZE)):
    print('!!!!!!!!!!!!!!!  bucket  !!!!!!!!!!!!!!!', bucket)
    batch_customer_ids = sub_customer_ids[bucket: bucket+BATCH_SIZE]
    batch_trans_cdf = raw_trans_cdf[raw_trans_cdf["customer_id"].isin(batch_customer_ids)]
    # if len(batch_trans_cdf)>0:
    
    # (要議論) speed upのためにclipped_trans_cdfに存在するart, custだけの情報を使う
    buyable_art_ids = batch_trans_cdf["article_id"].unique()
    buyable_art_cdf = raw_art_cdf[raw_art_cdf["article_id"].isin(buyable_art_ids)]
    bought_cust_ids = batch_trans_cdf["customer_id"].to_pandas().unique()
    bought_cust_cdf = raw_cust_cdf[raw_cust_cdf["customer_id"].isin(bought_cust_ids)]
    
    # ここに候補生成
    batch_base_df = candidate_generation(
            candidate_blocks_test,
            batch_trans_cdf, # 該当ユーザーのtransactionが欲しいのでこれでOK
            buyable_art_cdf,
            bought_cust_cdf,
            y_cdf=None,
            target_customers=batch_customer_ids, # todo: 本当はbatch_customer_ids(全ユーザーに対して候補生成をしたい)
        )
    print('before',batch_base_df.shape)
    batch_base_df = batch_base_df.drop_duplicates(subset=['customer_id', 'article_id'], keep='last')
    batch_base_df = reduce_mem_usage(batch_base_df)
    
    batch_base_df = batch_base_df.merge(cust_feat_df, how="left", on="customer_id")
    batch_base_df = batch_base_df.merge(art_feat_df, how="left", on="article_id")
        
#         for cust_id, df in batch_base_df.groupby('customer_id'):
#             batch_test_pred = clf.predict(df.drop(drop_cols, axis=1))
#             top12_index = np.argsort(batch_test_pred)[::-1][:12]
#             top12_article = df['article_id'].values[top12_index]
#             preds_dic[cust_id] = ' '.join(['0'+str(k) for k in top12_article])

# preds_df = pd.DataFrame.from_dict(preds_dic, orient='index').reset_index().rename(columns={'index':'customer_id', 0:'article_df'})
# print(preds_df)
            
            
        
    #batch_base_df = reduce_mem_usage(batch_base_df)
    # 不要?
    batch_base_df.sort_values(["customer_id"], inplace=True)
    batch_test_X = batch_base_df.drop(drop_cols, axis=1)

    # 推論
    batch_test_pred = clf.predict(batch_test_X)
    batch_base_df["pred"] = batch_test_pred

    # subの作成
    batch_submission_df = batch_base_df[['customer_id', 'article_id', 'pred']]
    batch_submission_df.sort_values(['customer_id', 'pred'], ascending=False, inplace = True)

    batch_submission_df = batch_submission_df.groupby('customer_id').head(12)

    batch_submission_df = (
        batch_submission_df
        .groupby('customer_id')[['article_id']]
        .aggregate(lambda x: x.tolist())
    )
    batch_submission_df['article_id'] = batch_submission_df['article_id'].apply(lambda x: ' '.join(['0'+str(k) for k in x]))
    
    preds_list.append(batch_submission_df)
    
preds_df = pd.concat(preds_list).reset_index()

""" ----------------------------------------------------------------提出用に変換----------------------------------------------------------------"""
# sample_subのcustomer_idをintに変換した上で、推論結果(preds_df)をmergeする
submission_df = pd.read_csv("/home/ec2-user/h_and_m/data/sample_submission.csv")
submission_df = submission_df[['customer_id']]
submission_df['customer_id'] = customer_hex_id_to_int(submission_df['customer_id'])
submission_df = submission_df.merge(preds_df, on="customer_id", how="left")
submission_df['customer_id'] = pd.read_csv("/home/ec2-user/h_and_m/data/sample_submission.csv")['customer_id']
submission_df.rename(columns={'article_id': 'prediction'},  inplace=True)
print(submission_df)

submission_df.to_csv(log_dir / "submission_df.csv", index=False)
