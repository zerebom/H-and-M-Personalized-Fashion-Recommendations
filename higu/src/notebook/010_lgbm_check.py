#%%
import gc
import json
import os
import pickle
import pprint
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from time import time

import cudf
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import make_scorer, mean_squared_error


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

# %%

root_dir = Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"

exp_name = "008_lgbm"
output_dir = root_dir / "output" / exp_name
#%%

valid_X = pd.read_pickle(output_dir / "valid_X.pkl")
valid_y = pd.read_pickle(output_dir / "valid_y.pkl")
valid_key = pd.read_pickle(output_dir / "valid_key.pkl")

lgbm_path = output_dir / "lgbm.pickle"
with open(lgbm_path, "rb") as f:
    clf = pickle.load(f)

#%%
valid_key = valid_key.reset_index(drop=True)
#%%

valid_key["target"] = valid_y
valid_key["pred"] = clf.predict_proba(valid_X)[:, 1]

#%%
valid_key

# %%

val_sub_fmt_df = squeeze_pred_df_to_submit_format(valid_key)
#%%
val_sub_fmt_df

# %%

mapk_val = mapk(
    val_sub_fmt_df["prediction"].map(lambda x: x.split()),
    val_sub_fmt_df["target"].map(lambda x: x.split()),
    k=12,
)
#%%
val_sub_fmt2 = val_sub_fmt_df[val_sub_fmt_df["target"] != ""]

mapk(
    val_sub_fmt2["prediction"].map(lambda x: x.split()),
    val_sub_fmt2["target"].map(lambda x: x.split()),
    k=12,
)
#%%
val_sub_fmt2

#%%
fig = px.violin(
    valid_key.head(50000),
    y="pred",
    x="true",
)
fig.show()

# %%


# %%

import cv2


def input_show(data):

    plt.title("Query Image")
    plt.imshow(data)


def show_result(df, result):
    fig = plt.figure(figsize=(12, 8))
    for i in range(0, 12):
        index_result = result[0][i]
        image_path = str(
            input_dir
            / f"images/0{str(df.article_id[index_result])[:2]}/0{int(df.article_id[index_result])}.jpg"
        )
        plt.subplot(3, 4, i + 1)
        im = cv2.imread(image_path)
        plt.imshow(im)
        plt.title(index_result)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#%%


#%%

#%%

true_valid = valid_key.query("true==1")
result_df = valid_key[
    valid_key["customer_id"] == true_valid["customer_id"].sample(1).values[0]
]

result_df.sort_values("pred", ascending=False).head(50)


#%%
val_sub_fmt_df = squeeze_pred_df_to_submit_format(valid_key)
# %%
import cv2


def read_img(art_id):
    image_path = str(input_dir / f"images/0{art_id[:2]}/0{art_id}.jpg")
    im = cv2.imread(image_path)
    return im


# %%


#%%

trues = result_df["true"].values
preds = result_df["pred"].values
art_ids = result_df["article_id"].astype(str).values

fig = plt.figure(figsize=(12, 8))
for idx, (true, pred, art_id) in enumerate(zip(trues, preds, art_ids)):
    im = read_img(art_id)
    plt.subplot(3, 4, idx + 1)
    plt.imshow(im)
    title = f"art_id:{art_id} \n true:{true} \n pred:{pred:.3f}"
    plt.title(title)
    if idx == 11:
        break

fig.tight_layout()
plt.show()

# %%
