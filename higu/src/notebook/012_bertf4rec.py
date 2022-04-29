#%%
import json
import os
import pickle
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import cudf
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder
from torch.nn import Linear
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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

# %%
path = "/home/kokoro/h_and_m/higu/input/transactions_train.parquet"
pk = pd.read_parquet(path)
#%%
pk


#%%
# ==================================================================================
# ================================= データのロード =================================
# ==================================================================================
raw_trans_cdf, raw_cust_cdf, raw_art_cdf = read_cdf(input_dir, DRY_RUN)

#%%
tdate = datetime(year=2019, month=9, day=1)
active_articles = raw_trans_cdf.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles.query("t_dat >= @tdate ").reset_index()

raw_trans_cdf = raw_trans_cdf[
    raw_trans_cdf["article_id"].isin(active_articles["article_id"])
].reset_index(drop=True)
raw_trans_cdf.shape
raw_trans_cdf["week"] = (
    raw_trans_cdf["t_dat"].max() - raw_trans_cdf["t_dat"]
).dt.days // 7
raw_trans_cdf["week"].value_counts()


#%%


article_ids = np.concatenate(
    [[0], np.unique(raw_trans_cdf.to_pandas()["article_id"].values)]
)
le_article = LabelEncoder()
le_article.fit(article_ids)
raw_trans_cdf["article_id"] = le_article.transform(
    raw_trans_cdf["article_id"].to_pandas()
)

#%%
WEEK_HIST_MAX = 5


def create_dataset(df, week):
    hist_df = df[(df["week"] > week) & (df["week"] <= week + WEEK_HIST_MAX)]
    hist_df = (
        hist_df.groupby("customer_id")
        .agg({"article_id": list, "week": list})
        .reset_index()
    )
    hist_df.rename(columns={"week": "week_history"}, inplace=True)

    target_df = df[df["week"] == week]
    target_df = target_df.groupby("customer_id").agg({"article_id": list}).reset_index()
    target_df.rename(columns={"article_id": "target"}, inplace=True)
    target_df["week"] = week

    return target_df.merge(hist_df, on="customer_id", how="left")


val_weeks = [0]
train_weeks = [1, 2, 3, 4]


val_df = cudf.concat([create_dataset(raw_trans_cdf, w) for w in val_weeks]).reset_index(
    drop=True
)
train_df = cudf.concat(
    [create_dataset(raw_trans_cdf, w) for w in train_weeks]
).reset_index(drop=True)


#%%
val_df.to_pandas()

#%%
class HMDataset(Dataset):
    def __init__(self, df, seq_len, is_test=False):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.is_test = is_test

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.is_test:
            target = torch.zeros(2).float()
        else:
            target = torch.zeros(len(article_ids)).float()
            for t in row.target:
                target[t] = 1.0

        article_hist = torch.zeros(self.seq_len).long()
        week_hist = torch.ones(self.seq_len).float()

        if isinstance(row.article_id, list):
            if len(row.article_id) >= self.seq_len:  # if seq_lenよりもそのユーザが商品を買っていたら...
                # 最後-seq_len ~ 最後に買った商品を使う
                article_hist = torch.LongTensor(row.article_id[-self.seq_len :])
                # 各商品の買ったweekも保存
                week_hist = (
                    (torch.LongTensor(row.week_history[-self.seq_len :]) - row.week)
                    / WEEK_HIST_MAX
                    / 2
                )
            else:
                # そうじゃないなら買った商品の量~末尾にアイテムを詰める
                article_hist[-len(row.article_id) :] = torch.LongTensor(row.article_id)
                week_hist[-len(row.article_id) :] = (
                    (torch.LongTensor(row.week_history) - row.week) / WEEK_HIST_MAX / 2
                )

        return article_hist, week_hist, target


#%%
HMDataset(val_df.to_pandas(), 64)[1]

#%%
import torch.nn as nn
import torch.nn.functional as F


class HMModel(nn.Module):
    def __init__(self, article_shape):
        super(HMModel, self).__init__()

        self.article_emb = nn.Embedding(
            article_shape[0], embedding_dim=article_shape[1]
        )

        self.article_likelihood = nn.Parameter(
            torch.zeros(article_shape[0]), requires_grad=True
        )
        self.top = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 8, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
        )

    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]
        x = self.article_emb(article_hist)
        print("x1", x.shape)
        x = F.normalize(x, dim=2)
        print("x2", x.shape)

        x = x @ F.normalize(self.article_emb.weight).T
        print("x3", x.shape)

        x, indices = x.max(axis=1)
        print("x4", x.shape, "shape", indices.shape)
        x = x.clamp(1e-3, 0.999)
        x = -torch.log(1 / x - 1)

        max_week = (
            week_hist.unsqueeze(2)
            .repeat(1, 1, x.shape[-1])
            .gather(1, indices.unsqueeze(1).repeat(1, week_hist.shape[1], 1))
        )
        print("week1", max_week.shape)
        max_week = max_week.mean(axis=1).unsqueeze(1)
        print("week2", max_week.shape)

        x1 = torch.cat(
            [
                x.unsqueeze(1),
                max_week,
                self.article_likelihood[None, None, :].repeat(x.shape[0], 1, 1),
            ],
            axis=1,
        )

        x2 = self.top(x1).squeeze(1)
        return x1, x2


model = HMModel((len(le_article.classes_), 512))
#%%
model.article_emb.weight.T.shape


#%%
SEQ_LEN = 64

BS = 8
NW = 8

train_dataset = HMDataset(train_df.to_pandas(), SEQ_LEN)
train_loader = DataLoader(
    train_dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=NW,
    pin_memory=False,
    drop_last=True,
)

#%%
tl = next(iter(train_loader))
tl

#%%
x1, x2 = model(tl)

#%%
x1.shape

#%%
x2.shape


#%%

x1[0]


#%%


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

#%%
PAD = 0
MASK = 1


def map_column(df: pd.DataFrame, col_name: str):
    """
    Maps column values to integers
    :param df:
    :param col_name:
    :return:
    """
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def get_context(
    df: pd.DataFrame, split: str, context_size: int = 120, val_context_size: int = 5
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param context_size:
    :param val_context_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(10, df.shape[0] - val_context_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    start_index = max(0, end_index - context_size)

    context = df[start_index:end_index]

    return context


def pad_arr(arr: np.ndarray, expected_size: int = 30):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def pad_list(list_integers, history_size: int, pad_val: int = PAD, mode="left"):
    """
    :param list_integers:
    :param history_size:
    :param pad_val:
    :param mode:
    :return:
    """

    if len(list_integers) < history_size:
        if mode == "left":
            list_integers = [pad_val] * (
                history_size - len(list_integers)
            ) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (
                history_size - len(list_integers)
            )

    return list_integers


def df_to_np(df, expected_size=30):
    arr = np.array(df)
    arr = pad_arr(arr, expected_size=expected_size)
    return arr


def genome_mapping(genome):
    genome.sort_values(by=["movieId", "tagId"], inplace=True)
    movie_genome = genome.groupby("movieId")["relevance"].agg(list).reset_index()

    movie_genome = {
        a: b for a, b in zip(movie_genome["movieId"], movie_genome["relevance"])
    }

    return


def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):

    _, predicted = torch.max(y_pred, 1)

    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)

    acc = (y_true == predicted).double().mean()

    return acc


def masked_ce(y_pred, y_true, mask):

    loss = F.cross_entropy(y_pred, y_true, reduction="none")

    loss = loss * mask

    return loss.sum() / (mask.sum() + 1e-8)


class Recommender(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=128,
        cap=0,
        mask=1,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()

        self.cap = cap
        self.mask = mask

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.item_embeddings = torch.nn.Embedding(
            self.vocab_size, embedding_dim=channels
        )

        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.linear_out = Linear(channels, self.vocab_size)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items):
        src_items = self.item_embeddings(src_items)

        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src_items += pos_encoder
        src = src_items.permute(1, 0, 2)
        src = self.encoder(src)
        return src.permute(1, 0, 2)

    def forward(self, src_items):
        src = self.encode_src(src_items)
        out = self.linear_out(src)

        return out

    def training_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


# %%
