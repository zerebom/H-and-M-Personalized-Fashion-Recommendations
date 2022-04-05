#%%
import json
import plotly.express as px
import umap
import gc
import torch
import transformers
from transformers import BertTokenizer
import os
import pickle
import pprint
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from time import time
from typing import Any, Dict, List, Tuple

import cudf
import cv2
import lightgbm as lgb
import matplotlib.pyplot as plt
import nltk
import nmslib
import numpy as np
import pandas as pd
import seaborn as sns
import swifter
from matplotlib.font_manager import FontProperties
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import column
from tqdm import tqdm

nltk.download('stopwords')
if True:
    sys.path.append("../")
    from candidacies import (AbstractCGBlock, BoughtItemsAtInferencePhase,
                             LastNWeekArticles, PopularItemsoftheLastWeeks,
                             candidates_dict2df)
    from eda_tools import visualize_importance
    from features import (AbstractBaseBlock, ModeCategoryBlock,
                          TargetEncodingBlock, TargetRollingBlock)
    from metrics import apk, mapk
    from utils import (Categorize, article_id_int_to_str,
                       article_id_str_to_int, arts_id_list2str,
                       customer_hex_id_to_int, phase_date, reduce_mem_usage,
                       timer)

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



# %%


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

#%%

art_cdf = cudf.read_csv(input_dir / 'articles.csv')
art_df = art_cdf.to_pandas()
stop_words_nltk = nltk.corpus.stopwords.words('english')

# NearestNeighbors (類似しているN個の商品を抽出)
def del_eng_stop_words(text):
    #!マークなども消したいので正規表現で空文字に置換
    regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    sample = regex.sub("", text)

    #小文字にして、stop wordsに入っていなければリストに追加
    words = [w.lower()  for w in sample.split(" ") \
                 if not w.lower()  in stop_words_nltk]
    words=' '.join(words)
    return words



#%%
# 欠損値補完
art_df["detail_desc"] = art_df["detail_desc"].fillna(" ")
# 商品説明以外のtxt情報も使用 (色とか)
art_df["text"] = art_df.apply(
    lambda x: " ".join(
        [
            str(x["prod_name"]),
            str(x["product_type_name"]),
            str(x["product_group_name"]),
            str(x["graphical_appearance_name"]),
            str(x["colour_group_name"]),
            str(x["perceived_colour_value_name"]),
            str(x["index_name"]),
            str(x["section_name"]),
            str(x["detail_desc"])
        ]
    ),
    axis=1,
)

# stop word削除
art_df["text"] = art_df["text"].swifter.apply(lambda x:del_eng_stop_words(x))
#%%
text = ' '.join(art_df["text"]) #各曲を長い一つのリストに統合

class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(
            self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor(
            [inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        seq_out = self.bert_model(inputs_tensor, masks_tensor)['last_hidden_state']
        # seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']
        if torch.cuda.is_available():
            # 0番目は [CLS] token, 768 dim の文章特徴量
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()

# %%
tqdm.pandas()
bert = BertSequenceVectorizer()
art_df["bert_vec"] = art_df["text"].progress_apply(lambda x:bert.vectorize(x))
# art_df["bert_vec"] = art_df["text"].swifter.progress_bar(True).apply(lambda x:bert.vectorize(x))




#%%
um = umap.UMAP(n_components=20)
emb = np.array([v for v in art_df["bert_vec"].values])
decomp_emb = um.fit_transform(emb)

# %%

emb_dic = {}
for art_id, emb_vec in zip(art_df['article_id'].values,decomp_emb):
    emb_dic[int(art_id)] = list(emb_vec.astype(float))

#%%
with open(str(input_dir / 'emb/customer_emb.json'), 'w') as f:
    json.dump(emb_dic, f, indent=4)

# %%

fig = px.scatter(art_df.head(1000),x='emb_1',y='emb_2',color='index_name')
fig.show()



# %%





# %%

# 画像系のコード
def input_show(data):

    plt.title("Query Image")
    plt.imshow(data)

def show_result(df, result):
    fig = plt.figure(figsize=(12,8))
    for i in range(0,12):
        index_result=result[0][i]
        image_path = str(input_dir / f'images/0{str(df.article_id[index_result])[:2]}/0{int(df.article_id[index_result])}.jpg')
        plt.subplot(3,4,i+1)
        im = cv2.imread(image_path)
        plt.imshow(im)
        plt.title(index_result)
    fig.tight_layout(rect=[0,0,1,0.96])
    plt.show()

N_NEIGHBORDHOOD = 12
def result_vector_cosine(feature_vector,new_feature):
    new_feature = np.array(new_feature)
    new_feature = new_feature.flatten()
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORDHOOD, metric="cosine").fit(feature_vector)
    distances, indices = nbrs.kneighbors([new_feature])
    return(indices)

i = 16500
emb_arr = np.array([v for v in emb_dic.values()])
result = result_vector_cosine(emb_arr,emb_arr[i])
print("########## query detail_desc ##########")
print(art_df.detail_desc[i])
output_path = str(input_dir / f'images/0{str(art_df.article_id[i])[:2]}/0{int(art_df.article_id[i])}.jpg')
input_show(cv2.imread(output_path))
show_result(art_df, result)

# %%

# %%
