# %%
'''
買うユーザ・買わないユーザを当てたい！
'''

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, f1_score
import gc
import os
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
from sqlalchemy import column
from tqdm import tqdm

if True:
    #sys.path.append("../")
    sys.path.append("/content/drive/MyDrive/competition/h_and_m/higu/src/")
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


root_dir = Path("/content/drive/MyDrive/competition/") #Path("/home/kokoro/h_and_m/higu")
input_dir = root_dir / "input"
output_dir = root_dir / "output"
DRY_RUN = False

LGBM_DEFAULT_PARAMS = {
    'objective': 'binary',
    'learning_rate': .1,
    # L2 Reguralization
    'reg_lambda': 1.,
    # こちらは L1
    'reg_alpha': .1,
    # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    'max_depth': 5,
    # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで,
    # まんべんなく特徴を使うという効果があります.
    'colsample_bytree': .5,
    # 最小分割でのデータ数. 小さいとより細かい粒度の分割方法を許容します.
    # 'min_child_samples': 10,
    # bagging の頻度と割合
    'subsample_freq': 3,
    'subsample': .9,
    'verbose': -1,
    # 特徴重要度計算のロジック(後述)
    'saved_feature_importance_type': 1,
    'random_state': 71,
}

def fit_single_lgbm(train_X, train_y, val_X, val_y, params, verbose=50):

    if params is None:
        params = {}

    lgb_train = lgb.Dataset(train_X,
                            train_y,
                            )

    lgb_eval = lgb.Dataset(val_X,
                           val_y,
                           reference=lgb_train,
                           )

    model = lgb.train(
        params, lgb_train, num_boost_round=2000, valid_sets=[
            lgb_train, lgb_eval], callbacks=[
            lgb.early_stopping(
                stopping_rounds=100), lgb.log_evaluation(
                    period=verbose)])
    val_pred = model.predict(val_X, num_iteration=model.best_iteration)
    return model, val_pred


# %load_ext autoreload
# %autoreload 2


pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.plotting.backend = "plotly"


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


# %%

# ================前処理================
# transaction が10以下のユーザは予測しない
# ids = trans_cdf.groupby('customer_id').size().to_frame(
#     'size').query('size >= 10').index
# raw_cust_cdf = cust_cdf.copy()
# cust_cdf = cust_cdf[cust_cdf['customer_id'].isin(ids)]

# 週だとデータ量的に厳しいので2ヶ月単位(8週間)で予測する
trans_cdf['month'] = trans_cdf['week'] // 8


# %%

keys = ['customer_id', 'month']


# monthとcustomerのcross join
cust_id_df = cust_cdf[['customer_id']]
month_df = cudf.DataFrame(list(range(0, 15)), columns=['month'])
cust_id_df['key'] = 1
month_df['key'] = 1
month_cust_prod_df = month_df.merge(
    cust_id_df,
    how='outer',
    on='key').drop(
        'key',
    axis=1).sort_values(keys).reset_index(drop=True)



# %%


# ================特徴量生成================
te_block = TargetEncodingBlock(
    key_col=keys,
    target_col='price',
    agg_list=[
        'count',
        'sum'])
month_trans_cdf = te_block.fit(trans_cdf)

# 購入数と購入額でrollingする
# 直近nヶ月の購入数
if 'price_sum' not in month_cust_prod_df.columns:
    month_cust_prod_df = month_cust_prod_df.merge(
        month_trans_cdf, how='left', on=keys).fillna(0)
    month_cust_prod_df = month_cust_prod_df.sort_values(
        ['customer_id', 'month']).reset_index(drop=True)

# n(2,3,6,12)ヶ月で買った(商品数,購入額)の(min,mean,max,sum)
for col in ['price_sum', 'price_count']:
    print(col)
    roll_block = TargetRollingBlock(
        keys, 'customer_id', col, [
            'mean', 'max', 'min', 'sum'], [
            3, 6, 12])
    rolling_df = roll_block.fit(month_cust_prod_df)

    month_cust_prod_df = month_cust_prod_df.merge(
        rolling_df, how='left', on=keys).fillna(0)
    month_cust_prod_df = month_cust_prod_df.sort_values(
        ['customer_id', 'month']).reset_index(drop=True)
    month_cust_prod_df

# デフォルトのuserの特徴(ageとか
month_cust_prod_df = month_cust_prod_df.merge(
    cust_cdf, how='left', on='customer_id')
month_cust_prod_df = month_cust_prod_df.sort_values(
    ['customer_id', 'month']).reset_index(drop=True)

#%%
month_cust_prod_df['next_perchase_cnt'] = month_cust_prod_df.to_pandas(
).groupby(['customer_id'])['price_count'].shift(-1)

#正例と負例の数を揃える
positive_uids = month_cust_prod_df.query('month==12 and next_perchase_cnt>0')['customer_id'].unique().values.get()
negative_uids = month_cust_prod_df.query('month==12 and next_perchase_cnt==0')['customer_id'].unique().values.get()

np.random.seed(42)
sample_nevative_uids = np.random.choice(negative_uids,replace=False,size=len(positive_uids))
train_ids = np.concatenate((sample_nevative_uids, positive_uids))
train_df = month_cust_prod_df[month_cust_prod_df['customer_id'].isin(train_ids)]

# %%

# ================前処理================
# X,yを作る。次の月の購入量をyとする
X = train_df.drop(columns=['next_perchase_cnt']).to_pandas()
# 連続値で解く
#y = month_cust_prod_df['next_perchase_cnt'].to_pandas()
# binaryで解く
y = (train_df['next_perchase_cnt'] != 0).astype(int).to_pandas()
X['postal_code'] = X['postal_code'].astype('category')






# %%
idx_tr, idx_val = X.query('month<12').index, X.query('month==12').index
train_X, val_X = X.loc[idx_tr, :], X.loc[idx_val, :]
train_y, val_y = y[idx_tr], y[idx_val]

#%%
model, val_pred = fit_single_lgbm(
    train_X, train_y, val_X, val_y, params=LGBM_DEFAULT_PARAMS)



#%%

# ================predict================
#ルールベースの追加(4ヶ月買わなかったユーザをターゲット期間でも買わないユーザと予測する)
last_month_cdf = trans_cdf.query('month < 13').groupby('customer_id')[
    'month'].max().to_frame('last_month').reset_index()
cust_cdf = cust_cdf.merge(last_month_cdf, how='left', on='customer_id')
cust_cdf
cust_cdf['is_active_4month'] = (cust_cdf['last_month'] >= 10).astype(int)
val_X = val_X.merge(cust_cdf[['customer_id', 'is_active_3month']].to_pandas(
), on='customer_id', how='left')
val_X




#%%

#validation期間の全ユーザに予測
val_df = month_cust_prod_df.query('month==12').to_pandas()
val_df['postal_code'] = val_df['postal_code'].astype('category')
val_df['true'] = (val_df['next_perchase_cnt'] != 0).astype(int)
val_df['pred'] = model.predict(val_df.drop(columns=['next_perchase_cnt','true']))


#%%
#test期間の全ユーザに予測
test_df = month_cust_prod_df.query('month==13').to_pandas().drop(columns=['next_perchase_cnt'])
test_df['postal_code'] = test_df['postal_code'].astype('category')
test_df['pred'] = model.predict(test_df)
test_df
#%%
#出力
test_df[['customer_id','pred']].to_parquet(output_dir/'005_preds_of_not_purchase_user.parquet',index=False)

#%%

# ================EDA================
#preriod = 12のヒストグラム
sample_val_df = val_df.sample(100000)
fig = px.histogram(
    sample_val_df,
    x="pred",
    color="true",
    # opacity=0.4,
    nbins=50,
    marginal="box",
    text_auto=False

)
fig.show()

#%%
#閾値ごとのclassification_report
for i in range(1, 10):
    print(i)
    preds = (val_X['pred'] > (i * 0.1)).astype(int).values
    print('pred==1:',len(preds[preds==1]),'pred==0:',len(preds[preds==0]))
    print(f1_score(val_X['true'], preds))
    print(classification_report(val_X['true'], preds))


# %%

#feature_importance
fig, ax = visualize_importance([model], X)
plt.show()
# %%

#roc曲線
fpr, tpr, thresholds = roc_curve(val_X['true'], val_X['pred'])
plt.plot(fpr, tpr)
plt.show()


# %%
fig = px.violin(
    val_X,
    y="pred",
    x="true",
)
fig.show()

# %%
# 使わなかったコード
first_day = trans_cdf['t_dat'].min()
last_day = trans_cdf['t_dat'].max()

date_df = pd.date_range(
    start=first_day,
    end=last_day,
    freq='D').to_frame(
        name='date').reset_index(
            drop=True)

date_df['month'] = date_df['date'].dt.month
date_df['day'] = date_df['date'].dt.day
date_df['monthofyear'] = date_df['date'].dt.monthofyear
date_df['month'] = date_df['date'].dt.month
date_df['monthday'] = date_df['date'].dt.monthday
date_df['dayofyear'] = date_df['date'].dt.dayofyear
date_df.tail(50)
