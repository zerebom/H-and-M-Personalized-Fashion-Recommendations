target_weeks = [5, 4, 3, 2, 1, 0]  # test_yから何週間離れているか

train_df, valid_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

for target_week in target_weeks:

    if target_week == 0:
        phase = "test"
    elif target_week == 1:
        phase = "validate"
    else:
        phase = "train"

    if phase in ["train", "validate"]:
      clipped_trans_cdf = trans_cdf.query("target_weeks+1 + {train_duration_weeks} <t-dat<target_week+1") #予測日の前週までのトランザクションが使える

      #(要議論) speed upのためにtrans_cdfに存在するart, custだけの情報を使う
      buyable_art_ids = clipped_trans_cdf['article_id'].unique()
      buyable_art_cdf = clipped_trans_cdf.query(f"art_id in {buyable_art_ids}")

      bought_cust_ids = clipped_trans_cdf['customer_id'].unique()
      bought_cust_cdf = clipped_trans_cdf.query(f"customer_id in {bought_cust_ids}")


      base_cols = [
          "customer_id",
          "article_id",
          "candidate_type",
          "target_week",
          "target",
      ]


      candidate_df = pd.DataFrame()

      #いろんなメソッドを使って候補生成
      for candidate_block in blocks:
          block_df = candidate_block(clipped_trans_cdf, buyable_art_cdf, bought_cust_cdf, target_customers = buyable_art_ids)
          candidate_df = pd.concat([candidate_df, block_df])

      candidate_df = sampling(candidate_df, clipped_trans_cdf) #適当に間引くはず

      # (Pending) 候補生成メソッドで取ってこれなかった正例を全部ブッコム
      # positive_df = get_postive_pair(clipped_trans_cdf, columns=base_cols)
      # positive_df["target"] = 1

      # base_df = pd.concat([candidate_df, positive_df]).drop_duplicates(keep='first')
    else:
      # 推論候補集合取得
      # クソデカになる & バッチで処理したいので後ほど作成でも良さそう。
      # ここの実装の解像度がまだ低い
      base_df = get_inference_data()

  # 特徴量作成
  base_df = base_df.merge(base_df, cust_feat_df, how="left", on="customer_id")
  base_df = base_df.merge(base_df, art_feat_df, how="left", on="article_id")
  base_df = base_df.merge(
      base_df, pair_feat_df, how="left", on=["article_id", "customer_id"]
  )

  if phase == "train":
      # 学習データは複数予測週があるのでconcatする
      train_df = pd.concat([train_df, base_df])
  elif phase == "validate":
      valid_df = base_df.copy()
  else:
      test_df = base_df.copy()


clf = lgb.LGBMRanker()
clf.fit(train_df,valid_df)
clf.predict(test_df)
