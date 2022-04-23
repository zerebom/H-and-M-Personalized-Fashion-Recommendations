target_weeks = [104,103,102,101,100]  # test_yから何週間離れているか

train_df, valid_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def make_train_valid_df():
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for target_week in target_weeks:
        phase = "validate" if target_week == 104 else "train"

        clipped_trans_cdf = trans_cdf.query(f"target_week - {train_duration_weeks} <= week < target_week") #予測日の前週までのトランザクションが使える
        clipped_trans_cdf["target_week"] = clipped_trans_cdf["week"] + 1

        #(要議論) speed upのためにclipped_trans_cdfに存在するart, custだけの情報を使う
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

        y_cdf = make_target_df(clipped_trans_cdf)

        #いろんなメソッドを使って候補生成
        base_df = candidate_generation(
            candidate_blocks, clipped_trans_cdf, buyable_art_cdf, bought_cust_cdf, y_cdf, target_customers = buyable_art_ids
        )

        base_df = negative_sampling(base_df, clipped_trans_cdf, y_cdf) #適当に間引くはず


        art_feat_df, cust_feat_df, pair_feat_df = feature_generation(
        feature_blocks, clipped_trans_cdf, buyable_art_cdf, bought_cust_cdf, target_customers = buyable_art_ids
    )

        # 特徴量作成
        base_df = base_df.merge(cust_feat_df, how="left", on="customer_id")
        base_df = base_df.merge(art_feat_df, how="left", on="article_id")
        base_df = base_df.merge(pair_feat_df, how="left", on=["article_id", "customer_id"]
        )

        if phase == "train":
            # 学習データは複数予測週があるのでconcatする
            train_df = pd.concat([train_df, base_df])
        elif phase == "validate":
            valid_df = base_df.copy()

    train_df['query_id'] = train_df.groupby(['customer_id','target_week']).size().values
    valid_df['query_id'] = valid_df.groupby(['customer_id','target_week']).size().values

    return train_df, valid_df

clf = lgb.LGBMRanker()
clf.fit(train_df,valid_df)
clf.predict(test_df)

base_df = get_inference_data()
