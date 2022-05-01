target_weeks = [104, 103, 102, 101, 100]  # test_yから何週間離れているか

train_df, valid_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def make_train_valid_df():
    train_df, valid_df = pd.DataFrame(), pd.DataFrame()
    for target_week in target_weeks:
        phase = "validate" if target_week == 104 else "train"

        clipped_trans_cdf = trans_cdf.query(
            f"target_week - {train_duration_weeks} <= week < target_week"
        )  # 予測日の前週までのトランザクションが使える

        # (要議論) speed upのためにclipped_trans_cdfに存在するart, custだけの情報を使う
        buyable_art_ids = clipped_trans_cdf["article_id"].unique()
        buyable_art_cdf = art_cdf.query(f"art_id in {buyable_art_ids}")

        bought_cust_ids = clipped_trans_cdf["customer_id"].unique()
        bought_cust_cdf = cust_cdf.query(f"customer_id in {bought_cust_ids}")

        y_cdf = make_target_df(clipped_trans_cdf)

        # いろんなメソッドを使って候補生成
        base_df = candidate_generation(
            candidate_blocks,
            clipped_trans_cdf,
            buyable_art_cdf,
            bought_cust_cdf,
            y_cdf,
            target_customers=None,
        )

        base_df = negative_sampling(base_df, clipped_trans_cdf, y_cdf)  # 適当に間引くはず

        art_feat_df, cust_feat_df, pair_feat_df = feature_generation(
            feature_blocks,
            clipped_trans_cdf,
            buyable_art_cdf,
            bought_cust_cdf,  # base_dfに入っているuser_id,article_idが必要
        )

        # 特徴量作成
        base_df = base_df.merge(cust_feat_df, how="left", on="customer_id")
        base_df = base_df.merge(art_feat_df, how="left", on="article_id")
        base_df = base_df.merge(pair_feat_df, how="left", on=["article_id", "customer_id"])
        base_df["target_week"] = target_week
        base_df["y"] = (
            base_df.merge(y_cdf, how="left", on=["customer_id", "article_id"])["y"]
            .fillna(0)
            .astype(int)
        )

        # base_df.columns ==  [
        #     "customer_id",
        #     "article_id",
        #     "candidate_block_name",
        #     "target_week",
        #     "target",
        # ] + [feature_columns]

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

    train_df["query_id"] = train_df.groupby(["customer_id", "target_week"]).size().values
    valid_df["query_id"] = valid_df.groupby(["customer_id", "target_week"]).size().values

    return train_df, valid_df


clf = lgb.LGBMRanker()
clf.fit(train_df, valid_df)
clf.predict(test_df)

base_df = get_inference_data()
