import cudf
import gc


def calc_pairs(train):
    dt = (
        train.groupby(["customer_id", "t_dat"])["article_id"]
        .agg(list)
        .rename("pair")
        .reset_index()
    )
    df = train[["customer_id", "t_dat", "article_id"]].merge(
        dt, on=["customer_id", "t_dat"], how="left"
    )
    del dt
    df = cudf.from_pandas(df[["article_id", "pair"]].to_pandas().explode(column="pair"))

    # df = df.loc[df['article_id']!=df['pair']].reset_index(drop=True)
    gc.collect()

    # Count how many times each pair combination happens
    df = df.groupby(["article_id", "pair"]).size().rename("count").reset_index()
    gc.collect()

    # Sort by frequency
    df = df.sort_values(["article_id", "count"], ascending=False).reset_index(drop=True)
    gc.collect()

    df = df.to_pandas()
    # Pick only top1 most frequent pair
    df["rank"] = df.groupby("article_id")["pair"].cumcount()
    df = df.loc[df["rank"] == 0].reset_index(drop=True)
    del df["rank"]
    gc.collect()
    df.to_csv(input_dir / "pair_df.csv", index=False)

    return df
