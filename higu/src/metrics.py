import numpy as np
import pandas as pd
from utils import customer_hex_id_to_int


def calc_map12(valid_df, logger, path_to_valid_true):
    valid_df.sort_values(["customer_id", "prediction"], ascending=False, inplace=True)
    valid_pred = valid_df.groupby("customer_id")[["article_id"]].aggregate(
        lambda x: x.tolist()
    )
    valid_pred["prediction"] = valid_pred["article_id"].apply(
        lambda x: " ".join(["0" + str(k) for k in x])
    )

    valid_true = pd.read_csv(path_to_valid_true)[["customer_id", "valid_true"]]
    valid_true["customer_id"] = customer_hex_id_to_int(valid_true["customer_id"])
    valid_true = valid_true.merge(
        valid_pred.reset_index(), how="left", on="customer_id"
    )

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


def calculate_apk(list_of_preds, list_of_gts):
    # for fast validation this can be changed to operate on dicts of {'cust_id_int': [art_id_int, ...]}
    # using 'data/val_week_purchases_by_cust.pkl'
    apks = []
    for preds, gt in zip(list_of_preds, list_of_gts):
        apks.append(apk(gt, preds, k=12))
    return np.mean(apks)


def eval_sub(sub_csv, skip_cust_with_no_purchases=True):
    sub = pd.read_csv(sub_csv)
    validation_set = pd.read_parquet("data/validation_ground_truth.parquet")

    apks = []

    no_purchases_pattern = []
    for pred, gt in zip(
        sub.prediction.str.split(), validation_set.prediction.str.split()
    ):
        if skip_cust_with_no_purchases and (gt == no_purchases_pattern):
            continue
        apks.append(apk(gt, pred, k=12))
    return np.mean(apks)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
