import pandas as pd
import cudf
from tqdm import tqdm

to_cdf = cudf.DataFrame.from_pandas

def candidates_dict2df(candidates_dict):
    tuples = []
    for cust, articles in tqdm(candidates_dict.items()):
        for art in articles:
            tuples.append((cust, art))

    df = pd.DataFrame(tuples)
    df.columns = ['customer_id', 'article_id']
    cdf = to_cdf(df)
    cdf = cdf.drop_duplicates().reset_index(drop=True)
    return cdf

class AbstractCGBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        raise NotImplementedError()

    def get_cudf(self):
        return to_cdf(self.out_df)


class LastNWeekArticles(AbstractCGBlock):
    """
    各ユーザ、最後に買った週から2週間前までの購入商品を取得する
    """
    def __init__(
            self,
            key_col='customer_id',
            item_col='article_id',
            n_weeks=2):
        self.key_col = key_col
        self.item_col = item_col
        self.n_weeks = n_weeks
        self.out_keys = [key_col, item_col]

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        input_df = _input_df.copy()

        week_df = input_df.\
            groupby(self.key_col)['week'].max()\
            .reset_index().\
            rename({'week': 'max_week'}, axis=1)

        input_df = input_df.merge(week_df, on=self.key_col, how='left')
        input_df['diff_week'] = input_df['max_week'] - input_df['week']
        self.out_df = input_df.loc[input_df['diff_week']
                                   <= self.n_weeks, self.out_keys].to_pandas()

        out_dic = self.out_df.groupby(self.key_col)[
            self.item_col].apply(list).to_dict()
        return out_dic



class BoughtItemsAtInferencePhase(AbstractCGBlock):
    def __init__(
            self,
            y_cdf,
            key_col='customer_id',
            item_col='article_id',
            ):

        self.y_cdf = y_cdf
        self.key_col = key_col
        self.item_col = item_col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        out_df = self.y_cdf\
            .to_pandas()\
            .groupby(self.key_col)[self.item_col]\
            .apply(list).to_dict()

        return out_df


class PopularItemsoftheLastWeeks(AbstractCGBlock):
    def __init__(
            self,
            customer_ids,
            key_col='customer_id',
            item_col='article_id',
            n=12):
        self.customer_ids = customer_ids
        self.key_col = key_col
        self.item_col = item_col
        self.n = n

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, _input_df):
        input_df = _input_df.copy()
        pop_item_dic = {}
        self.popular_items = input_df\
            .loc[input_df['week'] == input_df['week'].max()][self.item_col]\
            .value_counts()\
            .to_pandas()\
            .index[:self.n]\
            .values

        for cust_id in self.customer_ids:
            pop_item_dic[cust_id] = list(self.popular_items)

        return pop_item_dic
