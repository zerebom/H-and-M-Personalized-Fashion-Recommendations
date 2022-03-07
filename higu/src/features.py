import pandas as pd
import cudf

to_cdf = cudf.DataFrame.from_pandas

class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        raise NotImplementedError()


class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(self,
                 key_col,
                 target_col,
                 agg_list):

        self.key_col = key_col
        self.target_col = target_col
        self.agg_list = agg_list

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        out_df = input_df.groupby(
            self.key_col)[
            self.target_col].agg(
            self.agg_list)
        out_df.columns = ['_'.join([self.target_col, col])
                          for col in out_df.columns]

        out_df.reset_index(inplace=True)

        return out_df


class ModeCategoryBlock(AbstractBaseBlock):
    # key_colごとにtarget_colのmodeを算出
    def __init__(self, key_col, target_col):
        self.key_col = key_col
        self.target_col = target_col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame):
        col_name = "most_cat_" + self.target_col
        out_df = input_df.groupby([self.key_col, self.target_col])\
            .size()\
            .reset_index()\
            .sort_values([self.key_col, 0], ascending=False)\
            .groupby(self.key_col)[self.target_col]\
            .nth(1).\
            reset_index().rename({self.target_col: col_name} , axis=1)
        out_df[col_name] = out_df[col_name].astype('category')
        return out_df  # %%
