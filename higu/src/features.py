import cudf
import pandas as pd

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


class TargetRollingBlock(TargetEncodingBlock):
    def __init__(self,
                 key_cols,
                 group_col,
                 target_col,
                 agg_list,
                 windows):

        super().__init__(key_cols, target_col, agg_list)
        self.windows = windows
        self.group_col = group_col

    def transform(self, input_df: pd.DataFrame):
        out_df = input_df[self.key_col].copy()
        for window in self.windows:
            for agg_func in self.agg_list:
                col = f'{self.target_col}_{window}_rolling_{agg_func}'
                out_df[col] = input_df.groupby(self.group_col)[self.target_col].rolling(window,center=False,min_periods=1).apply(agg_func).reset_index(0)[self.target_col]

        out_df.reset_index(inplace=True, drop=True)
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
