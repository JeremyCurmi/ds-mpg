import pandas as pd

from df_transformer import DfTransformer
from sklearn.impute import SimpleImputer


class Imputer(DfTransformer):
    """
        Builds on SimpleImputer, keeps the dataframe structure, available methods:
        1. mean
        2. median
        3. most_frequent
        4. constant
        Note: When method -> “constant”, if fill_value is left to default value, fill_value 
        will be 0 for numerical data and “missing_value” for categorical data.

    """
    def __init__(self, method = "median", fill_value=None):
        self.name = "Imputer"
        super().log_start(self.name)

        self.method = method
        self.fill_value = fill_value
        self.imputer = None
        self.statistics_ = None


    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy = self.method, fill_value= self.fill_value)
        self.imputer.fit(X)
        self.statistics_ = pd.Series(self.imputer.statistics_, index = X.columns)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, index = X.index, columns = X.columns)
        
        super().log_end(self.name)
        return X_imputed_df

