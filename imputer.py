import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class Imputer(BaseEstimator, TransformerMixin):
    """
        Builds on SimpleImputer, keeps the dataframe structure, available methods:

    """
    def __init__(self, method = "median"):
        self.method = method
        self.imputer = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy = self.method)
        self.imputer.fit(X)
        self.statistics_ = pd.Series(self.imputer.statistics_, index = X.columns)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, index = X.index, columns = X.columns)
        return X_imputed_df

