import pandas as pd
import warnings

from df_transformer import DfTransformer


class CategoricalEncoder(DfTransformer):
    """
        Builds on pandas get_dummies
    """
    def __init__(self, drop_first = False, match_cols = True):
        self.name = "CategoricalEncoder"
        super().log_start(self.name)

        self.drop_first = drop_first
        self.columns = []
        self.match_cols = match_cols

    def fit(self, X, y=None):
        self.columns = []
        return self
    
    def transform(self, X):
        X_dummies = pd.get_dummies(X, drop_first = self.drop_first)
        
        if len(self.columns) > 0:
            if self.match_cols:
                X_dummies = self.match_columns(X_dummies)
            self.columns = X_dummies.columns
        else:
            self.columns = X_dummies.columns

        super().log_end(self.name)
        return X_dummies

    def match_columns(self, X):
        cols_missing_from_train = list(set(X.columns) - set(self.columns))
        cols_missing_from_test = list(set(self.columns) - set(X.columns))

        missing_features = False

        X_matching_features = X.copy()

        if len(cols_missing_from_train) > 0:
            for feature in cols_missing_from_train:
                del X_matching_features[feature]
                missing_features = True

        if len(cols_missing_from_test) > 0:
            for feature in cols_missing_from_test:
                X_matching_features[feature] = 0 
                missing_features = True

        if missing_features:
            warnings.warn("Certain Features in the Test df do not match the ones in the Training df. This issue was resolved automatically.",UserWarning)
        
        return X_matching_features

    def get_feature_names(self):
        return self.columns