import warnings

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTypeSelector(BaseEstimator, TransformerMixin):
    """
        Selects Features based on their type
    """

    def __init__(self, feature_type = "numeric"):
        self.feature_type = feature_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.feature_type == "numeric":
            num_features = X.columns[X.dtypes != object].tolist()
            return X[num_features]
        elif self.dtype == "categorical":
            cat_features = X.columns[X.dtypes == object].tolist()
            return X[cat_features]


class FeatureSelector(BaseEstimator,TransformerMixin):
    """
        Selects Features based on input list and type of feature, 
        its important that all features in the given feature list 
        must be of the same type.
    """
    def __init__(self, feature_type = "numeric", feature_list = []):
        self.feature_type = feature_type
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        if self.feature_list == []:
            X_subset = self.return_df_subset(X)
        else:
            X_subset = self.return_df_subset(X[self.feature_list])
            if len(X_subset.columns.tolist()) != len(self.feature_list):
                warnings.warn("Certain Features from the provided feature list were not found bu the Feature Selector, make sure that all features in the feature list are of the same type",UserWarning)
            
        return X_subset              

    def return_df_subset(self, X):
        if self.feature_type == "numeric":
            num_features = X.columns[X.dtypes != object].tolist()
            return X[num_features]
        elif self.dtype == "categorical":
            cat_features = X.columns[X.dtypes == object].tolist()
            return X[cat_features] 