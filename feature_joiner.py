import warnings

from sklearn.pipeline import FeatureUnion

from df_transformer import DfTransformer


class FeatureUnion(DfTransformer):
    """

    """
    def __init__(self, transformer_list, n_jobs=-1):
        self.name = "FeatureUnion"
        super().log_start(self.name)

        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.feature_union = FeatureUnion(self.transformer_list, self.n_jobs)

    def fit(self, X, y=None):
        self.feature_union.fit(X)
        return self

    def transform(self, X, y=None):
        X_transform = self.feature_union.transform(X)
        
        super().log_end(self.name)