
from df_transformer import DfTransformer


class FeatureRemover(DfTransformer):
    
    def __init__(self, feature_list):
        self.name = "FeatureRemover"
        super().log_start(self.name)
        
        self.feature_list = feature_list
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for feature in self.feature_list:
            del X[feature]
        
        super().log_end(self.name)
        return X
    