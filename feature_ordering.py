
from df_transformer import DfTransformer


class FeatureOrderer(DfTransformer):
    
    def __init__(self):
        self.name = "FeatureOrderer"
        self.log_start( self.name)
        self.columns = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ordered = X.sort_index(axis=1)
        self.columns = X_ordered.columns 
        
        self.log_end(self.name)
        return X_ordered

    def get_feature_names(self):
        return self.columns