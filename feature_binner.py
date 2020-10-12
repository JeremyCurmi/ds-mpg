from sklearn.preprocessing import KBinsDiscretizer
from df_transformer import DfTransformer

class FeatureKBinner(DfTransformer):
    
    def __init__(self, n_bins=5, encode='onehot', strategy='quantile'):
        self.name = "FeatureKBinner"
        self.log_start(self.name)

        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.kbinner = KBinsDiscretizer(self.n_bins,self.encode,self.strategy)
        self.columns = []
        
    def fit(self, X, y=None):
        self.kbinner.fit(X)
        return self
    
    def transform(self, X, y=None):
        X_binned = self.kbinner.transform(X)
        self.log_end(self.name)
        return X_binned
