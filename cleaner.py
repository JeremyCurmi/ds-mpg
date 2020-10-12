import pandas as pd
import numpy as np

from df_transformer import DfTransformer

class Cleaner(DfTransformer):
    """
        Data cleaning before preprocessing
    """
    
    def __init__(self):
        self.name = "Cleaner"
        super().log_start(self.name)    

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_clean = self.clean_features(X)

        super().log_end(self.name)        
        return X_clean

    def clean_features(self, X):
        X["horsepower"] = pd.to_numeric(X["horsepower"], errors = "coerce")
        X["weight"] = X["weight"].astype("float32")
        return X
