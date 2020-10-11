import pandas as pd
import numpy as np

from df_transformer import DfTransformer

class FeatureTypeTransformer(DfTransformer):
    """
        Input a list of features that needs to change type.
        If num_to_str is TRUE then input features are expected to be numeric and needs to be changed to str
        Otherwise if num_to_str is FALSE then input features are excpected to be str and needs to be changed to numeric
    """
    def __init__(self, feature_list=[], num_to_str = True):
        self.name = "FeatureTypeTransformer"
        super().log_start(self.name)
        
        self.feature_list = feature_list
        self.num_to_str = num_to_str
        self.columns = []
    
    def fit(self, X,y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = self.type_transformer(X)
        
        self.columns = X_transformed.columns
        super().log_end(self.name)
        return X_transformed
    
    def type_transformer(self, X):
        if self.num_to_str:
            for feature in self.feature_list:
                X[feature] = X[feature].astype(str)
        else:
            for feature in self.feature_list:
                X[feature] = X[feature].astype('int32')            
        return X
    
    def get_feature_names(self):
        return self.columns